//! Interactive raster preview for the 3D material view. Settled frames still use the CPU
//! renderer, which remains the image reference and the portable fallback.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::wgpu::util::DeviceExt as _;
use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor, wgpu};
use materialcraft_geom::glam::Vec4;
use materialcraft_render::{Lighting, Maps, Scene, Settings};

const SHADER: &str = include_str!("raster_preview.wgsl");
pub const GPU_RAYTRACE_MAE_TOLERANCE: f32 = 0.20;

pub struct RasterTarget {
    pub size: [u32; 2],
    pub texture_id: egui::TextureId,
    pub view: wgpu::TextureView,
    pub depth_view: wgpu::TextureView,
    _texture: wgpu::Texture,
    _depth: wgpu::Texture,
    pub timestamp_query: Option<wgpu::QuerySet>,
    pub timestamp_resolve: Option<wgpu::Buffer>,
    pub timestamp_readback: Option<wgpu::Buffer>,
    pub gpu_timing_supported: bool,
}

impl RasterTarget {
    pub fn new(rs: &egui_wgpu::RenderState, size: [u32; 2]) -> Self {
        let size = [size[0].max(1), size[1].max(1)];
        let texture = rs.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("materialcraft raster preview output"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let depth = rs.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("materialcraft raster preview depth"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());
        let texture_id = rs.renderer.write().register_native_texture(&rs.device, &view, wgpu::FilterMode::Linear);
        let timestamp_features = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let gpu_timing_supported = rs.device.features().contains(timestamp_features);
        let (timestamp_query, timestamp_resolve, timestamp_readback) = if gpu_timing_supported {
            let query = rs.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("materialcraft GPU frame timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = rs.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("materialcraft GPU timestamp resolve"),
                size: 256,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = rs.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("materialcraft GPU timestamp readback"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            (Some(query), Some(resolve), Some(readback))
        } else {
            (None, None, None)
        };
        Self {
            size,
            texture_id,
            view,
            depth_view,
            _texture: texture,
            _depth: depth,
            timestamp_query,
            timestamp_resolve,
            timestamp_readback,
            gpu_timing_supported,
        }
    }

    /// Read the most recent raster frame after it has been rendered. This is only used by the
    /// opt-in preview-vs-ray-tracer check after interaction settles.
    pub fn readback_rgba(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<u8>, String> {
        let row_bytes = self.size[0].saturating_mul(4);
        let padded_row_bytes = row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer_size = u64::from(padded_row_bytes) * u64::from(self.size[1]);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("materialcraft GPU preview comparison readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("materialcraft GPU preview comparison copy") });
        encoder.copy_texture_to_buffer(
            self._texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_row_bytes), rows_per_image: Some(self.size[1]) },
            },
            wgpu::Extent3d { width: self.size[0], height: self.size[1], depth_or_array_layers: 1 },
        );
        let submission = queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result.map_err(|e| e.to_string()));
        });
        device
            .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(std::time::Duration::from_secs(5)) })
            .map_err(|e| e.to_string())?;
        rx.recv_timeout(std::time::Duration::from_secs(5)).map_err(|e| e.to_string())??;
        let view = buffer.slice(..).get_mapped_range().map_err(|e| e.to_string())?;
        let mut pixels = Vec::with_capacity((self.size[0] * self.size[1] * 4) as usize);
        for y in 0..self.size[1] as usize {
            let start = y * padded_row_bytes as usize;
            pixels.extend_from_slice(&view[start..start + row_bytes as usize]);
        }
        drop(view);
        buffer.unmap();
        Ok(pixels)
    }

    /// Resolve the latest raster pass and return its elapsed GPU time in milliseconds.
    pub fn readback_gpu_frame_ms(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<f32, String> {
        let (Some(query), Some(resolve), Some(readback)) = (&self.timestamp_query, &self.timestamp_resolve, &self.timestamp_readback) else {
            return Err("Vulkan timestamp queries are unavailable on this adapter".into());
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("materialcraft GPU timestamp readback") });
        encoder.resolve_query_set(query, 0..2, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 16);
        let submission = queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..16).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result.map_err(|e| e.to_string()));
        });
        device
            .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(std::time::Duration::from_secs(5)) })
            .map_err(|e| e.to_string())?;
        rx.recv_timeout(std::time::Duration::from_secs(5)).map_err(|e| e.to_string())??;
        let view = readback.slice(..16).get_mapped_range().map_err(|e| e.to_string())?;
        let start = u64::from_le_bytes(view[0..8].try_into().map_err(|_| "timestamp start is truncated")?);
        let end = u64::from_le_bytes(view[8..16].try_into().map_err(|_| "timestamp end is truncated")?);
        drop(view);
        readback.unmap();
        let elapsed_ms = end.wrapping_sub(start) as f64 * f64::from(queue.get_timestamp_period()) / 1_000_000.0;
        if elapsed_ms.is_finite() && elapsed_ms >= 0.0 { Ok(elapsed_ms as f32) } else { Err("GPU timestamp result was not finite".into()) }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    tangent: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    config: [f32; 4], // tiling, exposure stops, environment rotation radians, show environment
    sh: [[f32; 4]; 9],
}

struct RasterResources {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniforms: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    mesh_id: usize,
    map_id: [usize; 8],
    env_name: String,
    _mesh: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    _maps: Vec<wgpu::Texture>,
    _env: Option<wgpu::Texture>,
}

impl RasterResources {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("materialcraft interactive raster preview"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("materialcraft preview resources"),
            entries: &[
                uniform_entry(0),
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                texture_entry(4),
                texture_entry(5),
                texture_entry(6),
                texture_entry(7),
                texture_entry(8),
                sampler_entry(9),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("materialcraft preview pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("materialcraft PBR preview"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                })],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("materialcraft preview uniforms"),
            size: size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("materialcraft preview sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            layout,
            sampler,
            uniforms,
            bind_group: None,
            mesh_id: 0,
            map_id: [0; 8],
            env_name: String::new(),
            _mesh: None,
            _maps: Vec::new(),
            _env: None,
        }
    }

    fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, call: &RasterCallback) {
        let scene_id = Arc::as_ptr(&call.scene) as usize;
        if scene_id != self.mesh_id {
            let vertices: Vec<_> = (0..call.scene.mesh.positions.len())
                .map(|i| Vertex {
                    position: call.scene.mesh.positions[i].to_array(),
                    normal: call.scene.mesh.normals.get(i).copied().unwrap_or(materialcraft_geom::glam::Vec3::Y).to_array(),
                    uv: call.scene.mesh.uvs.get(i).copied().unwrap_or_default().to_array(),
                    tangent: call.scene.mesh.tangents.get(i).copied().unwrap_or(Vec4::X).to_array(),
                })
                .collect();
            let indices = &call.scene.mesh.indices;
            let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("materialcraft preview mesh vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("materialcraft preview mesh indices"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            self._mesh = Some((vertex, index, indices.len() as u32));
            self.mesh_id = scene_id;
            self.bind_group = None;
        }
        let map_id = map_ids(&call.maps);
        if map_id != self.map_id {
            self._maps = [
                (&call.maps.base_color, [0.5, 0.5, 0.5, 1.0]),
                (&call.maps.roughness, [0.5, 0.5, 0.5, 1.0]),
                (&call.maps.metallic, [0.0, 0.0, 0.0, 1.0]),
                (&call.maps.ao, [1.0, 1.0, 1.0, 1.0]),
                (&call.maps.normal, [0.5, 0.5, 1.0, 1.0]),
                (&call.maps.emissive, [0.0, 0.0, 0.0, 1.0]),
                (&call.maps.opacity, [1.0, 1.0, 1.0, 1.0]),
            ]
            .into_iter()
            .map(|(map, fallback)| upload_map(device, queue, map.as_ref(), fallback))
            .collect();
            self.map_id = map_id;
            self.bind_group = None;
        }
        if self.env_name != call.light.name {
            self._env = Some(upload_environment(device, queue, &call.light));
            self.env_name.clone_from(&call.light.name);
            self.bind_group = None;
        }
        let camera = materialcraft_render::camera(&call.scene, &call.settings);
        let mut sh = [[0.0; 4]; 9];
        for (dst, src) in sh.iter_mut().zip(call.light.sh) {
            *dst = [src[0], src[1], src[2], 0.0];
        }
        let uniforms = Globals {
            view_proj: camera.view_proj(call.size[0] as f32 / call.size[1].max(1) as f32).to_cols_array_2d(),
            eye: camera.eye().extend(1.0).to_array(),
            config: [call.settings.tiling, call.settings.exposure, call.settings.env_rotation.to_radians(), if call.settings.show_env { 1.0 } else { 0.0 }],
            sh,
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniforms));
        if self.bind_group.is_none() {
            let Some(env) = self._env.as_ref() else { return };
            let views: Vec<_> = self._maps.iter().map(|t| t.create_view(&Default::default())).collect();
            let env_view = env.create_view(&Default::default());
            let entries: Vec<_> = std::iter::once(wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.as_entire_binding() })
                .chain(views.iter().enumerate().map(|(i, v)| wgpu::BindGroupEntry { binding: i as u32 + 1, resource: wgpu::BindingResource::TextureView(v) }))
                .chain(std::iter::once(wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&env_view) }))
                .chain(std::iter::once(wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&self.sampler) }))
                .collect();
            self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("materialcraft PBR preview bindings"),
                layout: &self.layout,
                entries: &entries,
            }));
        }
        let (Some((vertex, index, count)), Some(bind_group)) = (&self._mesh, &self.bind_group) else { return };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("materialcraft raster preview"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &call.target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.105, g: 0.11, b: 0.12, a: 1.0 }), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &call.depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: call.timestamp_query.as_ref().map(|query_set| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, vertex.slice(..));
        pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..*count, 0, 0..1);
        drop(pass);
        if let (Some(query), Some(resolve), Some(readback)) = (&call.timestamp_query, &call.timestamp_resolve, &call.timestamp_readback) {
            encoder.resolve_query_set(query, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 16);
        }
    }
}

pub struct RasterCallback {
    pub scene: Arc<Scene>,
    pub maps: Maps,
    pub light: Arc<Lighting>,
    pub settings: Settings,
    pub size: [u32; 2],
    pub target: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    pub timestamp_query: Option<wgpu::QuerySet>,
    pub timestamp_resolve: Option<wgpu::Buffer>,
    pub timestamp_readback: Option<wgpu::Buffer>,
}

impl CallbackTrait for RasterCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut state = resources.remove::<RasterResources>().unwrap_or_else(|| RasterResources::new(device));
        state.prepare(device, queue, encoder, self);
        resources.insert(state);
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, _pass: &mut wgpu::RenderPass<'static>, _resources: &CallbackResources) {}
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn map_ids(m: &Maps) -> [usize; 8] {
    [
        m.base_color.as_ref(),
        m.roughness.as_ref(),
        m.metallic.as_ref(),
        m.ao.as_ref(),
        m.normal.as_ref(),
        m.emissive.as_ref(),
        m.opacity.as_ref(),
        m.height.as_ref(),
    ]
    .map(|map| map.map_or(0, |i| Arc::as_ptr(&i.data) as usize))
}

fn upload_map(device: &wgpu::Device, queue: &wgpu::Queue, image: Option<&materialcraft_image::Image>, fallback: [f32; 4]) -> wgpu::Texture {
    let (w, h) = image.map_or((1, 1), |i| (i.w.min(1024), i.h.min(1024)));
    let pixels: Vec<u8> = (0..(w * h) as usize)
        .flat_map(|i| {
            let rgba = image.map_or(fallback, |img| {
                let x = i as u32 % w;
                let y = i as u32 / w;
                img.get(x as i64, y as i64, true)
            });
            rgba.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        })
        .collect();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("materialcraft material map"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        &pixels,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    texture
}

fn upload_environment(device: &wgpu::Device, queue: &wgpu::Queue, lighting: &Lighting) -> wgpu::Texture {
    let levels = lighting.specular.len().max(1) as u32;
    let first = lighting.specular.first();
    let (w, h) = first.map_or((16, 8), |e| (e.width, e.height));
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("materialcraft split-sum IBL environment"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, env) in lighting.specular.iter().enumerate() {
        let packed: Vec<u16> = env.data.iter().flat_map(|p| [p[0], p[1], p[2], 1.0].map(half::f16::from_f32).map(half::f16::to_bits)).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: level as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&packed),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(env.width * 8), rows_per_image: Some(env.height) },
            wgpu::Extent3d { width: env.width, height: env.height, depth_or_array_layers: 1 },
        );
    }
    texture
}

pub fn map_average_pixel_error(a: &[u8], b: &[u8]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let sum: u64 = a.iter().zip(b).map(|(x, y)| u64::from(x.abs_diff(*y))).sum();
    Some(sum as f32 / a.len() as f32 / 255.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_difference_is_normalized_per_channel() {
        assert_eq!(map_average_pixel_error(&[0, 64, 128, 255], &[0, 64, 128, 255]), Some(0.0));
        assert_eq!(map_average_pixel_error(&[0, 0, 0, 0], &[255, 255, 255, 255]), Some(1.0));
        assert_eq!(map_average_pixel_error(&[1, 2], &[1]), None);
    }
}
