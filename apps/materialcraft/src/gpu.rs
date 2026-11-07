//! Select the Vulkan hardware adapter for the MaterialCraft viewport.
//!
//! Agentbox has both a discrete NVIDIA adapter and llvmpipe in its Vulkan adapter list. The
//! default high-performance request is a hint; selecting explicitly keeps the software adapter
//! from winning when a hardware device can present to the window.

use std::sync::Arc;

use eframe::NativeOptions;
use eframe::egui_wgpu::WgpuSetup;
use eframe::wgpu::{self, DeviceType};

pub fn prefer_hardware_vulkan(options: &mut NativeOptions) {
    let WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup else {
        return;
    };

    setup.instance_descriptor.backends = wgpu::Backends::VULKAN;
    setup.power_preference = wgpu::PowerPreference::HighPerformance;
    let base_device_descriptor = Arc::clone(&setup.device_descriptor);
    setup.device_descriptor = Arc::new(move |adapter| {
        let mut descriptor = base_device_descriptor(adapter);
        let requested = adapter.features() & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
        descriptor.required_features |= requested;
        descriptor
    });
    setup.native_adapter_selector = Some(Arc::new(|adapters, surface| {
        let force_software = std::env::var("MATERIALCRAFT_FORCE_SOFTWARE").as_deref() == Ok("1");
        let mut candidates: Vec<_> = adapters
            .iter()
            .enumerate()
            .map(|(index, adapter)| {
                let info = adapter.get_info();
                let supported = surface.is_none_or(|surface| adapter.is_surface_supported(surface));
                (index, info, supported)
            })
            .collect();
        for (_, info, supported) in &candidates {
            log::info!("Vulkan adapter candidate: {} ({:?}, surface supported: {supported})", info.name, info.device_type);
        }
        candidates.sort_by_key(|(index, info, supported)| {
            let software = info.device_type == DeviceType::Cpu
                || ["llvmpipe", "lavapipe", "software"].iter().any(|needle| info.name.to_ascii_lowercase().contains(needle));
            let device_rank = match info.device_type {
                DeviceType::DiscreteGpu => 0,
                DeviceType::IntegratedGpu => 1,
                DeviceType::Other => 2,
                DeviceType::VirtualGpu => 3,
                DeviceType::Cpu => 4,
            };
            let nvidia_rank = if info.vendor == 0x10de { 0 } else { 1 };
            (!supported, if force_software { !software } else { software }, nvidia_rank, device_rank, *index)
        });
        let (index, info, supported) = candidates.first().ok_or_else(|| "no Vulkan adapter is available".to_string())?;
        if !supported {
            return Err("no Vulkan adapter can present to the MaterialCraft window".into());
        }
        let is_software =
            info.device_type == DeviceType::Cpu || ["llvmpipe", "lavapipe", "software"].iter().any(|needle| info.name.to_ascii_lowercase().contains(needle));
        if force_software && !is_software {
            return Err("MATERIALCRAFT_FORCE_SOFTWARE=1 was set but no software Vulkan adapter is available".into());
        }
        log::info!("selected Vulkan adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
        adapters.get(*index).cloned().ok_or_else(|| "selected Vulkan adapter disappeared".to_string())
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_hardware_is_requested_and_selector_is_installed() {
        let mut options = NativeOptions::default();
        prefer_hardware_vulkan(&mut options);
        let WgpuSetup::CreateNew(setup) = options.wgpu_options.wgpu_setup else { return };
        assert_eq!(setup.instance_descriptor.backends, wgpu::Backends::VULKAN);
        assert_eq!(setup.power_preference, wgpu::PowerPreference::HighPerformance);
        assert!(setup.native_adapter_selector.is_some());
    }
}
