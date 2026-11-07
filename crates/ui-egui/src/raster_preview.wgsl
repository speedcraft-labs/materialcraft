struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    config: vec4<f32>,
    sh: array<vec4<f32>, 9>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var base_map: texture_2d<f32>;
@group(0) @binding(2) var rough_map: texture_2d<f32>;
@group(0) @binding(3) var metal_map: texture_2d<f32>;
@group(0) @binding(4) var ao_map: texture_2d<f32>;
@group(0) @binding(5) var normal_map: texture_2d<f32>;
@group(0) @binding(6) var emissive_map: texture_2d<f32>;
@group(0) @binding(7) var opacity_map: texture_2d<f32>;
@group(0) @binding(8) var environment_map: texture_2d<f32>;
@group(0) @binding(9) var material_sampler: sampler;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip = g.view_proj * vec4<f32>(v.position, 1.0);
    out.world = v.position;
    out.normal = v.normal;
    out.uv = v.uv * g.config.x;
    out.tangent = v.tangent;
    return out;
}

fn env_uv(d0: vec3<f32>) -> vec2<f32> {
    let c = cos(g.config.z);
    let s = sin(g.config.z);
    let d = vec3<f32>(c * d0.x - s * d0.z, d0.y, s * d0.x + c * d0.z);
    let phi = atan2(d.z, d.x);
    return vec2<f32>(fract(phi / 6.2831853 + 1.0), acos(clamp(d.y, -1.0, 1.0)) / 3.14159265);
}

fn irradiance(n0: vec3<f32>) -> vec3<f32> {
    let c = cos(g.config.z);
    let s = sin(g.config.z);
    let n = vec3<f32>(c * n0.x - s * n0.z, n0.y, s * n0.x + c * n0.z);
    let x = n.x;
    let y = n.y;
    let z = n.z;
    var r = g.sh[0].xyz * 0.282095;
    r += g.sh[1].xyz * (0.488603 * y);
    r += g.sh[2].xyz * (0.488603 * z);
    r += g.sh[3].xyz * (0.488603 * x);
    r += g.sh[4].xyz * (1.092548 * x * y);
    r += g.sh[5].xyz * (1.092548 * y * z);
    r += g.sh[6].xyz * (0.315392 * (3.0 * z * z - 1.0));
    r += g.sh[7].xyz * (1.092548 * x * z);
    r += g.sh[8].xyz * (0.546274 * (x * x - y * y));
    return max(r, vec3<f32>(0.0));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn env_brdf(roughness: f32, ndv: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    return vec2<f32>(a004 * -1.04 + r.z, a004 * 1.04 + r.w);
}

@fragment
fn fs_main(v: VertexOut) -> @location(0) vec4<f32> {
    let uv = v.uv;
    let base = textureSample(base_map, material_sampler, uv).rgb;
    let rough = clamp(textureSample(rough_map, material_sampler, uv).r, 0.02, 1.0);
    let metal = clamp(textureSample(metal_map, material_sampler, uv).r, 0.0, 1.0);
    let ao = clamp(textureSample(ao_map, material_sampler, uv).r, 0.0, 1.0);
    let map_n = textureSample(normal_map, material_sampler, uv).xyz * 2.0 - 1.0;
    let n0 = normalize(v.normal);
    let t = normalize(v.tangent.xyz - n0 * dot(n0, v.tangent.xyz));
    let b = normalize(cross(n0, t)) * select(-1.0, 1.0, v.tangent.w >= 0.0);
    var n = normalize(t * map_n.x - b * map_n.y + n0 * max(map_n.z, 0.05));
    let view = normalize(g.eye.xyz - v.world);
    if dot(n, view) < 0.0 {
        n = n0;
    }
    let nov = max(dot(n, view), 0.001);
    let reflected = reflect(-view, n);
    let env_spec = textureSampleLevel(environment_map, material_sampler, env_uv(reflected), rough * 5.0).rgb;
    let f0 = mix(vec3<f32>(0.04), base, metal);
    let brdf = env_brdf(rough, nov);
    let diffuse = base * (1.0 - metal) * irradiance(n);
    let specular = env_spec * (f0 * brdf.x + vec3<f32>(brdf.y));
    let emissive = textureSample(emissive_map, material_sampler, uv).rgb;
    let opacity = textureSample(opacity_map, material_sampler, uv).r;
    let linear = ((diffuse + specular) * ao + emissive) * exp2(g.config.y);
    let alpha = clamp(opacity, 0.0, 1.0);
    return vec4<f32>(aces(linear), alpha);
}
