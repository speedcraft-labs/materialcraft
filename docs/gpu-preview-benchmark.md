# MaterialCraft interactive GPU preview

MaterialCraft uses a wgpu raster preview while the 3D material view is being orbited. The CPU
ray tracer remains the settled-frame renderer and the portable reference/fallback. The preview
is accepted against the CPU image when normalized per-channel mean absolute error is at most
`0.20` (`GPU_RAYTRACE_MAE_TOLERANCE`). Adapters without usable wgpu support keep the CPU path.

## RTX 2080 Ti result

The issue #237 corpus journey loaded Poly Haven's CC0 `Camera_01_1k.gltf` (26,987 triangles),
used the built-in studio lighting, and orbited the same material graph at the same view size on
both revisions. The comparison used ten measured runs per revision, two warmups, and 2,000
bootstrap resamples. The report is publishable and confirms the settled CPU output was checked.

| Measurement | CPU baseline | wgpu preview | Ratio (95% CI) |
|---|---:|---:|---:|
| Interactive frame median | 5.139 ms | 24.496 µs | 0.00477 [0.00456, 0.00520] |
| Process CPU time median | 382.101 µs | 374.122 µs | 0.979 [0.809, 1.220] |

All 36 drag frames reported an active GPU preview. Vulkan timestamp query median was 24.416 µs.
The preview-to-ray-tracer image comparison measured `0.183773` MAE, within the `0.20` bound.
After the drag, both revisions produced the same settled CPU PNG: SHA-256
`6cfe3dd9737b7af7025c61e2fac5ef7e69d07d9961285f2f4d9089b5d33b09c8` at 581×366 pixels.

The measured app revisions were `dc7d5a494e9ee392ee36264651690a9c0602463f` (CPU baseline) and
`54be555b10d33431b20a8375d8924481349ba104` (wgpu preview). The latter already contains the GPU
preview implementation on the speedcraft-labs fork; this PR adds its reproducible benchmark
record. The test harness and raw sidecars are in [benchmarks/issue-237](../benchmarks/issue-237/).

The 95% interval for process CPU time includes 1.0, so this run establishes a lower interactive
frame latency rather than a reduction in total process CPU use.
