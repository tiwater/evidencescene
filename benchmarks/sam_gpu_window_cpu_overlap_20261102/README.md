# SAM2 GPU 执行窗口内的同线程 CPU 准备重叠（零后台线程）

## 1. 既有 Producer 与状态 Owner 边界

本实验严格遵循 `AGENTS.md` 与 `docs/plans/2026-11-02-sam-gpu-window-cpu-overlap.md`：

- **感知与重建状态 Owner**：`OriginalPersistentWorldRegistry.advance`、`ObjectCentricObservationSession.advance`、`SAM2ObservationSession.advance`、`SAM2VideoMemoryTracker.track`、`PinnedBundleTrackSession.advance_registered`、`PinnedBundleTrackRuntime.track_registered` 与 `CausalObjectSurface.update` 保持唯一权威。
- **独立 Adapter 模块**：新增 `sam2_gpu_window_overlap.py`，未改动受既有 `20261101` 验证器 SHA-256 约束的根部生产源码文件。

## 2. 物理机制：为何同线程 GPU 窗口重叠优于后台线程池

在先前 `same-observation-sam-prepare-overlap-20261101` 中，候选方案在 GPU 启动前通过后台 `ThreadPoolExecutor` 并发执行 `SAM2VideoMemoryTracker._prepare_image` 与 `prepare_tracking_observation`，造成 CPU/GIL 争用且未利用 GPU 推理期间的主线程空闲等待窗口。

通过剖析 `SAM2VideoMemoryTracker.track_group` 的 CPU/GPU 交界可见：
1. 在畸变相机热帧（`active.frame_count > 0`、`initialize=False`）上，`SAM2ObservationSession.advance` 在 `self.tracker.track(image=image, anchor_mask=None)` 返回前仅读取 `frame.stream_key`、时间戳与传感器原图 `frame.sam_color_rgb`；去畸变后的 `frame.color_rgb`、`frame.depth_mm`、`frame.mask`、`frame.intrinsics` 直至第 175 行才首次被访问。
2. `self._predictor.propagate_in_video(...)` 在 CPU 上完成全部 CUDA kernel 入队并 `yield` 输出张量仅需 **`~9.0 ms`**（`gpu_launch_to_yield_ms`），随后主线程在 `(logits > 0.0).cpu().numpy()` 处同步阻塞等待 GPU 完成 `memory_attention`（4 层 FP32 `RoPEAttention`）达 **`~24.2 ms`**（`post_cpu_gpu_sync_wait_ms`）。
3. `PendingObservationCpuPreparation` 在进入 SAM2 前先以 **`~0.25 ms`** 完成全部输入/时间戳/配准/会话状态 fail-closed 预校验，立即启动 SAM2 `_prepare_image` 与 `propagate_in_video` GPU 推理；随后在 `propagate_in_video` 完成 CUDA kernel 入队、`yield` 返回至 `.cpu().numpy()` 阻塞之前的 **同一主线程 GPU 执行窗口内**，同步执行：
   - `rectify_registered_rgbd` 双目 RGB-D 去畸变重映射；
   - `source_evidence_hash` 与 `PreparedFeatureSourceBinding` 计算；
   - `RegisteredBundleTrackInput.evidence_hash` 的 `(header + color_rgb)` 与 `(header + color_rgb + depth_mm)` 双前缀 SHA-256（覆盖 `4.61 MB / 5.53 MB = 83.3%` 字节）；
   - `PinnedBundleTrackSession.advance_registered` 所需的私有 `color_bgr`（`RGB2BGR`）与 `depth_m`（`float32 / 1000.0`）缓冲区转换。
4. GPU 掩码读回后：
   - `SAM2ObservationSession.advance` 在同一帧作用域内通过持有 `sensor_mask` 强引用（`cached[0] is mask`）复用 `encode_mask_rle(sensor_mask)` 与 `_mask_evidence_sha256`;
   - `complete_evidence_hash_from_prefix` 在主线程用 **`0.374 ms`** 完成剩余 `mask + intrinsics`（或 `depth_far_zeroed` 帧的 `depth_mm + mask + intrinsics`）SHA-256，**消除全部 38 个热帧 `evidence-hash` 后台线程的创建与 GIL 争用**；
   - `advance_registered_with_prepared_buffers` 单次消费预计算的 `(color_bgr, depth_m)` 缓冲区。

## 3. 验证与评测结果（40 帧 `teleop_20260625_173830`，排除首帧锚点的 39 个热帧）

执行顺序为 `validate-1` $\rightarrow$ `control-1` $\rightarrow$ `overlap-1` $\rightarrow$ `overlap-2` $\rightarrow$ `control-2` 五个独立进程（均启用已验证的持久 `TestClient` portal 与 8 个 CPU 准备线程）。

| 指标（排除首帧的 39 个热帧） | `control-1` (p50 / p95) | `overlap-1` (p50 / p95) | `overlap-2` (p50 / p95) | `control-2` (p50 / p95) | 合并 78 热帧 $\Delta$ (`overlap - control`) |
|---|---:|---:|---:|---:|---:|
| `input_prepare_ms`（SAM2 前输入准备） | 3.957 / 4.146 ms | **0.251 / 0.266 ms** | **0.252 / 0.267 ms** | 3.956 / 4.122 ms | **-3.705 / -3.868 ms** |
| `gpu_launch_to_yield_ms`（CUDA 入队） | 9.055 / 10.373 ms | 8.982 / 9.366 ms | 9.009 / 9.386 ms | 9.074 / 9.415 ms | -0.069 / -0.518 ms |
| `gpu_window_cpu_prepare_ms`（GPU 窗口内 CPU 准备） | 0.000 / 0.000 ms | **5.717 / 5.927 ms** | **5.720 / 5.894 ms** | 0.000 / 0.000 ms | **+5.719 / +5.911 ms**（被 GPU 窗口吸收） |
| `post_cpu_gpu_sync_wait_ms`（`.cpu().numpy()` 阻塞等待） | 24.122 / 24.239 ms | **18.481 / 18.744 ms** | **18.425 / 18.672 ms** | 24.224 / 24.278 ms | **-5.720 / -5.551 ms** |
| `sam_track_ms`（`SAM2VideoMemoryTracker.track` 总计） | 36.161 / 37.027 ms | 36.447 / 36.929 ms | 36.234 / 37.090 ms | 36.436 / 37.234 ms | +0.042 / -0.121 ms（几乎不变） |
| `native_track_registered_ms`（38 次 BundleTrack） | 56.598 / 77.914 ms | **55.973 / 76.947 ms** | **55.811 / 77.921 ms** | 56.044 / 78.179 ms | **-0.429 / -0.613 ms** |
| `registry_advance_ms`（`OriginalPersistentWorldRegistry.advance`） | 102.447 / 129.099 ms | **96.377 / 123.375 ms** | **96.686 / 122.032 ms** | 102.112 / 129.469 ms | **-5.761 / -6.943 ms** |
| `endpoint_ms`（外层 HTTP Endpoint） | 155.549 / 214.587 ms | **148.538 / 208.902 ms** | **149.165 / 202.557 ms** | 155.116 / 213.698 ms | **-6.328 / -6.815 ms** |
| `pipeline_ms`（含回放辅助开销的全链路） | 201.060 / 249.908 ms | **195.335 / 245.483 ms** | **194.870 / 237.807 ms** | 201.389 / 252.925 ms | **-6.103 / -8.315 ms** |

### 精确性与不变量核验

- **逐字节一致性**：`validate-1` 对全部 39 个热帧逐帧比对延迟物化结果与即时 `prepare_tracking_observation` 的 `color_rgb`、`depth_mm`、`mask`、`intrinsics`、`source_evidence_hash`、`feature_source_binding`（`39/39` 完全一致），并比对全部 38 个进入 native BundleTrack 的热帧的 `evidence_hash`、`(color_bgr, depth_m)` 及 RLE 缓存（`38/38` 完全一致，含 `depth_far_zeroed=True` 的第 10 帧等 6 帧）。
- **跨运行输出恒等**：5 轮运行共 `5 × 80 = 400` 个 SAM 掩码 SHA-256、`5 × 38 = 190` 个 BundleTrack `input_evidence_hash`、每轮 `34` 个 camera-local candidate 与 `32` 个 current registered surface、以及 40 帧 `source_owner_evidence` 与冻结基线 `100%` 一致。
- **线程与 OpenMP 隔离**：后台 worker 线程创建数为 `0`；热帧 `evidence-hash` 线程创建数由每轮 `38` 降至 `0`；全部 `190` 次 native BundleTrack 调用前后 `torch_threads == torch_omp == native_omp == 1`。

## 4. 局限性与现场验收边界

本实验证明在同一主线程内利用 SAM2 `propagate_in_video` 的 GPU 执行窗口重叠同帧 CPU 准备与哈希前缀计算，可在不改动任何估计器输出、不引入后台线程池的前提下稳定降低 `registry_advance_ms`（p50 `-5.761 ms`、p95 `-6.943 ms`）与 `endpoint_ms`（p50 `-6.328 ms`、p95 `-6.815 ms`）。但离线 40 帧回放不代表现场 `15 Hz`、`<100 ms`、`15 min / 1 h` 资源有界或物理精度达标；现场验收仍保持 `0/6` 未通过（`gate_passed=false`、`wire_authorized=false`）。

## 5. 复现实验与独立验证

```sh
PYTHONPATH=apps/replay-api:apps/scene-runtime apps/scene-runtime/.venv/bin/python -m pytest apps/scene-runtime/tests/perception/test_sam2_gpu_window_overlap.py -v
.venv-bundlesdf/bin/python docs/papers/sam-gpu-window-cpu-overlap-20261102/score.py
.venv-bundlesdf/bin/python docs/papers/sam-gpu-window-cpu-overlap-20261102/verify.py
```
