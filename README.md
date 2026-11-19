# Edge0-145B Q3_K — one-command Podman deploy (AMD Strix Halo 128 GB)

Serves [Edge0/Edge0-145B-q3k](https://huggingface.co/Edge0/Edge0-145B-q3k) (Edge0 145B MoE, Q3_K quantisation) in a
rootless Podman container on an AMD Strix Halo (Ryzen AI Max, `gfx1151`), using the `edge0-strata` engine built with
its HIP backend plus the `edge0-infer` launcher.

**Unofficial / experimental — verified working on real hardware.** Edge0's support matrix lists 145B for
CUDA/Metal only, and `edge0-strata/cmake/hip_backend.cmake` still marks `gfx1151` as buildable-but-unvalidated
(`gfx1151 builds, but it is not validated on a real card yet`). This recipe is not vendor-supported; it runs the
gfx1151 HIP path end-to-end with five local patches (see `patches/`).

**Verified status on a 128 GB Strix Halo (2026-11-19).** The image builds (after the five upstream-gap patches
in `patches/`), the container starts, the engine loads the full checkpoint, `/health` answers, `/v1/models` serves
`edge0_145b`, and **chat completions and SSE streaming work** — measured ~27 tok/s decode at the default
`CONTEXT=262144` (model native 256K). The
last blocker was engine-side: `record_window`'s first-12-windows top3 diagnostic did a synchronous `cudaMemcpy`
inside the CUDA-graph capture window, which invalidates the capture on HIP
(`hipErrorStreamCaptureImplicit` → `verify: end capture: operation failed due to a previous error during
capture`). Patch `0005` skips that readback while a capture is active; see `patches/0005-*.patch`.

One command brings it up; one command tears it down.

```bash
cp .env.sample .env   # optional — every var has a default
./start.sh            # build (cached) -> download model -> run -> wait healthy
./stop.sh
```

## What's here

| File            | Role                                                                                                                                                                               |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Containerfile` | Self-contained runtime: pinned TheRock ROCm 7.14.1 SDK for gfx1151 (sha256-pinned), `edge0-strata` HIP build, `edge0-infer` (cpu preset), uv Python venv. Checkpoint not baked in. |
| `start.sh`      | One-command entrypoint: preflight → build (cached) → download checkpoint (resumable) → run → health-poll.                                                                          |
| `stop.sh`       | Stops the container with a 60 s grace period; idempotent.                                                                                                                          |
| `patches/`      | Five gfx1151 upstream-gap patches (`0001`–`0005`), applied in numeric order during the image build; each header explains its gap.                                                  |
| `.env.sample`   | Optional overrides — copy to `.env`.                                                                                                                                               |

The ~106 GB checkpoint is **not** baked into the image; it is bind-mounted read-only at `/models`. The image is built
on first run and cached for subsequent launches.

## Requirements

- **Podman** (rootless) — `podman --version` ≥ 5.x
- **AMD GPU driver** present: `/dev/kfd` and `/dev/dri` exist (amdgpu loaded)
- Current user in the **`render`** (and ideally `video`) group — check with `id -nG`
- **≥128 GB unified RAM** — experts (~40 GB) stay resident, the 51 GB PLE table is read on demand, plus KV cache;
  64 GB boxes are expected to fail at 8K context
- **~120 GB free disk**: ~106 GB weights + ~11 GB image (measured; TheRock SDK + build trees)
- `curl`, `python3`, and the Hugging Face CLI `hf` (`pip install -U huggingface_hub[cli]`) — or the
  `huggingface_hub` Python package — on the host for the model download

No root/sudo is required for the normal path. Podman runs rootless; GPU access comes from `--group-add keep-groups`
(carries your host `render` group into the container) plus SELinux `:z` bind relabeling.

## Quick start

```bash
cp .env.sample .env      # optional
./start.sh
```

First run:

1. Builds the image: downloads the TheRock gfx1151 SDK, compiles `edge0-strata` with the HIP backend for `gfx1151`
   (~25 min) and `edge0-infer` with the cpu preset.
2. Downloads the ~106 GB checkpoint to `./models` (resumable via `hf`, `huggingface-cli`, or `huggingface_hub`).
3. Starts the container and streams the engine log to `.edge0.log`.
4. Exits "ready" once `/health` returns 200 (cold load of ~106 GB weights takes a while — default timeout 2 h).

Then:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq -r '.data[0].id'
# -> edge0_145b
```

Tear down:

```bash
./stop.sh
```

Re-running `./start.sh` after the first time skips the build and the download and reaches ready as fast as the
checkpoint load allows.

## Configuration

Copy `.env.sample` to `.env` and edit. Precedence (highest first): **exported shell env → `.env` → `start.sh` defaults.**

| Var              | Default                   | Meaning                                                          |
| ---------------- | ------------------------- | ---------------------------------------------------------------- |
| `IMAGE`          | `edge0-145b-strix:latest` | Image tag                                                        |
| `CONTAINER_NAME` | `edge0-145b-strix`        | Container name                                                   |
| `PORT`           | `8080`                    | Host publish port (container always listens on 8000)             |
| `MODEL_DIR`      | `./models`                | Where the checkpoint lives / downloads to                        |
| `HF_MODEL`       | `Edge0/Edge0-145B-q3k`    | HuggingFace model repo                                           |
| `HF_REVISION`    | *(latest)*                | Pin a commit SHA for reproducibility                             |
| `LOG_FILE`       | `./.edge0.log`            | Engine output log                                                |
| `CONTEXT`        | `262144`                  | Engine context tokens (model native 256K); lower if memory-tight |
| `READY_TIMEOUT`  | `7200`                    | Seconds to wait for the health endpoint                          |
| `MAX_JOBS`       | `$(nproc)`                | Build parallelism (hipcc/ninja); also `podman build --jobs`      |
| `HF_TOKEN`       | *(unset)*                 | Optional HuggingFace token — higher download rate limits         |

## How it works

The `Containerfile` builds a self-contained image:

1. Ubuntu 24.04 base with build deps, `socat`, and `uv`.
2. **TheRock ROCm 7.14.1 SDK for gfx1151** unpacked to `/opt/rocm` (sha256-pinned tarball from `repo.amd.com`) —
   hipcc/clang, hipBLAS, rocBLAS, hipBLASLt. No system ROCm from apt; the SDK is self-contained.
3. Edge0 sources at a pinned revision unpack to `/opt` (engine at `/opt/edge0-strata`, launcher at
   `/opt/edge0-infer`); pinned llama.cpp (ggml) lands in `/opt/edge0-strata/third_party/llama.cpp`.
4. **`patches/` — five gfx1151 upstream-gap patches**, copied to `/patches` and applied in numeric order with
   `patch -p1 -d /opt` (all drop when fixed upstream):
   - `0001` appends the missing `cuda*→hip*` defines (`cudaGraphNode_t`, `cudaKernelNodeParams`,
     `cudaMemcpy3DParms`, `cudaPointerAttributes`, `cudaStreamGetCaptureInfo`, …) to
     `edge0-strata/include/strata/hip_compat/cuda_runtime.h` — used by `verify.cpp` / `prefill.cpp` in any HIP build.
   - `0002` wraps the `latent_kernel` launches in `moe_fused.cu` in `#if !defined(__HIPCC__)` (the kernel is
     CUDA-only; the HIP launcher exits before reaching them).
   - `0003` adds the latent host classes (`latent_dense.cpp`, `latent_tables.cpp`, `latent_moe.cpp`) to
     `strata_engine` for HIP — upstream wires them only under `STRATA_ENABLE_CUDA` while `verify.cpp`,
     `prefill.cpp`, `generate.cpp` and `peer_experts.cpp` reference them in every build.
   - `0004` adds `lat_q3k_fused.cu` / `moe_fused_q3k.cu` to the HIP `strata_mmq` list and applies the same
     launch guard to `moe_fused_q3k.cu`.
   - `0005` guards `verify.cpp`'s synchronous top3 readback with `cudaStreamGetCaptureInfo` so it is skipped
     while a graph capture is active — without it every request dies at `verify: end capture`.
5. `edge0-strata` is configured and built with `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151` into
   `build/` — **the directory name is load-bearing**: the launch template `edge0/models/edge0-145b.json` hardcodes
   `${STRATA_DIR}/build/strata`. CMake warns that gfx1151 is unvalidated; that is expected.
6. A uv venv (`.venv`) with `edge0-strata/requirements.txt` — `serve/server.py` runs on this interpreter.
7. `edge0-infer` is built with the `cpu` preset (the launcher needs no GPU backend of its own).

At runtime the container runs `edge0-infer serve --model /models --host 127.0.0.1 --port 8001` (the launcher refuses
a non-loopback `--host` without `--api-key-file`) and `socat` republishes it on `0.0.0.0:8000`, so host access stays
token-free. `HIP_VISIBLE_DEVICES=0` selects the APU — the launcher's `--device cuda:0` form is only its parser's
accepted spelling; the engine binary is HIP. The hipBLASLt tuning table
`edge0-strata/tools/hip/gfx1151-hipblaslt-100401.txt` is loaded via `STRATA_HIPBLASLT_TUNING`.

## Verify

```bash
curl -s http://127.0.0.1:8080/v1/models | jq -r '.data[0].id'
# -> edge0_145b

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0_145b",
    "messages": [{"role": "user", "content": "Name three moons of Jupiter."}],
    "max_tokens": 64
  }' | jq -r '.choices[0].message.content'
# -> Io, Europa, and Ganymede.

curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0_145b",
    "messages": [{"role": "user", "content": "Say hello in five words."}],
    "max_tokens": 24,
    "stream": true
  }' | head
# -> data: {...delta...} SSE chunks
```

Folder completeness (no GPU needed):

```bash
podman exec edge0-145b-strix /opt/edge0-infer/build/cpu/bin/edge0-infer inspect --model /models --json | jq .ok
# -> true
```

The hipBLASLt tuning table loads during engine startup
(`prefill gemm: hipBLASLt tuning enabled (90 rows, gfx1151, version 100401)`). The launcher does not forward that
line to `podman logs`; to see it, run the engine directly inside the container with the template's env block
(see `edge0/models/edge0-145b.json` for the variables).

## Troubleshooting

**`Cannot allocate memory` / HSA pin errors at startup** — the user session's `memlock` limit is too low for ROCm
memory pinning. `start.sh` only warns about this (it does not block). If the engine actually fails to pin, raise it
once (needs sudo):

```bash
sudo mkdir -p /etc/systemd/system/user@.service.d
printf '[Service]\nLimitMEMLOCK=infinity\n' | sudo tee /etc/systemd/system/user@.service.d/10-memlock.conf
sudo systemctl daemon-reload
# then log out and back in
```

**Container exits immediately** — check logs:

```bash
podman logs edge0-145b-strix
```

Common causes: missing GPU driver (`/dev/kfd` not present), `memlock` too low, or an incomplete model directory
(`start.sh` checks `model.json` + `experts.bin`).

**No tuning-table line in the log** — check `STRATA_HIPBLASLT_TUNING` points at
`/opt/edge0-strata/tools/hip/gfx1151-hipblaslt-100401.txt` (`podman exec edge0-145b-strix env | grep STRATA`).

**OOM / pin errors under load** — lower `CONTEXT` (default 262144 = model native; int8 KV costs ~12 GB at the
full window, on top of ~40 GB resident experts). Optionally, on the host,
`ttm.pages_limit=29360128 ttm.page_pool_size=29360128 amd_iommu=off` (from `edge0-strata/docs/STRIX_HALO.md`;
host kernel args — not required to boot the container).

**Engine crashes on the latent-MoE kernels** — see Contingencies below; there is no vendor-tested fix, this is the
experimental edge of the path.

**`hf: command not found`** — install or upgrade `huggingface_hub` on the host:

```bash
pip install -U huggingface_hub[cli]
```

`start.sh` tries `hf` (the current CLI), then the deprecated `huggingface-cli`, then a Python one-liner using
`huggingface_hub`.

**No `/dev/kfd` or `/dev/dri`** — the AMD GPU kernel driver is not loaded. Load it:

```bash
sudo modprobe amdgpu
sudo modprobe kfd
```

## Contingencies

- **`edge0-infer serve` and the `cuda:0` device form** — resolved in practice: the launcher accepts its default
  device and logs `starting edge0_145b ... on cuda:0` while running the HIP engine (`HIP_VISIBLE_DEVICES=0`
  selects the APU). No bypass needed.
- **Verify-window capture failure on chat requests** — **resolved** by `patches/0005`: the synchronous top3
  readback in `record_window` (first 12 windows) ran inside the graph capture and invalidated it on HIP
  (`hipErrorStreamCaptureImplicit`). No env knob could avoid it — `STRATA_LATENT_FUSED{,_DU}`,
  `STRATA_LATENT_U_GPU`, `STRATA_LAT_FUSED_Q3K`, `STRATA_GFX1151_DEFAULTS=0`, `STRATA_SH_STREAM=0` were all
  tried and irrelevant, and `--spec 0` is rejected (native IQ packs require `--spec >= 2`). Diagnosis path:
  `AMD_LOG_LEVEL=3` on a direct engine run shows the single illegal call (`hipMemcpy: Returned
  hipErrorStreamCaptureImplicit`) between `hipStreamBeginCapture` and `hipStreamEndCapture`. Worth reporting
  upstream (engine 0.1.55) — CUDA would reject the same call.
- **`repo.amd.com` tarball unavailable**: TheRock dists also ship on the ROCm/TheRock GitHub releases; update
  `THEROCK_URL` and keep the sha256 check.
- If the strata HIP build fails inside latent-MoE kernels there is no recipe-side fix — capture the failing target
  and report it upstream; do not silently fall back to CPU.

## Notes

- **No system ROCm in the image.** The TheRock SDK under `/opt/rocm` is self-contained; mixing apt ROCm with it is
  unsupported. The container distro (Ubuntu 24.04) is independent of the host distro — only the kernel is shared.
- **`--ipc host` is load-bearing** — the GPU runtime dies at startup without it; no `shm_size` substitutes.
- **SELinux `Permission denied` on `/models`** — confirm the `:z` suffix on the bind (`start.sh` sets it). If you
  relabeled the directory manually, `restorecon -R` it or just re-run `./start.sh`.
- **Docker instead of Podman** — the same `Containerfile` works under a rootful Docker daemon with
  `docker run --device /dev/kfd --device /dev/dri --group-add video,render --ipc host --ulimit memlock=-1:-1`.
  Swap the CLI in `start.sh` if you prefer Docker.

## Licence

The model weights are under the Edge0 community model licence (see the model card). The `edge0-strata` engine fork is
MIT-licensed (Strata).

- Model: [Edge0/Edge0-145B-q3k](https://huggingface.co/Edge0/Edge0-145B-q3k) — Edge0 community model licence
- Engine: [Edge0-AI/Edge0](https://github.com/Edge0-AI/Edge0) (`edge0-strata`, `edge0-infer`) — MIT
- This recipe: provided as-is, without warranty
