# Edge0-145B Q3_K — self-contained edge0-strata (HIP, gfx1151) runtime for AMD Strix Halo.
# Builds edge0-strata with the HIP backend for gfx1151 against a pinned TheRock ROCm 7.14.1 SDK,
# plus edge0-infer (cpu preset) which launches it. The ~106 GB checkpoint is NOT in the image —
# bind-mount it at /models (ro). Experimental: Edge0 does not vendor-test 145B on AMD.
FROM docker.io/library/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git curl ca-certificates \
      socat libnuma-dev python3 patch \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

ARG EDGE0_REVISION=6383f651c9387147140dd9f6a5198dfaea5f28d6
ARG GGML_REVISION=3cf03257f219afbe7334045ff7c6a06ac68c627d
ARG THEROCK_URL=https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.1.tar.gz
ARG THEROCK_SHA256=c40e8f2bd6630a7d11557c762b99c6fa8afb04c9bd0e51ed1675ee1ca24afb00
ARG MAX_JOBS=32

# TheRock ROCm 7.14.1 SDK for gfx1151 (hipcc/clang, hipBLAS, rocBLAS, hipBLASLt).
RUN curl -fL -o /tmp/therock.tar.gz "${THEROCK_URL}" \
    && echo "${THEROCK_SHA256}  /tmp/therock.tar.gz" | sha256sum -c \
    && mkdir -p /opt/rocm && tar -xzf /tmp/therock.tar.gz -C /opt/rocm \
    && rm /tmp/therock.tar.gz
ENV SDK=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/root/.local/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/rocm_sysdeps/lib:/opt/rocm/lib/llvm/lib

# Edge0 sources at a pinned revision + pinned ggml for the strata build.
RUN curl -fL "https://github.com/Edge0-AI/Edge0/archive/${EDGE0_REVISION}.tar.gz" \
      | tar -xz -C /opt --strip-components=1 \
    && mkdir -p /opt/edge0-strata/third_party/llama.cpp \
    && curl -fL "https://github.com/ggml-org/llama.cpp/archive/${GGML_REVISION}.tar.gz" \
      | tar -xz --strip-components=1 -C /opt/edge0-strata/third_party/llama.cpp

# gfx1151 upstream-gap patches, applied in numeric order (0001..0005). Each file
# carries its own header comment; see patches/*.patch. Summary:
#   0001 hip_compat: cuda* names the HIP build uses but the header never maps
#   0002 moe_fused.cu: guard CUDA-only latent_kernel launches for HIP
#   0003 CMakeLists: latent host classes (latent_dense/tables/moe) for HIP link
#   0004 CMakeLists + moe_fused_q3k.cu: q3k latent sources for HIP link + guard
#   0005 verify.cpp: skip the synchronous top3 readback during graph capture —
#        without it hipStreamEndCapture returns hipErrorStreamCaptureImplicit
#        and every request dies with "verify: end capture: operation failed
#        due to a previous error during capture"
COPY patches /patches
RUN for p in $(ls /patches/*.patch | sort); do patch -p1 -d /opt --silent < "$p" || exit 1; done

# edge0-strata, HIP backend for gfx1151. Build dir MUST be named `build`: the launch
# template (edge0/models/edge0-145b.json) hardcodes ${STRATA_DIR}/build/strata.
# Flags from edge0-strata/docs/STRIX_HALO.md §2 (gfx1151 section). CMake warns that
# gfx1151 is unvalidated — expected. ~25 min.
RUN cd /opt/edge0-strata \
    && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
         -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
         -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
         -DSTRATA_GGML_DIR=/opt/edge0-strata/third_party/llama.cpp \
         -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
         -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
         "-DCMAKE_PREFIX_PATH=/opt/rocm;/opt/rocm/lib/rocm_sysdeps;/opt/rocm/lib/llvm" \
         "-DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode" \
    && cmake --build build --target strata -j "${MAX_JOBS}"

# Server Python env (serve/server.py runs on this interpreter).
RUN cd /opt/edge0-strata \
    && uv venv .venv \
    && uv pip install --python .venv/bin/python -r requirements.txt

# edge0-infer launcher: cpu preset needs no GPU backend of its own.
RUN cd /opt/edge0-infer \
    && cmake --preset cpu \
    && cmake --build --preset cpu -j "${MAX_JOBS}"

# HIP_VISIBLE_DEVICES selects the AMD device (edge0-infer's --device cuda:0 is only a
# parser formality). Tuning table matches TheRock 7.14.1 hipBLASLt (100401).
ENV HIP_VISIBLE_DEVICES=0 \
    STRATA_HIPBLASLT_TUNING=/opt/edge0-strata/tools/hip/gfx1151-hipblaslt-100401.txt \
    MODEL_PATH=/models \
    CONTEXT=262144
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60m --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1
# edge0-infer serve refuses a non-loopback --host without --api-key-file, so it stays on
# 127.0.0.1:8001 and socat republishes it on 0.0.0.0:8000 (token-free host access).
# --context 262144 = model native (qwen4exp.context_length); override via -e CONTEXT.
CMD ["bash", "-c", "socat TCP-LISTEN:8000,fork,reuseaddr TCP:127.0.0.1:8001 & exec /opt/edge0-infer/build/cpu/bin/edge0-infer serve --model /models --host 127.0.0.1 --port 8001 --context \"${CONTEXT}\" --strata-ready-timeout 3600"]
