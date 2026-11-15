# The GPU workstation environment (QS-112, plan.md §4.5). Build and run it from the repository root:
#
#   docker build -f environment/gpu.Dockerfile -t ketjet-gpu .
#   docker run --rm --gpus all ketjet-gpu
#
# The default command checks the GPU (environment/check_gpu.py). Setup guide:
# docs/dev/gpu-workstation.md.
#
# Reproducible by construction: the base image and uv are pinned by digest, Python by
# .python-version, and every Python package, CUDA runtime included, by uv.lock. The CUDA 12
# runtime ships as pip wheels matched to jaxlib (the `gpu` dependency group), so the image needs
# no CUDA base image: only the host's NVIDIA driver, which the NVIDIA Container Toolkit passes in.

FROM ubuntu:24.04@sha256:2198e4c65076e9cb0b3d695219df667c0e3b0072da64ca3c2648aca17dcecaf5

COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 /uv /uvx /usr/local/bin/

# ca-certificates: uv downloads Python and wheels over HTTPS. Nothing else is needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_FROZEN=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_CACHE=1 \
    PATH=/opt/venv/bin:$PATH

# Jobs run as an unprivileged user (plan.md §4.5: the GPU runner is a non-admin account).
RUN useradd --create-home --uid 1001 ketjet
WORKDIR /home/ketjet/ketjet

# Dependencies first, in their own layer, so a code change doesn't reinstall CUDA (most of the image's ~10 GB).
COPY .python-version pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --no-install-project --no-default-groups --group test --group gpu --extra pennylane

COPY . .
RUN uv sync --no-default-groups --group test --group gpu --extra pennylane \
    && chown -R ketjet:ketjet /home/ketjet

USER ketjet
CMD ["python", "environment/check_gpu.py", "--expect-name", "RTX 4060"]
