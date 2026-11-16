# GPU workstation

How to set up the GPU development machine, from a clean Windows install to ketjet running on the GPU inside a
pinned container (QS-112). It takes about an hour, most of it downloads.

**The result:** Windows 11 runs Ubuntu 24.04 in WSL2. Inside it, Docker runs `environment/gpu.Dockerfile`, which
sees the RTX 4060 through the Windows NVIDIA driver. JAX, the CUDA 12 runtime and every other package come from
`uv.lock`, so the environment is the same on every rebuild.

```
Windows 11 ── NVIDIA driver (the only driver; WSL2 shares it)
└── WSL2: Ubuntu 24.04
    ├── Docker Engine + NVIDIA Container Toolkit
    └── container: ketjet-gpu image (Ubuntu 24.04, Python 3.13, uv.lock incl. jax[cuda12])
```

**Why this design** (plan.md §4.5):

- **WSL2, not native Windows:** JAX's CUDA builds are Linux-only.
- **Docker, not a virtualenv on the host:** one pinned image is the environment for development now and for the
  self-hosted GPU runner later (Sprint 6), and it can be rebuilt exactly.
- **No CUDA toolkit anywhere:** `jax[cuda12]` brings the CUDA runtime as pip wheels matched to jaxlib. The only
  NVIDIA software installed by hand is the Windows driver and the container toolkit.

## Hardware and versions

| | Required | Notes |
|---|---|---|
| GPU | NVIDIA RTX 4060 (8 GB, compute capability 8.9) | Any CUDA GPU of compute capability ≥ 5.2 works; the check expects this name by default. |
| OS | Windows 11, 64-bit | Windows 10 21H2+ also supports GPU in WSL2. |
| NVIDIA driver | Recent Game Ready or Studio driver whose `nvidia-smi` reports **CUDA 12.9 or later** (`CUDA Version` on older drivers, `CUDA UMD Version` on newer ones) | The image bundles the CUDA 12.9 runtime. |
| WSL | WSL 2, current version from `wsl --update` | |
| Disk | ~30 GB free | Ubuntu, Docker, and the image: 9.9 GB, measured in CI. |

## 1. Windows: driver and WSL2

In **PowerShell** (as your normal user):

1. Install the latest NVIDIA driver for the RTX 4060 from nvidia.com (Game Ready or Studio). Don't install anything
   NVIDIA inside Linux later: WSL2 uses this driver.
2. Check it:

    ```powershell
    nvidia-smi
    ```

    The header must show the RTX 4060 and CUDA 12.9 or later: `CUDA Version: 12.9` on older drivers, or
    `CUDA UMD Version: 13.3` (next to `KMD Version`) on newer ones such as 610.88.

3. Install WSL2 with Ubuntu 24.04 (an administrator prompt is needed for this step only), then restart if asked:

    ```powershell
    wsl --install -d Ubuntu-24.04
    wsl --update
    wsl --list --verbose          # Ubuntu-24.04 must show VERSION 2
    ```

4. Start Ubuntu from the Start menu and create your Linux user when it asks.

**Optional: memory.** WSL2 takes up to half of the PC's RAM by default. To set it explicitly, create
`%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=16GB
```

then run `wsl --shutdown` and start Ubuntu again.

## 2. Ubuntu: check the GPU and systemd

In the **Ubuntu** terminal:

```bash
nvidia-smi
```

It should print the same GPU and driver as on Windows. WSL provides this command from the Windows driver
(`/usr/lib/wsl/lib/nvidia-smi`). **Don't install `nvidia-driver-*` or `cuda-drivers` packages in Ubuntu:** they
replace the WSL driver shim and break GPU access.

Docker needs systemd, which new Ubuntu 24.04 WSL installs enable:

```bash
systemctl is-system-running    # "running" or "degraded" are both fine; "offline" means systemd is off
```

If it's off, add this to `/etc/wsl.conf`, then run `wsl --shutdown` from PowerShell and reopen Ubuntu:

```ini
[boot]
systemd=true
```

## 3. Ubuntu: Docker Engine

Docker Engine from Docker's own apt repository. It's free, and doesn't need Docker Desktop:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo usermod -aG docker "$USER"
```

Close and reopen the Ubuntu terminal (for the `docker` group), then:

```bash
docker run --rm hello-world
```

## 4. Ubuntu: NVIDIA Container Toolkit

This lets containers use the GPU. The version is pinned so the setup can be repeated exactly:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
TOOLKIT=1.20.1-1
sudo apt-get install -y \
  nvidia-container-toolkit=$TOOLKIT nvidia-container-toolkit-base=$TOOLKIT \
  libnvidia-container-tools=$TOOLKIT libnvidia-container1=$TOOLKIT
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Check that a container sees the GPU:

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

## 5. Build and check the ketjet image

Clone the repository **inside the Linux file system** (`~`, not `/mnt/c/...`, which is much slower):

```bash
cd ~
git clone https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator.git ketjet
cd ketjet
docker build -f environment/gpu.Dockerfile -t ketjet-gpu .
docker run --rm --gpus all ketjet-gpu
```

The last command runs `environment/check_gpu.py`. **Acceptance (QS-112):** it ends with `all checks passed`,
after showing:

- `nvidia-smi` with the RTX 4060, and a driver that supports the bundled CUDA runtime;
- `jax.devices()` with `gpu:0 NVIDIA GeForce RTX 4060`;
- a 16-qubit, 136-parameter circuit compiled on the GPU in both precisions, matching the NumPy oracle, and its
  `jax.jit(jax.grad)`.

Then run the JAX backend's tests on the GPU:

```bash
docker run --rm --gpus all ketjet-gpu pytest -p no:cacheprovider -q \
  tests/unit/test_jax_backend.py tests/unit/test_compile_circuit.py tests/unit/test_backprop.py
```

These were written for the CPU. If a complex64 test fails on the GPU by a small margin, record it below rather
than loosening the test: GPU precision is QS-607's subject (Sprint 6).

## 6. Record the setup

Paste the results into this table, in the PR that closes QS-112, so the environment's versions are on record
(plan.md §4.5: every benchmark records its hardware, driver and package versions):

```bash
cmd.exe /c ver; wsl.exe --version | head -3; uname -r
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
docker --version; nvidia-ctk --version | head -1
docker image inspect ketjet-gpu --format '{{.Id}}'
```

| Item | Value |
|---|---|
| Recorded | 2026-11-16, on the RTX 4060 workstation |
| Windows build | Windows 11, 10.0.26200.9711 |
| WSL version, kernel | WSL 3.0.5.0; kernel 6.18.58.1-microsoft-standard-WSL2; Ubuntu 24.04.5 LTS |
| GPU, NVIDIA driver | NVIDIA GeForce RTX 4060 (8 GB, compute capability 8.9), driver 610.88 (CUDA 13.3) |
| Docker Engine | 29.10.0, build fdd4b8e |
| NVIDIA Container Toolkit | 1.20.1 |
| Image ID (`ketjet-gpu`) | `sha256:be6aa75a52651eac15332e9638cd4fe8e8fe30ef865f7d75f05ef9f8561f6b84`, 9.86 GB, built from `5f4cbb6` |
| `check_gpu.py` | All checks passed (output below) |
| JAX backend tests on the GPU | 135 of 153 passed. 18 failed, all from TF32: 17 complex64 tests off by up to 8e-3, and the test without 64-bit mode, which runs in complex64 (details below) |

The `check_gpu.py` output, from `docker run --rm --gpus all ketjet-gpu`:

```text
== versions
   Python 3.13.18, Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.39
   ketjet 0.1.0, jax 0.12.0.1, numpy 2.5.4
   jax-cuda12-plugin 0.12.0.1
== nvidia-smi
   NVIDIA GeForce RTX 4060, 610.88, 8188 MiB, 8.9
   driver supports CUDA ?; bundled CUDA runtime 12.9.79
== jax.devices()
   gpu:0 NVIDIA GeForce RTX 4060
== ketjet on gpu: 16 qubits, 8 layers
   complex128: first call 5.43 s, then 64.15 ms; max |state - NumPy| = 1.9e-17 ok
   complex64: first call 4.51 s, then 10.79 ms; max |state - NumPy| = 2.6e-08 ok
   jax.jit(jax.grad): 136 parameters, |grad| = 0.0404
== result
   all checks passed
```

**What this run found:**

- **`driver supports CUDA ?`:** driver 610.88 labels the header line `CUDA UMD Version: 13.3`, which the check didn't read, so it skipped the driver check without failing. #144 fixes that; with it, the line reads `driver supports CUDA 13.3; bundled CUDA runtime 12.9.79`.
- **The 18 GPU test failures:** on this GPU, XLA's default precision computes float32 products in TF32, so complex64 results drifted from the NumPy oracle. The check's complex64 error also varied between runs, from 2.2e-8 to 7.3e-5 against its 1e-4 limit. With `JAX_DEFAULT_MATMUL_PRECISION=highest` all 153 tests pass. #145 sets full precision in the backend itself; with it, all 153 pass with no override, and the check's complex64 error stays at 2.3e-8 to 2.4e-8.
- **Installing WSL:** `wsl --install` failed with `403 Forbidden` on this PC, even with `--web-download`. Installing from the two files worked (see Troubleshooting).

## Day to day

- **Code changes:** `git pull`, then `docker build ...` again. The dependency layer is cached, so only the ketjet
  layer is rebuilt. Run anything in the image with `docker run --rm --gpus all ketjet-gpu <command>`.
- **Editing in place:** mount the checkout over the image's copy to run your working tree without rebuilding:
  `docker run --rm --gpus all -v "$PWD:/home/ketjet/ketjet" ketjet-gpu pytest -q tests/unit/test_backprop.py`.
- **Dependency changes** arrive through `uv.lock`, so a rebuild picks them up, and the
  [GPU image workflow](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/actions/workflows/gpu-image.yml)
  has already built the image in CI.
- **Updating the driver:** any newer Windows driver works. Re-run step 5 afterwards, and update the record.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `nvidia-smi: command not found` in Ubuntu | The Windows driver is missing or too old for WSL: update it, then `wsl --shutdown`. |
| `nvidia-smi` worked, then stopped after `apt upgrade` | A Linux NVIDIA driver package got installed: `sudo apt-get remove --purge 'nvidia-driver-*' 'cuda-drivers*'`, then `wsl --shutdown`. |
| `could not select device driver "" with capabilities: [[gpu]]` | The container toolkit isn't configured: re-run the two `nvidia-ctk`/`systemctl` lines of step 4. |
| `permission denied ... docker.sock` | Your user isn't in the `docker` group yet: close and reopen the terminal, or `newgrp docker`. |
| `check_gpu.py`: driver supports CUDA 12.x, older than the bundled runtime | Update the Windows driver. |
| `jax.devices()` shows only `cpu` | The container was started without `--gpus all`, or JAX logged why CUDA failed to initialize at the top of the output. |
| Docker can't start: `System has not been booted with systemd` | Enable systemd (step 2). |
| `docker build` is very slow | The clone is under `/mnt/c`: clone into `~` instead. |
| `wsl --install` fails with `403 Forbidden`, even with `--web-download` | The Store download is refused. Install from files instead: run the `wsl.<version>.x64.msi` from [WSL's releases](https://github.com/microsoft/WSL/releases), restart, download `ubuntu-24.04.5-wsl-amd64.wsl` from [releases.ubuntu.com/24.04.5](https://releases.ubuntu.com/24.04.5/) (check it against `SHA256SUMS`), then `wsl --install --from-file <the .wsl file>`. |
| Ubuntu should live on another drive | `wsl --shutdown`, then `wsl --manage Ubuntu-24.04 --move D:\WSL\Ubuntu-24.04`. Cheapest before the image is built: the disk is ~1.5 GB then. |
| `CUDA_ERROR_OUT_OF_MEMORY` lines at the start of a run, which then passes | JAX tries to reserve 75% of the GPU, and Windows already uses part of it, so JAX retries smaller. It's harmless; `-e XLA_PYTHON_CLIENT_PREALLOCATE=false` silences it. |
