# Environment Setup Guide

This guide covers setting up the LLM Testing Framework on Proxmox VE with LXC containers.

## Host System Requirements

- Proxmox VE 8.x (or Ubuntu 22.04+ for bare metal)
- NVIDIA Driver 535+ (CUDA 12.x support)
- Linux kernel 5.15+

## 1. Proxmox Host Setup

### Install NVIDIA Drivers

```bash
# Add non-free repository
echo "deb http://deb.debian.org/debian bookworm non-free non-free-firmware" >> /etc/apt/sources.list.d/non-free.list

# Update and install
apt update
apt install nvidia-driver nvidia-smi

# Verify installation
nvidia-smi
```

### Load NVIDIA Kernel Modules

```bash
# Add modules to load at boot
cat >> /etc/modules-load.d/nvidia.conf << EOF
nvidia
nvidia_uvm
EOF

# Load modules now
modprobe nvidia
modprobe nvidia_uvm

# Verify device nodes exist
ls -la /dev/nvidia*
```

## 2. LXC Container Setup

### Create Container

1. In Proxmox web UI, create a new LXC container:
   - Template: Ubuntu 22.04 or 24.04
   - Unprivileged: No (privileged required for GPU)
   - Memory: 32GB+
   - CPU: 8+ cores
   - Disk: 100GB+ for system

2. Mount shared storage for models

### GPU Passthrough Configuration

Edit `/etc/pve/lxc/<vmid>.conf`:

```bash
# NVIDIA GPU passthrough
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.cgroup2.devices.allow: c 509:* rwm

# Single GPU
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file

# Multi-GPU (add for each GPU)
lxc.mount.entry: /dev/nvidia1 dev/nvidia1 none bind,optional,create=file

# Shared storage mount
lxc.mount.entry: /mnt/nvme/llm-testing mnt/llm-testing none bind,create=dir 0 0

# Required features
features: nesting=1
```

### Install NVIDIA Drivers in Container

```bash
# Inside container
apt update
apt install -y nvidia-driver-535 nvidia-container-toolkit

# Verify
nvidia-smi
```

## 3. Docker Setup

### Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Add NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt update
apt install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Verify GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## 4. Inference Engine Setup

### Ollama

```bash
# Docker (recommended)
docker run -d --gpus all \
    -v /mnt/llm-testing/models/ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama

# Verify
curl http://localhost:11434/api/version
```

### llama.cpp

```bash
# Docker with CUDA
docker run -d --gpus all \
    -v /mnt/llm-testing/models/gguf:/models \
    -p 8080:8080 \
    --name llamacpp \
    ghcr.io/ggerganov/llama.cpp:full-cuda \
    --server -m /models/model.gguf --host 0.0.0.0 --n-gpu-layers 999

# For multi-GPU
docker run -d --gpus all \
    -v /mnt/llm-testing/models/gguf:/models \
    -p 8080:8080 \
    --name llamacpp \
    ghcr.io/ggerganov/llama.cpp:full-cuda \
    --server -m /models/model.gguf --host 0.0.0.0 \
    --n-gpu-layers 999 --tensor-split 0.5,0.5
```

### vLLM

```bash
# Docker
docker run -d --gpus all \
    -v /mnt/llm-testing/models/hf:/root/.cache/huggingface \
    -p 8000:8000 \
    --name vllm \
    vllm/vllm-openai \
    --model Qwen/Qwen2.5-3B --gpu-memory-utilization 0.95

# For multi-GPU with tensor parallelism
docker run -d --gpus all \
    -v /mnt/llm-testing/models/hf:/root/.cache/huggingface \
    -p 8000:8000 \
    --name vllm \
    vllm/vllm-openai \
    --model Qwen/Qwen2.5-14B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95
```

## 5. Python Environment

```bash
# Install Python 3.9+
apt install -y python3 python3-pip python3-venv

# Create virtual environment (optional)
python3 -m venv /opt/llm-testing-env
source /opt/llm-testing-env/bin/activate

# Install dependencies
pip install -r /mnt/llm-testing/scripts/requirements.txt
```

## 6. Shared Storage Setup

Create the directory structure on the NVMe:

```bash
# On Proxmox host
mkdir -p /mnt/nvme/llm-testing/{models,scripts,docs}
mkdir -p /mnt/nvme/llm-testing/models/{ollama,gguf,hf}
mkdir -p /mnt/nvme/llm-testing/gpus/nvidia/{1660super-1x,1660super-2x,1660super-4x,3060ti,3070,3090ti,3090ti-3060ti}/{configs,results}

# Set permissions
chmod -R 777 /mnt/nvme/llm-testing
```

## 7. Verification

Run the prerequisites check script:

```bash
#!/bin/bash
echo "=== Checking Prerequisites ==="

echo -n "Python 3.9+: "
python3 --version 2>/dev/null || echo "NOT FOUND"

echo -n "Docker: "
docker --version 2>/dev/null || echo "NOT FOUND"

echo -n "NVIDIA Driver: "
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "NOT FOUND"

echo -n "GPU in Docker: "
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi -L 2>/dev/null | head -1 || echo "NOT AVAILABLE"

echo -n "Ollama: "
curl -s http://localhost:11434/api/version 2>/dev/null | grep -o '"version":"[^"]*"' || echo "NOT RUNNING"

echo "=== Done ==="
```

## Multi-GPU Setup Notes

### Identical GPUs (2x or 4x)
- Use `--tensor-split` for llama.cpp with equal ratios
- Use `--tensor-parallel-size` for vLLM
- Set `CUDA_VISIBLE_DEVICES=0,1` if needed

### Mixed GPUs (e.g., 3090 Ti + 3060 Ti)
- Use proportional tensor split: `--tensor-split 0.75,0.25`
- vLLM: prefer pipeline parallelism
- Consider power limiting the larger GPU

### Environment Variables

```bash
# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES=0,1

# For PCIe systems (not NVLink)
export OLLAMA_GPU_COMMUNICATION=p2p
```

## Network Requirements

| Service | Port | Purpose |
|---------|------|---------|
| Ollama | 11434 | API endpoint |
| llama.cpp | 8080 | API endpoint |
| vLLM | 8000 | OpenAI-compatible API |

Outbound access needed:
- `ollama.com` - Model downloads
- `huggingface.co` - Model downloads
- `ghcr.io` - Docker images
