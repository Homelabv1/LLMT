# Troubleshooting Guide

Common issues and solutions for the LLM Testing Framework.

## GPU Issues

### GPU Not Detected

**Symptom**: `nvidia-smi` returns error or shows no GPUs.

**Solutions**:
1. Check driver installation:
   ```bash
   lsmod | grep nvidia
   dpkg -l | grep nvidia-driver
   ```

2. Load kernel modules:
   ```bash
   modprobe nvidia
   modprobe nvidia_uvm
   ```

3. Check device nodes:
   ```bash
   ls -la /dev/nvidia*
   ```

4. For LXC containers, verify passthrough config in `/etc/pve/lxc/<vmid>.conf`

### GPU Not Visible in Docker

**Symptom**: `docker run --gpus all nvidia/cuda:12.0-base nvidia-smi` fails.

**Solutions**:
1. Install NVIDIA Container Toolkit:
   ```bash
   apt install nvidia-container-toolkit
   nvidia-ctk runtime configure --runtime=docker
   systemctl restart docker
   ```

2. Check Docker runtime config:
   ```bash
   cat /etc/docker/daemon.json
   ```

   Should include:
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

### VRAM Not Clearing After Unload

**Symptom**: VRAM usage remains high after model unload.

**Solutions**:
1. For Ollama, verify unload with:
   ```bash
   curl http://localhost:11434/api/ps
   ```

2. Force garbage collection by loading a tiny model:
   ```bash
   curl -X POST http://localhost:11434/api/generate \
        -d '{"model":"tinyllama","prompt":"hi","keep_alive":0}'
   ```

3. Restart the inference engine container

4. In extreme cases, reset GPU:
   ```bash
   nvidia-smi --gpu-reset
   ```

## Model Loading Issues

### Download Timeout

**Symptom**: Model download times out.

**Solutions**:
1. Increase download timeout:
   ```bash
   python scripts/batch_test.py --download-timeout 7200  # 2 hours
   ```

2. Pre-download models:
   ```bash
   # Ollama
   ollama pull qwen3:4b

   # llama.cpp - download GGUF manually
   wget https://huggingface.co/.../model.gguf -O /mnt/llm-testing/models/gguf/
   ```

3. Check network connectivity and speed

### Load Timeout

**Symptom**: Model loads partially then times out.

**Solutions**:
1. Increase load timeout:
   ```bash
   python scripts/run_test.py --load-timeout 900  # 15 minutes
   ```

2. Check if model fits in VRAM:
   ```bash
   nvidia-smi -l 1  # Watch VRAM usage during load
   ```

3. Try a more quantized version (e.g., Q4_K_M instead of Q5_K_M)

### Model Not Found

**Symptom**: "Model not found" errors.

**Solutions**:
1. For Ollama, check available models:
   ```bash
   ollama list
   ```

2. For llama.cpp, verify GGUF file path:
   ```bash
   ls -la /mnt/llm-testing/models/gguf/
   ```

3. For vLLM, verify HuggingFace model exists:
   ```bash
   # Check on huggingface.co or use:
   python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen2.5-3B'))"
   ```

## Connection Issues

### Engine Not Reachable

**Symptom**: "Connection refused" or timeout.

**Solutions**:
1. Check if container is running:
   ```bash
   docker ps | grep ollama
   ```

2. Check port binding:
   ```bash
   netstat -tlnp | grep 11434
   ```

3. Check firewall:
   ```bash
   ufw status
   iptables -L
   ```

4. Test from inside container:
   ```bash
   docker exec ollama curl http://localhost:11434/api/version
   ```

### Intermittent Timeouts

**Symptom**: Queries randomly timeout.

**Solutions**:
1. Check GPU temperature:
   ```bash
   nvidia-smi -q -d temperature
   ```

2. Monitor system resources:
   ```bash
   htop
   watch -n 1 nvidia-smi
   ```

3. Check for thermal throttling in nvidia-smi output

4. Increase cooldown time:
   ```bash
   python scripts/batch_test.py --min-cooldown 60 --target-temp 45
   ```

## Performance Issues

### Slow Inference

**Symptom**: Much slower than expected tokens/second.

**Solutions**:
1. Check GPU utilization:
   ```bash
   nvidia-smi -l 1
   ```

2. Ensure model is on GPU:
   - For Ollama: Check that GPU shows usage during inference
   - For llama.cpp: Use `--n-gpu-layers 999`
   - For vLLM: Check `--gpu-memory-utilization 0.95`

3. Check for CPU offload (model too large for GPU)

4. For multi-GPU, check tensor split is correct

### High Latency First Query

**Symptom**: First query very slow, subsequent queries fast.

This is expected behavior - the model needs to be loaded into VRAM.

**Mitigation**:
1. Use warmup queries (default: 1)
2. Keep model loaded with `keep_alive`

## Results Issues

### Corrupted Results File

**Symptom**: JSON parse errors when reading results.

**Solutions**:
1. Check file for truncation:
   ```bash
   tail -c 100 questions.jsonl
   ```

2. Remove corrupted last line:
   ```bash
   head -n -1 questions.jsonl > questions.jsonl.fixed
   mv questions.jsonl.fixed questions.jsonl
   ```

3. Validate JSONL:
   ```bash
   while read line; do echo "$line" | python -m json.tool > /dev/null; done < questions.jsonl
   ```

### Missing Results

**Symptom**: Some questions not saved.

**Solutions**:
1. Check for errors in the test output
2. Look at failures.log:
   ```bash
   cat gpus/nvidia/1660super-1x/results/failures.log
   ```

3. Check load_failure.json if model failed to load

### Resume Not Working

**Symptom**: Tests restart from beginning.

**Solutions**:
1. Verify results directory path matches:
   ```bash
   ls -la gpus/nvidia/1660super-1x/results/ollama/qwen3-4b/
   ```

2. Check questions.jsonl exists and is readable

3. Use `--no-resume` to start fresh if needed

## Power and Thermal Issues

### GPU Overheating

**Symptom**: Temperature exceeds safe limits.

**Solutions**:
1. Increase cooldown:
   ```bash
   python scripts/batch_test.py --min-cooldown 120 --target-temp 40
   ```

2. Apply power limits:
   ```bash
   sudo nvidia-smi -i 0 -pl 200  # Limit to 200W
   ```

3. Improve case cooling

### Power Limit Errors

**Symptom**: "Permission denied" when setting power limit.

**Solutions**:
1. Run with sudo:
   ```bash
   sudo python scripts/batch_test.py --power-limit "280,160"
   ```

2. Enable persistence mode:
   ```bash
   sudo nvidia-smi -pm 1
   ```

3. In LXC, may need privileged container

## Container Issues

### Ollama Out of Memory

**Symptom**: Container crashes or becomes unresponsive.

**Solutions**:
1. Increase container memory limit

2. Use smaller/more quantized models

3. Set Ollama environment variables:
   ```bash
   docker run -e OLLAMA_NUM_PARALLEL=1 ...
   ```

### llama.cpp Server Crash

**Symptom**: Server exits during model load.

**Solutions**:
1. Check container logs:
   ```bash
   docker logs llamacpp
   ```

2. Reduce context size:
   ```bash
   --ctx-size 2048
   ```

3. Reduce GPU layers if model too large:
   ```bash
   --n-gpu-layers 20  # Partial offload
   ```

## Getting Help

If issues persist:

1. Check logs with full verbosity
2. Capture nvidia-smi output during failure
3. Note exact error messages
4. Report issue with system details:
   - GPU model and driver version
   - Docker version
   - Python version
   - Inference engine version
