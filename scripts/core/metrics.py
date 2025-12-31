"""
GPU monitoring and cooldown logic for the LLM Testing Framework.

This module provides utilities for monitoring NVIDIA GPU metrics via nvidia-smi
and implementing dynamic cooldown between model tests.
"""

import subprocess
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Union


class GPUMetrics:
    """Represents a snapshot of GPU metrics."""

    def __init__(
        self,
        vram_used_mb: int,
        vram_total_mb: int,
        gpu_util_pct: int,
        temp_c: int,
        power_draw_w: Optional[float] = None,
        gpu_index: int = 0
    ):
        """
        Initialize GPUMetrics instance.

        Args:
            vram_used_mb: VRAM used in MB
            vram_total_mb: Total VRAM in MB
            gpu_util_pct: GPU utilization percentage (0-100)
            temp_c: Temperature in Celsius
            power_draw_w: Power draw in watts
            gpu_index: GPU index (0-based)
        """
        self.vram_used_mb = vram_used_mb
        self.vram_total_mb = vram_total_mb
        self.gpu_util_pct = gpu_util_pct
        self.temp_c = temp_c
        self.power_draw_w = power_draw_w
        self.gpu_index = gpu_index
        self.timestamp = datetime.now().isoformat()

    @property
    def vram_free_mb(self) -> int:
        """Get free VRAM in MB."""
        return self.vram_total_mb - self.vram_used_mb

    @property
    def vram_used_pct(self) -> float:
        """Get VRAM usage as percentage."""
        if self.vram_total_mb == 0:
            return 0.0
        return round(self.vram_used_mb / self.vram_total_mb * 100, 1)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp,
            'gpu_index': self.gpu_index,
            'vram_used_mb': self.vram_used_mb,
            'vram_total_mb': self.vram_total_mb,
            'vram_free_mb': self.vram_free_mb,
            'vram_used_pct': self.vram_used_pct,
            'gpu_util_pct': self.gpu_util_pct,
            'temp_c': self.temp_c,
            'power_draw_w': self.power_draw_w
        }

    def __repr__(self) -> str:
        return (
            f"GPUMetrics(vram={self.vram_used_mb}/{self.vram_total_mb}MB, "
            f"util={self.gpu_util_pct}%, temp={self.temp_c}C)"
        )


def get_gpu_info(gpu_index: int = 0) -> Optional[Dict]:
    """
    Get basic GPU information via nvidia-smi.

    Args:
        gpu_index: GPU index to query

    Returns:
        Dictionary with GPU info or None if nvidia-smi fails
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--query-gpu=name,memory.total,driver_version,cuda_version',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(', ')
        if len(parts) < 3:
            return None

        return {
            'name': parts[0].strip(),
            'vram_total_mb': int(parts[1].strip()),
            'driver_version': parts[2].strip(),
            'cuda_version': parts[3].strip() if len(parts) > 3 else 'unknown'
        }

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


def get_gpu_count() -> int:
    """
    Get the number of available NVIDIA GPUs.

    Returns:
        Number of GPUs or 0 if nvidia-smi fails
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return 0
        return len([l for l in result.stdout.strip().split('\n') if l.startswith('GPU')])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0


def get_gpu_metrics(gpu_index: int = 0) -> Optional[GPUMetrics]:
    """
    Get current GPU metrics via nvidia-smi.

    Args:
        gpu_index: GPU index to query (default: 0)

    Returns:
        GPUMetrics instance or None if query fails
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '-i', str(gpu_index),
                '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(', ')
        if len(parts) < 4:
            return None

        power_draw = None
        if len(parts) > 4:
            try:
                power_draw = float(parts[4].strip())
            except ValueError:
                pass

        return GPUMetrics(
            vram_used_mb=int(parts[0].strip()),
            vram_total_mb=int(parts[1].strip()),
            gpu_util_pct=int(parts[2].strip()),
            temp_c=int(parts[3].strip()),
            power_draw_w=power_draw,
            gpu_index=gpu_index
        )

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        print(f"[WARN] Failed to get GPU metrics: {e}")
        return None


def get_all_gpu_metrics() -> List[GPUMetrics]:
    """
    Get metrics for all available GPUs.

    Returns:
        List of GPUMetrics instances
    """
    gpu_count = get_gpu_count()
    metrics = []

    for i in range(gpu_count):
        m = get_gpu_metrics(i)
        if m is not None:
            metrics.append(m)

    return metrics


def get_total_power_draw() -> float:
    """
    Get total power draw across all GPUs.

    Returns:
        Total power draw in watts, or 0 if unable to measure
    """
    metrics = get_all_gpu_metrics()
    total = 0.0
    for m in metrics:
        if m.power_draw_w is not None:
            total += m.power_draw_w
    return total


def estimate_system_power(overhead_watts: float = 200.0) -> float:
    """
    Estimate total system power consumption.

    Args:
        overhead_watts: Estimated power for CPU, RAM, etc.

    Returns:
        Estimated system power in watts
    """
    return get_total_power_draw() + overhead_watts


def apply_power_limit(gpu_index: int, watts: int) -> Tuple[bool, str]:
    """
    Apply power limit to a GPU.

    Requires root/sudo access to nvidia-smi.

    Args:
        gpu_index: GPU index
        watts: Power limit in watts

    Returns:
        Tuple of (success, message)
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '-i', str(gpu_index), '-pl', str(watts)],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return True, f"Power limit set to {watts}W for GPU {gpu_index}"
        else:
            return False, f"Failed to set power limit: {result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return False, "nvidia-smi command timed out"
    except FileNotFoundError:
        return False, "nvidia-smi not found"
    except Exception as e:
        return False, f"Error: {str(e)}"


def apply_power_limits(power_limits: str) -> List[Tuple[int, bool, str]]:
    """
    Apply power limits to multiple GPUs.

    Args:
        power_limits: Comma-separated power limits (e.g., "280,160")

    Returns:
        List of (gpu_index, success, message) tuples
    """
    results = []
    limits = [int(w.strip()) for w in power_limits.split(',')]

    for gpu_idx, watts in enumerate(limits):
        success, msg = apply_power_limit(gpu_idx, watts)
        results.append((gpu_idx, success, msg))
        print(f"  GPU {gpu_idx}: {msg}")

    return results


class GPUMonitor:
    """
    Background thread for monitoring GPU metrics.

    Captures metrics at regular intervals and tracks the current phase
    (idle, model_loading, inference).
    """

    def __init__(
        self,
        interval_sec: float = 1.0,
        gpu_indices: Optional[List[int]] = None,
        callback: Optional[Callable[[List[GPUMetrics], str], None]] = None
    ):
        """
        Initialize GPUMonitor.

        Args:
            interval_sec: Sampling interval in seconds
            gpu_indices: List of GPU indices to monitor (default: all)
            callback: Optional callback function for each sample
        """
        self.interval_sec = interval_sec
        self.gpu_indices = gpu_indices or list(range(get_gpu_count()))
        self.callback = callback

        self._phase = 'idle'
        self._running = False
        self._thread = None  # type: Optional[threading.Thread]
        self._samples = []  # type: List[Tuple[List[GPUMetrics], str]]
        self._lock = threading.Lock()

    @property
    def phase(self) -> str:
        """Get current phase."""
        return self._phase

    @phase.setter
    def phase(self, value: str) -> None:
        """Set current phase."""
        self._phase = value

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = []
                for idx in self.gpu_indices:
                    m = get_gpu_metrics(idx)
                    if m is not None:
                        metrics.append(m)

                if metrics:
                    with self._lock:
                        self._samples.append((metrics, self._phase))

                    if self.callback is not None:
                        self.callback(metrics, self._phase)

            except Exception as e:
                print(f"[WARN] GPU monitor error: {e}")

            time.sleep(self.interval_sec)

    def get_samples(self) -> List[Tuple[List[GPUMetrics], str]]:
        """
        Get all collected samples.

        Returns:
            List of (metrics_list, phase) tuples
        """
        with self._lock:
            return list(self._samples)

    def clear_samples(self) -> None:
        """Clear collected samples."""
        with self._lock:
            self._samples.clear()

    def get_summary(self) -> Dict:
        """
        Get summary statistics from collected samples.

        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return {'error': 'No samples collected'}

        # Aggregate by GPU index
        by_gpu = {}  # type: Dict[int, Dict]

        for metrics_list, phase in samples:
            for m in metrics_list:
                if m.gpu_index not in by_gpu:
                    by_gpu[m.gpu_index] = {
                        'vram_used_mb': [],
                        'gpu_util_pct': [],
                        'temp_c': [],
                        'power_draw_w': []
                    }

                by_gpu[m.gpu_index]['vram_used_mb'].append(m.vram_used_mb)
                by_gpu[m.gpu_index]['gpu_util_pct'].append(m.gpu_util_pct)
                by_gpu[m.gpu_index]['temp_c'].append(m.temp_c)
                if m.power_draw_w is not None:
                    by_gpu[m.gpu_index]['power_draw_w'].append(m.power_draw_w)

        # Calculate statistics
        summary = {'gpus': {}, 'total_samples': len(samples)}

        for gpu_idx, data in by_gpu.items():
            summary['gpus'][gpu_idx] = {
                'vram_mb': {
                    'min': min(data['vram_used_mb']),
                    'max': max(data['vram_used_mb']),
                    'avg': round(sum(data['vram_used_mb']) / len(data['vram_used_mb']), 1)
                },
                'gpu_util_pct': {
                    'min': min(data['gpu_util_pct']),
                    'max': max(data['gpu_util_pct']),
                    'avg': round(sum(data['gpu_util_pct']) / len(data['gpu_util_pct']), 1)
                },
                'temp_c': {
                    'min': min(data['temp_c']),
                    'max': max(data['temp_c']),
                    'avg': round(sum(data['temp_c']) / len(data['temp_c']), 1)
                }
            }

            if data['power_draw_w']:
                summary['gpus'][gpu_idx]['power_w'] = {
                    'min': round(min(data['power_draw_w']), 1),
                    'max': round(max(data['power_draw_w']), 1),
                    'avg': round(sum(data['power_draw_w']) / len(data['power_draw_w']), 1)
                }

        return summary


def wait_for_cooldown(
    target_temp_c: int = 50,
    min_wait_sec: int = 30,
    max_wait_sec: int = 300,
    check_interval_sec: float = 5.0,
    baseline_vram_mb: Optional[int] = None,
    gpu_index: int = 0,
    verbose: bool = True
) -> bool:
    """
    Wait for GPU to cool down before loading next model.

    Args:
        target_temp_c: Target temperature in Celsius
        min_wait_sec: Minimum wait time in seconds
        max_wait_sec: Maximum wait time in seconds
        check_interval_sec: Interval between checks in seconds
        baseline_vram_mb: Expected baseline VRAM usage (None = auto-detect)
        gpu_index: GPU index to monitor
        verbose: Print status updates

    Returns:
        True if target reached, False if timed out
    """
    start_time = time.time()
    target_vram_threshold = 0.1  # 10% above baseline

    # Get baseline VRAM if not specified
    if baseline_vram_mb is None:
        initial_metrics = get_gpu_metrics(gpu_index)
        if initial_metrics is not None:
            # Use total VRAM * 10% as baseline threshold
            baseline_vram_mb = int(initial_metrics.vram_total_mb * 0.1)
        else:
            baseline_vram_mb = 500  # Default fallback

    vram_target = int(baseline_vram_mb * (1 + target_vram_threshold))

    if verbose:
        print(f"  Cooldown: min={min_wait_sec}s | temp_target<{target_temp_c}C | vram_target<{vram_target}MB")

    while True:
        elapsed = time.time() - start_time

        # Get current metrics
        metrics = get_gpu_metrics(gpu_index)
        if metrics is None:
            if verbose:
                print("  [WARN] Could not read GPU metrics")
            time.sleep(check_interval_sec)
            continue

        # Check if we've met targets
        vram_ok = metrics.vram_used_mb <= vram_target
        temp_ok = metrics.temp_c <= target_temp_c
        min_wait_ok = elapsed >= min_wait_sec

        if verbose:
            status = (
                f"\r  Cooldown: {int(elapsed)}s | "
                f"VRAM: {metrics.vram_used_mb}MB (target: <{vram_target}) | "
                f"Temp: {metrics.temp_c}C (target: <{target_temp_c})"
            )
            print(status, end='', flush=True)

        if min_wait_ok and vram_ok and temp_ok:
            if verbose:
                print(f"\n  GPU idle: {metrics.vram_used_mb}MB VRAM, {metrics.temp_c}C")
            return True

        if elapsed >= max_wait_sec:
            if verbose:
                print(f"\n  [WARN] Cooldown timeout after {max_wait_sec}s (temp={metrics.temp_c}C)")
            return False

        # Adaptive target if GPU is very hot - relax target to avoid excessive waits
        if metrics.temp_c > 70 and not temp_ok:
            # Allow slightly higher temp target (up to 55C) for very hot GPUs
            target_temp_c = min(target_temp_c + 5, 55)

        time.sleep(check_interval_sec)


def verify_vram_cleared(
    expected_baseline_mb: int,
    tolerance_pct: float = 0.2,
    gpu_index: int = 0,
    timeout_sec: int = 30
) -> bool:
    """
    Verify that VRAM has been cleared after unloading a model.

    Args:
        expected_baseline_mb: Expected baseline VRAM in MB
        tolerance_pct: Tolerance percentage above baseline
        gpu_index: GPU index to check
        timeout_sec: Maximum wait time

    Returns:
        True if VRAM is at or near baseline
    """
    threshold = int(expected_baseline_mb * (1 + tolerance_pct))
    start_time = time.time()

    while time.time() - start_time < timeout_sec:
        metrics = get_gpu_metrics(gpu_index)
        if metrics is not None and metrics.vram_used_mb <= threshold:
            return True
        time.sleep(1)

    return False


class PowerMonitor:
    """
    Monitor power consumption and abort if threshold exceeded.

    Background thread that monitors GPU power draw and sets an abort
    flag if the estimated system power exceeds the threshold for a
    specified duration.
    """

    def __init__(
        self,
        threshold_watts: float,
        overhead_watts: float = 200.0,
        duration_sec: int = 10,
        check_interval_sec: float = 1.0
    ):
        """
        Initialize PowerMonitor.

        Args:
            threshold_watts: Power threshold in watts
            overhead_watts: Estimated non-GPU power consumption
            duration_sec: How long threshold must be exceeded
            check_interval_sec: Check interval in seconds
        """
        self.threshold_watts = threshold_watts
        self.overhead_watts = overhead_watts
        self.duration_sec = duration_sec
        self.check_interval_sec = check_interval_sec

        self._abort_flag = False
        self._running = False
        self._thread = None  # type: Optional[threading.Thread]
        self._exceeded_since = None  # type: Optional[float]
        self._lock = threading.Lock()

    @property
    def should_abort(self) -> bool:
        """Check if power threshold has been exceeded."""
        with self._lock:
            return self._abort_flag

    def start(self) -> None:
        """Start the power monitoring thread."""
        if self._running:
            return

        self._running = True
        self._abort_flag = False
        self._exceeded_since = None
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the power monitoring thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _monitor_loop(self) -> None:
        """Main power monitoring loop."""
        while self._running:
            try:
                total_power = estimate_system_power(self.overhead_watts)

                if total_power > self.threshold_watts:
                    if self._exceeded_since is None:
                        self._exceeded_since = time.time()
                    elif time.time() - self._exceeded_since >= self.duration_sec:
                        with self._lock:
                            self._abort_flag = True
                        print(
                            f"\n[POWER] Threshold {self.threshold_watts}W exceeded for "
                            f">{self.duration_sec}s (current: {total_power:.0f}W)"
                        )
                else:
                    self._exceeded_since = None

            except Exception as e:
                print(f"[WARN] Power monitor error: {e}")

            time.sleep(self.check_interval_sec)

    def reset(self) -> None:
        """Reset the abort flag and timer."""
        with self._lock:
            self._abort_flag = False
            self._exceeded_since = None
