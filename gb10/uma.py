"""
gb10/uma.py - GB10/DGX Spark unified memory support for SD-SCM inference

The Grace Blackwell Superchip (GB10) uses a single physical LPDDR5X memory
pool shared between the Grace CPU and Blackwell GPU. Standard discrete GPU
memory management assumptions do not apply.

On GB10:
- cudaMemGetInfo returns N/A — /proc/meminfo is ground truth
- KV cache from past_key_values accumulates in the shared pool across calls
- Thousands of chunk_continuation() calls without cleanup cause memory
  pressure to build until the system hangs or becomes unresponsive

This module provides:
- GB10 platform detection
- MemAvailable check before inference batches
- KV cache and tensor cleanup between sample batches
- PSI memory pressure monitoring during long generation runs
- gb10_safe_sample_sequences() — drop-in replacement for sample_sequences()

No new dependencies. No Docker. No vLLM.
Requires only torch and the existing sdscm environment.
"""

import os
import gc
import logging
from typing import Optional

import torch
from tqdm import tqdm

from gb10.memory import get_available_kb, get_swap_pressure, memory_snapshot

logger = logging.getLogger(__name__)


# Minimum MemAvailable in KB before refusing to start a new batch.
# On GB10 with 128GB unified pool this is conservative.
# Adjust based on model size and sequence length.
_MIN_AVAILABLE_KB = 4 * 1024 * 1024  # 4GB


def is_gb10() -> bool:
    """
    Detect GB10/DGX Spark platform.
    Returns True if running on Grace Blackwell unified memory system.
    """
    # Check for NVLink-C2C unified memory — GB10 specific
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip().lower()
            if "gb10" in gpu_name or "grace" in gpu_name or "blackwell" in gpu_name:
                return True
    except Exception:
        pass

    # Check for unified memory indicator — no discrete VRAM
    try:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            # On GB10, total_memory reflects the full unified pool
            # /proc/meminfo MemTotal should be close to cuda total
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total_kb = int(line.split()[1])
                        mem_total_bytes = mem_total_kb * 1024
                        # If CUDA total is within 10% of system total — unified memory
                        if abs(total - mem_total_bytes) / mem_total_bytes < 0.10:
                            return True
                        break
    except Exception:
        pass

    return False


def check_memory_available(min_available_kb: int = _MIN_AVAILABLE_KB) -> bool:
    """
    Check if sufficient memory is available before starting inference.
    Returns True if safe to proceed, False if memory is critically low.
    """
    available_kb = get_available_kb()
    available_gb = available_kb / (1024 * 1024)

    if available_kb < min_available_kb:
        logger.warning(
            f"GB10 UMA: MemAvailable={available_gb:.1f}GB below threshold "
            f"({min_available_kb // (1024*1024)}GB). Consider freeing memory before sampling."
        )
        return False

    logger.debug(f"GB10 UMA: MemAvailable={available_gb:.1f}GB — safe to proceed")
    return True


def flush_inference_cache(model, verbose: bool = False):
    """
    Release KV cache and unreferenced tensors between sample batches.
    On GB10 unified memory, these stay in the shared LPDDR5X pool
    and accumulate across calls to chunk_continuation().
    """
    if hasattr(model, "past_key_values"):
        model.past_key_values = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    if verbose:
        snap = memory_snapshot()
        logger.debug(
            f"GB10 UMA: After flush — MemAvailable={snap['available_gb']:.1f}GB "
            f"PSI={snap['psi_some_avg10']:.2f}"
        )


def gb10_safe_sample_sequences(
    model,
    tokenizer,
    sequence_sample_space,
    num_samples: int,
    verbose: bool = True,
    flush_every: int = 50,
    min_available_kb: int = _MIN_AVAILABLE_KB,
):
    """
    Drop-in replacement for sdscm.sample_sequences() with GB10 UMA safety.

    Adds:
    - MemAvailable check before starting
    - KV cache flush every flush_every samples
    - PSI pressure logging during long runs
    - Memory state logged at completion

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        sequence_sample_space: SD-SCM sample space definition
        num_samples: number of sequences to generate
        verbose: show progress bar
        flush_every: flush KV cache every N samples (default 50)
        min_available_kb: minimum MemAvailable before starting (default 4GB)

    Returns:
        List of samples — same format as sample_sequences()
    """
    from sdscm import sample_sequence

    # Pre-run memory check
    if not check_memory_available(min_available_kb):
        logger.warning(
            "GB10 UMA: Proceeding despite low memory. "
            "Monitor /proc/pressure/memory for stalls."
        )

    samples = []
    psi_warnings = 0

    for i in tqdm(range(num_samples), disable=not verbose):
        sample = sample_sequence(model, tokenizer, sequence_sample_space)
        samples.append(sample)

        # Periodic flush to prevent KV cache accumulation
        if (i + 1) % flush_every == 0:
            flush_inference_cache(model)

            # PSI check — log if memory pressure is building
            psi = get_swap_pressure()
            if psi > 5.0:
                psi_warnings += 1
                avail_gb = get_available_kb() / (1024 * 1024)
                logger.warning(
                    f"GB10 UMA: PSI memory pressure={psi:.1f}% "
                    f"MemAvailable={avail_gb:.1f}GB at sample {i+1}/{num_samples}"
                )

    # Post-run summary
    if verbose:
        snap = memory_snapshot()
        logger.info(
            f"GB10 UMA: Completed {num_samples} samples. "
            f"MemAvailable={snap['available_gb']:.1f}GB "
            f"PSI warnings={psi_warnings}"
        )

    return samples


def setup_gb10(model, verbose: bool = True) -> bool:
    """
    One-call GB10 setup. Call once after loading your model.

    Detects platform, logs memory state, returns True if GB10 detected.

    Usage:
        from gb10.uma import setup_gb10, gb10_safe_sample_sequences
        is_spark = setup_gb10(model)

    Then replace sample_sequences() with gb10_safe_sample_sequences().
    """
    on_gb10 = is_gb10()
    snap = memory_snapshot()

    if on_gb10:
        logger.info(
            f"GB10/DGX Spark detected. "
            f"MemAvailable={snap['available_gb']:.1f}GB "
            f"PSI={snap['psi_some_avg10']:.2f}"
        )
        if verbose:
            print(
                f"[gb10] DGX Spark unified memory platform detected.\n"
                f"[gb10] MemAvailable: {snap['available_gb']:.1f} GB\n"
                f"[gb10] PSI memory pressure: {snap['psi_some_avg10']:.2f}%\n"
                f"[gb10] Use gb10_safe_sample_sequences() for long generation runs."
            )
    else:
        if verbose:
            print(
                f"[gb10] Non-GB10 platform. "
                f"MemAvailable: {snap['available_gb']:.1f} GB. "
                f"Standard inference path."
            )

    return on_gb10
