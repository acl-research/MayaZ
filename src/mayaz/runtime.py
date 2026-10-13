"""ONNX Runtime session for the served model (numpy + onnxruntime only: the server imports this, never torch).

Inputs are the arrays `render.collate` builds; the graph returns raw logits [B, K] with -1e4 in padded slots.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

INPUT_NAMES = ("input_ids", "position_ids", "segment_ids", "q_mask", "opt_start", "opt_end", "opt_mask")


def session_options(threads: int = 1) -> ort.SessionOptions:
    """Low-memory CPU settings for a 512 MB box: no arena, no memory-pattern pre-allocation, one thread, no spinning."""
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.enable_cpu_mem_arena = False  # the arena keeps the largest request's buffers for the life of the process
    o.enable_mem_pattern = False  # no pre-planned buffers sized for each new input shape
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")  # no busy-wait on a 0.1 vCPU quota
    return o


def _libc():
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        return lib if hasattr(lib, "mallopt") and hasattr(lib, "malloc_trim") else None
    except OSError:
        return None


_LIBC = _libc()  # glibc only; elsewhere the allocator settings below are skipped
M_MMAP_THRESHOLD = -3


def low_memory_malloc() -> bool:
    """Fix glibc's mmap threshold (MAYAZ_MMAP_THRESHOLD bytes, default 1 MiB; 0 = leave glibc's default).

    glibc raises the threshold on the fly, after which large activation buffers come from the heap and stay resident
    after the pass. A fixed threshold serves them with mmap, which returns them to the OS on free: 68M's peak over
    eleven 600-token passes falls from ~480 MB to ~200 MB."""
    size = int(os.environ.get("MAYAZ_MMAP_THRESHOLD", 1024 * 1024))
    return bool(_LIBC and size and _LIBC.mallopt(M_MMAP_THRESHOLD, size))


class Session:
    def __init__(self, path: str | Path, threads: int = 1, providers: list[str] | None = None):
        self.low_memory = low_memory_malloc()
        self.options = session_options(threads)
        self.sess = ort.InferenceSession(str(path), self.options, providers=providers or ["CPUExecutionProvider"])
        self.provider = self.sess.get_providers()[0]  # the one actually used (CUDA may fall back to CPU)

    def logits(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        out = self.sess.run(["logits"], {k: batch[k] for k in INPUT_NAMES})[0]
        if self.low_memory:
            _LIBC.malloc_trim(0)  # hand freed heap pages back to the OS after every pass
        return out


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def bucket(k: int) -> str:
    """Cardinality bucket {2, 3-5, 6-20, 21+} (same as mayaz.eval.metrics.bucket_of)."""
    if k < 2:
        raise ValueError(f"need at least 2 options, got {k}")
    return "2" if k == 2 else "3-5" if k <= 5 else "6-20" if k <= 20 else "21+"


def temperature_for(temps: dict, qtype: str, k: int) -> float:
    b = bucket(k)
    return float(temps["by_key"].get(f"{qtype}|{b}", temps["by_bucket"].get(b, temps["global"])))
