# gb10/ — GB10/DGX Spark Unified Memory Support

The NVIDIA Grace Blackwell Superchip (GB10) combines Grace CPU and Blackwell GPU
compute behind a shared LPDDR5X memory subsystem.

This module provides GB10/DGX Spark-aware support for:
- unified memory pressure detection
- MemAvailable and PSI stall signal collection
- KV cache lifecycle management during long inference runs
- workload-aware sampling with memory pressure monitoring

---

## Module Structure

```text
gb10/
├── memory.py  — MemAvailable and PSI memory pressure readers
└── uma.py     — GB10 detection, cache management, inference wrappers
```

---

## Issues

https://github.com/parallelArchitect/sequence-driven-scms/issues
