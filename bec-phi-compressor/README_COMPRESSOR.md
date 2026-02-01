# BEC Φ-Compressor

**Deterministic numeric data compression using physics-inspired projection operators.**

No ML. No training. No heuristics. GPU-accelerated with bounded reconstruction.

---

## What this is

- **Deterministic compression** using Φ-projection operators
- **GPU-accelerated** with CPU fallback
- **Bounded reconstruction** with measurable error
- **Hash verification** for integrity
- **Format-agnostic** numeric data processing

## What this is not

- Not machine learning
- Not lossless compression
- Not semantic compression
- Not file format specific

## Quick Start

### Installation

```bash
# Install dependencies
pip install numpy scipy

# Optional: GPU support
pip install cupy
```

### Compression

```bash
python -m bec_phi_compress.cli compress \
  --input data.npy \
  --phi 0.0 10.0 \
  --resolution 5000 \
  --backend gpu \
  --out output/compressed_run
```

### Decompression

```bash
python -m bec_phi_compress.cli decompress \
  --input output/compressed_run/data_compressed.npy \
  --out output/reconstructed.npy
```

## Supported Formats

- **NumPy arrays** (.npy)
- **CSV numeric data** (.csv)
- **Binary float data** (.bin)

## Compression Contract

**Input**: Numeric array X ∈ ℝⁿ  
**Operator**: Π_Φ(X) using FMI ladder constraints  
**Output**: Compressed data C with bounded reconstruction X̂ = Π_Φ⁻¹(C)

**Guarantee**: ‖X − X̂‖₂ / ‖X‖₂ ≤ ε (ε reported, not tuned)

## Performance

Typical results on test datasets:
- **Compression ratio**: 2-10x depending on data structure
- **GPU speedup**: 5-20x over CPU
- **Reconstruction error**: 10⁻⁶ to 10⁻³ (data dependent)

## Outputs

Compression produces:
- `*_compressed.npy` - Compressed data
- `*_manifest.json` - Metadata and parameters
- `*_hash.txt` - SHA256 verification hashes
- `*_performance.json` - Timing and metrics

## Mathematical Foundation

Uses the FMI (Frozen Mathematical Interface) ladder constraints:
- **σ₈ containment**: Variance normalization
- **Kernel support**: Localization via Gaussian kernels
- **k-orthogonality**: Mode orthogonalization
- **Ordering/causality**: Φ-depth stratification

## License

MIT License - see LICENSE file

## Citation

See CITATION.cff for proper citation format.
