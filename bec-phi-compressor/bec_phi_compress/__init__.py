"""
BEC Φ-Compressor - Deterministic Numeric Data Compression

GPU-accelerated compression using Φ-projection operators.
No ML, no training, no heuristics - deterministic mathematical compression.
"""

from .compressor import PhiCompressor
from .decompressor import PhiDecompressor
from .projection import PhiProjection

__version__ = "1.0.0"
__all__ = ["PhiCompressor", "PhiDecompressor", "PhiProjection"]
