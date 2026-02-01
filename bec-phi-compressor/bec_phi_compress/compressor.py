"""
Φ-Compressor - File-Level Compression Interface

Handles file I/O, metadata generation, and hash verification.
"""

import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Union
from .projection import PhiProjection


class PhiCompressor:
    """
    File-level Φ-compression with deterministic outputs.
    
    Supports .npy, .csv, .bin files with hash verification.
    """
    
    def __init__(self, phi_range: Tuple[float, float] = (0.0, 10.0),
                 resolution: int = 5000, backend: str = "auto"):
        """
        Initialize compressor.
        
        Args:
            phi_range: Φ projection range
            resolution: Number of Φ levels
            backend: Computational backend
        """
        self.projection = PhiProjection(phi_range, resolution, backend)
        
    def compress_file(self, input_path: Union[str, Path], 
                     output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Compress a numeric file.
        
        Args:
            input_path: Path to input file
            output_dir: Directory for compressed outputs
            
        Returns:
            Compression metadata
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load input data
        data, file_metadata = self._load_file(input_path)
        
        # Apply compression
        compressed, projection_metadata = self.projection.compress(data)
        
        # Compute compression metrics
        compression_ratio = self.projection.compute_compression_ratio(data, compressed)
        
        # Create full metadata
        full_metadata = {
            "file_metadata": file_metadata,
            "projection_metadata": projection_metadata,
            "compression_metrics": {
                "compression_ratio": compression_ratio,
                "original_size_bytes": data.nbytes,
                "compressed_size_bytes": compressed.nbytes,
                "size_reduction_percent": (1 - 1/compression_ratio) * 100
            }
        }
        
        # Save outputs
        base_name = input_path.stem
        compressed_path = output_dir / f"{base_name}_compressed.npy"
        manifest_path = output_dir / f"{base_name}_manifest.json"
        hash_path = output_dir / f"{base_name}_hash.txt"
        
        # Save compressed data
        np.save(compressed_path, compressed)
        
        # Save manifest
        with open(manifest_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        # Save hash file
        input_hash = self._compute_file_hash(input_path)
        compressed_hash = self.projection._compute_hash(compressed)
        hash_content = f"Input file hash: {input_hash}\nCompressed data hash: {compressed_hash}\n"
        with open(hash_path, 'w') as f:
            f.write(hash_content)
        
        # Add output paths to metadata
        full_metadata["output_files"] = {
            "compressed_data": str(compressed_path),
            "manifest": str(manifest_path),
            "hash_file": str(hash_path)
        }
        
        return full_metadata
    
    def _load_file(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load numeric data from file."""
        file_metadata = {
            "filename": file_path.name,
            "file_size_bytes": file_path.stat().st_size,
            "file_extension": file_path.suffix.lower()
        }
        
        if file_path.suffix.lower() == '.npy':
            data = np.load(file_path)
            file_metadata["data_type"] = "numpy_array"
            
        elif file_path.suffix.lower() == '.csv':
            data = np.loadtxt(file_path, delimiter=',')
            file_metadata["data_type"] = "csv_numeric"
            
        elif file_path.suffix.lower() == '.bin':
            # Assume float32 binary data
            data = np.fromfile(file_path, dtype=np.float32)
            file_metadata["data_type"] = "binary_float32"
            
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        file_metadata.update({
            "data_shape": list(data.shape),
            "data_dtype": str(data.dtype),
            "data_size_bytes": data.nbytes
        })
        
        return data, file_metadata
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
