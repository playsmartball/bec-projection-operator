"""
Φ-Decompressor - File-Level Decompression Interface

Handles reconstruction and error verification.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Union
from .projection import PhiProjection


class PhiDecompressor:
    """
    File-level Φ-decompression with bounded reconstruction.
    """
    
    def __init__(self):
        """Initialize decompressor."""
        pass
    
    def decompress_file(self, compressed_path: Union[str, Path],
                      manifest_path: Union[str, Path],
                      output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Decompress a file and compute reconstruction error.
        
        Args:
            compressed_path: Path to compressed .npy file
            manifest_path: Path to manifest.json
            output_path: Path for reconstructed output
            
        Returns:
            Decompression metadata including reconstruction error
        """
        compressed_path = Path(compressed_path)
        manifest_path = Path(manifest_path)
        output_path = Path(output_path)
        
        # Load compressed data
        compressed = np.load(compressed_path)
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Reconstruct projection operator from metadata
        proj_metadata = manifest["projection_metadata"]
        projection = PhiProjection(
            phi_range=tuple(proj_metadata["phi_range"]),
            resolution=proj_metadata["resolution"],
            backend=proj_metadata["backend"]
        )
        
        # Decompress
        reconstructed, _ = projection.decompress(compressed, proj_metadata)
        
        # Load original data for error computation (if available)
        reconstruction_error = None
        original_path = self._find_original_file(manifest)
        if original_path and original_path.exists():
            original_data = self._load_original_data(original_path, manifest["file_metadata"])
            reconstruction_error = projection.compute_reconstruction_error(original_data, reconstructed)
        
        # Save reconstructed data
        np.save(output_path, reconstructed)
        
        # Create decompression metadata
        decomp_metadata = {
            "reconstruction_metrics": {
                "reconstruction_error": reconstruction_error,
                "reconstructed_shape": list(reconstructed.shape),
                "reconstructed_dtype": str(reconstructed.dtype),
                "reconstructed_size_bytes": reconstructed.nbytes
            },
            "compression_metrics": manifest["compression_metrics"],
            "output_file": str(output_path)
        }
        
        return decomp_metadata
    
    def _find_original_file(self, manifest: Dict[str, Any]) -> Path:
        """Attempt to find original file from manifest."""
        # This is a simple heuristic - in practice, the original file
        # location would be stored or provided by the user
        filename = manifest["file_metadata"]["filename"]
        
        # Try common locations
        search_paths = [
            Path("../data/raw") / filename,
            Path("data/raw") / filename,
            Path(filename)
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return Path(filename)  # Return as reference even if not found
    
    def _load_original_data(self, file_path: Path, file_metadata: Dict[str, Any]) -> np.ndarray:
        """Load original data using file metadata."""
        if not file_path.exists():
            raise FileNotFoundError(f"Original file not found: {file_path}")
        
        if file_metadata["file_extension"] == '.npy':
            return np.load(file_path)
        elif file_metadata["file_extension"] == '.csv':
            return np.loadtxt(file_path, delimiter=',')
        elif file_metadata["file_extension"] == '.bin':
            if file_metadata["data_dtype"] == "float32":
                return np.fromfile(file_path, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported binary dtype: {file_metadata['data_dtype']}")
        else:
            raise ValueError(f"Unsupported file type: {file_metadata['file_extension']}")
