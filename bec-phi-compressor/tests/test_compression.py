"""
Test Φ-Compressor Functionality

Tests compression, decompression, and deterministic behavior.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
sys.path.append('..')

from bec_phi_compress import PhiCompressor, PhiDecompressor


class TestPhiCompressor:
    """Test compression functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.compressor = PhiCompressor(phi_range=(0.0, 5.0), resolution=1000)
        self.decompressor = PhiDecompressor()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_numpy_compression(self):
        """Test compression of numpy array."""
        # Create test data
        data = np.random.randn(1000, 100)
        
        # Save test file
        test_file = Path(self.temp_dir) / "test_data.npy"
        np.save(test_file, data)
        
        # Compress
        metadata = self.compressor.compress_file(test_file, self.temp_dir)
        
        # Verify metadata
        assert "compression_metrics" in metadata
        assert metadata["compression_metrics"]["compression_ratio"] > 1.0
        assert metadata["file_metadata"]["data_shape"] == [1000, 100]
        
        # Verify output files exist
        compressed_file = Path(self.temp_dir) / "test_data_compressed.npy"
        manifest_file = Path(self.temp_dir) / "test_data_manifest.json"
        hash_file = Path(self.temp_dir) / "test_data_hash.txt"
        
        assert compressed_file.exists()
        assert manifest_file.exists()
        assert hash_file.exists()
    
    def test_csv_compression(self):
        """Test compression of CSV data."""
        # Create test CSV
        data = np.random.randn(500, 50)
        test_file = Path(self.temp_dir) / "test_data.csv"
        np.savetxt(test_file, data, delimiter=',')
        
        # Compress
        metadata = self.compressor.compress_file(test_file, self.temp_dir)
        
        # Verify
        assert metadata["compression_metrics"]["compression_ratio"] > 1.0
        assert metadata["file_metadata"]["file_extension"] == ".csv"
    
    def test_binary_compression(self):
        """Test compression of binary data."""
        # Create test binary file
        data = np.random.randn(10000).astype(np.float32)
        test_file = Path(self.temp_dir) / "test_data.bin"
        data.tofile(test_file)
        
        # Compress
        metadata = self.compressor.compress_file(test_file, self.temp_dir)
        
        # Verify
        assert metadata["compression_metrics"]["compression_ratio"] > 1.0
        assert metadata["file_metadata"]["data_dtype"] == "float32"
    
    def test_deterministic_compression(self):
        """Test deterministic compression behavior."""
        # Create test data
        data = np.random.RandomState(42).randn(1000)
        test_file = Path(self.temp_dir) / "test_data.npy"
        np.save(test_file, data)
        
        # Compress twice
        metadata1 = self.compressor.compress_file(test_file, self.temp_dir)
        metadata2 = self.compressor.compress_file(test_file, self.temp_dir)
        
        # Should be identical
        assert metadata1["compression_metrics"]["compression_ratio"] == metadata2["compression_metrics"]["compression_ratio"]
        assert metadata1["projection_metadata"]["compressed_hash"] == metadata2["projection_metadata"]["compressed_hash"]


class TestPhiDecompressor:
    """Test decompression functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.compressor = PhiCompressor(phi_range=(0.0, 5.0), resolution=1000)
        self.decompressor = PhiDecompressor()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_decompression_pipeline(self):
        """Test full compression/decompression pipeline."""
        # Create and save test data
        data = np.random.RandomState(123).randn(500, 200)
        test_file = Path(self.temp_dir) / "test_data.npy"
        np.save(test_file, data)
        
        # Compress
        metadata = self.compressor.compress_file(test_file, self.temp_dir)
        
        # Decompress
        compressed_file = Path(self.temp_dir) / "test_data_compressed.npy"
        manifest_file = Path(self.temp_dir) / "test_data_manifest.json"
        output_file = Path(self.temp_dir) / "reconstructed.npy"
        
        decomp_metadata = self.decompressor.decompress_file(
            compressed_file, manifest_file, output_file
        )
        
        # Verify reconstruction
        assert output_file.exists()
        reconstructed = np.load(output_file)
        assert reconstructed.shape == data.shape
        
        # Verify metadata
        assert "reconstruction_metrics" in decomp_metadata
        assert decomp_metadata["reconstruction_metrics"]["reconstructed_shape"] == list(data.shape)
    
    def test_reconstruction_error(self):
        """Test reconstruction error computation."""
        # Create test data
        data = np.random.RandomState(456).randn(200, 100)
        test_file = Path(self.temp_dir) / "test_data.npy"
        np.save(test_file, data)
        
        # Compress
        metadata = self.compressor.compress_file(test_file, self.temp_dir)
        
        # Decompress
        compressed_file = Path(self.temp_dir) / "test_data_compressed.npy"
        manifest_file = Path(self.temp_dir) / "test_data_manifest.json"
        output_file = Path(self.temp_dir) / "reconstructed.npy"
        
        decomp_metadata = self.decompressor.decompress_file(
            compressed_file, manifest_file, output_file
        )
        
        # Check that reconstruction error is computed
        error = decomp_metadata["reconstruction_metrics"]["reconstruction_error"]
        assert error is not None
        assert error >= 0  # Error should be non-negative
