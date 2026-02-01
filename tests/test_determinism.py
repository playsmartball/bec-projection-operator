"""
Test Determinism - Hash Validation and Parity

Tests deterministic behavior and hash validation.
"""

import pytest
import numpy as np
import tempfile
import shutil
import sys
sys.path.append('..')

from interface import HashValidator


class TestHashValidator:
    """Test hash validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = HashValidator(reference_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_array_hash(self):
        """Test array hash computation."""
        data = np.array([1.0, 2.0, 3.0])
        hash1 = self.validator.compute_hash(data)
        hash2 = self.validator.compute_hash(data)
        
        # Same data should produce same hash
        assert hash1 == hash2
        
        # Different data should produce different hash
        different_data = np.array([1.0, 2.0, 3.1])
        different_hash = self.validator.compute_hash(different_data)
        assert hash1 != different_hash
    
    def test_dict_hash(self):
        """Test dictionary hash computation."""
        data_dict = {"a": 1, "b": 2}
        hash1 = self.validator.compute_dict_hash(data_dict)
        hash2 = self.validator.compute_dict_hash(data_dict)
        
        # Same dict should produce same hash
        assert hash1 == hash2
        
        # Different order should produce same hash (sorted keys)
        different_order = {"b": 2, "a": 1}
        same_hash = self.validator.compute_dict_hash(different_order)
        assert hash1 == same_hash
    
    def test_save_load_reference(self):
        """Test reference save and load."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        filename = "test_data"
        
        # Save reference
        self.validator.save_reference(data, filename)
        
        # Load reference
        loaded_data, reference_hash = self.validator.load_reference(filename)
        
        # Check data integrity
        assert np.allclose(data, loaded_data)
        assert len(reference_hash) == 64  # SHA256 length
    
    def test_hash_validation(self):
        """Test hash validation."""
        data = np.array([1.0, 2.0, 3.0])
        reference_hash = self.validator.compute_hash(data)
        
        # Valid hash should pass
        assert self.validator.validate_hash(data, reference_hash) == True
        
        # Invalid hash should fail
        invalid_hash = "0" * 64
        assert self.validator.validate_hash(data, invalid_hash) == False
    
    def test_parity_validation(self):
        """Test CPU/GPU parity validation."""
        data1 = np.array([1.0, 2.0, 3.0])
        data2 = np.array([1.0, 2.0, 3.0])
        
        # Identical data should pass
        assert self.validator.validate_parity(data1, data2) == True
        
        # Different data should fail
        data3 = np.array([1.0, 2.0, 3.1])
        assert self.validator.validate_parity(data1, data3) == False
        
        # Different shapes should fail
        data4 = np.array([1.0, 2.0])
        assert self.validator.validate_parity(data1, data4) == False
