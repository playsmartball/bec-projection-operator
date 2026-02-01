"""
Determinism Module - Hash Validation and Parity Checks

Ensures CPU/GPU parity and deterministic behavior across executions.
"""

import hashlib
import json
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path


class HashValidator:
    """
    Validates deterministic behavior through SHA256 hashing.
    """
    
    def __init__(self, reference_dir: str = "data/reference"):
        """
        Initialize hash validator.
        
        Args:
            reference_dir: Directory containing reference hashes
        """
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_hash(self, data: np.ndarray) -> str:
        """
        Compute SHA256 hash of numpy array.
        
        Args:
            data: Input array to hash
            
        Returns:
            SHA256 hash string
        """
        # Convert to bytes in a deterministic way
        data_bytes = data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()
    
    def compute_dict_hash(self, data_dict: Dict[str, Any]) -> str:
        """
        Compute hash of dictionary with deterministic serialization.
        
        Args:
            data_dict: Dictionary to hash
            
        Returns:
            SHA256 hash string
        """
        # Sort keys for deterministic serialization
        json_str = json.dumps(data_dict, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def save_reference(self, data: np.ndarray, filename: str) -> None:
        """
        Save reference data and its hash.
        
        Args:
            data: Reference data array
            filename: Base filename (without extension)
        """
        # Save data
        data_file = self.reference_dir / f"{filename}.json"
        json.dump({"data": data.tolist()}, open(data_file, 'w'))
        
        # Save hash
        hash_file = self.reference_dir / f"{filename}.sha256"
        hash_value = self.compute_hash(data)
        open(hash_file, 'w').write(hash_value)
    
    def load_reference(self, filename: str) -> Tuple[np.ndarray, str]:
        """
        Load reference data and its hash.
        
        Args:
            filename: Base filename (without extension)
            
        Returns:
            Tuple of (data_array, reference_hash)
        """
        # Load data
        data_file = self.reference_dir / f"{filename}.json"
        data_dict = json.load(open(data_file, 'r'))
        data = np.array(data_dict["data"])
        
        # Load hash
        hash_file = self.reference_dir / f"{filename}.sha256"
        reference_hash = open(hash_file, 'r').read().strip()
        
        return data, reference_hash
    
    def validate_hash(self, data: np.ndarray, reference_hash: str) -> bool:
        """
        Validate data against reference hash.
        
        Args:
            data: Data to validate
            reference_hash: Reference hash to compare against
            
        Returns:
            True if hashes match
        """
        computed_hash = self.compute_hash(data)
        return computed_hash == reference_hash
    
    def validate_parity(self, cpu_data: np.ndarray, gpu_data: np.ndarray, tolerance: float = 1e-12) -> bool:
        """
        Validate CPU/GPU parity within tolerance.
        
        Args:
            cpu_data: CPU computation result
            gpu_data: GPU computation result
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if CPU and GPU results match within tolerance
        """
        if cpu_data.shape != gpu_data.shape:
            return False
        
        max_diff = np.max(np.abs(cpu_data - gpu_data))
        return max_diff <= tolerance
