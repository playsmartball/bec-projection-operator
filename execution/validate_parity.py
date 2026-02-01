"""
Parity Validation Script - CPU/GPU Determinism Check

Validates that CPU and GPU executions produce identical results within tolerance.
"""

import numpy as np
import json
from pathlib import Path
import sys
sys.path.append('..')

from interface import HashValidator


def main():
    """Main parity validation with hash checking."""
    print("Validating CPU/GPU parity...")
    
    validator = HashValidator()
    
    # Load CPU and GPU outputs
    output_dir = Path("data/outputs")
    
    try:
        with open(output_dir / "cpu_output.json", 'r') as f:
            cpu_output = json.load(f)
        
        with open(output_dir / "gpu_output.json", 'r') as f:
            gpu_output = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Missing output file - {e}")
        print("Please run both execution scripts first:")
        print("  python execution/run_cpu.py")
        print("  python execution/run_gpu.py")
        return False
    
    # Compare hashes
    cpu_hash = validator.compute_dict_hash(cpu_output)
    gpu_hash = validator.compute_dict_hash(gpu_output)
    
    print(f"CPU output hash: {cpu_hash}")
    print(f"GPU output hash: {gpu_hash}")
    
    # Validate reference trace
    try:
        reference_data, reference_hash = validator.load_reference("phi_trace")
        cpu_valid = validator.validate_hash(
            np.array(cpu_output["kernel_response"]), 
            reference_hash
        )
        gpu_valid = validator.validate_hash(
            np.array(gpu_output["kernel_response"]), 
            reference_hash
        )
        
        print(f"CPU vs reference: {'PASS' if cpu_valid else 'FAIL'}")
        print(f"GPU vs reference: {'PASS' if gpu_valid else 'FAIL'}")
        
    except FileNotFoundError:
        print("Warning: No reference data found")
        cpu_valid = gpu_valid = False
    
    # Validate CPU/GPU parity
    cpu_response = np.array(cpu_output["kernel_response"])
    gpu_response = np.array(gpu_output["kernel_response"])
    
    parity_valid = validator.validate_parity(cpu_response, gpu_response)
    
    if parity_valid:
        max_diff = np.max(np.abs(cpu_response - gpu_response))
        print(f"CPU/GPU parity: PASS (max diff: {max_diff:.2e})")
    else:
        max_diff = np.max(np.abs(cpu_response - gpu_response))
        print(f"CPU/GPU parity: FAIL (max diff: {max_diff:.2e})")
    
    # Overall validation result
    all_valid = cpu_valid and gpu_valid and parity_valid
    
    if all_valid:
        print("\n✅ All validations PASSED - Determinism verified")
    else:
        print("\n❌ Validation FAILED - Determinism compromised")
    
    return all_valid


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
