"""
Create Test Dataset for Φ-Compressor Proof Test

Generates a 500MB+ numeric dataset for compression testing.
"""

import numpy as np
from pathlib import Path


def create_test_dataset():
    """Create a large test dataset with mixed structure."""
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating test dataset...")
    
    # Create structured data (low entropy)
    print("  - Creating structured component...")
    structured_size = 100_000_000  # 100M elements
    structured = np.zeros(structured_size, dtype=np.float64)
    
    # Add periodic structure
    x = np.linspace(0, 100*np.pi, structured_size)
    structured += np.sin(x) + 0.5 * np.sin(3*x) + 0.25 * np.sin(5*x)
    
    # Add some smooth variations
    structured += 0.1 * np.random.RandomState(42).randn(structured_size)
    
    # Create random data (high entropy)
    print("  - Creating random component...")
    random_size = 50_000_000  # 50M elements
    random_data = np.random.RandomState(123).randn(random_size).astype(np.float64)
    
    # Create mixed data (medium entropy)
    print("  - Creating mixed component...")
    mixed_size = 75_000_000  # 75M elements
    mixed = np.zeros(mixed_size, dtype=np.float64)
    
    # Blocks of different patterns
    block_size = 10000
    num_blocks = mixed_size // block_size
    
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        
        if i % 3 == 0:
            # Smooth block
            mixed[start:end] = np.sin(np.linspace(0, 2*np.pi, block_size))
        elif i % 3 == 1:
            # Random block
            mixed[start:end] = np.random.RandomState(i).randn(block_size) * 0.5
        else:
            # Constant block
            mixed[start:end] = i % 7
    
    # Combine all data
    print("  - Combining components...")
    total_data = np.concatenate([structured, random_data, mixed])
    
    # Calculate final size
    size_mb = total_data.nbytes / (1024 * 1024)
    print(f"  - Dataset size: {size_mb:.1f} MB")
    
    # Save as different formats
    print("  - Saving dataset...")
    
    # Save as .npy
    np.save(output_dir / "test_dataset_500mb.npy", total_data)
    
    # Save as .csv (subset for testing)
    csv_subset = total_data[:1_000_000]  # 1M elements for CSV
    np.savetxt(output_dir / "test_dataset_subset.csv", csv_subset, delimiter=',')
    
    # Save as .bin
    total_data.astype(np.float32).tofile(output_dir / "test_dataset_500mb.bin")
    
    print(f"✅ Test dataset created in {output_dir}")
    print(f"   - test_dataset_500mb.npy ({size_mb:.1f} MB)")
    print(f"   - test_dataset_subset.csv ({csv_subset.nbytes / (1024*1024):.1f} MB)")
    print(f"   - test_dataset_500mb.bin ({total_data.astype(np.float32).nbytes / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    create_test_dataset()
