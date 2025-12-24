# V9 BEC Crust CLASS Patches

These files contain the modifications to CLASS for the V9 BEC Crust model.

## How to Apply

1. Clone CLASS v3.2.0:
```bash
git clone https://github.com/lesgourg/class_public.git class_v9_bec
cd class_v9_bec
git checkout v3.2.0
```

2. Copy the patched files:
```bash
cp ../class_v9_bec_patches/background.h include/
cp ../class_v9_bec_patches/background.c source/
cp ../class_v9_bec_patches/input.c source/
```

3. Build CLASS:
```bash
make clean
make -j4
```

4. Install Python wrapper:
```bash
cd python
pip install .
```

## Key Modifications

### background.h
- Added `n_bec` parameter (BEC dilution exponent)
- Added `has_bec_crust` flag
- Added indices for BEC crust quantities in background table

### background.c
- Implemented BEC density evolution: `ρ_BEC = ρ_BEC_0 × (1+z)^n_BEC`
- Equation of state: `w = n_BEC/3 - 1`
- Disabled `has_lambda` when `has_bec_crust = TRUE` to prevent double-counting

### input.c
- Added parsing for `bec_crust` (yes/no) and `n_bec` parameters
- Default: `has_bec_crust = FALSE`, `n_bec = 0.0`

## Usage

In CLASS input files or Python:
```python
cosmo.set({
    'bec_crust': 'yes',
    'n_bec': 0.5,  # V9 prediction
    # ... other parameters
})
```

When `n_bec = 0`, the model exactly recovers ΛCDM.
