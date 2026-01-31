#!/usr/bin/env bash
set -euo pipefail

# Environment
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH="${PYTHONPATH:-/mnt/e/CascadeProjects/bec-projection-operator-git}"
OUTDIR="${OUTDIR:-/mnt/e/CascadeProjects/bec-projection-operator-git/analysis/phase8_runs}"
mkdir -p "$OUTDIR"

# Python interpreter
PY="${PY:-$HOME/.micromamba/envs/dedalus/bin/python}"
if [ ! -x "$PY" ]; then PY=python3; fi

# Parameter grid
KAPPAS="${KAPPAS:-1e-4 5e-4 2e-3}"
AMPS="${AMPS:-1e-6 1e-4}"
EPSS="${EPSS:-0 0.05}"

date
echo "START PERSISTENT PHASE-8A SWEEP"
for K in $KAPPAS; do
  for A in $AMPS; do
    for E in $EPSS; do
      CSV="$OUTDIR/charbc_nl_eta1e-3_t20_robin_kappa${K}_amp${A}_eps${E}.csv"
      if [ -f "$CSV" ]; then
        echo "SKIP_EXISTING kappa=$K amp=$A eps=$E"
        continue
      fi
      echo "RUN_START kappa=$K amp=$A eps=$E"
      "$PY" -m examples.dedalus_alfven_2d_nl_ivp \
        --bc characteristic --nl 1 --eta 1e-3 --kappa "$K" \
        --dt 1e-3 --tmax 20 --Lx 128 --Nx 128 --Lz 128 --Nz 129 \
        --m 1 --n 1 --amp "$A" --eps "$E" --csv "$CSV"
      echo "EXTRACT kappa=$K amp=$A eps=$E"
      "$PY" /mnt/e/CascadeProjects/bec-projection-operator-git/analysis/phase6_extract_metrics.py \
        --csv "$CSV" || true
      echo "RUN_DONE kappa=$K amp=$A eps=$E"
      sync
    done
  done
done
echo "ALL_DONE"
date
