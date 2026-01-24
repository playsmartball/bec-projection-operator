#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
ROOT="/mnt/e/CascadeProjects/bec-projection-operator-git"
export PYTHONPATH="$ROOT"
OUT="$ROOT/analysis/phase8_runs"
mkdir -p "$OUT"

PY="$HOME/.micromamba/envs/dedalus/bin/python"
if [ ! -x "$PY" ]; then
  PY=python3
fi

KAPPAS=(5e-4 2e-3)
OC_MULTS=(0.25 0.5 1.0)
FORCINGS=(bulk edge)

OMEGA0_FILE="$OUT/p8b_omega0.txt"
if [ -f "$OMEGA0_FILE" ]; then
  OMEGA0=$(cat "$OMEGA0_FILE")
else
  MEASURE_CSV="$OUT/p8b_measure.csv"
  MEASURE_LOG="$OUT/p8b_measure.log"
  echo "[MEASURE] Estimating omega0 via short baseline run..."
  "$PY" -m examples.dedalus_alfven_2d_nl_ivp \
    --bc characteristic --nl 1 --eta 1e-3 --kappa 5e-4 \
    --kappa_model constant --omega_c 0 --dt 1e-3 --tmax 2 \
    --Lx 128 --Nx 128 --Lz 128 --Nz 129 --m 1 --n 1 --amp 1e-6 --eps 0 \
    --csv "$MEASURE_CSV" | tee "$MEASURE_LOG" || true
  OMEGA0=$(grep -F "omega (Hilbert)" "$MEASURE_LOG" | tail -n1 | awk '{print $4}')
  if [ -z "${OMEGA0:-}" ]; then
    # Fallback to prior smoke observation
    OMEGA0=2.444506
  fi
  echo "$OMEGA0" > "$OMEGA0_FILE"
  echo "[MEASURE] omega0=$OMEGA0"
fi

for kappa in "${KAPPAS[@]}"; do
  for mult in "${OC_MULTS[@]}"; do
    omega_c=$("$PY" - <<PY
mult = float("$mult")
omega0 = float("$OMEGA0")
print(omega0*mult)
PY
)
    oc_tag=$(printf "%0.3f" "$mult" | tr -d '.')
    for forcing in "${FORCINGS[@]}"; do
      CSV="$OUT/p8b_k${kappa}_oc${oc_tag}_${forcing}.csv"
      if [ -s "$CSV" ]; then
        echo "[SKIP] Existing CSV: $CSV"
        "$PY" "$ROOT/analysis/phase6_extract_metrics.py" --csv "$CSV" --omega_c "$omega_c" || true
        continue
      fi
      LOG="$OUT/p8b_k${kappa}_oc${oc_tag}_${forcing}.log"
      echo "[RUN] kappa=$kappa mult=$mult omega_c=$omega_c forcing=$forcing"
      "$PY" -m examples.dedalus_alfven_2d_nl_ivp \
        --bc characteristic --nl 1 --eta 1e-3 --kappa "$kappa" \
        --kappa_model lowpass --omega_c "$omega_c" --dt 1e-3 --tmax 20 \
        --Lx 128 --Nx 128 --Lz 128 --Nz 129 --m 1 --n 1 --amp 1e-6 --eps 0 \
        --forcing "$forcing" --csv "$CSV" | tee "$LOG"
      echo "[EXTRACT] $CSV"
      "$PY" "$ROOT/analysis/phase6_extract_metrics.py" --csv "$CSV" --omega_c "$omega_c" || true
    done
  done
done

echo "[DONE] Phase-8B sweep complete."
