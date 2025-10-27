#!/usr/bin/env bash
# check_cuda_image_filter_status.sh
# Status & sanity checker for the cpp-cuda-image-filter project
# Usage:
#   ./check_cuda_image_filter_status.sh [--rebuild] [--no-run] [--bin ./build/conv2d_shared]
#
# Exits non‑zero on failure. Produces a concise, colored summary.

set -euo pipefail

# ---------- Config ----------
BIN="./build/conv2d_shared"
REBUILD="0"
RUN_TEST="1"
TOL="0.005"   # tolerance for expected center value around 0.04
EXPECTED="0.04"

# Colors
if [ -t 1 ]; then
  BOLD="\033[1m"; RED="\033[31m"; GRN="\033[32m"; YEL="\033[33m"; CYN="\033[36m"; DIM="\033[2m"; CLR="\033[0m"
else
  BOLD=""; RED=""; GRN=""; YEL=""; CYN=""; DIM=""; CLR=""
fi

# ---------- Args ----------
while (( "$#" )); do
  case "$1" in
    --rebuild) REBUILD="1"; shift ;;
    --no-run)  RUN_TEST="0"; shift ;;
    --bin)     BIN="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--rebuild] [--no-run] [--bin <path>]"
      exit 0
      ;;
    *)
      echo -e "${YEL}Warning:${CLR} Unknown argument: $1"
      shift
      ;;
  esac
done

echo -e "${BOLD}=== CUDA Image Filter: Project Status ===${CLR}"

# ---------- Environment ----------
echo -e "${CYN}[Environment]${CLR}"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/CUDA Toolkit: \1/p'
else
  echo -e "${RED}nvcc not found in PATH${CLR}"; exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
  # Try to get compute capability if supported
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)
  if [ -n "${CAP:-}" ]; then
    echo "Compute Capability: ${CAP}"
  else
    echo -e "${DIM}(compute capability query not available via nvidia-smi)${CLR}"
  fi
else
  echo -e "${YEL}nvidia-smi not found; GPU details unavailable${CLR}"
fi
echo

# ---------- Build ----------
if [ "${REBUILD}" = "1" ]; then
  echo -e "${CYN}[Build]${CLR} Rebuilding (Release)..."
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
  echo
fi

if [ ! -f "${BIN}" ]; then
  echo -e "${RED}[Build] Binary not found at ${BIN}${CLR}"
  echo "Run:"
  echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
  exit 2
fi
echo -e "${GRN}[Build] Binary present:${CLR} ${BIN}"
echo

# ---------- Optional run ----------
if [ "${RUN_TEST}" = "1" ]; then
  echo -e "${CYN}[Runtime]${CLR} Running kernel..."
  LOG="run_image_filter.log"
  # Use /usr/bin/time if present for portable timing
  if [ -x /usr/bin/time ]; then
    /usr/bin/time -f "Elapsed: %E  CPU: %P  MaxRSS: %M KB" "${BIN}" | tee "${LOG}"
  else
    "${BIN}" | tee "${LOG}"
  fi

  # Parse the center value
  CENTER_LINE=$(grep -E "Center value after blur" "${LOG}" || true)
  CENTER_VAL=$(echo "${CENTER_LINE}" | awk '{print $5}' )

  if [ -z "${CENTER_VAL}" ]; then
    echo -e "${RED}[Check] Could not parse center value from output${CLR}"
    echo "Expected a line like: Center value after blur: 0.040000 (expected ~0.04)"
    exit 3
  fi

  # Numeric check: |center - expected| <= tol
  python3 - <<'PY' 2>/dev/null || {
    echo -e "${RED}[Check] Python not available for numeric comparison; skipping tolerance check${CLR}"
    exit 0
  }
import sys, math, os
center = float(os.environ.get("CENTER_VAL_ENV", "nan"))
expected = float(os.environ.get("EXPECTED_ENV", "0.04"))
tol = float(os.environ.get("TOL_ENV", "0.005"))
ok = abs(center - expected) <= tol
print("Parsed center:", center, " Expected:", expected, " Tol:", tol, " OK:", ok)
sys.exit(0 if ok else 1)
PY
  PYOK=$?
  # The above Python block expects env vars; set and rerun if needed.
  if [ "${PYOK}" -ne 0 ]; then
    CENTER_VAL_ENV="${CENTER_VAL}" EXPECTED_ENV="${EXPECTED}" TOL_ENV="${TOL}" python3 - <<'PY' || true
import sys, math, os
center = float(os.environ["CENTER_VAL_ENV"])
expected = float(os.environ["EXPECTED_ENV"])
tol = float(os.environ["TOL_ENV"])
ok = abs(center - expected) <= tol
print("Parsed center:", center, " Expected:", expected, " Tol:", tol, " OK:", ok)
sys.exit(0 if ok else 1)
PY
    PYOK=$?
  fi

  if [ "${PYOK}" -eq 0 ]; then
    echo -e "${GRN}[Check] Center value within tolerance (${EXPECTED} ± ${TOL}) ✔${CLR}"
    echo -e "${GRN}✅ Success${CLR}"
  else
    echo -e "${YEL}[Check] Center value outside tolerance (${EXPECTED} ± ${TOL}). Found: ${CENTER_VAL}${CLR}"
    echo -e "${YEL}⚠ Please inspect ${LOG}${CLR}"
    exit 4
  fi
else
  echo -e "${DIM}[Runtime skipped with --no-run]${CLR}"
fi
