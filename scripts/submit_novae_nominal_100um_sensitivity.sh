#!/usr/bin/env bash
# Submit the predeclared nominal-100um calibration sensitivity on HPG.
# The protocol is fixed; test-only path substitutions use the explicit
# NOVAE_SENSITIVITY_* variables below rather than inherited baseline overrides.
set -euo pipefail

if [[ "${1:-}" != "--render-only" && "${1:-}" != "--no-submit" && $# -ne 0 ]]; then
  echo "usage: $0 [--render-only|--no-submit]" >&2
  exit 2
fi

REPO_DIR="${NOVAE_REPO_DIR:-/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool}"
BASELINE_INPUT="/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_spatial.h5ad"
BASELINE_MODEL="/blue/kejun.huang/vasco.hinostroza/models/novae-human-0"
BASELINE_REVISION="b8c0a5d7612bac6bc719ab57ed3cd16ad814728c"
SCALE_MANIFEST_DEFAULT="${REPO_DIR}/presets/novae_nominal_100um_scales.csv"

reject_conflict() {
  local name="$1" expected="$2" value="${!1-}"
  if [[ -n "${value}" && "${value}" != "${expected}" ]]; then
    echo "${name} conflicts with the fixed nominal-100um sensitivity protocol; use an explicit NOVAE_SENSITIVITY_* override only for test paths" >&2
    exit 2
  fi
}
reject_conflict NOVAE_RESOLUTIONS "0.5 1.0 2.0"
reject_conflict NOVAE_PRIMARY_RESOLUTION "1.0"
reject_conflict NOVAE_EXPECTED_NEIGHBOR_DISTANCE_UM "100"
reject_conflict NOVAE_MIN_DOMAIN_ASSIGNMENT_COVERAGE "0.70"
reject_conflict NOVAE_SEED "42"
reject_conflict NOVAE_MODEL_REVISION "${BASELINE_REVISION}"
reject_conflict NOVAE_COORDINATE_STRATEGY "visium_explicit_scale"
reject_conflict NOVAE_OMIT_GRAPH_RADIUS_PRUNING "1"

INPUT_H5AD="${NOVAE_SENSITIVITY_INPUT_H5AD:-${BASELINE_INPUT}}"
MODEL_PATH="${NOVAE_SENSITIVITY_MODEL_PATH:-${BASELINE_MODEL}}"
SCALE_MANIFEST="${NOVAE_SENSITIVITY_SCALE_MANIFEST:-${SCALE_MANIFEST_DEFAULT}}"
if [[ -n "${NOVAE_INPUT_H5AD-}" && "${NOVAE_INPUT_H5AD}" != "${BASELINE_INPUT}" ]]; then
  echo "NOVAE_INPUT_H5AD conflicts with fixed baseline source; use NOVAE_SENSITIVITY_INPUT_H5AD for tests" >&2; exit 2
fi
if [[ -n "${NOVAE_MODEL_PATH-}" && "${NOVAE_MODEL_PATH}" != "${BASELINE_MODEL}" ]]; then
  echo "NOVAE_MODEL_PATH conflicts with fixed cached baseline checkpoint; use NOVAE_SENSITIVITY_MODEL_PATH for tests" >&2; exit 2
fi
if [[ -n "${NOVAE_SAMPLE_MANIFEST-}" && "${NOVAE_SAMPLE_MANIFEST}" != "${SCALE_MANIFEST}" ]]; then
  echo "NOVAE_SAMPLE_MANIFEST conflicts with fixed reviewed scale manifest; use NOVAE_SENSITIVITY_SCALE_MANIFEST for tests" >&2; exit 2
fi
export NOVAE_INPUT_H5AD="${INPUT_H5AD}"
export NOVAE_MODEL_PATH="${MODEL_PATH}"
export NOVAE_MODEL_REVISION="${BASELINE_REVISION}"
export NOVAE_SAMPLE_MANIFEST="${SCALE_MANIFEST}"
export NOVAE_RESOLUTIONS="0.5 1.0 2.0"
export NOVAE_PRIMARY_RESOLUTION="1.0"
export NOVAE_EXPECTED_NEIGHBOR_DISTANCE_UM="100"
export NOVAE_MIN_DOMAIN_ASSIGNMENT_COVERAGE="0.70"
export NOVAE_SEED="42"
export NOVAE_COORDINATE_STRATEGY="visium_explicit_scale"
export NOVAE_OMIT_GRAPH_RADIUS_PRUNING="1"
export NOVAE_REPO_DIR="${REPO_DIR}"
export NOVAE_DATASET_ID="${NOVAE_SENSITIVITY_DATASET_ID:-skin_visium_ssc_nominal_100um_sensitivity}"
export NOVAE_OUTPUT_DIR="${NOVAE_SENSITIVITY_OUTPUT_DIR:-${REPO_DIR}/runs/novae_skin_pilot/${NOVAE_DATASET_ID}}"
export NOVAE_RUN_ROOT="${NOVAE_SENSITIVITY_RUN_ROOT:-${REPO_DIR}/runs/novae_skin_pilot/nominal_100um_sensitivity}"
export NOVAE_LOG_DIR="${NOVAE_SENSITIVITY_LOG_DIR:-${NOVAE_RUN_ROOT}/logs}"
export NOVAE_JOB_SCRIPT="${NOVAE_SENSITIVITY_JOB_SCRIPT:-${NOVAE_RUN_ROOT}/submit_novae_nominal_100um_sensitivity.sbatch}"
export NOVAE_JOB_NAME="${NOVAE_SENSITIVITY_JOB_NAME:-novae_nominal_100um_sensitivity}"

exec "${REPO_DIR}/scripts/submit_novae_skin_pilot.sh" "$@"
