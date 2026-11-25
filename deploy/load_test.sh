#!/usr/bin/env bash
# Load the stateless docs-search endpoint through the Ingress and record how the HPA reacts.
# Writes <out>/load_<stamp>.log (k6 report) and <out>/hpa_<stamp>.tsv (one sample every 5 s).
#   deploy/load_test.sh [duration=180s] [virtual_users=50] [out=deploy/verification]
# PRE_SECONDS / POST_SECONDS set how long to sample before and after the load (default 30 / 360:
# the HPA waits 5 minutes before scaling down).
set -euo pipefail
cd "$(dirname "$0")/.."

DURATION=${1:-180s}
CONCURRENCY=${2:-50}
OUT=${3:-deploy/verification}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
SAMPLES="$OUT/hpa_$STAMP.tsv"

sample() {
  printf 'utc\tphase\thpa_cpu_current/target\thpa_desired\tready_pods\n'
  local phase=${1}
  while :; do
    local hpa ready
    hpa=$(kubectl -n triage-agent get hpa api -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}%/60%{"\t"}{.status.desiredReplicas}' 2>/dev/null || echo "?\t?")
    ready=$(kubectl -n triage-agent get deploy api -o jsonpath='{.status.readyReplicas}')
    printf '%s\t%s\t%b\t%s\n' "$(date -u +%H:%M:%S)" "$(cat "$phase")" "$hpa" "${ready:-0}"
    sleep 5
  done
}

PHASE=$(mktemp)
echo before >"$PHASE"
sample "$PHASE" >"$SAMPLES" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true; rm -f "$PHASE"' EXIT

sleep "${PRE_SECONDS:-30}"
echo load >"$PHASE"
echo "[$(date -u +%H:%M:%S)] k6: $CONCURRENCY virtual users for $DURATION on /api/docs/search"
k6 run --quiet -e VUS="$CONCURRENCY" -e DURATION="$DURATION" deploy/load_test.js 2>&1 | tee "$OUT/load_$STAMP.log"
echo after >"$PHASE"
echo "[$(date -u +%H:%M:%S)] load finished; sampling ${POST_SECONDS:-360}s more to see the scale-down"
sleep "${POST_SECONDS:-360}"
kill $SAMPLER
cat "$SAMPLES"
