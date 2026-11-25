#!/usr/bin/env bash
# One command from nothing to a running deployment on a local kind cluster:
# cluster -> images -> ingress-nginx + metrics-server -> API-key Secret -> app manifests -> smoke check.
# Needs docker, kind, kubectl, and GEMINI_API_KEY or ZHIPUAI_API_KEY in the environment.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER=triage-agent
INGRESS_NGINX=controller-v1.12.1
METRICS_SERVER=v0.7.2
URL=http://127.0.0.1:18080
log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ -z "${GEMINI_API_KEY:-}${ZHIPUAI_API_KEY:-}" ]; then
  echo "kind_up: set GEMINI_API_KEY or ZHIPUAI_API_KEY first (it goes into a Kubernetes Secret, never into an image)" >&2
  exit 2
fi

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  log "cluster $CLUSTER exists, reusing it"
else
  log "creating kind cluster $CLUSTER"
  kind create cluster --config k8s/kind-cluster.yaml --wait 120s
fi
kubectl config use-context "kind-$CLUSTER" >/dev/null

log "building images"
docker build -q -t triage-agent-api:0.1.0 .
docker build -q -t triage-agent-web:0.1.0 web
log "loading images into the cluster"
kind load docker-image triage-agent-api:0.1.0 triage-agent-web:0.1.0 --name "$CLUSTER"

log "installing ingress-nginx $INGRESS_NGINX and metrics-server $METRICS_SERVER"
kubectl apply -f "https://raw.githubusercontent.com/kubernetes/ingress-nginx/$INGRESS_NGINX/deploy/static/provider/kind/deploy.yaml" >/dev/null
kubectl apply -f "https://github.com/kubernetes-sigs/metrics-server/releases/download/$METRICS_SERVER/components.yaml" >/dev/null
# kind's kubelets use self-signed serving certificates.
if ! kubectl -n kube-system get deploy metrics-server -o jsonpath='{.spec.template.spec.containers[0].args}' | grep -q kubelet-insecure-tls; then
  kubectl -n kube-system patch deploy metrics-server --type=json \
    -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' >/dev/null
fi
kubectl -n ingress-nginx rollout status deploy/ingress-nginx-controller --timeout=180s
kubectl -n kube-system rollout status deploy/metrics-server --timeout=180s

log "creating the API-key Secret (values read from the environment, not from the command line)"
kubectl apply -f k8s/namespace.yaml >/dev/null
kubectl -n triage-agent create secret generic triage-agent-keys --dry-run=client -o yaml \
  --from-env-file=<(printf 'GEMINI_API_KEY=%s\nZHIPUAI_API_KEY=%s\n' "${GEMINI_API_KEY:-}" "${ZHIPUAI_API_KEY:-}") |
  kubectl apply -f - >/dev/null

log "deploying the app"
kubectl apply -k k8s/
kubectl -n triage-agent rollout status deploy/api --timeout=180s
kubectl -n triage-agent rollout status deploy/web --timeout=120s

log "smoke check through the Ingress at $URL"
for i in $(seq 1 30); do
  if curl -fsS "$URL/readyz" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS "$URL/readyz"; echo
curl -fsS -o /dev/null -w "web index: HTTP %{http_code}\n" "$URL/"
kubectl -n triage-agent get pods,svc,ingress,hpa
