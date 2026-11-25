// k6 load on the stateless docs-search endpoint (BM25, CPU-bound, no model call).
//   k6 run -e VUS=50 -e DURATION=180s deploy/load_test.js
import http from "k6/http";
import { check } from "k6";

const BASE = __ENV.BASE_URL || "http://127.0.0.1:18080";
const QUERIES = ["model_dump exclude none", "alias generator camel case", "validate_call memory", "frozen model immutable", "field_serializer"];

export const options = {
  vus: Number(__ENV.VUS || 50),
  duration: __ENV.DURATION || "180s",
  summaryTrendStats: ["avg", "med", "p(95)", "p(99)", "max"],
};

export default function () {
  const q = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const res = http.get(`${BASE}/api/docs/search?q=${encodeURIComponent(q)}&k=5`);
  check(res, { "status 200": (r) => r.status === 200 });
}
