import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs on :8000; in compose and Kubernetes the gateway routes /api.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000", "/readyz": "http://localhost:8000" } },
});
