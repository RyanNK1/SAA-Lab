import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    // The API runs separately. Proxying in development means the app calls
    // same-origin paths in both dev and production, so no environment-specific
    // base URL has to be threaded through the client.
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
