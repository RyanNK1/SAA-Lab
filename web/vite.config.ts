import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  build: {
    rollupOptions: {
      output: {
        // Charting is a large dependency used on one tab. Splitting it out
        // keeps it off the first paint and lets it stay cached across deploys
        // that only touch application code.
        manualChunks: { charts: ["recharts"] },
      },
    },
  },
  server: {
    // The API runs separately in development. Forwarding /api untouched -- not
    // stripping the prefix -- keeps the path identical to production, where a
    // rewrite sends /api/* to the API service. Stripping it here would mean
    // the two environments disagree about where the API lives, and the
    // difference would only surface after deploying.
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
