import { defineConfig, loadEnv } from "vite";
import vue from "@vitejs/plugin-vue";
import { fileURLToPath, URL } from "node:url";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backend = env.VITE_DEV_PROXY_TARGET || "http://127.0.0.1:8000";

  return {
    plugins: [vue()],
    base: "/web/",
    build: {
      outDir: fileURLToPath(new URL("../web", import.meta.url)),
      emptyOutDir: true,
    },
    server: {
      port: 5173,
      proxy: {
        "/ask": backend,
        "/health": backend,
        "/docs": backend,
      },
    },
  };
});
