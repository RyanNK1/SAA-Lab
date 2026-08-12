import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

// Results are pure functions of the request, and the dataset does not change
// while the server is running, so a cached answer never goes stale. Retries are
// off because a rejected mandate is a valid answer, not a transient failure.
const client = new QueryClient({
  defaultOptions: { queries: { staleTime: Infinity, retry: false, refetchOnWindowFocus: false } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
