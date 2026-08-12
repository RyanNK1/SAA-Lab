/** Typed client for the allocation API.
 *
 * Requests go to same-origin `/api/...`, which Vite proxies to the Python
 * server in development and a reverse proxy handles in production. Keeping the
 * path identical in both means no environment-specific base URL has to be
 * threaded through the components.
 */

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    // The API returns a useful message in `detail`, either as a string or as
    // pydantic's list of field errors. Surfacing it verbatim is better than a
    // generic failure notice, because it names the field that was wrong.
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      } else if (Array.isArray(payload.detail)) {
        detail = payload.detail
          .map(
            (e: { loc?: unknown[]; msg?: string }) =>
              `${e.loc?.slice(1).join(".") ?? "request"}: ${e.msg ?? "invalid"}`,
          )
          .join("; ");
      }
    } catch {
      /* keep the status-based message */
    }
    throw new ApiError(detail, response.status);
  }

  return response.json() as Promise<T>;
}

export interface AssetMeta {
  key: string;
  label: string;
  proxy: string;
  caveat: string;
  allocatable: boolean;
}

export interface SleeveComponent {
  key: string;
  label: string;
  proxy: string;
  caveat: string;
}

export interface Meta {
  coverage: { start: string; end: string; months: number };
  assets: AssetMeta[];
  sleeve: { key: string; components: SleeveComponent[]; note: string };
  objectives: string[];
  rebalance_schedules: string[];
  rankable: string[];
  groups: Record<string, string[]>;
  regimes: { label: string; start: string; end: string }[];
  disclaimer: string;
}

export interface ConstraintSpec {
  caps?: Record<string, number>;
  floors?: Record<string, number>;
  group_caps?: Record<string, number>;
  group_floors?: Record<string, number>;
}

export interface MandateBody {
  start?: string;
  end?: string;
  gold_weight: number;
  assets?: string[];
  rebalance: string;
  cost_bps: number;
  samples: number;
  target_return?: number | null;
  max_volatility?: number | null;
  max_drawdown?: number | null;
  max_recovery_months?: number | null;
  constraints: ConstraintSpec;
  rank_by: string;
  limit: number;
}

export type Allocation = Record<string, number | null> & {
  realised_return: number;
  volatility: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  months_to_recover: number | null;
  months_underwater: number;
};

export interface EnvelopeRow {
  asset: string;
  min: number;
  median: number;
  max: number;
  spread: number;
}

export interface Relaxation {
  what: string;
  current: number;
  required: number;
  note: string;
  description: string;
}

export interface SleeveSplit {
  sleeve_weight: number;
  gold_weight: number;
  gold: number;
  commodities_ex_gold: number;
}

export interface MandateResult {
  mandate: string;
  feasible: boolean;
  n_sampled: number;
  n_qualifying: number;
  explanation: string;
  relaxations?: Relaxation[];
  ranked_by?: string;
  allocations?: Allocation[];
  envelope?: EnvelopeRow[];
  sleeve_split?: SleeveSplit;
}

export const api = {
  meta: () => request<Meta>("/meta"),
  mandate: (body: MandateBody, signal?: AbortSignal) =>
    request<MandateResult>("/mandate", body, signal),
};
