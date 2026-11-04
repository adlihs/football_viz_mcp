/**
 * Cloudflare Worker in front of the mplsoccer MCP server.
 *
 * The Python server (../Dockerfile) runs inside a Cloudflare Container. This Worker:
 *  - checks the API key BEFORE waking the container (bots and scanners never cost you container time),
 *  - forwards every request (MCP at /mcp, /upload, /charts/<id>.png, /health) to the container,
 *  - passes secrets and settings to the container as environment variables.
 */
import { Container, getContainer, getRandom } from "@cloudflare/containers";

export interface Env {
  MPLSOCCER: DurableObjectNamespace<MplsoccerContainer>;
  // secrets (wrangler secret put ...)
  MCP_API_KEY?: string;
  R2_ACCESS_KEY_ID?: string;
  R2_SECRET_ACCESS_KEY?: string;
  // vars (wrangler.jsonc)
  R2_ACCOUNT_ID?: string;
  R2_BUCKET_NAME?: string;
  PUBLIC_BASE_URL?: string;
  SLEEP_AFTER?: string;
  INSTANCES?: string;
  DATASET_TTL_HOURS?: string;
}

const PASS_THROUGH_VARS = [
  "MCP_API_KEY",
  "R2_ACCESS_KEY_ID",
  "R2_SECRET_ACCESS_KEY",
  "R2_ACCOUNT_ID",
  "R2_BUCKET_NAME",
  "PUBLIC_BASE_URL",
] as const;

export class MplsoccerContainer extends Container<Env> {
  defaultPort = 8000;
  // How long the container stays awake after the last request. Datasets are kept in R2,
  // so a sleeping container loses nothing; it only costs a cold start (~10-20 s) on the next call.
  sleepAfter = "30m";
  pingEndpoint = "localhost/health";
  enableInternet = true; // needed to reach R2 and to load datasets from user URLs

  constructor(ctx: ConstructorParameters<typeof Container<Env>>[0], env: Env) {
    super(ctx, env);
    if (env.SLEEP_AFTER) this.sleepAfter = env.SLEEP_AFTER;
    const vars: Record<string, string> = { PORT: "8000" };
    for (const k of PASS_THROUGH_VARS) {
      const v = env[k];
      if (v) vars[k] = v;
    }
    if (env.DATASET_TTL_HOURS) vars.MPLSOCCER_MCP_DATASET_TTL_HOURS = env.DATASET_TTL_HOURS;
    this.envVars = vars;
  }

  override onError(error: unknown) {
    console.error("container error", error);
    throw error;
  }
}

function timingSafeEqual(a: string, b: string): boolean {
  const enc = new TextEncoder();
  const x = enc.encode(a);
  const y = enc.encode(b);
  if (x.length !== y.length) return false;
  let diff = 0;
  for (let i = 0; i < x.length; i++) diff |= x[i] ^ y[i];
  return diff === 0;
}

function authorized(request: Request, env: Env): boolean {
  if (!env.MCP_API_KEY) return true;
  const url = new URL(request.url);
  const auth = request.headers.get("authorization") ?? "";
  const supplied =
    request.headers.get("x-api-key") ??
    url.searchParams.get("key") ??
    (auth.toLowerCase().startsWith("bearer ") ? auth.slice(7).trim() : "");
  return supplied !== "" && timingSafeEqual(supplied, env.MCP_API_KEY);
}

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "*",
  "Access-Control-Expose-Headers": "Mcp-Session-Id",
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { pathname } = new URL(request.url);

    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
    if (pathname === "/" ) {
      return Response.json({ name: "mplsoccer MCP server", mcp: "/mcp", upload: "/upload", health: "/health" });
    }
    // chart links use unguessable random ids and are public; everything else needs the key
    const isPublic = pathname === "/health" || pathname.startsWith("/charts/");
    if (!isPublic && !authorized(request, env)) {
      return Response.json(
        { error: "unauthorized: send Authorization: Bearer <key>, X-API-Key, or ?key=<key>" },
        { status: 401, headers: CORS },
      );
    }

    const instances = Math.max(1, parseInt(env.INSTANCES ?? "1", 10) || 1);
    const container =
      instances > 1 ? await getRandom(env.MPLSOCCER, instances) : getContainer(env.MPLSOCCER, "main");
    return container.fetch(request);
  },
} satisfies ExportedHandler<Env>;
