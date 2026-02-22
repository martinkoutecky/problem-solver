import { Elysia } from "elysia"
import { cors } from "@elysiajs/cors"

import { problems_router } from "./problems"
import { research_router } from "./research"
import { admin_router } from "./admin"
import { profile_router } from "./profile"
import { auth_router } from "./auth"

import { jobs } from "./jobs"
import { get_server_url } from "@backend/server"
import { ensure_local_profile } from "@backend/auth/local"

const api_router = new Elysia({ prefix: "/api" })
  .get("/health", { status: "ok" })
  .use(auth_router)
  .use(problems_router)
  .use(research_router)
  .use(admin_router)
  .use(profile_router)

const backend = new Elysia({ name: "backend" })
  .use(cors({
    origin: get_server_url("frontend"),
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
  }))
  .use(api_router)

const frontend = new Elysia({ name: "frontend" })
// NOTE: This is a workaround for serving React SPA with Browser History.
// It could be possible to use @elysiajs/static and Hash History setting in Tanstack Router
// but who even likes Hash History??? (can'use the plugin with Browser History
// because it's completely broken... most likely a Bun issue)
// (hopefully it's not vulnerable to path-traversal attack haha)
if (Bun.env.NODE_ENV === "production") {
  frontend.get("/*", async ({ params }) => {
    const static_file = Bun.file(`../frontend/dist/${params["*"]}`)
    const react_app = Bun.file("../frontend/dist/index.html")
    return (await static_file.exists()) ? static_file : react_app
  })
}

export const app = new Elysia()
  // PRODUCTION [RAILWAY]: The '/health' check is required by Railway for successful deployment
  .get("/health", { status: "ok" })
  .use(backend)
  .use(frontend)
  .onStart(async () => {
    await ensure_local_profile()
    await jobs.start()
    console.log(`✌️ [BACKEND] is running at http://${app.server?.hostname}:${app.server?.port}.`)
  })
  .onStop(async () => {
    // TODO: How to do the gentle shutdown? To wait for all active jobs to finish??
    console.log("🔥 [BACKEND] stopped!")
    await jobs.stop()
    process.exit(0)
  })
  .listen(Bun.env.NODE_ENV === "production" ? Bun.env.PORT! : (Bun.env.BACKEND_PORT || 3942))

let shutting_down = false
function handle_shutdown() {
  if (!shutting_down) {
    shutting_down = true
    app.stop(true)
  }
}

// SIGINT: interactive interrupt (Ctrl + C)
// SIGTERM: termination request (e.g. from platform)
process.on("SIGINT", () => handle_shutdown())
process.on("SIGTERM", () => handle_shutdown())

// DEV Hack to let TypeScript know we will always specify these in .env
declare module "bun" {
  interface Env {
    AUTH_MODE?: "local" | "supabase",
    LOCAL_AUTH_USERNAME?: string,
    LOCAL_AUTH_PASSWORD?: string,
    LOCAL_AUTH_USER_ID?: string,
    LOCAL_AUTH_NAME?: string,
    LOCAL_AUTH_EMAIL?: string,

    SUPABASE_URL?: string,
    SUPABASE_SECRET_KEY?: string,

    BACKEND_PORT?: number,
    FRONTEND_PORT?: number,

    DATABASE_URL: string,
    DATABASE_PASSWORD: string,

    OPENROUTER_API_KEY?: string,
    OPENROUTER_PROVISION_KEY?: string,

    REDIS_URL: string,
    CODEX_TIMEOUT_MS?: string,
    CODEX_BIN?: string,
  }
}

export type App = typeof app
