import { z } from "zod"
import { mkdirSync, existsSync } from "node:fs"
import { get_model_by_id, get_model_transport } from "@shared/types/research"
import type { ModelConfig, ModelID } from "@shared/types/research"
import { save_llm_log } from "./research_utils"
import type { DbOrTx } from "./research_utils"
import { get_user_openrouter_key } from "@backend/openrouter/provider"

export interface LLMUsage {
  cost: number | null,
  transport: "openrouter" | "codex_cli" | "opencode_cli",
  [key: string]: unknown,
}

interface OpenRouterContentBlock {
  type: string,
  text?: string,
}

interface OpenRouterOutputBlock {
  type: string,
  status?: string,
  content?: OpenRouterContentBlock[],
}

interface OpenRouterAPIResponse {
  output: OpenRouterOutputBlock[],
  usage?: Record<string, unknown>,
  status?: string,
}

interface LLMSuccessResponse {
  success: true,
  usage: LLMUsage,
  time: number,
  model_id: ModelID,
  warnings?: string[],
}

export interface LLMTextSuccessResponse extends LLMSuccessResponse{
  output: string
}

export interface LLMStructuredSuccessResponse<T> extends LLMSuccessResponse {
  output: T,
}

type LLMErrorResponse = {
  success: false,
  error: Error,
  model_id: ModelID,
}

export type LLMStructuredResponse<T> = LLMStructuredSuccessResponse<T> | LLMErrorResponse
export type LLMTextResponse = LLMTextSuccessResponse | LLMErrorResponse
export type LLMResponse<T> = LLMStructuredResponse<T> | LLMTextResponse

interface LLMMessages {
  role: "system" | "user" | "assistant",
  content: string,
}

interface GenerateLLMBaseParams {
  db: DbOrTx,
  model: ModelConfig,
  user_id: string,
  messages: LLMMessages[],
  prompt_file_id: string,
  context: string,
  max_retries?: number,
  save_to_db?: boolean,
  temperature?: number,
}

interface GenerateLLMParamsWithSchema<T> extends GenerateLLMBaseParams {
  schema: z.ZodType<T>
}

interface GenerateLLMParamsWithoutSchema extends GenerateLLMBaseParams {
  schema?: undefined
}

export type GenerateLLMParams<T> = GenerateLLMParamsWithSchema<T> | GenerateLLMParamsWithoutSchema

interface OpenRouterRequestBody {
  model: string,
  input: (LLMMessages & { type: "message" })[],
  temperature: number,
  reasoning: any,
  provider: {
    only: string[],
  },
  user: string,
  text: {
    verbosity: "high",
    format: {
      type: "text"
    } | {
      type: "json_schema",
      strict: true,
      name: string,
      schema: any,
    }
  },
  max_output_tokens?: number,
}

function extract_text_output(data: OpenRouterAPIResponse) {
  const message = data.output?.find((o) => o.type === "message")
  const content = message?.content?.find((c) => c.type === "output_text" || c.type === "text")
  if (!content?.text) return null
  return {
    text: content.text
      .normalize("NFC")
      .replace(/\u0000/g, "")
      .replace(/[\u0001-\u0008\u000B\u000C\u000E-\u001F]/g, ""),
  }
}

function parse_json_output<T>(text: string, schema: z.ZodType<T>, log_prefix: string): T {
  const candidates = extract_json_candidates(text)
  let parsed_json: unknown

  let parsed = false
  for (const candidate of candidates) {
    try {
      parsed_json = JSON.parse(candidate)
      parsed = true
      console.log(`${log_prefix} JSON parsed successfully`)
      break
    } catch {
      // try next candidate
    }
  }

  if (!parsed) {
    console.error(`${log_prefix} JSON parse error: no valid JSON candidate found`)
    throw new Error("Invalid JSON in response")
  }

  try {
    const result = schema.parse(parsed_json) as T
    console.log(`${log_prefix} Zod validation passed`)
    return result
  } catch (err) {
    console.error(`${log_prefix} Zod validation error:`, err)
    throw new Error("Response doesn't match expected schema")
  }
}

function extract_json_candidates(text: string) {
  const candidates: string[] = []
  const trimmed = text.trim()
  if (trimmed) candidates.push(trimmed)

  const fenced = trimmed.match(/```json\s*([\s\S]*?)\s*```/i)
  if (fenced?.[1]) candidates.push(fenced[1].trim())

  const first_object = extract_first_balanced_json(trimmed)
  if (first_object) candidates.push(first_object)

  // de-duplicate while preserving order
  const seen = new Set<string>()
  return candidates.filter((candidate) => {
    if (seen.has(candidate)) return false
    seen.add(candidate)
    return true
  })
}

function extract_first_balanced_json(text: string) {
  const start = text.indexOf("{")
  if (start < 0) return null

  let depth = 0
  let in_string = false
  let escaped = false

  for (let i = start; i < text.length; i++) {
    const ch = text[i]

    if (in_string) {
      if (escaped) escaped = false
      else if (ch === "\\") escaped = true
      else if (ch === "\"") in_string = false
      continue
    }

    if (ch === "\"") {
      in_string = true
      continue
    }

    if (ch === "{") depth++
    if (ch === "}") {
      depth--
      if (depth === 0) {
        return text.slice(start, i + 1).trim()
      }
    }
  }

  return null
}

function get_reasoning_config(model: ModelConfig) {
  const reasoning = model.config.reasoning_effort

  if (reasoning === null) return undefined
  if (typeof reasoning === "boolean") return { enabled: reasoning }

  if (reasoning === "none") {
    if (model.id === "google/gemini-3-flash-preview") return { enabled: false }
  }

  return { effort: reasoning }
}

function get_codex_reasoning_effort(model: ModelConfig): "low" | "medium" | "high" | "xhigh" {
  const value = model.config.reasoning_effort
  if (value === "low" || value === "medium" || value === "high" || value === "xhigh") return value
  if (value === "none" || value === false || value === null) return "low"
  return "medium"
}

function build_codex_prompt(messages: LLMMessages[]) {
  const prefix = [
    "IMPORTANT RUNTIME CONSTRAINTS:",
    "- This is a non-interactive one-shot run.",
    "- Do NOT ask follow-up questions.",
    "- Do NOT run tools or commands.",
    "- Do NOT attempt command execution, shell calls, file edits, or environment inspection.",
    "- Return exactly one final assistant message as the answer.",
    "- Follow the task instructions as written.",
    "- Partial, rigorous progress is acceptable when full resolution is not possible.",
    "",
  ].join("\n")

  return prefix + messages.map((message) =>
    `### ${message.role.toUpperCase()}\n${message.content}`.trim()
  ).join("\n\n")
}

function get_codex_cli_model_id(model_id: string) {
  if (model_id === "chatgpt-5.2") return "gpt-5.2"
  if (model_id === "chatgpt-5.3-codex") return "gpt-5.3-codex"
  return model_id
}

const OPENCODE_GEMINI_PRO_MODEL_ID = "opencode/google/gemini-3-pro-preview" as const
const OPENCODE_GEMINI_FLASH_MODEL_ID = "opencode/google/gemini-3-flash-preview" as const

function get_opencode_cli_model_id(model_id: string) {
  if (model_id === OPENCODE_GEMINI_PRO_MODEL_ID) return "google/gemini-3-pro-preview"
  if (model_id === OPENCODE_GEMINI_FLASH_MODEL_ID) return "google/gemini-3-flash-preview"
  return model_id
}

function is_opencode_quota_error(text: string) {
  return /(quota|resource[_\s-]?exhausted|rate[_\s-]?limit|daily[_\s-]?limit|too many requests|status[^0-9]*429|\b429\b)/i.test(text)
}

function ensure_local_temp_dir() {
  const dir = `/tmp/codex-llm-${process.pid}`
  mkdirSync(dir, { recursive: true })
  if (!existsSync(dir)) {
    throw new Error(`Failed to create temp dir for Codex: ${dir}`)
  }
  return dir
}

function get_debug_dir(base_dir: string) {
  const debug_dir = `${base_dir}/debug`
  mkdirSync(debug_dir, { recursive: true })
  return debug_dir
}

function should_debug_codex() {
  return Bun.env.CODEX_DEBUG === "1" || Bun.env.CODEX_DEBUG === "true"
}

function should_debug_opencode() {
  return Bun.env.OPENCODE_DEBUG === "1" || Bun.env.OPENCODE_DEBUG === "true"
}

async function run_codex_exec(command: string[], timeout_ms: number, input?: string) {
  const proc = Bun.spawn(command, {
    stdout: "pipe",
    stderr: "pipe",
    stdin: input !== undefined ? "pipe" : "ignore",
  })

  if (input !== undefined && proc.stdin) {
    proc.stdin.write(input)
    proc.stdin.end()
  }

  let timed_out = false
  const timer = setTimeout(() => {
    timed_out = true
    try {
      proc.kill()
    } catch {
      // noop
    }
  }, timeout_ms)

  const exit_code = await proc.exited
  clearTimeout(timer)

  const stdout = await new Response(proc.stdout).text()
  const stderr = await new Response(proc.stderr).text()

  return { exit_code, stdout, stderr, timed_out }
}

async function run_opencode_exec(command: string[], timeout_ms: number) {
  const proc = Bun.spawn(command, {
    stdout: "pipe",
    stderr: "pipe",
    stdin: "ignore",
  })

  let timed_out = false
  const timer = setTimeout(() => {
    timed_out = true
    try {
      proc.kill()
    } catch {
      // noop
    }
  }, timeout_ms)

  const exit_code = await proc.exited
  clearTimeout(timer)

  const stdout = await new Response(proc.stdout).text()
  const stderr = await new Response(proc.stderr).text()

  return { exit_code, stdout, stderr, timed_out }
}

function summarize_codex_jsonl(stdout: string) {
  const lines = stdout.split("\n").map(line => line.trim()).filter(Boolean)
  const event_types: Record<string, number> = {}
  const errors: string[] = []
  let turn_completed = false
  let turn_failed = false

  for (const line of lines) {
    if (!line.startsWith("{")) continue
    try {
      const event = JSON.parse(line) as Record<string, unknown>
      const type = typeof event.type === "string" ? event.type : "unknown"
      event_types[type] = (event_types[type] ?? 0) + 1
      if (type === "turn.completed") turn_completed = true
      if (type === "turn.failed") turn_failed = true
      if (type === "error" && typeof event.message === "string") errors.push(event.message)
    } catch {
      // ignore malformed line
    }
  }

  return {
    turn_completed,
    turn_failed,
    errors,
    event_types,
    line_count: lines.length,
  }
}

function extract_codex_output_from_jsonl(stdout: string) {
  const chunks: string[] = []
  const lines = stdout.split("\n").map(line => line.trim()).filter(Boolean)

  const collect_text_blocks = (content: unknown): string[] => {
    if (!Array.isArray(content)) return []
    const text_chunks: string[] = []
    for (const block of content) {
      if (!block || typeof block !== "object") continue
      const typed = block as Record<string, unknown>
      const block_type = typeof typed.type === "string" ? typed.type : ""
      if (block_type !== "output_text" && block_type !== "text") continue
      if (typeof typed.text === "string") text_chunks.push(typed.text)
    }
    return text_chunks
  }

  for (const line of lines) {
    if (!line.startsWith("{")) continue
    try {
      const event = JSON.parse(line) as Record<string, unknown>
      const type = typeof event.type === "string" ? event.type : ""

      if (type === "agent_message" || type === "assistant_message") {
        const payload = event.payload
        if (payload && typeof payload === "object") {
          const p = payload as Record<string, unknown>
          if (typeof p.text === "string") chunks.push(p.text)
          if (typeof p.message === "string") chunks.push(p.message)
        }
        continue
      }

      if (type === "item.completed") {
        const item = event.item
        if (!item || typeof item !== "object") continue
        const i = item as Record<string, unknown>
        const item_type = typeof i.type === "string" ? i.type : ""
        if (item_type !== "agent_message" && item_type !== "assistant_message") continue
        if (typeof i.text === "string") chunks.push(i.text)
        if (typeof i.message === "string") chunks.push(i.message)
        continue
      }

      if (type === "response_item") {
        const payload = event.payload
        if (!payload || typeof payload !== "object") continue
        const p = payload as Record<string, unknown>
        const payload_type = typeof p.type === "string" ? p.type : ""

        if (payload_type === "message") {
          chunks.push(...collect_text_blocks(p.content))
          continue
        }

        if ((payload_type === "output_text" || payload_type === "text") && typeof p.text === "string") {
          chunks.push(p.text)
          continue
        }
      }
    } catch {
      // noop
    }
  }

  const text = chunks.join("").trim()
  return text || null
}

function extract_codex_output_from_stdout_fallback(stdout: string) {
  const candidates: string[] = []

  const push_candidate = (value: string) => {
    const cleaned = value
      .replace(/\\n/g, "\n")
      .replace(/\\"/g, "\"")
      .replace(/\\\\/g, "\\")
      .trim()
    if (!cleaned) return
    if (cleaned.length < 40) return
    if (/^item\./i.test(cleaned)) return
    candidates.push(cleaned)
  }

  // 1) Try to parse each JSONL line and collect explicit text/message fields.
  const lines = stdout.split("\n").map(line => line.trim()).filter(Boolean)
  for (const line of lines) {
    if (!line.startsWith("{")) continue
    try {
      const event = JSON.parse(line) as Record<string, unknown>
      const walk = (value: unknown): void => {
        if (!value || typeof value !== "object") return
        if (Array.isArray(value)) {
          for (const item of value) walk(item)
          return
        }
        const obj = value as Record<string, unknown>
        if (typeof obj.text === "string") push_candidate(obj.text)
        if (typeof obj.message === "string") push_candidate(obj.message)
        if (typeof obj.delta === "string") push_candidate(obj.delta)
        for (const nested of Object.values(obj)) walk(nested)
      }
      walk(event)
    } catch {
      // noop
    }
  }

  // 2) Regex fallback for JSON-like fragments in stdout.
  const regexes = [
    /"text"\s*:\s*"((?:[^"\\]|\\.)*)"/g,
    /"message"\s*:\s*"((?:[^"\\]|\\.)*)"/g,
  ]
  for (const regex of regexes) {
    let match: RegExpExecArray | null = null
    while ((match = regex.exec(stdout)) !== null) {
      if (match[1]) push_candidate(match[1])
    }
  }

  if (candidates.length === 0) return null
  candidates.sort((a, b) => b.length - a.length)
  return candidates[0]
}

function extract_opencode_output_from_json(stdout: string) {
  const lines = stdout.split("\n").map(line => line.trim()).filter(Boolean)
  const strong_candidates: string[] = []
  const weak_candidates: string[] = []

  const push_candidate = (target: string[], value: string) => {
    const cleaned = value.trim()
    if (!cleaned) return
    if (cleaned.length < 2) return
    target.push(cleaned)
  }

  const collect_string_fields = (value: unknown, target: string[]): void => {
    if (!value || typeof value !== "object") return
    if (Array.isArray(value)) {
      for (const item of value) collect_string_fields(item, target)
      return
    }
    const obj = value as Record<string, unknown>
    if (typeof obj.text === "string") push_candidate(target, obj.text)
    if (typeof obj.message === "string") push_candidate(target, obj.message)
    if (typeof obj.content === "string") push_candidate(target, obj.content)
  }

  for (const line of lines) {
    if (!line.startsWith("{")) continue
    try {
      const event = JSON.parse(line) as Record<string, unknown>
      const type = typeof event.type === "string" ? event.type : ""
      const role = typeof event.role === "string" ? event.role : ""

      if (typeof event.text === "string" && role === "assistant") {
        push_candidate(strong_candidates, event.text)
      }
      if (typeof event.message === "string" && role === "assistant") {
        push_candidate(strong_candidates, event.message)
      }

      if (type === "item.completed") {
        const item = event.item
        if (item && typeof item === "object") {
          const i = item as Record<string, unknown>
          const item_type = typeof i.type === "string" ? i.type : ""
          if (item_type === "agent_message" || item_type === "assistant_message" || item_type === "message") {
            collect_string_fields(i, strong_candidates)
          } else if (item_type === "reasoning") {
            // Ignore reasoning blocks; these often contain JSON-like snippets
            // that are not the requested structured output schema.
          } else {
            collect_string_fields(i, weak_candidates)
          }
        }
      } else if (type === "agent_message" || type === "assistant_message") {
        collect_string_fields(event, strong_candidates)
      } else if (/assistant|message|response|output|completed/i.test(type)) {
        collect_string_fields(event, weak_candidates)
      } else {
        collect_string_fields(event, weak_candidates)
      }
    } catch {
      // noop
    }
  }

  const ordered = [...strong_candidates, ...weak_candidates]
  if (ordered.length === 0) return null
  return ordered[ordered.length - 1]
}

function parse_json_output_candidates<T>(candidates: string[], schema: z.ZodType<T>): T | null {
  const parsed_objects: unknown[] = []

  for (const candidate of candidates) {
    const json_candidates = extract_json_candidates(candidate)
    for (const json_candidate of json_candidates) {
      try {
        const parsed = JSON.parse(json_candidate) as unknown
        parsed_objects.push(parsed)
        const validated = schema.safeParse(parsed)
        if (validated.success) return validated.data
      } catch {
        // try next candidate
      }
    }
  }

  for (const parsed of parsed_objects) {
    const normalized = normalize_common_structured_output(parsed)
    const validated = schema.safeParse(normalized)
    if (validated.success) return validated.data
  }

  return null
}

function parse_markdown_structured_fallback<T>(candidates: string[], schema: z.ZodType<T>): T | null {
  const text = candidates.find((candidate) => candidate.trim().length > 0)
  if (!text) return null

  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
  const first_line = lines[0]?.replace(/^[-*#\s`]+/, "").trim() ?? ""
  const one_line_summary = first_line.slice(0, 100)

  const payload = {
    summary: text,
    one_line_summary,
  }

  const validated = schema.safeParse(payload)
  if (validated.success) return validated.data
  return null
}

function normalize_common_structured_output(value: unknown): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) return value
  const obj = { ...(value as Record<string, unknown>) }

  // Common verifier shape issue on Gemini/OpenCode:
  // notes/proofs/output updates returned as plain strings instead of { action, content }.
  for (const key of ["notes_update", "proofs_update", "output_update"] as const) {
    const update = obj[key]
    if (typeof update === "string") {
      obj[key] = { action: "append", content: update }
      continue
    }
    if (update && typeof update === "object" && !Array.isArray(update)) {
      const u = update as Record<string, unknown>
      const action = u.action === "replace" ? "replace" : "append"
      const content = typeof u.content === "string"
        ? u.content
        : typeof u.text === "string"
          ? u.text
          : ""
      obj[key] = { action, content }
      continue
    }
    if (update == null) {
      obj[key] = { action: "append", content: "" }
    }
  }

  if (typeof obj.verdict !== "string" || !["promising", "uncertain", "unlikely"].includes(obj.verdict)) {
    obj.verdict = "uncertain"
  }

  if (!Array.isArray(obj.blocking_issues)) obj.blocking_issues = []

  if (!Array.isArray(obj.per_prover)) {
    if (obj.per_prover && typeof obj.per_prover === "object" && !Array.isArray(obj.per_prover)) {
      const entries = Object.entries(obj.per_prover as Record<string, unknown>)
      obj.per_prover = entries.map(([prover_id, value]) => {
        if (typeof value === "string") {
          const lowered = value.toLowerCase()
          const score = (["promising", "uncertain", "unlikely"] as const).find((s) => lowered.includes(s)) ?? "uncertain"
          return { prover_id, brief_feedback: value, score }
        }
        if (value && typeof value === "object") {
          const v = value as Record<string, unknown>
          const score = v.score === "promising" || v.score === "uncertain" || v.score === "unlikely"
            ? v.score
            : "uncertain"
          const brief_feedback = typeof v.brief_feedback === "string"
            ? v.brief_feedback
            : typeof v.feedback === "string"
              ? v.feedback
              : ""
          return { prover_id, brief_feedback, score }
        }
        return { prover_id, brief_feedback: "", score: "uncertain" }
      })
    } else {
      obj.per_prover = []
    }
  } else {
    obj.per_prover = obj.per_prover.map((entry) => {
      if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
        return { prover_id: "unknown", brief_feedback: "", score: "uncertain" as const }
      }
      const e = entry as Record<string, unknown>

      if (!("prover_id" in e) && Object.keys(e).length === 1) {
        const [key, value] = Object.entries(e)[0]
        if (typeof value === "string") {
          const lowered = value.toLowerCase()
          const score = (["promising", "uncertain", "unlikely"] as const).find((s) => lowered.includes(s)) ?? "uncertain"
          return { prover_id: key, brief_feedback: value, score }
        }
      }

      const prover_id = typeof e.prover_id === "string" ? e.prover_id : "unknown"
      const brief_feedback = typeof e.brief_feedback === "string"
        ? e.brief_feedback
        : typeof e.feedback === "string"
          ? e.feedback
          : ""
      const score = e.score === "promising" || e.score === "uncertain" || e.score === "unlikely"
        ? e.score
        : "uncertain"
      return { prover_id, brief_feedback, score }
    })
  }

  if (typeof obj.feedback_md !== "string") {
    if (typeof obj.feedback === "string") obj.feedback_md = obj.feedback
    else if (typeof obj.analysis === "string") obj.feedback_md = obj.analysis
  }

  if (typeof obj.summary_md !== "string") {
    if (typeof obj.summary === "string") obj.summary_md = obj.summary
  }

  return obj
}

function get_opencode_output_candidates(stdout: string) {
  const by_message = collect_opencode_text_by_message(stdout)
  const primary = extract_opencode_output_from_json(stdout)
  const fallback = extract_codex_output_from_stdout_fallback(stdout)
  const candidates = [...by_message, primary, fallback].filter((value): value is string => Boolean(value))
  const deduped: string[] = []
  const seen = new Set<string>()
  for (const candidate of candidates) {
    if (seen.has(candidate)) continue
    seen.add(candidate)
    deduped.push(candidate)
  }
  return deduped
}

function collect_opencode_text_by_message(stdout: string) {
  const lines = stdout.split("\n").map(line => line.trim()).filter(Boolean)
  const by_message = new Map<string, string>()

  for (const line of lines) {
    if (!line.startsWith("{")) continue
    try {
      const event = JSON.parse(line) as Record<string, unknown>
      if (event.type !== "text") continue

      const part = event.part
      if (!part || typeof part !== "object") continue
      const p = part as Record<string, unknown>
      if (typeof p.text !== "string") continue

      const message_id = typeof p.messageID === "string"
        ? p.messageID
        : typeof p.id === "string"
          ? p.id
          : crypto.randomUUID()

      by_message.set(message_id, (by_message.get(message_id) ?? "") + p.text)
    } catch {
      // noop
    }
  }

  return Array.from(by_message.values()).map((value) => value.trim()).filter(Boolean)
}

async function generate_via_openrouter<T>(
  params: GenerateLLMParams<T>,
  schema: z.ZodType<T> | undefined,
  is_structured: boolean
): Promise<LLMStructuredResponse<T> | LLMTextResponse> {
  const {
    db,
    model,
    user_id,
    messages,
    prompt_file_id,
    context,
    max_retries = 3,
    save_to_db = true,
    temperature = 1,
  } = params

  const log_prefix = `[OpenRouter][${model.id}][${context}]`
  const model_id = model.id
  const model_info = get_model_by_id(model_id)!
  const api_key = await get_user_openrouter_key(db, user_id)

  const openrouter_input = messages.map((msg) => ({
    type: "message" as const,
    role: msg.role,
    content: msg.content,
  }))

  let request_body: OpenRouterRequestBody = {
    model: model_id,
    input: openrouter_input,
    temperature,
    reasoning: get_reasoning_config(model),
    provider: {
      only: [model_info.provider],
    },
    user: user_id,
    text: {
      verbosity: "high",
      format: { type: "text" }
    }
  }

  if (schema) {
    request_body.text.format = {
      type: "json_schema",
      strict: true,
      name: `${context}-schema`,
      schema: schema.toJSONSchema(),
    }
  }

  if (model_info.max_output_tokens) {
    request_body.max_output_tokens = model_info.max_output_tokens
  }

  for (let attempt = 1; attempt <= max_retries; attempt++) {
    const start_time = performance.now()
    console.log(`${log_prefix} attempt ${attempt}/${max_retries}`)

    try {
      const response = await fetch("https://openrouter.ai/api/v1/responses", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${api_key}`,
          "Content-Type": "application/json",
          "HTTP-Referer": "https://bolzano.app",
          "X-Title": "Bolzano",
        },
        body: JSON.stringify(request_body),
      })

      if (!response.ok) {
        const error_text = await response.text()
        console.error(`${log_prefix} API error:`, error_text.slice(0, 1000))
        throw new Error(`API error ${response.status}`)
      }

      const data = (await response.json()) as OpenRouterAPIResponse
      const time = (performance.now() - start_time) / 1000
      const extracted = extract_text_output(data)
      if (!extracted) throw new Error("No text output in response")

      let output: T | string
      if (is_structured && schema) output = parse_json_output(extracted.text, schema, log_prefix)
      else output = extracted.text

      const usage: LLMUsage = {
        transport: "openrouter",
        cost: typeof data.usage?.cost === "number" ? data.usage.cost : null,
        ...(data.usage ?? {})
      }

      if (save_to_db) {
        await save_llm_log(db, prompt_file_id, { output, request_body }, usage, model_id)
      }

      if (is_structured) {
        return { success: true, output: output as T, usage, time, model_id } as LLMStructuredSuccessResponse<T>
      }
      return { success: true, output: output as string, usage, time, model_id } as LLMTextSuccessResponse
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e))
      console.warn(`${log_prefix} [${attempt}/${max_retries}] Failed: ${error.message}`)
      if (attempt === max_retries) return { success: false, error, model_id }
    }
  }

  return { success: false, error: new Error("Unexpected error"), model_id }
}

async function generate_via_codex_cli<T>(
  params: GenerateLLMParams<T>,
  schema: z.ZodType<T> | undefined,
  is_structured: boolean
): Promise<LLMStructuredResponse<T> | LLMTextResponse> {
  const {
    db,
    model,
    prompt_file_id,
    context,
    max_retries = 3,
    save_to_db = true,
    messages,
  } = params

  const model_id = model.id
  const log_prefix = `[Codex][${model_id}][${context}]`
  const codex_bin = Bun.env.CODEX_BIN ?? "codex"
  const timeout_ms = Number(Bun.env.CODEX_TIMEOUT_MS ?? "1800000")
  const prompt = build_codex_prompt(messages)
  console.log(`${log_prefix} config: timeout_ms=${timeout_ms} env_CODEX_TIMEOUT_MS=${Bun.env.CODEX_TIMEOUT_MS ?? "<unset>"}`)

  for (let attempt = 1; attempt <= max_retries; attempt++) {
    const start_time = performance.now()
    const temp_dir = ensure_local_temp_dir()
    const output_file_name = `codex-last-message-${crypto.randomUUID()}.txt`
    const schema_file_name = is_structured ? `codex-schema-${crypto.randomUUID()}.json` : null
    const output_file = `${temp_dir}/${output_file_name}`
    const schema_file = schema_file_name ? `${temp_dir}/${schema_file_name}` : null

    try {
      if (schema && schema_file) {
        await Bun.write(schema_file, JSON.stringify(schema.toJSONSchema(), null, 2))
      }

      const command = [
        codex_bin, "exec",
        "--json",
        "-m", get_codex_cli_model_id(model_id),
        "--cd", temp_dir,
        "--skip-git-repo-check",
        "--sandbox", "read-only",
        "--disable", "apps",
        "-c", `model_reasoning_effort=\"${get_codex_reasoning_effort(model)}\"`,
        "--output-last-message", output_file_name,
      ]
      if (schema && schema_file) {
        // Codex resolves this before applying --cd in some paths; absolute path is robust.
        command.push("--output-schema", schema_file)
      }
      command.push("-")

      const result = await run_codex_exec(command, timeout_ms, prompt)
      const diagnostics = summarize_codex_jsonl(result.stdout)
      const debug_enabled = should_debug_codex()
      if (debug_enabled) {
        const debug_dir = get_debug_dir(temp_dir)
        const attempt_prefix = `${debug_dir}/attempt-${attempt}-${crypto.randomUUID()}`
        await Bun.write(`${attempt_prefix}.command.txt`, command.join(" "))
        await Bun.write(`${attempt_prefix}.prompt.txt`, prompt)
        await Bun.write(`${attempt_prefix}.stdout.jsonl`, result.stdout)
        await Bun.write(`${attempt_prefix}.stderr.log`, result.stderr)
        await Bun.write(`${attempt_prefix}.summary.json`, JSON.stringify({
          diagnostics,
          exit_code: result.exit_code,
          timed_out: result.timed_out,
          output_file,
          output_exists: await Bun.file(output_file).exists(),
          schema_file,
          schema_exists: schema_file ? await Bun.file(schema_file).exists() : null,
        }, null, 2))
      }

      if (result.timed_out) {
        throw new Error(
          `codex exec timed out after ${timeout_ms}ms ` +
          `codex_events=${JSON.stringify(diagnostics.event_types)} ` +
          `turn_completed=${diagnostics.turn_completed} turn_failed=${diagnostics.turn_failed} ` +
          `errors=${JSON.stringify(diagnostics.errors.slice(-3))}`
        )
      }

      if (result.exit_code !== 0) {
        const stderr = result.stderr ?? ""
        const head = stderr.slice(0, 2000)
        const tail = stderr.length > 2000 ? stderr.slice(-2000) : ""
        const rendered = tail
          ? `${head}\n...\n${tail}`
          : head
        throw new Error(
          `codex exec failed (${result.exit_code}): ${rendered}\n` +
          `codex_events=${JSON.stringify(diagnostics.event_types)} ` +
          `turn_completed=${diagnostics.turn_completed} turn_failed=${diagnostics.turn_failed} ` +
          `errors=${JSON.stringify(diagnostics.errors.slice(-3))}`
        )
      }

      const output_handle = Bun.file(output_file)
      if (!(await output_handle.exists())) {
        const json_text = extract_codex_output_from_jsonl(result.stdout)
        const stdout_fallback_text = json_text ?? extract_codex_output_from_stdout_fallback(result.stdout)

        if (stdout_fallback_text) {
          const output = is_structured && schema
            ? parse_json_output(stdout_fallback_text, schema, log_prefix)
            : stdout_fallback_text

          const time = (performance.now() - start_time) / 1000
          const usage: LLMUsage = { transport: "codex_cli", cost: 0 }
          if (save_to_db) {
            await save_llm_log(db, prompt_file_id, { output, command, fallback: "json_stdout_or_regex" }, usage, model_id)
          }
          if (is_structured) return {
            success: true,
            output: output as T,
            usage,
            time,
            model_id,
          } as LLMStructuredSuccessResponse<T>
          return {
            success: true,
            output: output as string,
            usage,
            time,
            model_id,
          } as LLMTextSuccessResponse
        }

        const stderr = result.stderr ?? ""
        const stdout = result.stdout ?? ""
        throw new Error(
          `codex exec did not produce output file (${output_file}). ` +
          `turn_completed=${diagnostics.turn_completed} turn_failed=${diagnostics.turn_failed} ` +
          `events=${JSON.stringify(diagnostics.event_types)} ` +
          `errors=${JSON.stringify(diagnostics.errors.slice(-3))} ` +
          `stderr=${stderr.slice(-1200)} stdout=${stdout.slice(-600)}`
        )
      }

      const text = (await output_handle.text()).trim()
      if (!text) throw new Error("Codex returned empty output")

      let output: T | string
      if (is_structured && schema) output = parse_json_output(text, schema, log_prefix)
      else output = text

      const time = (performance.now() - start_time) / 1000
      const usage: LLMUsage = { transport: "codex_cli", cost: 0 }

      if (save_to_db) {
        await save_llm_log(db, prompt_file_id, { output, command }, usage, model_id)
      }

      if (is_structured) return {
        success: true,
        output: output as T,
        usage,
        time,
        model_id,
      } as LLMStructuredSuccessResponse<T>

      return {
        success: true,
        output: output as string,
        usage,
        time,
        model_id,
      } as LLMTextSuccessResponse
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e))
      console.warn(`${log_prefix} [${attempt}/${max_retries}] Failed: ${error.message}`)
      if (attempt === max_retries) return { success: false, error, model_id }
    }
  }

  return { success: false, error: new Error("Unexpected error"), model_id }
}

async function generate_via_opencode_cli<T>(
  params: GenerateLLMParams<T>,
  schema: z.ZodType<T> | undefined,
  is_structured: boolean
): Promise<LLMStructuredResponse<T> | LLMTextResponse> {
  const {
    db,
    model,
    prompt_file_id,
    context,
    max_retries = 3,
    save_to_db = true,
    messages,
  } = params

  const initial_model_id = model.id
  const log_prefix = `[OpenCode][${initial_model_id}][${context}]`
  const opencode_bin = Bun.env.OPENCODE_BIN ?? "opencode"
  const timeout_ms = Number(Bun.env.OPENCODE_TIMEOUT_MS ?? "1800000")
  const prompt = build_codex_prompt(messages)
  let active_model_id = initial_model_id
  let fallback_used = false
  let fallback_warning: string | null = null

  for (let attempt = 1; attempt <= max_retries; attempt++) {
    const start_time = performance.now()
    const temp_dir = ensure_local_temp_dir()
    const command = [
      opencode_bin,
      "run",
      "-m",
      get_opencode_cli_model_id(active_model_id),
      "--format",
      "json",
      prompt,
    ]

    try {
      const result = await run_opencode_exec(command, timeout_ms)
      const diagnostics = summarize_codex_jsonl(result.stdout)

      if (should_debug_opencode()) {
        const debug_dir = get_debug_dir(temp_dir)
        const attempt_prefix = `${debug_dir}/opencode-attempt-${attempt}-${crypto.randomUUID()}`
        await Bun.write(`${attempt_prefix}.command.txt`, command.join(" "))
        await Bun.write(`${attempt_prefix}.prompt.txt`, prompt)
        await Bun.write(`${attempt_prefix}.stdout.jsonl`, result.stdout)
        await Bun.write(`${attempt_prefix}.stderr.log`, result.stderr)
        await Bun.write(`${attempt_prefix}.summary.json`, JSON.stringify({
          diagnostics,
          exit_code: result.exit_code,
          timed_out: result.timed_out,
          active_model_id,
        }, null, 2))
      }

      const combined_output = `${result.stderr}\n${result.stdout}`
      if (result.timed_out) {
        throw new Error(
          `opencode run timed out after ${timeout_ms}ms ` +
          `events=${JSON.stringify(diagnostics.event_types)} ` +
          `turn_completed=${diagnostics.turn_completed} turn_failed=${diagnostics.turn_failed} ` +
          `errors=${JSON.stringify(diagnostics.errors.slice(-3))}`
        )
      }

      if (result.exit_code !== 0) {
        if (!fallback_used && active_model_id === OPENCODE_GEMINI_PRO_MODEL_ID && is_opencode_quota_error(combined_output)) {
          fallback_used = true
          active_model_id = OPENCODE_GEMINI_FLASH_MODEL_ID
          fallback_warning = `Gemini Pro quota exhausted for ${context}; automatically retried with Gemini Flash.`
          console.warn(`${log_prefix} quota exhausted, retrying with Gemini Flash`)
          continue
        }

        throw new Error(
          `opencode run failed (${result.exit_code}): ` +
          `${(result.stderr || result.stdout).slice(0, 2000)}`
        )
      }

      const output_candidates = get_opencode_output_candidates(result.stdout)
      if (output_candidates.length === 0) {
        if (!fallback_used && active_model_id === OPENCODE_GEMINI_PRO_MODEL_ID && is_opencode_quota_error(combined_output)) {
          fallback_used = true
          active_model_id = OPENCODE_GEMINI_FLASH_MODEL_ID
          fallback_warning = `Gemini Pro quota exhausted for ${context}; automatically retried with Gemini Flash.`
          console.warn(`${log_prefix} quota exhausted (empty output), retrying with Gemini Flash`)
          continue
        }
        throw new Error(
          `opencode run did not produce parseable output. ` +
          `stdout=${result.stdout.slice(-600)} stderr=${result.stderr.slice(-600)}`
        )
      }

      let output: T | string
      if (is_structured && schema) {
        const parsed_json = parse_json_output_candidates(output_candidates, schema)
        const parsed_markdown = parsed_json ? null : parse_markdown_structured_fallback(output_candidates, schema)
        const parsed = parsed_json ?? parsed_markdown
        if (!parsed) {
          throw new Error("Response doesn't match expected schema")
        }
        output = parsed
      } else {
        output = output_candidates[output_candidates.length - 1]
      }

      const time = (performance.now() - start_time) / 1000
      const usage: LLMUsage = { transport: "opencode_cli", cost: 0 }
      const warnings = fallback_warning ? [fallback_warning] : undefined

      if (save_to_db) {
        await save_llm_log(db, prompt_file_id, { output, command, warnings }, usage, active_model_id as ModelID)
      }

      if (is_structured) return {
        success: true,
        output: output as T,
        usage,
        time,
        model_id: active_model_id as ModelID,
        warnings,
      } as LLMStructuredSuccessResponse<T>

      return {
        success: true,
        output: output as string,
        usage,
        time,
        model_id: active_model_id as ModelID,
        warnings,
      } as LLMTextSuccessResponse
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e))
      const error_text = error.message ?? ""
      if (!fallback_used && active_model_id === OPENCODE_GEMINI_PRO_MODEL_ID && is_opencode_quota_error(error_text)) {
        fallback_used = true
        active_model_id = OPENCODE_GEMINI_FLASH_MODEL_ID
        fallback_warning = `Gemini Pro quota exhausted for ${context}; automatically retried with Gemini Flash.`
        console.warn(`${log_prefix} quota exhausted, retrying with Gemini Flash`)
        continue
      }

      console.warn(`${log_prefix} [${attempt}/${max_retries}] Failed: ${error.message}`)
      if (attempt === max_retries) return { success: false, error, model_id: active_model_id as ModelID }
    }
  }

  return { success: false, error: new Error("Unexpected error"), model_id: active_model_id as ModelID }
}

export async function generate_llm_response(params: GenerateLLMParamsWithoutSchema): Promise<LLMTextResponse>
export async function generate_llm_response<T>(params: GenerateLLMParamsWithSchema<T>): Promise<LLMStructuredResponse<T>>

export async function generate_llm_response<T>(
  params: GenerateLLMParams<T>
): Promise<LLMStructuredResponse<T> | LLMTextResponse> {
  const schema = "schema" in params ? params.schema : undefined
  const is_structured = schema !== undefined
  const transport = get_model_transport(params.model.id)

  if (transport === "codex_cli") return generate_via_codex_cli(params, schema, is_structured)
  if (transport === "opencode_cli") return generate_via_opencode_cli(params, schema, is_structured)
  return generate_via_openrouter(params, schema, is_structured)
}
