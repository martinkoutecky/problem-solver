import { get_model_by_id, models, type ModelConfig, type ModelID, type ReasoningConfig, type ReasoningEffort, type ReasoningEffortValue } from "@shared/types/research"

export type Transport = "openrouter" | "codex_cli" | "opencode_cli" | "claude_cli" | "metacentrum_openai"

export type ModelVisibility = Record<ModelID, boolean>

interface LegacyTransportVisibility {
  openrouter: boolean
  codex_cli: boolean
  opencode_cli: boolean
  claude_cli: boolean
  metacentrum_openai: boolean
}

interface LegacyResearchUIPreferences {
  transport_visibility: LegacyTransportVisibility
  default_models: {
    prover: ModelConfig
    verifier: ModelConfig
    summarizer: ModelConfig
  }
}

export interface ResearchUIPreferences {
  model_visibility: ModelVisibility
  default_models: {
    prover: ModelConfig
    verifier: ModelConfig
    summarizer: ModelConfig
  }
}

const STORAGE_KEY_V2 = "bolzano:research-ui-preferences:v2"
const STORAGE_KEY_V1 = "bolzano:research-ui-preferences:v1"

const HARD_DEFAULTS: ResearchUIPreferences["default_models"] = {
  prover: {
    id: "gpt-5.2",
    config: { reasoning_effort: "xhigh", web_search: false },
    role: "prover",
  },
  verifier: {
    id: "gpt-5.2",
    config: { reasoning_effort: "high", web_search: false },
    role: "verifier",
  },
  summarizer: {
    id: "gpt-5.3-codex",
    config: { reasoning_effort: "low", web_search: false },
    role: "summarizer",
  },
}

export function load_research_ui_preferences(): ResearchUIPreferences {
  const v2_raw = read_storage<Partial<ResearchUIPreferences>>(STORAGE_KEY_V2)
  if (v2_raw) return normalize_research_ui_preferences(v2_raw)

  // Best-effort migration from legacy transport-level settings.
  const v1_raw = read_storage<Partial<LegacyResearchUIPreferences>>(STORAGE_KEY_V1)
  if (v1_raw) {
    const migrated = normalize_research_ui_preferences({
      model_visibility: map_legacy_transport_visibility_to_models(v1_raw.transport_visibility),
      default_models: v1_raw.default_models,
    })
    save_research_ui_preferences(migrated)
    return migrated
  }

  return normalize_research_ui_preferences(null)
}

export function normalize_research_ui_preferences(raw: Partial<ResearchUIPreferences> | null | undefined): ResearchUIPreferences {
  const visibility = normalize_model_visibility(raw?.model_visibility)

  return {
    model_visibility: visibility,
    default_models: {
      prover: sanitize_model_for_role(raw?.default_models?.prover, "prover", visibility),
      verifier: sanitize_model_for_role(raw?.default_models?.verifier, "verifier", visibility),
      summarizer: sanitize_model_for_role(raw?.default_models?.summarizer, "summarizer", visibility),
    },
  }
}

export function save_research_ui_preferences(preferences: ResearchUIPreferences) {
  if (typeof localStorage === "undefined") return
  localStorage.setItem(STORAGE_KEY_V2, JSON.stringify(normalize_research_ui_preferences(preferences)))
}

export function get_available_model_ids_for_role(role: "prover" | "verifier" | "summarizer", visibility: ModelVisibility): ModelID[] {
  type ModelEntry = {
    id: ModelID
    structured_output: boolean
  }

  const ids: ModelID[] = []
  for (const provider_models of Object.values(models) as Array<readonly ModelEntry[]>) {
    for (const model of provider_models) {
      if (visibility[model.id] === false) continue
      if (role !== "prover" && !model.structured_output) continue
      ids.push(model.id)
    }
  }
  return ids
}

function all_model_ids() {
  const ids: ModelID[] = []
  for (const provider_models of Object.values(models) as Array<Array<{ id: ModelID }>>) {
    for (const model of provider_models) ids.push(model.id)
  }
  return ids
}

function fallback_model_id() {
  const ids = all_model_ids()
  return (ids.includes("gpt-5.2") ? "gpt-5.2" : ids[0]) as ModelID
}

function normalize_model_visibility(input: unknown): ModelVisibility {
  const candidate = (input && typeof input === "object" ? input : {}) as Partial<Record<ModelID, unknown>>
  const normalized = {} as ModelVisibility

  for (const id of all_model_ids()) {
    normalized[id] = candidate[id] !== false
  }

  const has_any_enabled = Object.values(normalized).some(Boolean)
  if (!has_any_enabled) {
    normalized[fallback_model_id()] = true
  }

  return normalized
}

function map_legacy_transport_visibility_to_models(input: unknown): ModelVisibility {
  const candidate = (input && typeof input === "object" ? input : {}) as Partial<LegacyTransportVisibility>
  const defaults: LegacyTransportVisibility = {
    openrouter: candidate.openrouter !== false,
    codex_cli: candidate.codex_cli !== false,
    opencode_cli: candidate.opencode_cli !== false,
    claude_cli: candidate.claude_cli !== false,
    metacentrum_openai: candidate.metacentrum_openai !== false,
  }

  const visibility = {} as ModelVisibility
  for (const provider_models of Object.values(models) as Array<Array<{ id: ModelID, transport?: Transport }>>) {
    for (const model of provider_models) {
      const transport = model.transport ?? "openrouter"
      visibility[model.id] = defaults[transport]
    }
  }

  return normalize_model_visibility(visibility)
}

function sanitize_model_for_role(
  input: unknown,
  role: "prover" | "verifier" | "summarizer",
  visibility: ModelVisibility,
): ModelConfig {
  const candidate = input as Partial<ModelConfig> | null | undefined
  const available_ids = get_available_model_ids_for_role(role, visibility)

  if (candidate && typeof candidate === "object" && typeof candidate.id === "string") {
    const model_info = get_model_by_id(candidate.id as ModelID)
    if (model_info && available_ids.includes(model_info.id)) {
      const normalized_reasoning = normalize_reasoning(model_info.config.reasoning, candidate.config?.reasoning_effort)
      return {
        id: model_info.id,
        role,
        config: {
          reasoning_effort: normalized_reasoning,
          web_search: false,
        },
      }
    }
  }

  const hard_default = HARD_DEFAULTS[role]
  if (available_ids.includes(hard_default.id)) {
    return {
      ...hard_default,
      role,
      config: {
        ...hard_default.config,
        reasoning_effort: normalize_reasoning(get_model_by_id(hard_default.id)!.config.reasoning, hard_default.config.reasoning_effort),
      },
    }
  }

  if (available_ids.length === 0) {
    const fallback = HARD_DEFAULTS[role]
    return {
      ...fallback,
      role,
      config: {
        ...fallback.config,
        reasoning_effort: normalize_reasoning(get_model_by_id(fallback.id)!.config.reasoning, fallback.config.reasoning_effort),
      },
    }
  }

  const first = get_model_by_id(available_ids[0])!
  return {
    id: first.id,
    role,
    config: {
      reasoning_effort: default_reasoning(first.config.reasoning),
      web_search: false,
    },
  }
}

function default_reasoning(config: ReasoningConfig): ReasoningEffortValue {
  if (config === null) return null
  if (config === "toggle") return true
  if (config.includes("high")) return "high"
  return config[config.length - 1]
}

function normalize_reasoning(
  config: ReasoningConfig,
  value: unknown,
): ReasoningEffortValue {
  if (config === null) return null
  if (config === "toggle") {
    if (typeof value === "boolean") return value
    return true
  }
  if (typeof value === "string") {
    const lowered = value.toLowerCase() as ReasoningEffort
    if (config.includes(lowered)) return lowered
  }
  return default_reasoning(config)
}

function read_storage<T>(key: string): T | null {
  if (typeof localStorage === "undefined") return null
  const raw = localStorage.getItem(key)
  if (!raw) return null

  try {
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== "object") return null
    return parsed as T
  } catch {
    return null
  }
}
