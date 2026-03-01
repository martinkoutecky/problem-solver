import { get_model_by_id, models, type ModelConfig, type ModelID, type ReasoningConfig, type ReasoningEffort, type ReasoningEffortValue } from "@shared/types/research"

export type Transport = "openrouter" | "codex_cli" | "opencode_cli"

export interface TransportVisibility {
  openrouter: boolean
  codex_cli: boolean
  opencode_cli: boolean
}

export interface ResearchUIPreferences {
  transport_visibility: TransportVisibility
  default_models: {
    prover: ModelConfig
    verifier: ModelConfig
    summarizer: ModelConfig
  }
}

const STORAGE_KEY = "bolzano:research-ui-preferences:v1"

export const DEFAULT_TRANSPORT_VISIBILITY: TransportVisibility = {
  openrouter: true,
  codex_cli: true,
  opencode_cli: true,
}

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
  const raw = read_storage()
  return normalize_research_ui_preferences(raw)
}

export function normalize_research_ui_preferences(raw: Partial<ResearchUIPreferences> | null | undefined): ResearchUIPreferences {
  const visibility = normalize_transport_visibility(raw?.transport_visibility)

  return {
    transport_visibility: visibility,
    default_models: {
      prover: sanitize_model_for_role(raw?.default_models?.prover, "prover", visibility),
      verifier: sanitize_model_for_role(raw?.default_models?.verifier, "verifier", visibility),
      summarizer: sanitize_model_for_role(raw?.default_models?.summarizer, "summarizer", visibility),
    },
  }
}

export function save_research_ui_preferences(preferences: ResearchUIPreferences) {
  if (typeof localStorage === "undefined") return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(normalize_research_ui_preferences(preferences)))
}

export function get_available_model_ids_for_role(role: "prover" | "verifier" | "summarizer", visibility: TransportVisibility): ModelID[] {
  type ModelEntry = {
    id: ModelID
    transport?: Transport
    structured_output: boolean
  }

  const ids: ModelID[] = []
  for (const provider_models of Object.values(models) as Array<readonly ModelEntry[]>) {
    for (const model of provider_models) {
      if (!is_transport_visible(model.transport ?? "openrouter", visibility)) continue
      if (role !== "prover" && !model.structured_output) continue
      ids.push(model.id)
    }
  }
  return ids
}

function is_transport_visible(transport: Transport, visibility: TransportVisibility) {
  return visibility[transport]
}

function normalize_transport_visibility(input: unknown): TransportVisibility {
  const candidate = input as Partial<TransportVisibility> | null | undefined

  const normalized: TransportVisibility = {
    openrouter: candidate?.openrouter !== false,
    codex_cli: candidate?.codex_cli !== false,
    opencode_cli: candidate?.opencode_cli !== false,
  }

  if (!normalized.openrouter && !normalized.codex_cli && !normalized.opencode_cli) {
    return {
      ...normalized,
      codex_cli: true,
    }
  }

  return normalized
}

function sanitize_model_for_role(
  input: unknown,
  role: "prover" | "verifier" | "summarizer",
  visibility: TransportVisibility,
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

function read_storage(): Partial<ResearchUIPreferences> | null {
  if (typeof localStorage === "undefined") return null
  const raw = localStorage.getItem(STORAGE_KEY)
  if (!raw) return null

  try {
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== "object") return null
    return parsed as Partial<ResearchUIPreferences>
  } catch {
    return null
  }
}
