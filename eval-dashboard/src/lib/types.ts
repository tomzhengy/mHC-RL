// eval results JSON schema (from eval_compare.py)

export interface GpuInfo {
  gpu_name: string;
  gpu_vram_gb: number;
}

export interface EvalConfig {
  evals: string[];
  max_per_task: number;
  max_problems: number | null;
}

export interface ModelConfig {
  n_layer?: number | null;
  n_embd?: number | null;
  n_head?: number | null;
  mhc_enabled?: boolean;
  mhc_static?: boolean | null;
  mhc_num_streams?: number | null;
}

export interface CoreResults {
  results: Record<string, number>;
  centered_results: Record<string, number>;
  core_metric: number;
  wall_time: number;
}

export interface ChatResults {
  results: Record<string, number>;
  centered_results: Record<string, number>;
  chatcore_metric: number | null;
  wall_time: number;
}

export interface ModelResult {
  model_spec: string;
  model_name: string;
  param_count: number;
  config: ModelConfig;
  core?: CoreResults;
  chat?: ChatResults;
}

export interface EvalResults {
  timestamp: string;
  gpu_info: GpuInfo;
  eval_config: EvalConfig;
  models: ModelResult[];
}

// job system

export interface JobMeta {
  id: string;
  models: string[];       // model spec strings
  evals: string[];        // "core", "chat"
  gpu: string;
  hf_ckpt_repo: string | null;
  ckpt_model_tag: string | null;
  ckpt_source: string;
  max_per_task: number;
  max_problems: number;
  created_at: string;
}

export interface JobStatus {
  status: "pending" | "running" | "completed" | "failed";
  started_at?: string;
  completed_at?: string;
  result_id?: string;
  exit_code?: number;
  error?: string;
}

export interface Job {
  meta: JobMeta;
  status: JobStatus;
}

// models.json config

export interface ModelEntry {
  id: string;
  name: string;
  spec: string;
  description: string;
  hf_ckpt_repo: string | null;
  ckpt_model_tag: string | null;
  ckpt_source: string;
}

export interface ModelsConfig {
  models: ModelEntry[];
  defaults: {
    gpu: string;
    max_per_task: number;
    max_problems: number;
  };
}

// API response for results list

export interface ResultSummary {
  id: string;
  timestamp: string;
  gpu: string;
  evals: string[];
  models: {
    name: string;
    core_metric: number | null;
    chatcore_metric: number | null;
  }[];
}
