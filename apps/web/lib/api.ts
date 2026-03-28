export type Run = {
  id: number;
  run_name: string;
  status: string;
  stage?: string | null;
  run_dir?: string | null;
  output_dir?: string | null;
  config_path?: string | null;
  job_id?: string | null;
  slurm_state?: string | null;
  slurm_reason?: string | null;
  slurm_exit_code?: number | null;
  slurm_exit_signal?: number | null;
  slurm_elapsed?: string | null;
  submitted_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  message?: string | null;
  created_at: string;
  updated_at: string;
};

export type RunSummary = {
  generated_at: string;
  run_name: string;
  output_dir: string;
  stages: string[];
  report_title?: string;
  report_notes?: string;
  report_path: string;
  manifest_path: string;
  figures_count: number;
  tables_count: number;
};

export type RunCreatePayload = {
  run_name: string;
  preset_path?: string;
  config?: Record<string, unknown>;
  submit?: boolean;
  queue?: boolean;
};

export type RunRerunPayload = {
  run_name: string;
  submit?: boolean;
  queue?: boolean;
};

export type Preset = {
  id: string;
  label?: string;
  organ?: string;
  platform?: string;
  path?: string;
  stages?: string[];
  run_name?: string;
  cosmx_h5ad_path?: string;
  reference_h5ad_path?: string;
  output_dir?: string;
  ref_model_dir?: string;
  template_path?: string;
  post_nmf_notebook_path?: string;
  post_nmf_mode?: string;
  rcausal_notebook_path?: string;
  rcausal_script_path?: string;
  rcausal_mode?: string;
  rcausal_parameters?: Record<string, unknown>;
  rcausal_args?: string[];
  rcausal_output_dir?: string;
  rcausal_h5ad_path?: string;
  cosmx_with_nmf_path?: string;
  rcausal_niche_h5ad_path?: string;
  rcausal_neighborhood_h5ad_path?: string;
  mlp_script_path?: string;
  default_resources?: {
    time?: string;
    mem?: string;
    cpus_per_task?: number;
    qos?: string;
  };
  default_params?: {
    mode?: string;
    n_components?: number;
    k_min?: number;
    k_max?: number;
  };
  slurm?: {
    enabled?: boolean;
    job_name?: string;
    time?: string;
    mem?: string;
    cpus_per_task?: number;
    account?: string;
    partition?: string;
    qos?: string;
    mail_user?: string;
    mail_type?: string;
    conda_env?: string;
  };
  preflight_slurm?: Record<string, unknown>;
  version?: string;
};

export type Dataset = {
  id: string;
  label?: string;
  organ?: string;
  platform?: string;
  staged_path?: string;
  cosmx_with_nmf_path?: string;
  reference_h5ad_path?: string;
  cell_metadata_path?: string;
  recommended_preset?: string;
  schema_manifest?: Record<string, unknown>;
  metadata_columns?: string[];
  notes?: string;
  public?: boolean;
  uploaded_by?: string;
  uploaded_at?: string;
  source?: string;
  updated_by?: string;
  updated_at?: string;
  checksums?: Record<string, string | null>;
};

export type ShareRunLink = {
  run_id: number;
  token: string;
  url: string;
  expires_at: string;
};

export type PublicRunProgress = {
  id: number;
  run_name: string;
  status: string;
  stage?: string | null;
  job_id?: string | null;
  slurm_state?: string | null;
  slurm_reason?: string | null;
  slurm_exit_code?: number | null;
  slurm_exit_signal?: number | null;
  slurm_elapsed?: string | null;
  submitted_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  message?: string | null;
  created_at: string;
  updated_at: string;
};

export type CreateUserPayload = {
  username: string;
  password: string;
};

export type CreateUserResponse = {
  username: string;
  created_by: string;
  created_at: string;
};

export type PreflightPayload = {
  preset_path?: string;
  config?: Record<string, unknown>;
  check_paths?: boolean;
};

export type PreflightResponse = {
  ok: boolean;
  errors: string[];
  warnings: string[];
  checks: Record<string, unknown>;
};

export type DryRunPayload = {
  run_name: string;
  preset_path?: string;
  config?: Record<string, unknown>;
  check_paths?: boolean;
  emit_sbatch?: boolean;
};

export type DryRunResponse = {
  ok: boolean;
  errors: string[];
  warnings: string[];
  checks: Record<string, unknown>;
  run_dir?: string | null;
  output_dir?: string | null;
  config_path?: string | null;
  resolved_config_path?: string | null;
  resolved_config?: Record<string, unknown> | null;
  pipeline_stdout?: string | null;
  pipeline_stderr?: string | null;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const CSRF_COOKIE_NAME = "sptx_csrf";
const CSRF_HEADER_NAME = "X-CSRF-Token";
let csrfToken: string | null = null;

function readCookie(name: string) {
  if (typeof document === "undefined") {
    return null;
  }
  const match = document.cookie.split("; ").find((item) => item.startsWith(`${name}=`));
  if (!match) {
    return null;
  }
  return decodeURIComponent(match.split("=")[1] || "");
}

function getCsrfToken() {
  return csrfToken || readCookie(CSRF_COOKIE_NAME);
}

function setCsrfToken(token?: string) {
  if (token) {
    csrfToken = token;
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method || "GET").toUpperCase();
  const providedHeaders = (init?.headers || {}) as Record<string, string>;
  const isFormData = typeof FormData !== "undefined" && init?.body instanceof FormData;
  const headers: Record<string, string> = { ...providedHeaders };
  if (!isFormData && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    const token = getCsrfToken();
    if (token) {
      headers[CSRF_HEADER_NAME] = token;
    }
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function fetchRuns() {
  return apiFetch<Run[]>("/runs");
}

export async function fetchRun(runId: number) {
  return apiFetch<Run>(`/runs/${runId}`);
}

export async function fetchPresets(params?: { organ?: string; platform?: string }) {
  const query = new URLSearchParams();
  if (params?.organ) {
    query.set("organ", params.organ);
  }
  if (params?.platform) {
    query.set("platform", params.platform);
  }
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiFetch<Preset[]>(`/presets${suffix}`);
}

export async function fetchDatasets(params?: {
  organ?: string;
  platform?: string;
  preset_id?: string;
}) {
  const query = new URLSearchParams();
  if (params?.organ) {
    query.set("organ", params.organ);
  }
  if (params?.platform) {
    query.set("platform", params.platform);
  }
  if (params?.preset_id) {
    query.set("preset_id", params.preset_id);
  }
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiFetch<Dataset[]>(`/datasets${suffix}`);
}

export async function fetchPublicDatasets() {
  return apiFetch<Dataset[]>("/datasets/public");
}

export async function createRun(payload: RunCreatePayload) {
  return apiFetch<Run>("/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function rerunRun(runId: number, payload: RunRerunPayload) {
  return apiFetch<Run>(`/runs/${runId}/rerun`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function preflightRun(payload: PreflightPayload) {
  return apiFetch<PreflightResponse>("/runs/preflight", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function dryRun(payload: DryRunPayload) {
  return apiFetch<DryRunResponse>("/runs/dry-run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchArtifacts(runId: number, path = "") {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return apiFetch<{ items: { path: string; size: string }[] }>(`/runs/${runId}/artifacts${query}`);
}

export async function fetchRunSummary(runId: number) {
  return apiFetch<RunSummary>(`/runs/${runId}/summary`);
}

export async function fetchLogs(runId: number, path?: string) {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return apiFetch<{ path: string; content: string }>(`/runs/${runId}/logs${query}`);
}

export function artifactUrl(runId: number, path: string) {
  return `${API_BASE}/runs/${runId}/artifact?path=${encodeURIComponent(path)}`;
}

export async function cancelRun(runId: number) {
  return apiFetch<{ status: string }>(`/runs/${runId}/cancel`, { method: "POST" });
}

export async function createShareLink(runId: number, expiresHours?: number) {
  return apiFetch<ShareRunLink>(`/runs/${runId}/share`, {
    method: "POST",
    body: JSON.stringify({ expires_hours: expiresHours }),
  });
}

export async function fetchPublicRunProgress(token: string) {
  return apiFetch<PublicRunProgress>(`/public/runs/progress?token=${encodeURIComponent(token)}`);
}

export type DatasetUploadPayload = {
  dataset_id: string;
  label: string;
  organ: string;
  platform: string;
  notes?: string;
  recommended_preset?: string;
  staged_file: File;
  cell_metadata_file: File;
  reference_file?: File;
};

export async function uploadDataset(payload: DatasetUploadPayload) {
  const body = new FormData();
  body.append("dataset_id", payload.dataset_id);
  body.append("label", payload.label);
  body.append("organ", payload.organ);
  body.append("platform", payload.platform);
  if (payload.notes) {
    body.append("notes", payload.notes);
  }
  if (payload.recommended_preset) {
    body.append("recommended_preset", payload.recommended_preset);
  }
  body.append("staged_file", payload.staged_file);
  body.append("cell_metadata_file", payload.cell_metadata_file);
  if (payload.reference_file) {
    body.append("reference_file", payload.reference_file);
  }
  return apiFetch<{ ok: boolean; dataset: Dataset }>("/datasets/upload", {
    method: "POST",
    body,
  });
}

export type UploadInitPayload = {
  dataset_id: string;
  file_role: "staged" | "metadata" | "reference" | "nmf_artifact";
  file_name: string;
  total_size: number;
  content_type?: string;
  expected_sha256?: string;
};

export type UploadSession = {
  upload_id: string;
  dataset_id: string;
  file_role: string;
  file_name: string;
  total_size: number;
  received_bytes: number;
  completed: boolean;
  final_path?: string | null;
  created_at: string;
  updated_at: string;
};

export async function initUploadSession(payload: UploadInitPayload) {
  return apiFetch<UploadSession>("/uploads/init", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchUploadStatus(uploadId: string) {
  return apiFetch<UploadSession>(`/uploads/${encodeURIComponent(uploadId)}/status`);
}

export async function uploadChunk(uploadId: string, offset: number, chunk: Blob) {
  const headers: Record<string, string> = {};
  const token = getCsrfToken();
  if (token) {
    headers[CSRF_HEADER_NAME] = token;
  }
  const response = await fetch(
    `${API_BASE}/uploads/${encodeURIComponent(uploadId)}/chunk?offset=${offset}`,
    {
      method: "PUT",
      credentials: "include",
      headers,
      body: chunk,
    }
  );
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Chunk upload failed: ${response.status}`);
  }
  return (await response.json()) as UploadSession;
}

export async function completeUploadSession(uploadId: string) {
  return apiFetch<UploadSession>(`/uploads/${encodeURIComponent(uploadId)}/complete`, {
    method: "POST",
  });
}

export type FinalizeDatasetUploadPayload = {
  dataset_id: string;
  label: string;
  organ: string;
  platform: string;
  notes?: string;
  recommended_preset?: string;
  staged_upload_id?: string;
  cell_metadata_upload_id?: string;
  reference_upload_id?: string;
  nmf_artifact_upload_id?: string;
  public?: boolean;
};

export async function finalizeDatasetUpload(payload: FinalizeDatasetUploadPayload) {
  return apiFetch<{ ok: boolean; dataset: Dataset }>("/datasets/upload/finalize", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function updateDataset(datasetId: string, payload: { label?: string; notes?: string; public?: boolean }) {
  return apiFetch<{ ok: boolean; dataset: Dataset }>(`/datasets/${encodeURIComponent(datasetId)}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export async function deleteDataset(datasetId: string) {
  return apiFetch<{ ok: boolean; deleted_id: string; deleted_by: string }>(
    `/datasets/${encodeURIComponent(datasetId)}`,
    {
      method: "DELETE",
    }
  );
}

export async function login(username: string, password: string) {
  const data = await apiFetch<{ username: string; csrf_token?: string }>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  setCsrfToken(data.csrf_token);
  return data;
}

export async function logout() {
  const data = await apiFetch<{ status: string }>("/auth/logout", { method: "POST" });
  csrfToken = null;
  return data;
}

export async function fetchMe() {
  const data = await apiFetch<{ username: string; csrf_token?: string }>("/auth/me");
  setCsrfToken(data.csrf_token);
  return data;
}

export async function createUser(payload: CreateUserPayload) {
  return apiFetch<CreateUserResponse>("/auth/users", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
