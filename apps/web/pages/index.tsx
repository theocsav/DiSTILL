import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createShareLink,
  createRun,
  dryRun,
  fetchDatasets,
  fetchMe,
  fetchPresets,
  fetchRuns,
  logout,
  preflightRun,
  rerunRun,
  Dataset,
  Preset,
  Run,
} from "../lib/api";

const DEFAULT_MIN_STAGES = ["cell2loc_nmf"];
const DEFAULT_PAPER_STAGES = ["cell2loc_nmf", "post_nmf", "rcausal_mgm", "mlp", "report"];
const RESOURCE_TIME_OPTIONS = ["1h", "2h", "6h", "12h", "24h", "96h"];
const RESOURCE_CPU_OPTIONS = [4, 8, 16, 32];
const RESOURCE_MEM_OPTIONS = ["16gb", "32gb", "64gb", "128gb", "400gb"];
const PIPELINE_COMMIT = process.env.NEXT_PUBLIC_PIPELINE_COMMIT || "not set";
const WEB_BASE = process.env.NEXT_PUBLIC_WEB_BASE;

type PreflightResult = {
  ok: boolean;
  errors: string[];
  warnings: string[];
  checks: Record<string, unknown>;
};

type DryRunResult = {
  ok: boolean;
  errors: string[];
  warnings: string[];
  checks: Record<string, unknown>;
  run_dir?: string | null;
  output_dir?: string | null;
  config_path?: string | null;
  resolved_config_path?: string | null;
  pipeline_stdout?: string | null;
  pipeline_stderr?: string | null;
};

type RunType = "minimal" | "paper";

type SlurmDefaults = {
  time?: string;
  mem?: string;
  cpus_per_task?: number;
  account?: string;
  partition?: string;
  qos?: string;
  mail_user?: string;
  mail_type?: string;
  conda_env?: string;
  enabled?: boolean;
};

function formatCheck(value: unknown) {
  if (value === true) {
    return { label: "ok", tone: "ok" as const };
  }
  if (value === false) {
    return { label: "fail", tone: "bad" as const };
  }
  if (value === "skipped") {
    return { label: "skipped", tone: "muted" as const };
  }
  return { label: "unknown", tone: "muted" as const };
}

function buildRerunName(baseName: string) {
  const now = new Date();
  const pad2 = (value: number) => value.toString().padStart(2, "0");
  const stamp = `${now.getFullYear()}${pad2(now.getMonth() + 1)}${pad2(now.getDate())}-${pad2(
    now.getHours()
  )}${pad2(now.getMinutes())}${pad2(now.getSeconds())}`;
  const safeBase = baseName.trim() || "run";
  return `${safeBase}-rerun-${stamp}`;
}

export default function Home() {
  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>("");
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [runType, setRunType] = useState<RunType>("paper");
  const [runName, setRunName] = useState("");
  const [mode, setMode] = useState("fixed_k");
  const [nComponents, setNComponents] = useState(4);
  const [kMin, setKMin] = useState(2);
  const [kMax, setKMax] = useState(20);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [referencePath, setReferencePath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [refModelDir, setRefModelDir] = useState("");
  const [slurmTime, setSlurmTime] = useState("96h");
  const [slurmCpus, setSlurmCpus] = useState(8);
  const [slurmMem, setSlurmMem] = useState("400gb");
  const [slurmAccount, setSlurmAccount] = useState("");
  const [slurmPartition, setSlurmPartition] = useState("");
  const [slurmQos, setSlurmQos] = useState("");
  const [slurmMailUser, setSlurmMailUser] = useState("");
  const [slurmCondaEnv, setSlurmCondaEnv] = useState("");
  const [submit, setSubmit] = useState(true);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [rerunLoadingId, setRerunLoadingId] = useState<number | null>(null);
  const [shareLoadingId, setShareLoadingId] = useState<number | null>(null);
  const [preflightLoading, setPreflightLoading] = useState(false);
  const [preflightResult, setPreflightResult] = useState<PreflightResult | null>(null);
  const [dryRunLoading, setDryRunLoading] = useState(false);
  const [dryRunResult, setDryRunResult] = useState<DryRunResult | null>(null);
  const [autoRefreshRuns, setAutoRefreshRuns] = useState(true);
  const [step, setStep] = useState(1);

  const selectedPreset = useMemo(
    () => presets.find((preset) => preset.id === selectedPresetId) || null,
    [presets, selectedPresetId]
  );
  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId) || null,
    [datasets, selectedDatasetId]
  );

  const stages = useMemo(() => {
    if (runType === "minimal") {
      return DEFAULT_MIN_STAGES;
    }
    if (selectedPreset?.stages && selectedPreset.stages.length > 0) {
      return selectedPreset.stages;
    }
    return DEFAULT_PAPER_STAGES;
  }, [runType, selectedPreset]);

  const refreshRuns = useCallback(async () => {
    try {
      const data = await fetchRuns();
      setRuns(data);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    fetchMe()
      .then((data) => setCurrentUser(data.username))
      .catch(() => setCurrentUser(null));
  }, []);

  useEffect(() => {
    if (!currentUser) {
      return;
    }
    refreshRuns();
    fetchPresets()
      .then((data) => setPresets(data))
      .catch((error) => setStatus(error instanceof Error ? error.message : String(error)));
  }, [currentUser, refreshRuns]);

  useEffect(() => {
    if (!currentUser || !selectedPresetId) {
      setDatasets([]);
      setSelectedDatasetId("");
      return;
    }
    fetchDatasets({ preset_id: selectedPresetId })
      .then((data) => setDatasets(data))
      .catch((error) => setStatus(error instanceof Error ? error.message : String(error)));
  }, [currentUser, selectedPresetId]);

  useEffect(() => {
    if (!selectedPreset) {
      return;
    }
    setStatus("");
    setPreflightResult(null);
    setDryRunResult(null);
    setRunName(selectedPreset.run_name || selectedPreset.id || "");

    const defaults = selectedPreset.default_params || {};
    setMode((defaults.mode as string) || "fixed_k");
    setNComponents(Number(defaults.n_components ?? 4));
    setKMin(Number(defaults.k_min ?? 2));
    setKMax(Number(defaults.k_max ?? 20));

    const resources = selectedPreset.default_resources || {};
    setSlurmTime((resources.time as string) || "96h");
    setSlurmCpus(Number(resources.cpus_per_task ?? 8));
    setSlurmMem((resources.mem as string) || "400gb");

    const slurm = selectedPreset.slurm || {};
    setSlurmAccount((slurm.account as string) || "");
    setSlurmPartition((slurm.partition as string) || "");
    setSlurmQos((slurm.qos as string) || "");
    setSlurmMailUser((slurm.mail_user as string) || "");
    setSlurmCondaEnv((slurm.conda_env as string) || "");

    setReferencePath((selectedPreset.reference_h5ad_path as string) || "");
    setOutputDir((selectedPreset.output_dir as string) || "");
    setRefModelDir((selectedPreset.ref_model_dir as string) || "");
  }, [selectedPreset]);

  useEffect(() => {
    setPreflightResult(null);
    setDryRunResult(null);
  }, [selectedDatasetId, mode, nComponents, kMin, kMax, runType]);

  useEffect(() => {
    if (!currentUser || !autoRefreshRuns) {
      return;
    }
    const timer = window.setInterval(() => {
      refreshRuns();
    }, 15000);
    return () => window.clearInterval(timer);
  }, [currentUser, autoRefreshRuns, refreshRuns]);


  async function handleLogout() {
    setStatus("");
    try {
      await logout();
      setCurrentUser(null);
      setRuns([]);
      setPresets([]);
      setDatasets([]);
      setSelectedPresetId("");
      setSelectedDatasetId("");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  function buildSlurmConfig(): SlurmDefaults {
    const base = { ...(selectedPreset?.slurm || {}) } as SlurmDefaults;
    return {
      ...base,
      enabled: true,
      time: slurmTime,
      mem: slurmMem,
      cpus_per_task: slurmCpus,
      account: slurmAccount || undefined,
      partition: slurmPartition || undefined,
      qos: slurmQos || undefined,
      mail_user: slurmMailUser || undefined,
      mail_type: base.mail_type || "ALL",
      conda_env: slurmCondaEnv || base.conda_env,
    };
  }

  function buildConfig() {
    const config: Record<string, unknown> = {
      run_name: runName,
      mode,
      stages,
    };

    if (selectedDataset?.id) {
      config.dataset_id = selectedDataset.id;
    }

    if (mode === "fixed_k") {
      config.n_components = nComponents;
    } else {
      config.k_min = kMin;
      config.k_max = kMax;
    }

    const resolvedReference = referencePath || (selectedPreset?.reference_h5ad_path as string);
    if (resolvedReference) {
      config.reference_h5ad_path = resolvedReference;
    }

    if (outputDir) {
      config.output_dir = outputDir;
    }
    if (refModelDir) {
      config.ref_model_dir = refModelDir;
    }

    if (selectedPreset?.template_path) {
      config.template_path = selectedPreset.template_path;
    }
    if (selectedPreset?.post_nmf_notebook_path) {
      config.post_nmf_notebook_path = selectedPreset.post_nmf_notebook_path;
    }
    if (selectedPreset?.post_nmf_mode) {
      config.post_nmf_mode = selectedPreset.post_nmf_mode;
    }
    if (selectedPreset?.rcausal_notebook_path) {
      config.rcausal_notebook_path = selectedPreset.rcausal_notebook_path;
    }
    if (selectedPreset?.rcausal_script_path) {
      config.rcausal_script_path = selectedPreset.rcausal_script_path;
    }
    if (selectedPreset?.rcausal_mode) {
      config.rcausal_mode = selectedPreset.rcausal_mode;
    }
    if (selectedPreset?.rcausal_parameters) {
      config.rcausal_parameters = selectedPreset.rcausal_parameters;
    }
    if (selectedPreset?.rcausal_args) {
      config.rcausal_args = selectedPreset.rcausal_args;
    }
    if (selectedPreset?.rcausal_output_dir) {
      config.rcausal_output_dir = selectedPreset.rcausal_output_dir;
    }
    if (selectedPreset?.rcausal_h5ad_path) {
      config.rcausal_h5ad_path = selectedPreset.rcausal_h5ad_path;
    }
    if (selectedPreset?.rcausal_niche_h5ad_path) {
      config.rcausal_niche_h5ad_path = selectedPreset.rcausal_niche_h5ad_path;
    }
    if (selectedPreset?.rcausal_neighborhood_h5ad_path) {
      config.rcausal_neighborhood_h5ad_path = selectedPreset.rcausal_neighborhood_h5ad_path;
    }
    if (selectedPreset?.mlp_script_path) {
      config.mlp_script_path = selectedPreset.mlp_script_path;
    }
    if (selectedPreset?.preflight_slurm) {
      config.preflight_slurm = selectedPreset.preflight_slurm;
    }

    config.slurm = buildSlurmConfig();
    config.report_title = `NicheRunner ${runName}`;

    return config;
  }

  async function handlePreflight() {
    if (!selectedPreset?.path) {
      setStatus("Select a preset before running preflight.");
      return;
    }
    setStatus("");
    setPreflightLoading(true);
    try {
      const config = buildConfig();
      const result = await preflightRun({
        preset_path: selectedPreset.path,
        config,
        check_paths: true,
      });
      setPreflightResult(result);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setPreflightLoading(false);
    }
  }

  async function handleDryRun() {
    if (!selectedPreset?.path) {
      setStatus("Select a preset before running a dry run.");
      return;
    }
    setStatus("");
    setDryRunLoading(true);
    try {
      const config = buildConfig();
      const result = await dryRun({
        run_name: runName,
        preset_path: selectedPreset.path,
        config,
        check_paths: true,
        emit_sbatch: true,
      });
      setDryRunResult(result);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setDryRunLoading(false);
    }
  }

  async function handleCreate() {
    if (!selectedPreset?.path) {
      setStatus("Select a preset before creating a run.");
      return;
    }
    setLoading(true);
    setStatus("");
    try {
      const config = buildConfig();
      const payload = { run_name: runName, preset_path: selectedPreset.path, config, submit };
      const run = await createRun(payload);
      setStatus(`Created run ${run.id} (${run.status})`);
      await refreshRuns();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }

  async function handleRerun(run: Run) {
    if (!currentUser) {
      setStatus("Sign in to rerun a job.");
      return;
    }
    const proposed = buildRerunName(run.run_name || "run");
    const name = window.prompt("Rerun name", proposed);
    if (!name || !name.trim()) {
      return;
    }
    setStatus("");
    setRerunLoadingId(run.id);
    try {
      const rerun = await rerunRun(run.id, { run_name: name.trim(), submit });
      setStatus(`Rerun created ${rerun.id} (${rerun.status})`);
      await refreshRuns();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setRerunLoadingId(null);
    }
  }

  async function handleShare(run: Run) {
    if (!currentUser) {
      setStatus("Sign in to create a share link.");
      return;
    }
    setStatus("");
    setShareLoadingId(run.id);
    try {
      const link = await createShareLink(run.id);
      const fallbackOrigin = typeof window !== "undefined" ? window.location.origin : "";
      const base = (WEB_BASE || fallbackOrigin).replace(/\/$/, "");
      const progressUrl = `${base}/progress/${link.token}`;
      if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(progressUrl);
      }
      setStatus(`Progress link copied: ${progressUrl}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setShareLoadingId(null);
    }
  }

  const datasetValidated = Boolean(preflightResult?.ok);
  const presetVersion = selectedPreset?.version || "not set";
  const sessionInitial = (currentUser?.[0] || "G").toUpperCase();

  return (
    <main className="page">
      <details className="session-chip">
        <summary>
          <span className="session-avatar">{sessionInitial}</span>
          <span>{currentUser ? currentUser : "Guest"}</span>
        </summary>
        <div className="session-menu">
          {currentUser ? (
            <>
              <div className="muted">Signed in as {currentUser}</div>
              <Link className="link" href="/users/create">
                Create User
              </Link>
              <button type="button" className="ghost" onClick={handleLogout}>
                Sign Out
              </button>
            </>
          ) : (
            <>
              <div className="muted">Sign in to create runs and preflight datasets.</div>
              <Link className="button-link" href="/login">
                Sign In
              </Link>
            </>
          )}
        </div>
      </details>
      <header className="hero">
        <div className="hero-copy">
          <h1>DiSTILL</h1>
          <p>Disease Diagnosis from Spatial Transcriptomics via Interpretable Latent Learning</p>
          <div className="inline-actions">
            <Link className="button-link" href="/datasets/upload">
              Public Dataset Upload
            </Link>
          </div>
        </div>
      </header>

      {status ? (
        <section className="status">
          <p>{status}</p>
        </section>
      ) : null}

      <section className="wizard">
        <div className="stepper">
          {["Setup", "Dataset", "Parameters", "Compute"].map((label, index) => {
            const stepIndex = index + 1;
            return (
              <button
                key={label}
                className={`step ${step === stepIndex ? "active" : ""}`}
                onClick={() => setStep(stepIndex)}
              >
                <span className="step-index">{stepIndex}</span>
                <span>{label}</span>
              </button>
            );
          })}
        </div>

        {step === 1 ? (
          <div className="panel step-panel">
            <h2>Setup</h2>
            <div className="row">
              <div>
                <label>Preset</label>
                <select
                  value={selectedPresetId}
                  onChange={(event) => setSelectedPresetId(event.target.value)}
                >
                  <option value="">Select a preset</option>
                  {presets.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.label || preset.id}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>Run name</label>
                <input value={runName} onChange={(event) => setRunName(event.target.value)} />
              </div>
            </div>
            <div className="row">
              <div>
                <label>Run type</label>
                <div className="segmented">
                  <button
                    type="button"
                    className={runType === "minimal" ? "active" : ""}
                    onClick={() => setRunType("minimal")}
                  >
                    Minimal
                  </button>
                  <button
                    type="button"
                    className={runType === "paper" ? "active" : ""}
                    onClick={() => setRunType("paper")}
                  >
                    Paper-ready
                  </button>
                </div>
              </div>
              <div>
                <label>Stages</label>
                <div className="tag-list">
                  {stages.map((stage) => (
                    <span key={stage} className="tag">
                      {stage}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : null}

        {step === 2 ? (
          <div className="panel step-panel">
            <h2>Dataset Selection</h2>
            <div className="row">
              <div>
                <label>Dataset</label>
                <select
                  value={selectedDatasetId}
                  onChange={(event) => setSelectedDatasetId(event.target.value)}
                >
                  <option value="">Select a dataset</option>
                  {datasets.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.label || dataset.id}
                    </option>
                  ))}
                </select>
                {datasetValidated ? (
                  <p className="pill ok">Dataset validated</p>
                ) : (
                  <p className="pill muted">Preflight pending</p>
                )}
              </div>
              <div className="card stack tight">
                <strong>Dataset context</strong>
                <div>ID: {selectedDataset?.id || "-"}</div>
                <div>Organ: {selectedDataset?.organ || "-"}</div>
                <div>Platform: {selectedDataset?.platform || "-"}</div>
                <div>Notes: {selectedDataset?.notes || "-"}</div>
              </div>
            </div>
            <div className="card stack tight inset">
              <strong>Resolved paths</strong>
              <div>Spatial H5AD: {selectedDataset?.staged_path || "-"}</div>
              <div>Cell metadata: {selectedDataset?.cell_metadata_path || "-"}</div>
            </div>
          </div>
        ) : null}

        {step === 3 ? (
          <div className="panel step-panel">
            <h2>Analysis Parameters</h2>
            <div className="row">
              <div>
                <label>Mode</label>
                <select value={mode} onChange={(event) => setMode(event.target.value)}>
                  <option value="fixed_k">Fixed K</option>
                  <option value="elbow_k">Elbow K</option>
                </select>
              </div>
              {mode === "fixed_k" ? (
                <div>
                  <label>NMF K</label>
                  <input
                    type="number"
                    min={2}
                    value={nComponents}
                    onChange={(event) => setNComponents(Number(event.target.value))}
                  />
                </div>
              ) : (
                <div className="row">
                  <div>
                    <label>K min</label>
                    <input
                      type="number"
                      min={2}
                      value={kMin}
                      onChange={(event) => setKMin(Number(event.target.value))}
                    />
                  </div>
                  <div>
                    <label>K max</label>
                    <input
                      type="number"
                      min={kMin}
                      value={kMax}
                      onChange={(event) => setKMax(Number(event.target.value))}
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="row">
              <div>
                <label>Advanced inputs</label>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => setShowAdvanced((prev) => !prev)}
                >
                  {showAdvanced ? "Hide" : "Show"} advanced paths
                </button>
              </div>
            </div>

            {showAdvanced ? (
              <div className="card inset">
                <div className="row">
                  <div>
                    <label>Reference h5ad path</label>
                    <input
                      value={referencePath}
                      onChange={(event) => setReferencePath(event.target.value)}
                    />
                  </div>
                  <div>
                    <label>Output dir</label>
                    <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
                  </div>
                  <div>
                    <label>Reference model dir</label>
                    <input
                      value={refModelDir}
                      onChange={(event) => setRefModelDir(event.target.value)}
                    />
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        {step === 4 ? (
          <div className="panel step-panel">
            <h2>Compute & Reproducibility</h2>
            <div className="row">
              <div>
                <label>Time</label>
                <select value={slurmTime} onChange={(event) => setSlurmTime(event.target.value)}>
                  {RESOURCE_TIME_OPTIONS.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>CPUs</label>
                <select value={slurmCpus} onChange={(event) => setSlurmCpus(Number(event.target.value))}>
                  {RESOURCE_CPU_OPTIONS.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label>Memory</label>
                <select value={slurmMem} onChange={(event) => setSlurmMem(event.target.value)}>
                  {RESOURCE_MEM_OPTIONS.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="row">
              <div>
                <label>SLURM account</label>
                <input
                  value={slurmAccount}
                  onChange={(event) => setSlurmAccount(event.target.value)}
                />
              </div>
              <div>
                <label>SLURM partition</label>
                <input
                  value={slurmPartition}
                  onChange={(event) => setSlurmPartition(event.target.value)}
                />
              </div>
              <div>
                <label>SLURM QoS</label>
                <input value={slurmQos} onChange={(event) => setSlurmQos(event.target.value)} />
              </div>
            </div>
            <div className="row">
              <div>
                <label>Mail user</label>
                <input
                  value={slurmMailUser}
                  onChange={(event) => setSlurmMailUser(event.target.value)}
                />
              </div>
              <div>
                <label>Conda env</label>
                <input
                  value={slurmCondaEnv}
                  onChange={(event) => setSlurmCondaEnv(event.target.value)}
                />
              </div>
              <div>
                <label>Submit to SLURM</label>
                <div className="checkbox">
                  <input
                    type="checkbox"
                    checked={submit}
                    onChange={(event) => setSubmit(event.target.checked)}
                  />
                  <span>{submit ? "Submit" : "Prepare only"}</span>
                </div>
              </div>
            </div>

            <div className="grid-two inset">
              <div className="card stack">
                <h3>Reproducibility</h3>
                <div>Pipeline commit: {PIPELINE_COMMIT}</div>
                <div>Preset version: {presetVersion}</div>
                <div>Preset: {selectedPreset?.label || selectedPreset?.id || "-"}</div>
                <div>Dataset: {selectedDataset?.label || selectedDataset?.id || "-"}</div>
                <div>Reference h5ad: {referencePath || selectedPreset?.reference_h5ad_path || "-"}</div>
                <div>Output dir: {outputDir || selectedPreset?.output_dir || "-"}</div>
              </div>

              <div className="card stack">
                <h3>Preflight & Dry Run</h3>
                <div className="inline-actions">
                  <button onClick={handlePreflight} disabled={!currentUser || preflightLoading}>
                    {preflightLoading ? "Running preflight..." : "Run preflight"}
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    onClick={handleDryRun}
                    disabled={!currentUser || dryRunLoading}
                  >
                    {dryRunLoading ? "Generating..." : "Dry run"}
                  </button>
                </div>
                <p className="muted">
                  Permissions reflect API host access; enable SLURM fallback to validate compute-user
                  readability when needed.
                </p>

                {preflightResult ? (
                  <div className="preflight">
                    <div className={`pill ${preflightResult.ok ? "ok" : "bad"}`}>
                      {preflightResult.ok ? "Preflight OK" : "Preflight issues"}
                    </div>
                    {preflightResult.errors.length > 0 ? (
                      <div>
                        <strong>Errors</strong>
                        <ul>
                          {preflightResult.errors.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {preflightResult.warnings.length > 0 ? (
                      <div>
                        <strong>Warnings</strong>
                        <ul>
                          {preflightResult.warnings.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}

                    {typeof preflightResult.checks === "object" ? (
                      <div className="preflight-grid">
                        {(["exists", "roots", "permissions"] as const).map((key) => {
                          const record = preflightResult.checks[key] as Record<string, unknown> | undefined;
                          if (!record) {
                            return null;
                          }
                          return (
                            <div key={key}>
                              <strong>{key}</strong>
                              {Object.entries(record).map(([pathKey, value]) => {
                                const meta = formatCheck(value);
                                return (
                                  <div key={pathKey} className={`check-row ${meta.tone}`}>
                                    <span>{pathKey}</span>
                                    <span>{meta.label}</span>
                                  </div>
                                );
                              })}
                            </div>
                          );
                        })}
                        {preflightResult.checks.join_keys ? (
                          <div>
                            <strong>join_keys</strong>
                            <pre>{JSON.stringify(preflightResult.checks.join_keys, null, 2)}</pre>
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <p className="muted">Run preflight to see validation results.</p>
                )}

                <div className="card inset">
                  <strong>Dry run output</strong>
                  {dryRunResult ? (
                    <div className="preflight">
                      <div className={`pill ${dryRunResult.ok ? "ok" : "bad"}`}>
                        {dryRunResult.ok ? "Dry run OK" : "Dry run issues"}
                      </div>
                      {dryRunResult.errors.length > 0 ? (
                        <div>
                          <strong>Errors</strong>
                          <ul>
                            {dryRunResult.errors.map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                      {dryRunResult.warnings.length > 0 ? (
                        <div>
                          <strong>Warnings</strong>
                          <ul>
                            {dryRunResult.warnings.map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                      {dryRunResult.run_dir ? <div>Run dir: {dryRunResult.run_dir}</div> : null}
                      {dryRunResult.output_dir ? <div>Output dir: {dryRunResult.output_dir}</div> : null}
                      {dryRunResult.config_path ? <div>Config: {dryRunResult.config_path}</div> : null}
                      {dryRunResult.resolved_config_path ? (
                        <div>Resolved config: {dryRunResult.resolved_config_path}</div>
                      ) : null}
                      {dryRunResult.pipeline_stdout ? (
                        <div>
                          <strong>Pipeline stdout</strong>
                          <pre>{dryRunResult.pipeline_stdout}</pre>
                        </div>
                      ) : null}
                      {dryRunResult.pipeline_stderr ? (
                        <div>
                          <strong>Pipeline stderr</strong>
                          <pre>{dryRunResult.pipeline_stderr}</pre>
                        </div>
                      ) : null}
                      {typeof dryRunResult.checks === "object" ? (
                        <div className="preflight-grid">
                          {(["exists", "roots", "permissions"] as const).map((key) => {
                            const record = dryRunResult.checks[key] as Record<string, unknown> | undefined;
                            if (!record) {
                              return null;
                            }
                            return (
                              <div key={key}>
                                <strong>{key}</strong>
                                {Object.entries(record).map(([pathKey, value]) => {
                                  const meta = formatCheck(value);
                                  return (
                                    <div key={pathKey} className={`check-row ${meta.tone}`}>
                                      <span>{pathKey}</span>
                                      <span>{meta.label}</span>
                                    </div>
                                  );
                                })}
                              </div>
                            );
                          })}
                          {dryRunResult.checks.join_keys ? (
                            <div>
                              <strong>join_keys</strong>
                              <pre>{JSON.stringify(dryRunResult.checks.join_keys, null, 2)}</pre>
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <p className="muted">Run a dry run to generate scripts without submitting.</p>
                  )}
                </div>
              </div>
            </div>

            <div className="inline-actions">
              <button onClick={handleCreate} disabled={loading || !currentUser}>
                {loading ? "Creating..." : "Create Run"}
              </button>
              <button type="button" className="ghost" onClick={refreshRuns} disabled={!currentUser}>
                Refresh runs
              </button>
            </div>
          </div>
        ) : null}

        <div className="wizard-actions">
          <button onClick={() => setStep((prev) => Math.max(prev - 1, 1))} disabled={step === 1}>
            Back
          </button>
          <button onClick={() => setStep((prev) => Math.min(prev + 1, 4))} disabled={step === 4}>
            Next
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Runs</h2>
          <div className="inline-actions">
            <div className="checkbox">
              <input
                type="checkbox"
                checked={autoRefreshRuns}
                onChange={(event) => setAutoRefreshRuns(event.target.checked)}
                disabled={!currentUser}
              />
              <span>Auto refresh</span>
            </div>
            <button onClick={refreshRuns} disabled={!currentUser}>
              Refresh
            </button>
          </div>
        </div>
        <div className="run-grid">
          {runs.map((run) => (
            <div key={run.id} className="card">
              <div className="run-header">
                <strong>{run.run_name}</strong>
                <span className="pill muted">{run.status}</span>
              </div>
              <div>Run ID: {run.id}</div>
              <div>Created: {new Date(run.created_at).toLocaleString()}</div>
              <div>Output: {run.output_dir || "-"}</div>
              <div className="inline-actions">
                <Link className="link" href={`/runs/${run.id}`}>
                  Logs
                </Link>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => handleShare(run)}
                  disabled={!currentUser || shareLoadingId === run.id}
                >
                  {shareLoadingId === run.id ? "Sharing..." : "Share Progress"}
                </button>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => handleRerun(run)}
                  disabled={!currentUser || rerunLoadingId === run.id}
                >
                  {rerunLoadingId === run.id ? "Rerunning..." : "Rerun"}
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      <footer className="site-footer">
        <p>
          Made by Vasco Hinostroza.
          <a href="https://github.com/theocsav">GitHub</a>
          <a href="https://www.linkedin.com/in/vasco-hinostroza/">LinkedIn</a>
          <a href="https://www.researchgate.net/profile/Vasco-Hinostroza-Fuentes">ResearchGate</a>
          <a href="mailto:vasco.hinostroza@ufl.edu">vasco.hinostroza@ufl.edu</a>
        </p>
      </footer>
    </main>
  );
}

