import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/router";
import {
  artifactUrl,
  cancelRun,
  createShareLink,
  fetchArtifacts,
  fetchLogs,
  fetchMe,
  fetchRun,
  fetchRunSummary,
  login,
  logout,
  Run,
  RunSummary,
} from "../../lib/api";

const WEB_BASE = process.env.NEXT_PUBLIC_WEB_BASE;

export default function RunDetail() {
  const router = useRouter();
  const { id } = router.query;
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [runSummary, setRunSummary] = useState<RunSummary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [artifacts, setArtifacts] = useState<{ path: string; size: string }[]>([]);
  const [logs, setLogs] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [shareLoading, setShareLoading] = useState(false);

  useEffect(() => {
    fetchMe()
      .then((data) => setCurrentUser(data.username))
      .catch(() => setCurrentUser(null));
  }, []);

  const loadData = useCallback(async () => {
    if (!id) {
      return;
    }
    const runId = Number(id);
    try {
      const runData = await fetchRun(runId);
      setRun(runData);
      const [artifactResult, logResult, summaryResult] = await Promise.allSettled([
        fetchArtifacts(runId),
        fetchLogs(runId),
        fetchRunSummary(runId),
      ]);

      if (artifactResult.status === "fulfilled") {
        setArtifacts(artifactResult.value.items);
      } else {
        setArtifacts([]);
      }

      if (logResult.status === "fulfilled" && logResult.value.content.trim()) {
        setLogs(logResult.value.content);
      } else {
        setLogs(runData.message || "");
      }

      if (summaryResult.status === "fulfilled") {
        setRunSummary(summaryResult.value);
      } else {
        setRunSummary(null);
      }
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }, [id]);

  useEffect(() => {
    if (currentUser) {
      loadData();
    }
  }, [currentUser, id, loadData]);

  useEffect(() => {
    if (!currentUser || !id || !autoRefresh) {
      return;
    }
    const timer = window.setInterval(() => {
      loadData();
    }, 15000);
    return () => window.clearInterval(timer);
  }, [currentUser, id, autoRefresh, loadData]);

  async function handleLogin() {
    setStatus("");
    try {
      const data = await login(username, password);
      setCurrentUser(data.username);
      setPassword("");
      await loadData();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleLogout() {
    setStatus("");
    try {
      await logout();
      setCurrentUser(null);
      setRun(null);
      setRunSummary(null);
      setArtifacts([]);
      setLogs("");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleCancel() {
    if (!id) {
      return;
    }
    setStatus("");
    try {
      await cancelRun(Number(id));
      await loadData();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleRefreshSummary() {
    if (!id) {
      return;
    }
    setSummaryLoading(true);
    try {
      const summaryData = await fetchRunSummary(Number(id));
      setRunSummary(summaryData);
    } catch {
      setRunSummary(null);
    } finally {
      setSummaryLoading(false);
    }
  }

  async function handleShare() {
    if (!id) {
      return;
    }
    setStatus("");
    setShareLoading(true);
    try {
      const link = await createShareLink(Number(id));
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
      setShareLoading(false);
    }
  }

  return (
    <main>
      <h1>Run Detail</h1>
      <p>Run ID: {id}</p>
      <section>
        <h2>Auth</h2>
        <div className="row">
          <div>
            <label>Username</label>
            <input
              value={username}
              onChange={(event) => setUsername(event.target.value)}
            />
          </div>
          <div>
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </div>
        </div>
        <div style={{ marginTop: 12, display: "flex", gap: 12 }}>
          <button onClick={handleLogin}>Sign In</button>
          <button onClick={handleLogout} disabled={!currentUser}>
            Sign Out
          </button>
          {currentUser ? <span>Signed in as {currentUser}</span> : null}
        </div>
      </section>

      {status ? (
        <section>
          <p>{status}</p>
        </section>
      ) : null}

      {run ? (
        <section>
          <h2>Summary</h2>
          <div className="inline-actions">
            <button onClick={loadData} disabled={!currentUser}>
              Refresh
            </button>
            <div className="checkbox">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(event) => setAutoRefresh(event.target.checked)}
                disabled={!currentUser}
              />
              <span>Auto refresh</span>
            </div>
          </div>
          <div>Run name: {run.run_name}</div>
          <div>Status: {run.status}</div>
          <div>Output dir: {run.output_dir || "-"}</div>
          <div>Job ID: {run.job_id || "-"}</div>
          <div>SLURM state: {run.slurm_state || "-"}</div>
          <div>SLURM reason: {run.slurm_reason || "-"}</div>
          <div>SLURM exit: {run.slurm_exit_code ?? "-"}:{run.slurm_exit_signal ?? "-"}</div>
          <div>SLURM elapsed: {run.slurm_elapsed || "-"}</div>
          <div>Submitted: {run.submitted_at || "-"}</div>
          <div>Started: {run.started_at || "-"}</div>
          <div>Finished: {run.finished_at || "-"}</div>
          <div>Message: {run.message || "-"}</div>
          <button onClick={handleCancel} disabled={!currentUser} style={{ marginTop: 12 }}>
            Cancel Run
          </button>
          <button
            onClick={handleShare}
            disabled={!currentUser || shareLoading}
            style={{ marginTop: 12, marginLeft: 12 }}
          >
            {shareLoading ? "Sharing..." : "Share Progress"}
          </button>
        </section>
      ) : null}

      <section>
        <h2>Run Summary</h2>
        {runSummary ? (
          <button onClick={handleRefreshSummary} disabled={!id || summaryLoading}>
            {summaryLoading ? "Refreshing..." : "Refresh summary"}
          </button>
        ) : null}
        {runSummary ? (
          <>
            <div>Report title: {runSummary.report_title || "-"}</div>
            <div>Stages: {runSummary.stages.join(", ")}</div>
            <div>Figures: {runSummary.figures_count}</div>
            <div>Tables: {runSummary.tables_count}</div>
            {id ? (
              <a href={artifactUrl(Number(id), runSummary.report_path)} target="_blank" rel="noreferrer">
                Open report
              </a>
            ) : null}
          </>
        ) : (
          <div className="muted">Run summary not available yet.</div>
        )}
      </section>

      <section>
        <h2>Artifacts</h2>
        <div className="list">
          {artifacts.map((item) => (
            <div key={item.path} className="card">
              <div>{item.path}</div>
              <div>Size: {item.size} bytes</div>
              {id ? (
                <a href={artifactUrl(Number(id), item.path)} target="_blank" rel="noreferrer">
                  Download
                </a>
              ) : null}
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2>Latest Log Tail</h2>
        <pre style={{ whiteSpace: "pre-wrap" }}>{logs || run?.message || "No logs yet."}</pre>
      </section>
    </main>
  );
}
