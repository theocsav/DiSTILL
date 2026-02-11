import { useRouter } from "next/router";
import { useEffect, useState } from "react";
import Link from "next/link";

import { PublicRunProgress, fetchPublicRunProgress } from "../../lib/api";

export default function PublicProgressPage() {
  const router = useRouter();
  const { token } = router.query;
  const [progress, setProgress] = useState<PublicRunProgress | null>(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!token || typeof token !== "string") {
      return;
    }
    setLoading(true);
    fetchPublicRunProgress(token)
      .then((data) => {
        setProgress(data);
        setStatus("");
      })
      .catch((error) => {
        setStatus(error instanceof Error ? error.message : String(error));
        setProgress(null);
      })
      .finally(() => setLoading(false));
  }, [token]);

  return (
    <main className="page">
      <section className="panel">
        <h1>Run Progress</h1>
        <p className="muted">Share this page to track status without signing in.</p>
        <div className="inline-actions">
          <button
            type="button"
            onClick={() => {
              if (typeof token !== "string") {
                return;
              }
              setLoading(true);
              fetchPublicRunProgress(token)
                .then((data) => {
                  setProgress(data);
                  setStatus("");
                })
                .catch((error) => {
                  setStatus(error instanceof Error ? error.message : String(error));
                  setProgress(null);
                })
                .finally(() => setLoading(false));
            }}
            disabled={loading || typeof token !== "string"}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </button>
          <Link className="link" href="/">
            Back to home
          </Link>
        </div>
      </section>

      {status ? (
        <section className="status">
          <p>{status}</p>
        </section>
      ) : null}

      {progress ? (
        <section className="panel">
          <h2>{progress.run_name}</h2>
          <div className="run-grid">
            <div className="card stack tight">
              <strong>Status</strong>
              <div>{progress.status}</div>
            </div>
            <div className="card stack tight">
              <strong>Stage</strong>
              <div>{progress.stage || "-"}</div>
            </div>
            <div className="card stack tight">
              <strong>SLURM</strong>
              <div>{progress.slurm_state || "-"}</div>
            </div>
            <div className="card stack tight">
              <strong>Job ID</strong>
              <div>{progress.job_id || "-"}</div>
            </div>
          </div>
          <div className="card stack inset">
            <strong>Timeline</strong>
            <div>Submitted: {progress.submitted_at || "-"}</div>
            <div>Started: {progress.started_at || "-"}</div>
            <div>Finished: {progress.finished_at || "-"}</div>
            <div>Updated: {progress.updated_at || "-"}</div>
            <div>Message: {progress.message || "-"}</div>
          </div>
        </section>
      ) : null}
    </main>
  );
}
