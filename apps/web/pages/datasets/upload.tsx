import { FormEvent, useEffect, useState } from "react";
import Link from "next/link";

import {
  Dataset,
  UploadSession,
  completeUploadSession,
  deleteDataset,
  fetchPublicDatasets,
  fetchMe,
  fetchUploadStatus,
  finalizeDatasetUpload,
  initUploadSession,
  updateDataset,
  uploadChunk,
} from "../../lib/api";

const CHUNK_SIZE_BYTES = 16 * 1024 * 1024;
const DEFAULT_MAX_CLIENT_HASH_MB = 512;
const parsedMaxHashMb = Number(process.env.NEXT_PUBLIC_MAX_CLIENT_HASH_MB);
const MAX_CLIENT_HASH_MB =
  Number.isFinite(parsedMaxHashMb) && parsedMaxHashMb > 0
    ? parsedMaxHashMb
    : DEFAULT_MAX_CLIENT_HASH_MB;
const MAX_CLIENT_HASH_BYTES = Math.floor(MAX_CLIENT_HASH_MB * 1024 * 1024);
const MAX_CHUNK_RETRIES = 3;

export default function DatasetUploadPage() {
  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const [datasetId, setDatasetId] = useState("");
  const [label, setLabel] = useState("");
  const [organ, setOrgan] = useState("");
  const [platform, setPlatform] = useState("");
  const [recommendedPreset, setRecommendedPreset] = useState("");
  const [notes, setNotes] = useState("");
  const [stagedFile, setStagedFile] = useState<File | null>(null);
  const [metadataFile, setMetadataFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [publicDatasets, setPublicDatasets] = useState<Dataset[]>([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeFile, setActiveFile] = useState("");
  const [checksumNote, setChecksumNote] = useState("");
  const [paused, setPaused] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  const [fileProgress, setFileProgress] = useState<Record<string, number>>({});
  const [editDatasetId, setEditDatasetId] = useState("");
  const [editLabel, setEditLabel] = useState("");
  const [editNotes, setEditNotes] = useState("");

  async function refreshPublicDatasets() {
    try {
      const items = await fetchPublicDatasets();
      setPublicDatasets(items);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  useEffect(() => {
    refreshPublicDatasets();
    fetchMe()
      .then((data) => setCurrentUser(data.username))
      .catch(() => setCurrentUser(null));
  }, []);

  function formatEta(seconds: number | null): string {
    if (!seconds || !Number.isFinite(seconds) || seconds < 0) {
      return "-";
    }
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  }

  async function ensureSession(
    dataset: string,
    role: "staged" | "metadata" | "reference",
    file: File,
    expectedSha256?: string
  ): Promise<UploadSession> {
    const key = `upload:${dataset}:${role}:${file.name}:${file.size}`;
    const cached = typeof window !== "undefined" ? window.localStorage.getItem(key) : null;
    if (cached) {
      try {
        const existing = await fetchUploadStatus(cached);
        if (
          existing.dataset_id === dataset &&
          existing.file_role === role &&
          existing.file_name === file.name &&
          Number(existing.total_size) === file.size
        ) {
          return existing;
        }
      } catch {
        // stale cached upload id, create a new one below
      }
    }
    const created = await initUploadSession({
      dataset_id: dataset,
      file_role: role,
      file_name: file.name,
      total_size: file.size,
      content_type: file.type || undefined,
      expected_sha256: expectedSha256,
    });
    if (typeof window !== "undefined") {
      window.localStorage.setItem(key, created.upload_id);
    }
    return created;
  }

  async function transferFile(
    dataset: string,
    role: "staged" | "metadata" | "reference",
    file: File,
    totalBytes: number,
    bytesCompletedBeforeFile: number,
    expectedSha256?: string
  ): Promise<string> {
    const key = `upload:${dataset}:${role}:${file.name}:${file.size}`;
    const session = await ensureSession(dataset, role, file, expectedSha256);
    let offset = Number(session.received_bytes || 0);
    setFileProgress((prev) => ({ ...prev, [role]: Math.floor((offset / file.size) * 100) }));

    while (offset < file.size) {
      while (paused) {
        await new Promise((resolve) => setTimeout(resolve, 200));
      }
      const end = Math.min(offset + CHUNK_SIZE_BYTES, file.size);
      const chunk = file.slice(offset, end);
      let chunkUploaded = false;
      for (let attempt = 1; attempt <= MAX_CHUNK_RETRIES; attempt += 1) {
        try {
          const result = await uploadChunk(session.upload_id, offset, chunk);
          offset = Number(result.received_bytes || end);
          chunkUploaded = true;
          break;
        } catch (error) {
          setRetryCount((prev) => prev + 1);
          const recovered = await fetchUploadStatus(session.upload_id);
          offset = Number(recovered.received_bytes || 0);
          if (offset >= end) {
            chunkUploaded = true;
            break;
          }
          if (attempt === MAX_CHUNK_RETRIES) {
            const detail = error instanceof Error ? error.message : String(error);
            throw new Error(
              `Chunk upload failed for ${file.name} at offset ${offset} after ${MAX_CHUNK_RETRIES} retries: ${detail}`
            );
          }
          await new Promise((resolve) => setTimeout(resolve, attempt * 400));
        }
      }
      if (!chunkUploaded) {
        throw new Error(`Chunk upload failed for ${file.name}.`);
      }
      const uploadedTotal = bytesCompletedBeforeFile + offset;
      setProgress(Math.min(100, Math.floor((uploadedTotal / totalBytes) * 100)));
      setFileProgress((prev) => ({ ...prev, [role]: Math.floor((offset / file.size) * 100) }));
    }

    const completed = await completeUploadSession(session.upload_id);
    if (!completed.completed) {
      throw new Error(`Upload did not complete for ${file.name}.`);
    }
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(key);
    }
    return completed.upload_id;
  }

  async function computeSha256Hex(file: File): Promise<string | undefined> {
    if (typeof window === "undefined" || !window.crypto?.subtle) {
      return undefined;
    }
    if (file.size > MAX_CLIENT_HASH_BYTES) {
      return undefined;
    }
    const buffer = await file.arrayBuffer();
    const digest = await window.crypto.subtle.digest("SHA-256", buffer);
    const bytes = new Uint8Array(digest);
    return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!datasetId.trim() || !label.trim() || !organ.trim() || !platform.trim()) {
      setStatus("dataset_id, label, organ, and platform are required.");
      return;
    }
    if (!stagedFile || !metadataFile) {
      setStatus("Upload both staged h5ad and cell metadata files.");
      return;
    }
    setStatus("");
    setLoading(true);
    setProgress(0);
    setPaused(false);
    setRetryCount(0);
    setEtaSeconds(null);
    setFileProgress({ staged: 0, metadata: 0, reference: 0 });
    try {
      const uploadStart = Date.now();
      const dataset = datasetId.trim().toLowerCase();
      const uploads: Array<{ role: "staged" | "metadata" | "reference"; file: File }> = [
        { role: "staged", file: stagedFile },
        { role: "metadata", file: metadataFile },
      ];
      if (referenceFile) {
        uploads.push({ role: "reference", file: referenceFile });
      }
      const totalBytes = uploads.reduce((sum, entry) => sum + entry.file.size, 0);
      let completedBytes = 0;
      const expectedHashes: Record<string, string | undefined> = {};
      let stagedUploadId = "";
      let metadataUploadId = "";
      let referenceUploadId: string | undefined;

      setChecksumNote("Computing client-side checksums...");
      for (const entry of uploads) {
        setActiveFile(entry.file.name);
        try {
          expectedHashes[entry.role] = await computeSha256Hex(entry.file);
        } catch {
          expectedHashes[entry.role] = undefined;
        }
      }
      const hashedCount = Object.values(expectedHashes).filter(Boolean).length;
      if (hashedCount === uploads.length) {
        setChecksumNote("Checksums ready.");
      } else if (hashedCount > 0) {
        setChecksumNote(
          `Some files exceeded ${MAX_CLIENT_HASH_MB} MB browser hashing limit; server-side checksum validation still runs.`
        );
      } else {
        setChecksumNote(
          `Using server-side checksum validation (browser hashing limit: ${MAX_CLIENT_HASH_MB} MB).`
        );
      }

      for (const entry of uploads) {
        setActiveFile(entry.file.name);
        const uploadId = await transferFile(
          dataset,
          entry.role,
          entry.file,
          totalBytes,
          completedBytes,
          expectedHashes[entry.role]
        );
        if (entry.role === "staged") {
          stagedUploadId = uploadId;
        } else if (entry.role === "metadata") {
          metadataUploadId = uploadId;
        } else {
          referenceUploadId = uploadId;
        }
        completedBytes += entry.file.size;
        const percent = Math.min(100, Math.floor((completedBytes / totalBytes) * 100));
        setProgress(percent);
        const elapsedSec = Math.max((Date.now() - uploadStart) / 1000, 1);
        const throughput = completedBytes / elapsedSec;
        const remainingBytes = Math.max(totalBytes - completedBytes, 0);
        setEtaSeconds(throughput > 0 ? remainingBytes / throughput : null);
      }

      await finalizeDatasetUpload({
        dataset_id: dataset,
        label: label.trim(),
        organ: organ.trim(),
        platform: platform.trim(),
        notes: notes.trim() || undefined,
        recommended_preset: recommendedPreset.trim() || undefined,
        staged_upload_id: stagedUploadId,
        cell_metadata_upload_id: metadataUploadId,
        reference_upload_id: referenceUploadId,
        public: true,
      });
      setStatus("Dataset uploaded and published.");
      setActiveFile("");
      setProgress(100);
      setChecksumNote("");
      setEtaSeconds(0);
      setDatasetId("");
      setLabel("");
      setOrgan("");
      setPlatform("");
      setRecommendedPreset("");
      setNotes("");
      setStagedFile(null);
      setMetadataFile(null);
      setReferenceFile(null);
      await refreshPublicDatasets();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(`Upload failed. ${message} You can click "Upload dataset" again to resume.`);
    } finally {
      setLoading(false);
    }
  }

  async function handleDatasetUpdate(dataset: Dataset) {
    setStatus("");
    try {
      const payload: { label?: string; notes?: string; public?: boolean } = {};
      if (editLabel.trim()) {
        payload.label = editLabel.trim();
      }
      payload.notes = editNotes;
      const nextPublic = !dataset.public;
      payload.public = nextPublic;
      await updateDataset(dataset.id, payload);
      setEditDatasetId("");
      setEditLabel("");
      setEditNotes("");
      await refreshPublicDatasets();
      setStatus(`Dataset ${dataset.id} updated.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleDatasetDelete(dataset: Dataset) {
    if (!window.confirm(`Delete dataset ${dataset.id} from registry?`)) {
      return;
    }
    setStatus("");
    try {
      await deleteDataset(dataset.id);
      await refreshPublicDatasets();
      setStatus(`Dataset ${dataset.id} deleted from registry.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <main className="page">
      <section className="panel">
        <h1>Public Dataset Upload</h1>
        <p className="muted">Uploaded datasets are visible to everyone in the shared registry.</p>
        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="row">
            <div>
              <label>Dataset ID</label>
              <input value={datasetId} onChange={(event) => setDatasetId(event.target.value)} />
            </div>
            <div>
              <label>Label</label>
              <input value={label} onChange={(event) => setLabel(event.target.value)} />
            </div>
          </div>
          <div className="row">
            <div>
              <label>Organ</label>
              <input value={organ} onChange={(event) => setOrgan(event.target.value)} />
            </div>
            <div>
              <label>Platform</label>
              <input value={platform} onChange={(event) => setPlatform(event.target.value)} />
            </div>
          </div>
          <div className="row">
            <div>
              <label>Recommended preset (optional)</label>
              <input
                value={recommendedPreset}
                onChange={(event) => setRecommendedPreset(event.target.value)}
              />
            </div>
            <div>
              <label>Notes (optional)</label>
              <input value={notes} onChange={(event) => setNotes(event.target.value)} />
            </div>
          </div>
          <div className="row">
            <div>
              <label>Staged H5AD</label>
              <input
                type="file"
                accept=".h5ad"
                onChange={(event) => setStagedFile(event.target.files?.[0] || null)}
              />
            </div>
            <div>
              <label>Cell metadata</label>
              <input type="file" onChange={(event) => setMetadataFile(event.target.files?.[0] || null)} />
            </div>
            <div>
              <label>Reference H5AD (optional)</label>
              <input
                type="file"
                accept=".h5ad"
                onChange={(event) => setReferenceFile(event.target.files?.[0] || null)}
              />
            </div>
          </div>
          <div className="inline-actions">
            <button type="submit" disabled={loading}>
              {loading ? "Uploading..." : "Upload dataset"}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => setPaused((prev) => !prev)}
              disabled={!loading}
            >
              {paused ? "Resume" : "Pause"}
            </button>
            <Link className="link" href="/">
              Back to home
            </Link>
          </div>
          {loading ? (
            <div className="card stack inset">
              <strong>Upload progress</strong>
              <div>{progress}% complete</div>
              <div>Current file: {activeFile || "-"}</div>
              <div>Checksum: {checksumNote || "-"}</div>
              <div>Retries: {retryCount}</div>
              <div>Estimated time remaining: {formatEta(etaSeconds)}</div>
              <div>
                <strong>Per-file progress</strong>
                <div>staged: {fileProgress.staged ?? 0}%</div>
                <div>metadata: {fileProgress.metadata ?? 0}%</div>
                <div>reference: {fileProgress.reference ?? 0}%</div>
              </div>
            </div>
          ) : null}
        </form>
      </section>

      {status ? (
        <section className="status">
          <p>{status}</p>
        </section>
      ) : null}

      <section className="panel">
        <div className="panel-header">
          <h2>Public Datasets</h2>
          <button type="button" className="ghost" onClick={refreshPublicDatasets}>
            Refresh
          </button>
        </div>
        <div className="run-grid">
          {publicDatasets.map((item) => (
            <div key={item.id} className="card stack tight">
              <strong>{item.label || item.id}</strong>
              <div>ID: {item.id}</div>
              <div>Organ: {item.organ || "-"}</div>
              <div>Platform: {item.platform || "-"}</div>
              <div>Public: {item.public ? "yes" : "no"}</div>
              <div>Uploaded by: {item.uploaded_by || "-"}</div>
              <div>Uploaded at: {item.uploaded_at || "-"}</div>
              <div>Updated by: {item.updated_by || "-"}</div>
              <div>Updated at: {item.updated_at || "-"}</div>
              <div>Notes: {item.notes || "-"}</div>
              <div>
                Checksums:
                <pre>{JSON.stringify(item.checksums || {}, null, 2)}</pre>
              </div>
              {currentUser ? (
                <div className="card inset">
                  <div className="row">
                    <div>
                      <label>Label</label>
                      <input
                        placeholder={item.label || item.id}
                        value={editDatasetId === item.id ? editLabel : ""}
                        onChange={(event) => {
                          setEditDatasetId(item.id);
                          setEditLabel(event.target.value);
                        }}
                      />
                    </div>
                    <div>
                      <label>Notes</label>
                      <input
                        placeholder={item.notes || ""}
                        value={editDatasetId === item.id ? editNotes : ""}
                        onChange={(event) => {
                          setEditDatasetId(item.id);
                          setEditNotes(event.target.value);
                        }}
                      />
                    </div>
                  </div>
                  <div className="inline-actions">
                    <button type="button" className="ghost" onClick={() => handleDatasetUpdate(item)}>
                      {item.public ? "Unpublish + Save" : "Publish + Save"}
                    </button>
                    <button type="button" className="ghost" onClick={() => handleDatasetDelete(item)}>
                      Delete
                    </button>
                  </div>
                </div>
              ) : null}
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
