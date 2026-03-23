# NicheRunner Web

## Setup

```bash
npm install
```

## Run

```bash
npm run dev
```

Set `NEXT_PUBLIC_API_BASE` to point at the FastAPI backend.
Set `NEXT_PUBLIC_WEB_BASE` to point at the deployed web app origin (for share links), for example `https://sptx-tool.vercel.app`.
Login sets a secure cookie; the UI does not store credentials.

## Data input (MVP)

Provide HPG paths to a prepared bundle (h5ad + metadata). NIH/GEO ingestion is deferred.
