# Environment Template (HPG API + Vercel FE)

Use this as a starting point for production.

## API (HPG host)

```bash
# Core paths
RUNS_DIR=/blue/kejun.huang/vasco.hinostroza/nicherunner/runs
ARTIFACT_ROOTS=/blue/kejun.huang/vasco.hinostroza/nicherunner/runs,/blue/kejun.huang/vasco.hinostroza/data,/orange/kejun.huang/vasco.hinostroza/data,/blue/kejun.huang/vasco.hinostroza/nicherunner/public_uploads/datasets
DATASETS_REGISTRY_PATH=/blue/kejun.huang/vasco.hinostroza/nicherunner/registries/datasets.json
PRESETS_DIR=/blue/kejun.huang/vasco.hinostroza/nicherunner/presets
DB_PATH=/blue/kejun.huang/vasco.hinostroza/nicherunner/runs.db
DATA_UPLOADS_DIR=/blue/kejun.huang/vasco.hinostroza/nicherunner/public_uploads/datasets

# Auth/session
SESSION_SECRET=<strong-random-secret>
BASIC_AUTH_USER=<login-id>
BASIC_AUTH_PASS=<strong-password>
AUTH_IDENTIFIER_DOMAIN=ufl.edu
# Only admins may call POST /auth/users. BASIC_AUTH_USER and users with
# role=admin in the users registry are admins; ADMIN_USERS adds more.
ADMIN_USERS=

# Subprocess timeouts (seconds): hard caps so a wedged child cannot pin a worker
PIPELINE_TIMEOUT_SECONDS=900
SLURM_COMMAND_TIMEOUT_SECONDS=120
SSH_COMMAND_TIMEOUT_SECONDS=300
SCP_TIMEOUT_SECONDS=1800

# SQLite busy/lock wait
DB_TIMEOUT_SECONDS=30

# Per-request cap for PUT /uploads/{id}/chunk (bounds server memory)
UPLOAD_MAX_CHUNK_BYTES=67108864

# Cross-site browser auth (Vercel FE -> API)
COOKIE_SECURE=true
COOKIE_SAMESITE=none
ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app

# Upload controls and cleanup
UPLOAD_MAX_CONCURRENT_PER_USER=6
UPLOAD_MAX_SIZE_STAGED_GB=100
UPLOAD_MAX_SIZE_METADATA_GB=5
UPLOAD_MAX_SIZE_REFERENCE_GB=50
UPLOAD_ALLOWED_EXT_STAGED=.h5ad
UPLOAD_ALLOWED_EXT_METADATA=.csv,.tsv,.gz
UPLOAD_ALLOWED_EXT_REFERENCE=.h5ad
UPLOAD_SESSION_TTL_HOURS=72
UPLOAD_CLEANUP_ENABLED=true
UPLOAD_CLEANUP_INTERVAL_SECONDS=900
```

## Frontend (Vercel)

```bash
NEXT_PUBLIC_API_BASE=https://<your-api-domain>
NEXT_PUBLIC_MAX_CLIENT_HASH_MB=512
```

## Reverse proxy recommendations (API domain)

For chunk uploads, set body/timeout values above your chunk size and expected transfer time:

- Max request body: >= chunk size (default app chunk: 16 MB)
- Read timeout: >= 300s
- Send timeout: >= 300s
- Keepalive timeout: >= 75s

Example Nginx snippet:

```nginx
client_max_body_size 32m;
proxy_read_timeout 300s;
proxy_send_timeout 300s;
keepalive_timeout 75s;
```

## Notes

- If API is on HPG and FE is on Vercel, API must be HTTPS and internet-reachable.
- Install API deps including multipart support:
  - `pip install -r apps/api/requirements.txt`
- If uploads fail due to CORS/cookies, verify `ALLOWED_ORIGINS`, `COOKIE_SECURE`, and `COOKIE_SAMESITE`.
