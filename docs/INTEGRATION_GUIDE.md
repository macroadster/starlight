# Starlight ↔️ Stargate Integration Guide

**Updated:** December 2, 2025  
**Scope:** Production-ready REST handoff (Postgres + PVC) with no filesystem tricks.

---

## Components (current reality)
- **Starlight API (FastAPI)**  
  - `/inscribe` embeds text into an image.  
  - If `STARGATE_INGEST_URL` is set (e.g. `http://stargate-backend:3001/api/ingest-inscription`), it POSTs the stego image as base64 with metadata (includes the embedded message) and optional `X-Ingest-Token`.  
  - `/scan/block` reads blocks from `BLOCKS_DIR` (mounted PVC, default `/data/blocks`) and scans inscriptions.
- **Stargate Backend (Go)**  
  - `/api/ingest-inscription` writes to Postgres table `starlight_ingestions` (JSONB metadata).  
  - `/api/pending-transactions` reads `starlight_ingestions`, materializes images to `/data/uploads/<id>_<filename>`, and returns pending items (including embedded text).  
  - `/api/inscribe` proxies to Starlight `/inscribe` (using `STARGATE_PROXY_BASE`), then falls back to local placeholder if needed; still returns a success envelope to the UI.  
  - Serves uploads from PVC at `/uploads/**` (backed by `/data/uploads`).
- **Stargate Frontend (React)**  
  - Calls backend `/api/search`, `/api/block-images`, `/api/pending-transactions`, `/api/inscribe`.  
  - Pending grid now shows image previews and text (hidden messages) from ingestion metadata.
- **Storage**  
  - **Blocks PVC** mounted at `/data/blocks` in backend and starlight-api.  
  - **Uploads PVC** mounted at `/data` in backend; Starlight writes ingest images directly into Postgres; backend materializes them to PVC for previews.

---

## Handoff Flow (prod path)
1) Client/FE → `stargate-backend /api/inscribe` (multipart `image`, `message`, `method=alpha|lsb|...`).  
2) Backend proxies the request to Starlight `/inscribe` (auth via `Authorization: Bearer $STARGATE_API_KEY`).  
3) Starlight embeds the message, POSTs to `STARGATE_INGEST_URL` (`/api/ingest-inscription`) with base64 payload and metadata containing `embedded_message`.  
4) Backend stores row in Postgres (`starlight_ingestions`) and re-materializes the image to `/data/uploads/<id>_<filename>`.  
5) Backend exposes pending via `/api/pending-transactions` (used by UI). When broadcasted/mined, a downstream job should mark status and remove from pending (not yet automated).

No watcher/symlink flow is required anymore.

---

## Key Endpoints
- Starlight API (behind cluster DNS `starlight-api:8080` or port-forward):  
  - `POST /inscribe` (requires `Authorization: Bearer $STARGATE_API_KEY`)  
  - `POST /scan/block` (`{"block_height":926058}`) reads from `$BLOCKS_DIR`
- Stargate Backend:  
  - `POST /api/inscribe` (frontend entry)  
  - `POST /api/ingest-inscription` (Starlight → Stargate ingestion), header `X-Ingest-Token` if configured  
  - `GET /api/pending-transactions` (pending grid)  
  - `GET /uploads/<file>` (serves images from `/data/uploads`)  
  - `GET /api/block-images?height=H` (served from blocks PVC)

---

## Configuration (Helm defaults)
- Blocks PVC mounted at `/data/blocks` in `stargate-backend` and `starlight-api` (`BLOCKS_DIR=/data/blocks`).  
- Uploads PVC mounted at `/data` in `stargate-backend`; uploads served from `/data/uploads`.  
- Env vars:  
  - `STARGATE_API_KEY` (shared between FE → backend proxy → Starlight)  
  - `STARGATE_PROXY_BASE=http://starlight-api:8080`  
  - `STARGATE_INGEST_URL=http://stargate-backend:3001/api/ingest-inscription`  
  - `STARGATE_INGEST_TOKEN` (optional)  
  - `BLOCKS_DIR=/data/blocks` (starlight-api)  
  - `UPLOADS_DIR=/data/uploads` (backend)

---

## Notes / TODOs
- Status transition for ingestions (pending → broadcasted/mined) is not yet automated; add a job or RPC to mark completion.  
- `/scan/block` remains synchronous; consider queueing for very large batches.  
- Keep Postgres JSONB as the source of truth; PVCs are for cached artifacts (blocks, uploaded previews).  
- ACL: frontend should only talk to Go backend; Python API is private behind the proxy.  
- Watcher script remains in repo for reference, but is not required in the current deployment. 
