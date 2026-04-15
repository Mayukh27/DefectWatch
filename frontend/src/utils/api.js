// src/utils/api.js
// Centralised API calls to FastAPI backend

const BASE = process.env.REACT_APP_API_URL || '';

export const api = {
  // ── Detection control — legacy (int camera_id) ────────────────
  start: (cameraId, method) =>
    fetch(`${BASE}/start?camera_id=${cameraId}&method=${method}`, { method: 'POST' })
      .then(r => r.json()),

  stop: (cameraId) =>
    fetch(`${BASE}/stop?camera_id=${cameraId}`, { method: 'POST' })
      .then(r => r.json()),

  // ── Detection control — v2 (string camera_id) ─────────────────
  addRtsp: (camera_id, rtsp_url, method) =>
    fetch(`${BASE}/add_rtsp_camera`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ camera_id, rtsp_url, method }),
    }).then(r => r.json()),

  registerClient: (camera_id, method) =>
    fetch(`${BASE}/register_client_camera`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ camera_id, method }),
    }).then(r => r.json()),

  stopV2: (camera_id) =>
    fetch(`${BASE}/stop_camera`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ camera_id }),
    }).then(r => r.json()),

  // ── Status ────────────────────────────────────────────────────
  // /status now returns legacy + v2 cameras in one merged list
  statusAll: () =>
    fetch(`${BASE}/status`).then(r => r.json()),

  statusOne: (cameraId) =>
    fetch(`${BASE}/status/${cameraId}`).then(r => r.json()),

  // ── Stream URL ─────────────────────────────────────────────────
  // Legacy int ids  → /stream/{id}
  // v2 string ids   → /v2/stream/{id}
  streamUrl: (cameraId) => {
    const id = String(cameraId);
    return !isNaN(Number(id))
      ? `${BASE}/stream/${id}`
      : `${BASE}/v2/stream/${encodeURIComponent(id)}`;
  },

  // ── Methods ───────────────────────────────────────────────────
  methods: () =>
    fetch(`${BASE}/methods`).then(r => r.json()),

  // ── v2 camera list ────────────────────────────────────────────
  camerasV2: () =>
    fetch(`${BASE}/v2/cameras`).then(r => r.json()),

  // ── Evaluation (backend computes + stores; UI just triggers) ──
  // ⚠️  METRICS / GRAPHS — for future research use
  // evaluate: () =>
  //   fetch(`${BASE}/evaluate`).then(r => r.json()),
  //
  // compare: () =>
  //   fetch(`${BASE}/compare`).then(r => r.json()),

  // ── Health ────────────────────────────────────────────────────
  health: () =>
    fetch(`${BASE}/health`).then(r => r.json()),
};
