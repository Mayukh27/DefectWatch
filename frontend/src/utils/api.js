// src/utils/api.js
// Centralised API calls to FastAPI backend.
// All calls go through apiFetch() which checks res.ok and throws
// a proper Error on non-2xx responses with the detail from the body.

const BASE = process.env.REACT_APP_API_URL || '';

async function apiFetch(url, options = {}) {
  const res = await fetch(`${BASE}${url}`, options);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body?.detail || body?.message || JSON.stringify(body);
    } catch (_) { /* body not JSON */ }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  // ── Detection control — legacy (int camera_id) ────────────────
  start: (cameraId, method) =>
    apiFetch(`/start?camera_id=${cameraId}&method=${method}`, { method: 'POST' }),

  stop: (cameraId) =>
    apiFetch(`/stop?camera_id=${cameraId}`, { method: 'POST' }),

  // ── Detection control — v2 (string camera_id) ─────────────────
  addRtsp: (camera_id, rtsp_url, method) =>
    apiFetch('/add_rtsp_camera', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ camera_id, rtsp_url, method }),
    }),

  registerClient: (camera_id, method) =>
    apiFetch('/register_client_camera', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ camera_id, method }),
    }),

  stopV2: (camera_id) =>
    apiFetch('/stop_camera', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ camera_id }),
    }),

  // ── Status ────────────────────────────────────────────────────
  statusAll: () => apiFetch('/status'),

  statusOne: (cameraId) => apiFetch(`/status/${cameraId}`),

  // ── Stream URL ─────────────────────────────────────────────────
  // Legacy int ids  → /stream/{id}
  // v2 string ids   → /v2/stream/{id}
  streamUrl: (cameraId) => {
    const id = String(cameraId);
    return !isNaN(Number(id))
      ? `${BASE}/stream/${id}`
      : `${BASE}/v2/stream/${encodeURIComponent(id)}`;
  },

  snapshotRawUrl: (cameraId) =>
    `${BASE}/snapshot_raw/${encodeURIComponent(String(cameraId))}`,

  // ── ROI management ────────────────────────────────────────────
  setRois: (camera_id, rois) =>
    apiFetch('/set_rois', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ camera_id: String(camera_id), rois }),
    }),

  getRois: (cameraId) => apiFetch(`/get_rois/${cameraId}`),

  resetReference: (camera_id) =>
    apiFetch('/reset_reference', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ camera_id: String(camera_id) }),
    }),

  // ── Snapshots ─────────────────────────────────────────────────
  popupData: (cameraId) => apiFetch(`/popup_data/${cameraId}`),

  // ── Methods ───────────────────────────────────────────────────
  methods: () => apiFetch('/methods'),

  // ── v2 camera list ────────────────────────────────────────────
  camerasV2: () => apiFetch('/v2/cameras'),

  // ── Health ────────────────────────────────────────────────────
  health: () => apiFetch('/health'),
};