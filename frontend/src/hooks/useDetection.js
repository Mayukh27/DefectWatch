// src/hooks/useDetection.js
// Handles legacy cameras (int id) + v2 cameras (string id, RTSP/client-push)

import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../utils/api';

const POLL_INTERVAL = 1200;

function mergeStatus(prev, incoming) {
  const next = new Map(incoming.map(s => [String(s.camera_id), s]));
  const merged = prev.map(cam => {
    const upd = next.get(String(cam.id));
    if (!upd) return cam;
    next.delete(String(cam.id));
    return {
      ...cam,
      active:    upd.active,
      defect:    upd.defect,
      method:    upd.method    ?? cam.method,
      fps:       upd.fps       ?? cam.fps,
      timestamp: upd.timestamp ?? cam.timestamp,
      camType:        upd.type      ?? cam.camType ?? 'local',
      roi_configured: upd.roi_configured ?? cam.roi_configured ?? false,
    };
  });
  // Append brand-new cameras discovered via polling (RTSP / client-push)
  next.forEach((s, id) => {
    merged.push({
      id:        id,
      active:    s.active,
      defect:    s.defect,
      method:    s.method    ?? 'custom',
      fps:       s.fps       ?? 0,
      timestamp: s.timestamp ?? null,
      camType:        s.type      ?? 'unknown',
      roi_configured: s.roi_configured ?? false,
    });
  });
  return merged;
}

export function useDetection(numCameras = 4) {
  const [cameras, setCameras] = useState(() =>
    Array.from({ length: numCameras }, (_, i) => ({
      id:        String(i),
      active:    false,
      defect:    false,
      method:    'custom',
      fps:       0,
      timestamp: null,
      camType:        'local',
      roi_configured: false,
    }))
  );

  const [availableMethods, setAvailableMethods] = useState([]);
  const [backendOnline, setBackendOnline]         = useState(false);
  const pollRef = useRef(null);

  useEffect(() => {
    api.methods().then(d => setAvailableMethods(d.methods ?? [])).catch(() => {});
  }, []);

  useEffect(() => {
    api.health().then(() => setBackendOnline(true)).catch(() => setBackendOnline(false));
  }, []);

  useEffect(() => {
    const poll = async () => {
      try {
        const statuses = await api.statusAll();
        setBackendOnline(true);
        setCameras(prev => mergeStatus(prev, statuses));
      } catch {
        setBackendOnline(false);
      }
    };
    pollRef.current = setInterval(poll, POLL_INTERVAL);
    poll();
    return () => clearInterval(pollRef.current);
  }, []);

  const startCamera = useCallback(async (cameraId, method) => {
    setCameras(prev => prev.map(c => String(c.id) === String(cameraId) ? { ...c, method } : c));
    if (!isNaN(Number(cameraId))) {
      await api.start(Number(cameraId), method);   // legacy local camera
    }
    // RTSP/client v2 cameras are started via AddCameraModal → api.addRtsp / api.registerClient
  }, []);

  const stopCamera = useCallback(async (cameraId) => {
    if (!isNaN(Number(cameraId))) {
      await api.stop(Number(cameraId));
    } else {
      await api.stopV2(String(cameraId));
    }
  }, []);

  const setMethod = useCallback((cameraId, method) => {
    setCameras(prev => prev.map(c => String(c.id) === String(cameraId) ? { ...c, method } : c));
  }, []);

  return { cameras, availableMethods, backendOnline, startCamera, stopCamera, setMethod };
}
