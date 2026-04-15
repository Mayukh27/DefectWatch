// src/components/ROISelectorModal.jsx
// ROI zone drawer with per-zone sensitivity controls.
//
// Key workflow note shown to user:
//   1. Clear the scene (remove objects)
//   2. Save zones → reference is captured on first frame (clean scene)
//   3. Place object / introduce defect → detection triggers
//   OR: Use "Reset Reference" to retake baseline at any time

import { useState, useRef, useEffect, useCallback } from 'react';

function genId() { return 'roi_' + Math.random().toString(36).slice(2, 7); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function normRect(x1, y1, x2, y2) {
  return {
    x: Math.round(Math.min(x1, x2)), y: Math.round(Math.min(y1, y2)),
    w: Math.round(Math.abs(x2 - x1)), h: Math.round(Math.abs(y2 - y1)),
  };
}

const ZONE_COLOR = '#388bfd';
const CANVAS_W   = 800;
const CANVAS_H   = 450;

// Default ROI settings — sensitive enough for thin wires/cracks
const DEFAULT_THRESHOLD = 8;   // lower = more sensitive (detects subtle changes)
const DEFAULT_MIN_AREA  = 150; // smaller = detects thinner objects

export default function ROISelectorModal({ camera, onClose, onSaved }) {
  const bgRef      = useRef(null);
  const fgRef      = useRef(null);
  const liveImgRef = useRef(null);

  const [mode,      setMode]      = useState('loading');
  const [staticImg, setStaticImg] = useState(null);
  const liveTimerRef = useRef(null);

  const [rois,       setRois]       = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [drawing,    setDrawing]    = useState(false);
  const [dragStart,  setDragStart]  = useState(null);
  const [dragCur,    setDragCur]    = useState(null);
  const [status,     setStatus]     = useState(null);
  const [scale,      setScale]      = useState({ x: 1, y: 1 });

  // ── Load snapshot ────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    const tryLoad = () => {
      const img = new Image();
      img.onload = () => {
        if (cancelled) return;
        setStaticImg(img);
        setScale({ x: img.naturalWidth / CANVAS_W, y: img.naturalHeight / CANVAS_H });
        setMode('static');
      };
      img.onerror = () => { if (!cancelled) setMode('live'); };
      img.src = `/snapshot_raw/${camera.id}?t=${Date.now()}`;
    };
    const t = setTimeout(tryLoad, 300);
    return () => { cancelled = true; clearTimeout(t); };
  }, []); // eslint-disable-line

  // ── Live MJPEG canvas loop ────────────────────────────────────────
  useEffect(() => {
    if (mode !== 'live') { clearInterval(liveTimerRef.current); return; }
    setScale({ x: 640 / CANVAS_W, y: 480 / CANVAS_H });
    liveTimerRef.current = setInterval(() => {
      const img    = liveImgRef.current;
      const canvas = bgRef.current;
      if (!img || !canvas || !img.complete || img.naturalWidth === 0) return;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      drawZones(ctx);
    }, 150);
    return () => clearInterval(liveTimerRef.current);
  }, [mode, rois, selectedId]); // eslint-disable-line

  // ── Draw zones helper ─────────────────────────────────────────────
  const drawZones = useCallback((ctx) => {
    rois.forEach(roi => {
      const cx = Math.round(roi.x / scale.x), cy = Math.round(roi.y / scale.y);
      const cw = Math.round(roi.w / scale.x), ch = Math.round(roi.h / scale.y);
      const sel = roi.roi_id === selectedId;
      ctx.strokeStyle = ZONE_COLOR; ctx.lineWidth = sel ? 2.5 : 1.5;
      ctx.setLineDash(sel ? [] : [6, 3]);
      ctx.strokeRect(cx, cy, cw, ch);
      ctx.setLineDash([]);
      ctx.fillStyle = ZONE_COLOR + '1a'; ctx.fillRect(cx, cy, cw, ch);
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(cx + 3, cy + 3, roi.roi_id.length * 6 + 8, 16);
      ctx.fillStyle = ZONE_COLOR; ctx.font = 'bold 10px IBM Plex Mono, monospace';
      ctx.fillText(roi.roi_id, cx + 6, cy + 14);
    });
  }, [rois, selectedId, scale]);

  // ── Background canvas redraw ──────────────────────────────────────
  useEffect(() => {
    if (mode !== 'static' && mode !== 'placeholder') return;
    const canvas = bgRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (mode === 'static' && staticImg) {
      ctx.drawImage(staticImg, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#30363d'; ctx.font = '13px IBM Plex Mono, monospace';
      ctx.fillText('Start the camera first — then reopen this dialog', 30, canvas.height / 2 - 8);
    }
    drawZones(ctx);
  }, [mode, staticImg, rois, selectedId, drawZones]);

  // ── Drag rect on fg canvas ────────────────────────────────────────
  useEffect(() => {
    const canvas = fgRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!drawing || !dragStart || !dragCur) return;
    const r = normRect(dragStart.x, dragStart.y, dragCur.x, dragCur.y);
    ctx.strokeStyle = ZONE_COLOR; ctx.lineWidth = 2; ctx.setLineDash([7, 3]);
    ctx.strokeRect(r.x, r.y, r.w, r.h); ctx.setLineDash([]);
    ctx.fillStyle = ZONE_COLOR + '18'; ctx.fillRect(r.x, r.y, r.w, r.h);
  }, [drawing, dragStart, dragCur]);

  // ── Load existing ROIs ────────────────────────────────────────────
  useEffect(() => {
    fetch(`/get_rois/${camera.id}`).then(r => r.json())
      .then(d => { if (d.rois?.length) setRois(d.rois.map(r => ({ ...r, roi_id: r.roi_id || genId() }))); })
      .catch(() => {});
  }, [camera.id]);

  // ── Mouse handlers ────────────────────────────────────────────────
  const toCanvas = useCallback((e) => {
    const c = fgRef.current; if (!c) return { x: 0, y: 0 };
    const r = c.getBoundingClientRect();
    return {
      x: clamp((e.clientX - r.left) * (c.width  / r.width),  0, c.width),
      y: clamp((e.clientY - r.top)  * (c.height / r.height), 0, c.height),
    };
  }, []);

  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    const pt = toCanvas(e); setDrawing(true); setDragStart(pt); setDragCur(pt);
  }, [toCanvas]);

  const onMouseMove = useCallback((e) => { if (!drawing) return; setDragCur(toCanvas(e)); }, [drawing, toCanvas]);

  const onMouseUp = useCallback((e) => {
    if (!drawing || !dragStart) return;
    setDrawing(false);
    const pt = toCanvas(e);
    const r  = normRect(dragStart.x, dragStart.y, pt.x, pt.y);
    setDragStart(null); setDragCur(null);
    if (r.w < 12 || r.h < 12) return;
    const newRoi = {
      roi_id: genId(), type: 'detect',
      x: Math.round(r.x * scale.x), y: Math.round(r.y * scale.y),
      w: Math.round(r.w * scale.x), h: Math.round(r.h * scale.y),
      threshold: DEFAULT_THRESHOLD,
      min_area:  DEFAULT_MIN_AREA,
    };
    setRois(p => [...p, newRoi]);
    setSelectedId(newRoi.roi_id);
  }, [drawing, dragStart, scale, toCanvas]);

  // ── ROI actions ───────────────────────────────────────────────────
  const deleteRoi = id => { setRois(p => p.filter(r => r.roi_id !== id)); if (selectedId === id) setSelectedId(null); };
  const updateRoi = (id, patch) => setRois(p => p.map(r => r.roi_id === id ? { ...r, ...patch } : r));

  // ── Save ──────────────────────────────────────────────────────────
  const handleSave = async () => {
    if (!rois.length) { setStatus({ ok: false, msg: 'Draw at least one zone first.' }); return; }
    try {
      setStatus({ ok: null, msg: 'Saving…' });
      const res = await fetch('/set_rois', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: String(camera.id), rois }),
      });
      const d = await res.json();
      if (res.ok) {
        setStatus({ ok: true, msg: `Saved ${d.roi_count} zone(s). First clean frame = reference.` });
        setTimeout(() => { onSaved ? onSaved() : onClose(); }, 900);
      } else { setStatus({ ok: false, msg: JSON.stringify(d) }); }
    } catch (e) { setStatus({ ok: false, msg: String(e) }); }
  };

  const handleResetRef = async () => {
    await fetch('/reset_reference', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_id: String(camera.id) }),
    });
    setStatus({ ok: true, msg: 'Reference cleared. Next frame will be the new clean baseline.' });
  };

  const streamUrl = `/stream/${camera.id}`;
  const selectedRoi = rois.find(r => r.roi_id === selectedId);

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 200,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)',
    }}>
      {mode === 'live' && (
        <img ref={liveImgRef} src={streamUrl} alt="" crossOrigin="anonymous"
          style={{ position: 'absolute', opacity: 0, pointerEvents: 'none', width: 1, height: 1 }}
          onError={() => setMode('placeholder')} />
      )}

      <div style={{
        background: '#161b22', border: '1px solid #30363d', borderRadius: 10,
        overflow: 'hidden', display: 'flex', flexDirection: 'column',
        width: '92vw', maxWidth: 1080, maxHeight: '94vh',
        boxShadow: '0 24px 64px rgba(0,0,0,0.6)',
      }}>

        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
          padding: '12px 16px', borderBottom: '1px solid #21262d',
          background: '#0d1117', flexShrink: 0,
        }}>
          <div style={{ flex: 1 }}>
            <p style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 4 }}>
              Camera {camera.id} — ROI Editor
            </p>
            <h2 style={{ fontFamily: 'IBM Plex Mono', fontSize: 14, fontWeight: 700, color: '#e6edf3', marginBottom: 6 }}>
              Draw Detection Zone
            </h2>
            {/* Workflow instructions */}
            <div style={{
              display: 'flex', gap: 12, flexWrap: 'wrap',
              padding: '6px 10px', borderRadius: 5,
              background: 'rgba(56,139,253,0.06)', border: '1px solid rgba(56,139,253,0.15)',
            }}>
              {['1. Remove objects from scene', '2. Draw zone → Save (clean frame = reference)', '3. Place object/defect → detection fires', '4. Reset Reference anytime to retake baseline'].map((s, i) => (
                <span key={i} style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: i === 1 ? '#388bfd' : '#484f58' }}>{s}</span>
              ))}
            </div>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#484f58', fontSize: 20, lineHeight: 1, padding: '0 4px', marginLeft: 16, flexShrink: 0 }}>✕</button>
        </div>

        {/* Body */}
        <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>

          {/* Canvas */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 14, minWidth: 0, overflow: 'hidden' }}>
            {mode === 'live' && (
              <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#388bfd', marginBottom: 8, padding: '3px 8px', borderRadius: 4, background: 'rgba(56,139,253,0.06)', border: '1px solid rgba(56,139,253,0.2)', display: 'inline-block' }}>
                ● Live preview
              </div>
            )}
            <div style={{
              position: 'relative', width: CANVAS_W, height: CANVAS_H, maxWidth: '100%',
              borderRadius: 6, overflow: 'hidden', border: '1px solid #21262d',
              cursor: 'crosshair', background: '#0a0c0e', flexShrink: 0,
            }}>
              <canvas ref={bgRef} width={CANVAS_W} height={CANVAS_H}
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
              <canvas ref={fgRef} width={CANVAS_W} height={CANVAS_H}
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
                onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp}
                onMouseLeave={() => { setDrawing(false); setDragStart(null); setDragCur(null); }} />
            </div>
            <p style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#30363d', marginTop: 6 }}>
              🖱 Click + drag to draw a zone · Select zone in right panel to adjust sensitivity
            </p>
          </div>

          {/* Right panel */}
          <div style={{ width: 240, flexShrink: 0, borderLeft: '1px solid #21262d', display: 'flex', flexDirection: 'column', background: '#0d1117' }}>
            <div style={{ flex: 1, overflowY: 'auto', padding: '10px 10px 0' }}>
              <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#30363d', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>
                Zones ({rois.length})
              </div>

              {!rois.length ? (
                <p style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#30363d', textAlign: 'center', paddingTop: 32 }}>
                  No zones yet.<br/>Draw on the frame.
                </p>
              ) : rois.map(roi => (
                <div key={roi.roi_id} onClick={() => setSelectedId(roi.roi_id)} style={{
                  padding: '8px 10px', marginBottom: 6, borderRadius: 6, cursor: 'pointer',
                  background: selectedId === roi.roi_id ? '#1c2128' : '#161b22',
                  border: `1px solid ${selectedId === roi.roi_id ? '#388bfd44' : '#21262d'}`,
                }}>
                  {/* Zone name */}
                  <input value={roi.roi_id} onChange={e => updateRoi(roi.roi_id, { roi_id: e.target.value })}
                    onClick={e => e.stopPropagation()} style={{
                      width: '100%', background: 'transparent',
                      fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#c9d1d9',
                      border: 'none', borderBottom: '1px solid #21262d', outline: 'none', paddingBottom: 3, marginBottom: 8,
                    }} />

                  {/* Coords */}
                  <p style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58', marginBottom: 8 }}>
                    {roi.x},{roi.y} · {roi.w}×{roi.h}px
                  </p>

                  {/* Sensitivity controls */}
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58' }}>Sensitivity (threshold)</span>
                      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#388bfd' }}>{roi.threshold ?? DEFAULT_THRESHOLD}</span>
                    </div>
                    <input type="range" min="2" max="40" value={roi.threshold ?? DEFAULT_THRESHOLD}
                      onChange={e => updateRoi(roi.roi_id, { threshold: parseInt(e.target.value) })}
                      onClick={e => e.stopPropagation()}
                      style={{ width: '100%', accentColor: '#388bfd', cursor: 'pointer' }}
                      title="Lower = more sensitive (detects thin wires). Higher = less noise."
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 8, color: '#30363d' }}>sensitive</span>
                      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 8, color: '#30363d' }}>robust</span>
                    </div>
                  </div>

                  <div style={{ marginBottom: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58' }}>Min size (px²)</span>
                      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#388bfd' }}>{roi.min_area ?? DEFAULT_MIN_AREA}</span>
                    </div>
                    <input type="range" min="50" max="1000" step="50" value={roi.min_area ?? DEFAULT_MIN_AREA}
                      onChange={e => updateRoi(roi.roi_id, { min_area: parseInt(e.target.value) })}
                      onClick={e => e.stopPropagation()}
                      style={{ width: '100%', accentColor: '#388bfd', cursor: 'pointer' }}
                      title="Minimum contour area. Lower = detects thinner objects (wires, cracks)."
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 8, color: '#30363d' }}>thin objects</span>
                      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 8, color: '#30363d' }}>large objects</span>
                    </div>
                  </div>

                  <button onClick={e => { e.stopPropagation(); deleteRoi(roi.roi_id); }} style={{
                    width: '100%', padding: '3px 0', borderRadius: 4, cursor: 'pointer',
                    fontFamily: 'IBM Plex Mono', fontSize: 9,
                    background: 'transparent', color: '#484f58', border: '1px solid #21262d',
                  }}
                    onMouseEnter={e => { e.currentTarget.style.color = '#da3633'; e.currentTarget.style.borderColor = 'rgba(218,54,51,0.3)'; }}
                    onMouseLeave={e => { e.currentTarget.style.color = '#484f58'; e.currentTarget.style.borderColor = '#21262d'; }}
                  >Remove</button>
                </div>
              ))}
            </div>

            {/* Status */}
            {status && (
              <div style={{
                margin: '0 10px 8px', padding: '6px 10px', borderRadius: 5,
                fontFamily: 'IBM Plex Mono', fontSize: 9,
                background: status.ok === true ? 'rgba(35,134,54,0.1)' : status.ok === false ? 'rgba(218,54,51,0.1)' : '#161b22',
                color: status.ok === true ? '#238636' : status.ok === false ? '#da3633' : '#8b949e',
                border: `1px solid ${status.ok === true ? 'rgba(35,134,54,0.2)' : status.ok === false ? 'rgba(218,54,51,0.2)' : '#21262d'}`,
              }}>{status.msg}</div>
            )}

            {/* Buttons */}
            <div style={{ padding: 10, borderTop: '1px solid #21262d', display: 'flex', flexDirection: 'column', gap: 6 }}>
              <button onClick={handleSave} style={{
                width: '100%', padding: '8px 0', borderRadius: 6, cursor: 'pointer',
                fontFamily: 'IBM Plex Mono', fontSize: 11, fontWeight: 700,
                background: 'rgba(56,139,253,0.12)', color: '#388bfd', border: '1px solid rgba(56,139,253,0.3)',
              }}>💾 Save Zones</button>
              <button onClick={handleResetRef} style={{
                width: '100%', padding: '6px 0', borderRadius: 6, cursor: 'pointer',
                fontFamily: 'IBM Plex Mono', fontSize: 10,
                background: 'rgba(158,106,3,0.08)', color: '#9e6a03', border: '1px solid rgba(158,106,3,0.2)',
              }}>↺ Reset Reference (retake clean baseline)</button>
              <button onClick={() => { setRois([]); setSelectedId(null); }} style={{
                width: '100%', padding: '5px 0', borderRadius: 6, cursor: 'pointer',
                fontFamily: 'IBM Plex Mono', fontSize: 9,
                background: 'transparent', color: '#484f58', border: '1px solid #21262d',
              }}>Clear All Zones</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}