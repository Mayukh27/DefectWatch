const METHOD_LABELS = {
  fd: 'Frame Diff', mog2: 'MOG2', running_avg: 'Running Avg',
  custom: 'Custom', dl: 'DL (inactive)',
};

// src/components/CameraCard.jsx
//
// Correct workflow:
//   1. Click "Start" → camera stream begins immediately
//   2. Stream shows "SELECT ROI TO START DETECTION" overlay (drawn server-side)
//   3. User clicks "Set ROI zones" → draws on the LIVE stream snapshot
//   4. After saving ROI → detection starts on next frame
//   5. Red rect + filled mask shown on detected defects

import { useState, useRef, useEffect } from 'react';
import { api } from '../utils/api';
import ROISelectorModal from './ROISelectorModal';



export default function CameraCard({ camera, onStart, onStop, availableMethods }) {
  const { id, active, defect, fps, method, timestamp, camType, roi_configured } = camera;
  const [selectedMethod, setSelectedMethod] = useState(method || 'custom');
  const imgRef   = useRef(null);
  const [imgErr, setImgErr] = useState(false);
  const [showROI, setShowROI] = useState(false);

  useEffect(() => {
    if (active && imgRef.current) {
      imgRef.current.src = api.streamUrl(id) + '?t=' + Date.now();
      setImgErr(false);
    }
  }, [active, id]);

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      background: '#161b22',
      border: `1px solid ${defect && active ? 'rgba(218,54,51,0.25)' : '#21262d'}`,
      borderRadius: 8, overflow: 'hidden',
    }}>

      {/* ── Header ─────────────────────────────── */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '6px 10px', background: '#0d1117', borderBottom: '1px solid #1a1f26',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
            background: !active ? '#30363d' : defect ? '#da3633' : '#238636',
          }} />
          <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 11, fontWeight: 600, color: '#e6edf3', letterSpacing: '0.04em' }}>
            CAM {String(id).padStart(2, '0')}
          </span>
          {camType && camType !== 'local' && (
            <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 8, padding: '1px 4px', borderRadius: 3, background: '#0d1117', color: '#484f58', border: '1px solid #21262d' }}>
              {camType}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {active && fps > 0 && (
            <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#484f58' }}>{fps} fps</span>
          )}
          {active && !roi_configured && (
            <span style={{
              fontFamily: 'IBM Plex Mono', fontSize: 9, fontWeight: 600,
              color: '#9e6a03', padding: '1px 5px', borderRadius: 3,
              background: 'rgba(158,106,3,0.1)', border: '1px solid rgba(158,106,3,0.2)',
            }}>NO ROI</span>
          )}
          {defect && active && (
            <span style={{
              fontFamily: 'IBM Plex Mono', fontSize: 9, fontWeight: 700,
              color: '#da3633', padding: '1px 5px', borderRadius: 3,
              background: 'rgba(218,54,51,0.08)', border: '1px solid rgba(218,54,51,0.22)',
            }}>DEFECT</span>
          )}
        </div>
      </div>

      {/* ── Video feed ─────────────────────────── */}
      <div style={{ position: 'relative', aspectRatio: '16/9', background: '#0a0c0e', overflow: 'hidden' }}>
        {active && !imgErr ? (
          <img
            ref={imgRef}
            src={api.streamUrl(id)}
            alt={`Camera ${id}`}
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
            onError={() => setImgErr(true)}
          />
        ) : (
          <OfflinePlaceholder error={imgErr} />
        )}
        {active && (
          <div style={{
            position: 'absolute', bottom: 5, right: 6, zIndex: 10,
            fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58',
            background: 'rgba(10,12,14,0.65)', padding: '2px 5px', borderRadius: 3,
          }}>
            {METHOD_LABELS[method] || method}
          </div>
        )}
      </div>

      {/* ── Controls ───────────────────────────── */}
      <div style={{ padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        <select
          value={selectedMethod}
          onChange={e => setSelectedMethod(e.target.value)}
          disabled={active}
          style={{
            width: '100%', background: '#0d1117', color: '#8b949e',
            border: '1px solid #21262d', borderRadius: 4,
            fontFamily: 'IBM Plex Mono', fontSize: 10, padding: '4px 6px',
            opacity: active ? 0.4 : 1, cursor: active ? 'not-allowed' : 'default', outline: 'none',
          }}
        >
          {(availableMethods.length ? availableMethods : Object.keys(METHOD_LABELS)).map(m => (
            <option key={m} value={m}>{METHOD_LABELS[m] || m}</option>
          ))}
        </select>

        <div style={{ display: 'flex', gap: 6 }}>
          {/* Start — immediately starts camera stream */}
          <CtrlBtn onClick={() => onStart(id, selectedMethod)} disabled={active} color="#238636">
            ▶ Start
          </CtrlBtn>
          <CtrlBtn onClick={() => onStop(id)} disabled={!active} color="#da3633">
            ■ Stop
          </CtrlBtn>
        </div>

        {/* Set ROI — enabled once camera is active so snapshot can load */}
        <button
          onClick={() => setShowROI(true)}
          disabled={!active}
          style={{
            width: '100%', padding: '5px 0', borderRadius: 4,
            cursor: active ? 'pointer' : 'not-allowed',
            fontFamily: 'IBM Plex Mono', fontSize: 10, fontWeight: 500,
            background: roi_configured ? 'rgba(56,139,253,0.06)' : (active ? '#0d1117' : '#0d1117'),
            color: roi_configured ? '#388bfd' : (active ? '#8b949e' : '#30363d'),
            border: `1px solid ${roi_configured ? 'rgba(56,139,253,0.25)' : '#21262d'}`,
            outline: 'none', opacity: active ? 1 : 0.4,
          }}
          onMouseEnter={e => { if (active) { e.currentTarget.style.color = '#388bfd'; e.currentTarget.style.borderColor = 'rgba(56,139,253,0.3)'; }}}
          onMouseLeave={e => {
            e.currentTarget.style.color = roi_configured ? '#388bfd' : '#8b949e';
            e.currentTarget.style.borderColor = roi_configured ? 'rgba(56,139,253,0.25)' : '#21262d';
          }}
        >
          ⊞ {roi_configured ? 'Edit ROI zones' : 'Set ROI zones'}
        </button>

        {timestamp && (
          <p style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#30363d', textAlign: 'center' }}>
            {new Date(timestamp).toLocaleTimeString()}
          </p>
        )}
      </div>

      {/* ROI Modal — opens only when camera is already active */}
      {showROI && (
        <ROISelectorModal
          camera={camera}
          onClose={() => setShowROI(false)}
          onSaved={() => setShowROI(false)}
        />
      )}
    </div>
  );
}

function OfflinePlaceholder({ error }) {
  return (
    <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.18 }}>
        <path d="M23 7l-7 5 7 5V7z" stroke="#8b949e" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        <rect x="1" y="5" width="15" height="14" rx="2" stroke="#8b949e" strokeWidth="1.5"/>
        <line x1="1" y1="1" x2="23" y2="23" stroke="#8b949e" strokeWidth="1.5"/>
      </svg>
      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#30363d', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
        {error ? 'stream error' : 'offline'}
      </span>
    </div>
  );
}

function CtrlBtn({ onClick, disabled, color, children }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      flex: 1, padding: '5px 0', borderRadius: 4,
      cursor: disabled ? 'not-allowed' : 'pointer',
      fontFamily: 'IBM Plex Mono', fontSize: 10, fontWeight: 600, letterSpacing: '0.03em',
      background: color + '10', color, border: `1px solid ${color}28`,
      opacity: disabled ? 0.3 : 1, outline: 'none', transition: 'background 0.12s',
    }}
      onMouseEnter={e => { if (!disabled) e.currentTarget.style.background = color + '22'; }}
      onMouseLeave={e => { e.currentTarget.style.background = color + '10'; }}
    >
      {children}
    </button>
  );
}