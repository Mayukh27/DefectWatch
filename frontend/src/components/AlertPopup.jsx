// src/components/AlertPopup.jsx
// Before / After popup — fetches real reference + current crops from /popup_data/{id}
import { useState, useEffect } from 'react';

const TYPE_LABEL = { crack: 'Crack detection', movement: 'Movement detection', general: 'General' };

export default function AlertPopup({ camera, onClose }) {
  const [data, setData]     = useState(null);
  const [loading, setLoad]  = useState(true);
  const [error, setError]   = useState('');

  useEffect(() => {
    if (!camera) return;
    setLoad(true); setError('');
    fetch(`/popup_data/${camera.id}?t=${Date.now()}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(d => { setData(d); setLoad(false); })
      .catch(e => { setError(`Failed to load data (${e})`); setLoad(false); });
  }, [camera]);

  if (!camera) return null;

  const roiId   = data?.roi_id   || '—';
  const roiType = data?.roi_type || 'general';
  const ts      = data?.timestamp || new Date().toLocaleString();
  const refSrc  = data?.reference;
  const curSrc  = data?.current;

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 200,
        background: 'rgba(0,0,0,0.72)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        backdropFilter: 'blur(5px)',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: '#0d1117', borderRadius: 10,
          border: `1px solid ${camera.defect ? 'rgba(218,54,51,0.40)' : '#21262d'}`,
          width: '88vw', maxWidth: 860,
          boxShadow: '0 32px 64px rgba(0,0,0,0.6)',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
          padding: '13px 18px', background: '#161b22', borderBottom: '1px solid #21262d',
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              {camera.defect && (
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#da3633', display: 'inline-block' }} />
              )}
              <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 13, fontWeight: 600, color: '#e6edf3' }}>
                CAM {String(camera.id).padStart(2, '0')} — Before / After
              </span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 18px' }}>
              <MetaItem label="Camera"    value={`CAM ${String(camera.id).padStart(2,'0')}`} />
              <MetaItem label="ROI"       value={roiId} />
              <MetaItem label="Type"      value={TYPE_LABEL[roiType] || roiType} />
              <MetaItem label="Status"    value={camera.defect ? 'DEFECT' : 'OK'}
                        color={camera.defect ? '#da3633' : '#238636'} />
              <MetaItem label="Timestamp" value={ts} />
            </div>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: '#484f58', fontSize: 18, lineHeight: 1, padding: '0 4px', marginLeft: 16,
          }}>✕</button>
        </div>

        {/* Image panels */}
        {loading ? (
          <div style={{ padding: 40, textAlign: 'center', fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#484f58' }}>
            Loading frames…
          </div>
        ) : error ? (
          <div style={{ padding: 40, textAlign: 'center', fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#da3633' }}>
            {error}
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', background: '#21262d', gap: 1 }}>
            {/* Before — reference */}
            <ImagePanel
              src={refSrc}
              title="Before"
              subtitle="Reference (fixed baseline)"
              dotColor="#238636"
              note="First frame captured — never auto-updates"
            />
            {/* After — current with defect overlay */}
            <ImagePanel
              src={curSrc}
              title="After"
              subtitle="Current (with defect overlay)"
              dotColor={camera.defect ? '#da3633' : '#238636'}
              note="Red overlay = exact defect region only"
              highlight={camera.defect}
            />
          </div>
        )}

        {/* Footer */}
        <div style={{
          padding: '9px 18px', borderTop: '1px solid #21262d',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58' }}>
            Click outside to close · Defect mask traces exact changed pixel regions
          </span>
          <button onClick={onClose} style={{
            fontFamily: 'IBM Plex Mono', fontSize: 11, fontWeight: 600,
            padding: '5px 14px', borderRadius: 4, cursor: 'pointer',
            background: '#21262d', color: '#8b949e', border: '1px solid #30363d',
          }}>Close</button>
        </div>
      </div>
    </div>
  );
}

function ImagePanel({ src, title, subtitle, dotColor, note, highlight }) {
  return (
    <div style={{ background: '#0a0c0e', padding: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 8 }}>
        <span style={{ width: 7, height: 7, borderRadius: '50%', background: dotColor, display: 'inline-block' }} />
        <div>
          <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 11, fontWeight: 600, color: '#c9d1d9', letterSpacing: '0.04em' }}>
            {title}
          </div>
          <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58', marginTop: 1 }}>
            {subtitle}
          </div>
        </div>
      </div>

      <div style={{
        borderRadius: 6, overflow: 'hidden',
        border: `1px solid ${highlight ? 'rgba(218,54,51,0.30)' : '#21262d'}`,
        background: '#161b22', aspectRatio: '4/3',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {src ? (
          <img src={src} alt={title}
            style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }} />
        ) : (
          <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#30363d' }}>No image available</span>
        )}
      </div>
      <p style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58', marginTop: 5, textAlign: 'center' }}>
        {note}
      </p>
    </div>
  );
}

function MetaItem({ label, value, color }) {
  return (
    <span>
      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
        {label}:{' '}
      </span>
      <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: color || '#8b949e', fontWeight: color ? 600 : 400 }}>
        {value}
      </span>
    </span>
  );
}
