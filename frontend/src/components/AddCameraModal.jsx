// src/components/AddCameraModal.jsx
// Modal for adding RTSP or client-push cameras.
// Uses inline styles throughout — tailwind.config.js only defines bg/text/ok/defect/warn/info/mask,
// so surface-800, accent-cyan etc. are not available and produce invisible elements.

import { useState } from 'react';
import { api } from '../utils/api';

const S = { fontFamily: 'IBM Plex Mono' };

const INPUT = {
  width: '100%', background: '#0d1117', border: '1px solid #21262d',
  color: '#c9d1d9', fontFamily: 'IBM Plex Mono', fontSize: 11,
  borderRadius: 5, padding: '7px 10px', outline: 'none', boxSizing: 'border-box',
};

export default function AddCameraModal({ onClose, availableMethods }) {
  const [camType,  setCamType]  = useState('rtsp');
  const [cameraId, setCameraId] = useState('');
  const [rtspUrl,  setRtspUrl]  = useState('rtsp://');
  const [method,   setMethod]   = useState('custom');
  const [status,   setStatus]   = useState(null);
  const [message,  setMessage]  = useState('');

  const handleSubmit = async () => {
    if (!cameraId.trim()) { setMessage('Camera ID is required.'); setStatus('error'); return; }
    if (camType === 'rtsp' && !rtspUrl.startsWith('rtsp')) {
      setMessage('RTSP URL must start with rtsp://'); setStatus('error'); return;
    }
    setStatus('loading');
    try {
      const res = camType === 'rtsp'
        ? await api.addRtsp(cameraId.trim(), rtspUrl.trim(), method)
        : await api.registerClient(cameraId.trim(), method);

      if (res.status === 'started' || res.status === 'registered' || res.status === 'already_active') {
        setStatus('ok');
        setMessage(`Camera '${cameraId}' registered. Stream will appear in the grid.`);
        setTimeout(onClose, 1800);
      } else {
        throw new Error(JSON.stringify(res));
      }
    } catch (e) {
      setStatus('error');
      setMessage(String(e));
    }
  };

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 200,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)',
    }}>
      <div style={{
        background: '#161b22', border: '1px solid #30363d', borderRadius: 10,
        width: '100%', maxWidth: 460, margin: '0 16px', overflow: 'hidden',
        boxShadow: '0 20px 48px rgba(0,0,0,0.6)',
      }}>

        {/* Header */}
        <div style={{ display:'flex', alignItems:'flex-start', justifyContent:'space-between', padding:'14px 18px', borderBottom:'1px solid #21262d', background:'#0d1117' }}>
          <div>
            <p style={{ ...S, fontSize:9, color:'#484f58', textTransform:'uppercase', letterSpacing:'0.1em', marginBottom:3 }}>Add Camera</p>
            <h2 style={{ ...S, fontSize:13, fontWeight:700, color:'#e6edf3' }}>Register New Source</h2>
          </div>
          <button onClick={onClose} style={{ background:'none', border:'none', cursor:'pointer', color:'#484f58', fontSize:18, lineHeight:1, padding:'0 4px' }}>✕</button>
        </div>

        {/* Body */}
        <div style={{ padding:'18px', display:'flex', flexDirection:'column', gap:14 }}>

          {/* Type toggle */}
          <div>
            <label style={{ ...S, fontSize:9, color:'#484f58', textTransform:'uppercase', letterSpacing:'0.08em', display:'block', marginBottom:6 }}>Camera Type</label>
            <div style={{ display:'flex', gap:8 }}>
              {[['rtsp','📡 RTSP / IP Camera'],['client','💻 Client Laptop (push)']].map(([val, label]) => (
                <button key={val} onClick={() => setCamType(val)} style={{
                  flex:1, padding:'7px 0', borderRadius:6, cursor:'pointer',
                  ...S, fontSize:10, fontWeight:600,
                  background: camType === val ? 'rgba(56,139,253,0.1)' : '#0d1117',
                  color:      camType === val ? '#388bfd' : '#484f58',
                  border:     `1px solid ${camType === val ? 'rgba(56,139,253,0.35)' : '#21262d'}`,
                }}>{label}</button>
              ))}
            </div>
          </div>

          {/* Camera ID */}
          <div>
            <label style={{ ...S, fontSize:9, color:'#484f58', textTransform:'uppercase', letterSpacing:'0.08em', display:'block', marginBottom:6 }}>
              Camera ID <span style={{ color:'#30363d' }}>(unique string)</span>
            </label>
            <input type="text" value={cameraId} onChange={e => setCameraId(e.target.value)}
              placeholder="e.g. factory_floor_1 or laptop_mayukh"
              style={{ ...INPUT }}
            />
          </div>

          {/* RTSP URL */}
          {camType === 'rtsp' && (
            <div>
              <label style={{ ...S, fontSize:9, color:'#484f58', textTransform:'uppercase', letterSpacing:'0.08em', display:'block', marginBottom:6 }}>RTSP URL</label>
              <input type="text" value={rtspUrl} onChange={e => setRtspUrl(e.target.value)}
                placeholder="rtsp://admin:pass@192.168.1.64:554/stream1"
                style={{ ...INPUT }}
              />
              <p style={{ ...S, fontSize:9, color:'#30363d', marginTop:4 }}>Format: rtsp://[user:pass@]ip:port/path</p>
            </div>
          )}

          {/* Client push instructions */}
          {camType === 'client' && (
            <div style={{ background:'#0d1117', border:'1px solid #21262d', borderRadius:6, padding:'10px 12px' }}>
              <p style={{ ...S, fontSize:9, color:'#484f58', textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:6 }}>Client Setup</p>
              <p style={{ ...S, fontSize:10, color:'#8b949e', lineHeight:1.7 }}>
                After registering, run on the laptop:<br/>
                <code style={{ color:'#388bfd' }}>python client/client_v2.py --server http://SERVER:8000 --id {cameraId || 'YOUR_ID'}</code>
              </p>
            </div>
          )}

          {/* Method */}
          <div>
            <label style={{ ...S, fontSize:9, color:'#484f58', textTransform:'uppercase', letterSpacing:'0.08em', display:'block', marginBottom:6 }}>Detection Method</label>
            <select value={method} onChange={e => setMethod(e.target.value)} style={{ ...INPUT, cursor:'pointer' }}>
              {(availableMethods?.length ? availableMethods : ['fd','mog2','running_avg','custom','dl']).map(m => (
                <option key={m} value={m}>{m.toUpperCase()}</option>
              ))}
            </select>
          </div>

          {/* Status */}
          {message && (
            <p style={{
              ...S, fontSize:10, padding:'7px 10px', borderRadius:5,
              background: status==='ok' ? 'rgba(35,134,54,0.1)' : status==='error' ? 'rgba(218,54,51,0.1)' : '#161b22',
              color:      status==='ok' ? '#238636'              : status==='error' ? '#da3633'              : '#8b949e',
              border:    `1px solid ${status==='ok' ? 'rgba(35,134,54,0.2)' : status==='error' ? 'rgba(218,54,51,0.2)' : '#21262d'}`,
            }}>{message}</p>
          )}
        </div>

        {/* Footer */}
        <div style={{ padding:'12px 18px', borderTop:'1px solid #21262d', display:'flex', gap:8, justifyContent:'flex-end' }}>
          <button onClick={onClose} style={{
            padding:'6px 14px', borderRadius:5, cursor:'pointer',
            ...S, fontSize:10, color:'#484f58',
            background:'transparent', border:'1px solid #21262d',
          }}>Cancel</button>
          <button onClick={handleSubmit} disabled={status==='loading'} style={{
            padding:'6px 16px', borderRadius:5,
            cursor: status==='loading' ? 'not-allowed' : 'pointer',
            ...S, fontSize:10, fontWeight:700,
            background:'rgba(56,139,253,0.1)', color:'#388bfd',
            border:'1px solid rgba(56,139,253,0.3)',
            opacity: status==='loading' ? 0.5 : 1,
          }}>
            {status === 'loading' ? 'Registering…' : 'Register Camera'}
          </button>
        </div>
      </div>
    </div>
  );
}