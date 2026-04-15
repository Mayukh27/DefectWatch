// src/components/AddCameraModal.jsx
// Modal for adding RTSP or client-push cameras.
// Appears as a floating panel; does NOT redesign the existing dashboard layout.

import { useState } from 'react';
import { api } from '../utils/api';

const TYPE_LABELS = {
  rtsp:   'RTSP / IP Camera',
  client: 'Client Laptop (push)',
};

export default function AddCameraModal({ onClose, availableMethods }) {
  const [camType,   setCamType]   = useState('rtsp');
  const [cameraId,  setCameraId]  = useState('');
  const [rtspUrl,   setRtspUrl]   = useState('rtsp://');
  const [method,    setMethod]    = useState('custom');
  const [status,    setStatus]    = useState(null);   // null | 'loading' | 'ok' | 'error'
  const [message,   setMessage]   = useState('');

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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="bg-surface-800 border border-surface-500 rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-surface-600">
          <div>
            <p className="font-mono text-[10px] text-gray-500 uppercase tracking-widest">Add Camera</p>
            <h2 className="font-display text-sm font-bold text-white mt-0.5">Register New Source</h2>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300 transition-colors text-lg leading-none">✕</button>
        </div>

        {/* Body */}
        <div className="px-5 py-5 flex flex-col gap-4">

          {/* Type toggle */}
          <div>
            <label className="font-mono text-[10px] text-gray-500 uppercase tracking-wider block mb-1.5">Camera Type</label>
            <div className="flex gap-2">
              {Object.entries(TYPE_LABELS).map(([val, label]) => (
                <button
                  key={val}
                  onClick={() => setCamType(val)}
                  className={`
                    flex-1 py-2 rounded-lg font-mono text-xs font-semibold tracking-wider border transition-all
                    ${camType === val
                      ? 'bg-accent-cyan/15 text-accent-cyan border-accent-cyan/40'
                      : 'bg-surface-700 text-gray-500 border-surface-500 hover:border-gray-500'}
                  `}
                >
                  {val === 'rtsp' ? '📡' : '💻'} {label}
                </button>
              ))}
            </div>
          </div>

          {/* Camera ID */}
          <div>
            <label className="font-mono text-[10px] text-gray-500 uppercase tracking-wider block mb-1.5">
              Camera ID <span className="text-gray-600">(unique string)</span>
            </label>
            <input
              type="text"
              value={cameraId}
              onChange={e => setCameraId(e.target.value)}
              placeholder="e.g. factory_floor_1 or laptop_mayukh"
              className="
                w-full bg-surface-700 border border-surface-500 text-gray-200
                font-mono text-xs rounded-lg px-3 py-2 placeholder-gray-600
                focus:outline-none focus:border-accent-cyan/50 transition-colors
              "
            />
          </div>

          {/* RTSP URL (only for rtsp type) */}
          {camType === 'rtsp' && (
            <div>
              <label className="font-mono text-[10px] text-gray-500 uppercase tracking-wider block mb-1.5">
                RTSP URL
              </label>
              <input
                type="text"
                value={rtspUrl}
                onChange={e => setRtspUrl(e.target.value)}
                placeholder="rtsp://admin:pass@192.168.1.64:554/stream1"
                className="
                  w-full bg-surface-700 border border-surface-500 text-gray-200
                  font-mono text-xs rounded-lg px-3 py-2 placeholder-gray-600
                  focus:outline-none focus:border-accent-cyan/50 transition-colors
                "
              />
              <p className="font-mono text-[9px] text-gray-600 mt-1">
                Format: rtsp://[user:pass@]ip:port/path
              </p>
            </div>
          )}

          {/* Client push instructions */}
          {camType === 'client' && (
            <div className="bg-surface-700 border border-surface-500 rounded-lg px-3 py-3">
              <p className="font-mono text-[10px] text-gray-500 uppercase tracking-wider mb-2">Client Setup</p>
              <p className="font-mono text-[10px] text-gray-400 leading-relaxed">
                After registering, run on the laptop:<br/>
                <code className="text-accent-cyan">python client/client_v2.py --server http://SERVER:8000 --id {cameraId || 'YOUR_ID'}</code>
              </p>
            </div>
          )}

          {/* Method */}
          <div>
            <label className="font-mono text-[10px] text-gray-500 uppercase tracking-wider block mb-1.5">Detection Method</label>
            <select
              value={method}
              onChange={e => setMethod(e.target.value)}
              className="w-full bg-surface-700 border border-surface-500 text-gray-300 font-mono text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-accent-cyan/50 transition-colors"
            >
              {(availableMethods.length ? availableMethods : ['fd','mog2','running_avg','custom','dl']).map(m => (
                <option key={m} value={m}>{m.toUpperCase()}</option>
              ))}
            </select>
          </div>

          {/* Status message */}
          {message && (
            <p className={`font-mono text-xs px-3 py-2 rounded-lg ${
              status === 'ok'    ? 'bg-accent-green/10 text-accent-green border border-accent-green/20' :
              status === 'error' ? 'bg-accent-red/10   text-accent-red   border border-accent-red/20'   :
              'bg-surface-700 text-gray-400'
            }`}>
              {message}
            </p>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-surface-600 flex gap-3 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg font-mono text-xs text-gray-500 hover:text-gray-300 border border-surface-500 hover:border-gray-500 transition-all"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={status === 'loading'}
            className="
              px-5 py-2 rounded-lg font-mono text-xs font-bold tracking-wider
              bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/30
              hover:bg-accent-cyan/25 active:scale-95 transition-all
              disabled:opacity-40 disabled:cursor-not-allowed
            "
          >
            {status === 'loading' ? 'Registering…' : 'Register Camera'}
          </button>
        </div>
      </div>
    </div>
  );
}
