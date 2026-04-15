// src/components/AlertLog.jsx
import { useState, useEffect, useRef } from 'react';

export default function AlertLog({ cameras }) {
  const [log, setLog] = useState([]);
  const prevDefects = useRef({});

  useEffect(() => {
    cameras.forEach(cam => {
      const was = prevDefects.current[cam.id] ?? false;
      if (cam.defect && cam.active && !was) {
        setLog(prev => [{
          key:      Date.now() + String(cam.id),
          cameraId: cam.id,
          method:   cam.method,
          time:     new Date().toLocaleTimeString('en-GB', { hour12: false }),
        }, ...prev.slice(0, 49)]);
      }
      prevDefects.current[cam.id] = cam.defect && cam.active;
    });
  }, [cameras]);

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: '#0d1117', borderTop: '1px solid #21262d',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '6px 14px', borderBottom: '1px solid #21262d', flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#9e6a03', display: 'inline-block' }} />
          <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#8b949e', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600 }}>
            Event Log
          </span>
        </div>
        {log.length > 0 && (
          <button onClick={() => setLog([])} style={{
            fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58',
            background: 'none', border: 'none', cursor: 'pointer',
          }}>clear</button>
        )}
      </div>

      {/* Events */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '6px 10px' }}>
        {log.length === 0 ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#30363d', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              No events
            </span>
          </div>
        ) : log.map(e => (
          <div key={e.key} style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '4px 8px', marginBottom: 3, borderRadius: 4,
            background: 'rgba(218,54,51,0.05)',
            border: '1px solid rgba(218,54,51,0.12)',
            animation: 'slideUp 0.2s ease-out',
          }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#da3633', flexShrink: 0, display: 'inline-block' }} />
            <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58', flexShrink: 0 }}>{e.time}</span>
            <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#8b949e' }}>
              CAM {String(e.cameraId).padStart(2, '0')}
            </span>
            <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#da3633', marginLeft: 'auto', flexShrink: 0 }}>
              [{e.method?.toUpperCase()}]
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
