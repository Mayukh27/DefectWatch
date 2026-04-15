// src/components/TopBar.jsx
import { useState, useEffect } from 'react';

export default function TopBar({ backendOnline, totalDefects }) {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <header style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 20px', height: 48,
      background: '#0d1117',
      borderBottom: '1px solid #21262d',
      flexShrink: 0,
    }}>
      {/* Brand */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <rect x="1" y="1" width="16" height="16" rx="2" stroke="#388bfd" strokeWidth="1.2"/>
          <circle cx="9" cy="9" r="3.5" stroke="#388bfd" strokeWidth="1.2"/>
          <circle cx="9" cy="9" r="1.2" fill="#388bfd"/>
        </svg>
        <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 13, fontWeight: 600, color: '#e6edf3', letterSpacing: '0.02em' }}>
          DefectWatch
        </span>
        <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#484f58', letterSpacing: '0.08em' }}>
          v2.3
        </span>
      </div>

      {/* Defect alert — only when active */}
      {totalDefects > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '3px 10px', borderRadius: 4,
          background: 'rgba(218,54,51,0.08)',
          border: '1px solid rgba(218,54,51,0.25)',
        }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#da3633', display: 'inline-block' }} />
          <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#da3633', fontWeight: 600, letterSpacing: '0.05em' }}>
            {totalDefects} DEFECT{totalDefects > 1 ? 'S' : ''}
          </span>
        </div>
      )}

      {/* Right: status + clock */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%',
            background: backendOnline ? '#238636' : '#da3633',
            display: 'inline-block',
          }} />
          <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 11, color: '#8b949e' }}>
            {backendOnline ? 'API online' : 'API offline'}
          </span>
        </div>
        <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 12, color: '#8b949e', tabularNums: true }}>
          {time.toLocaleTimeString('en-GB', { hour12: false })}
        </span>
      </div>
    </header>
  );
}
