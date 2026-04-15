// src/components/Sidebar.jsx  — RIGHT side
// Clicking a camera row opens the AlertPopup (Before/After).

import { useState } from 'react';

const DOT = ({ color }) => (
  <span style={{
    width: 7, height: 7, borderRadius: '50%',
    background: color, display: 'inline-block', flexShrink: 0,
  }} />
);

export default function Sidebar({ cameras, backendOnline, onStartAll, onStopAll, onAddCamera, onCameraClick }) {
  const activeCams = cameras.filter(c => c.active).length;
  const defectCams = cameras.filter(c => c.defect && c.active).length;
  const [collapsed, setCollapsed] = useState(false);
  const W = collapsed ? 44 : 220;

  return (
    <aside style={{
      width: W, flexShrink: 0,
      background: '#0d1117',
      borderLeft: '1px solid #21262d',   // LEFT border because sidebar is on RIGHT
      display: 'flex', flexDirection: 'column',
      transition: 'width 0.2s ease',
      overflow: 'hidden',
    }}>

      {/* Collapse toggle */}
      <div style={{
        height: 40, display: 'flex', alignItems: 'center',
        justifyContent: collapsed ? 'center' : 'flex-start',
        padding: '0 10px', borderBottom: '1px solid #21262d',
      }}>
        <button onClick={() => setCollapsed(p => !p)} style={{
          background: 'none', border: 'none', cursor: 'pointer',
          color: '#484f58', fontSize: 14, padding: 4, borderRadius: 4,
        }}>
          {collapsed ? '‹' : '›'}   {/* flipped arrows for right sidebar */}
        </button>
      </div>

      {!collapsed && (
        <>
          {/* Stats */}
          <div style={{ padding: '12px 12px', borderBottom: '1px solid #21262d', display: 'flex', gap: 8 }}>
            <StatBox label="Active"  value={activeCams} color="#e6edf3" />
            <StatBox label="Defects" value={defectCams} color={defectCams > 0 ? '#da3633' : '#484f58'}
              border={defectCams > 0 ? 'rgba(218,54,51,0.3)' : '#21262d'} />
          </div>

          {/* Backend status */}
          <div style={{ padding: '8px 14px', borderBottom: '1px solid #21262d' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <DOT color={backendOnline ? '#238636' : '#da3633'} />
              <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 10, color: '#8b949e' }}>
                {backendOnline ? 'API online' : 'API offline'}
              </span>
            </div>
          </div>

          {/* Camera list — clicking opens Before/After popup */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 10px 0' }}>
            <div style={{
              fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58',
              textTransform: 'uppercase', letterSpacing: '0.1em',
              marginBottom: 8, paddingLeft: 4,
            }}>
              Cameras — click to inspect
            </div>

            {cameras.map(cam => {
              const dotColor = !cam.active ? '#30363d' : cam.defect ? '#da3633' : '#238636';
              return (
                <div key={cam.id}
                  onClick={() => onCameraClick && onCameraClick(cam)}
                  title="Click to see Before / After"
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '7px 8px', marginBottom: 4, borderRadius: 6,
                    cursor: 'pointer',
                    background: cam.defect && cam.active ? 'rgba(218,54,51,0.06)' : '#161b22',
                    border: `1px solid ${cam.defect && cam.active ? 'rgba(218,54,51,0.2)' : '#21262d'}`,
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = cam.defect && cam.active
                    ? 'rgba(218,54,51,0.1)' : '#1c2128'}
                  onMouseLeave={e => e.currentTarget.style.background = cam.defect && cam.active
                    ? 'rgba(218,54,51,0.06)' : '#161b22'}
                >
                  <DOT color={dotColor} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 11, fontWeight: 500, color: '#c9d1d9' }}>
                      CAM {String(cam.id).padStart(2, '0')}
                    </div>
                    <div style={{
                      fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58',
                      marginTop: 1, textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap',
                    }}>
                      {cam.active ? (cam.method || 'custom') : 'idle'}
                    </div>
                  </div>
                  {cam.active && cam.fps > 0 && (
                    <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58', flexShrink: 0 }}>
                      {cam.fps}
                    </span>
                  )}
                  {cam.defect && cam.active && (
                    <span style={{
                      fontFamily: 'IBM Plex Mono', fontSize: 8, color: '#da3633',
                      fontWeight: 700, flexShrink: 0,
                    }}>●</span>
                  )}
                </div>
              );
            })}
          </div>

          {/* Controls */}
          <div style={{ padding: '10px', borderTop: '1px solid #21262d', display: 'flex', flexDirection: 'column', gap: 6 }}>
            <SideBtn onClick={onStartAll} color="#238636">▶ Start All</SideBtn>
            <SideBtn onClick={onStopAll}  color="#da3633">■ Stop All</SideBtn>
            <SideBtn onClick={onAddCamera} color="#388bfd">+ Add Camera</SideBtn>
          </div>
        </>
      )}

      {/* Collapsed dots */}
      {collapsed && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, paddingTop: 12 }}>
          {cameras.map(cam => (
            <div key={cam.id} title={`CAM ${cam.id}`}
              onClick={() => onCameraClick && onCameraClick(cam)}
              style={{ cursor: 'pointer' }}>
              <DOT color={!cam.active ? '#30363d' : cam.defect ? '#da3633' : '#238636'} />
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}

function StatBox({ label, value, color, border = '#21262d' }) {
  return (
    <div style={{
      flex: 1, background: '#161b22', borderRadius: 6, padding: '8px 10px',
      border: `1px solid ${border}`,
    }}>
      <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 18, fontWeight: 600, color, lineHeight: 1 }}>{value}</div>
      <div style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#484f58', marginTop: 3, textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</div>
    </div>
  );
}

function SideBtn({ onClick, color, children }) {
  return (
    <button onClick={onClick} style={{
      width: '100%', padding: '7px 0', borderRadius: 6, cursor: 'pointer',
      fontFamily: 'IBM Plex Mono', fontSize: 11, fontWeight: 600, letterSpacing: '0.04em',
      background: color + '10', color, border: `1px solid ${color}28`,
      transition: 'background 0.15s',
    }}
      onMouseEnter={e => e.currentTarget.style.background = color + '22'}
      onMouseLeave={e => e.currentTarget.style.background = color + '10'}
    >
      {children}
    </button>
  );
}
