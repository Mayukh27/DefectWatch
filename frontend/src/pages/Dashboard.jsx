// src/pages/Dashboard.jsx
// Sidebar is on the RIGHT. Clicking a camera row opens the Before/After popup.
import { useCallback, useState } from 'react';
import CameraCard     from '../components/CameraCard';
import Sidebar        from '../components/Sidebar';
import TopBar         from '../components/TopBar';
import AlertLog       from '../components/AlertLog';
import AlertPopup     from '../components/AlertPopup';
import AddCameraModal from '../components/AddCameraModal';
import { useDetection } from '../hooks/useDetection';

const NUM_CAMERAS = 4;

export default function Dashboard() {
  const { cameras, availableMethods, backendOnline, startCamera, stopCamera } = useDetection(NUM_CAMERAS);

  const totalDefects = cameras.filter(c => c.defect && c.active).length;
  const [showAdd,  setShowAdd]  = useState(false);
  const [alertCam, setAlertCam] = useState(null);

  const handleStartAll = useCallback(() => {
    cameras.forEach(cam => { if (!cam.active) startCamera(cam.id, cam.method); });
  }, [cameras, startCamera]);

  const handleStopAll = useCallback(() => {
    cameras.forEach(cam => { if (cam.active) stopCamera(cam.id); });
  }, [cameras, stopCamera]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden', background: '#0a0c0e' }}>
      {showAdd   && <AddCameraModal onClose={() => setShowAdd(false)} availableMethods={availableMethods} />}
      {alertCam  && <AlertPopup camera={alertCam} onClose={() => setAlertCam(null)} />}

      <TopBar backendOnline={backendOnline} totalDefects={totalDefects} />

      {/* Main row — grid LEFT, sidebar RIGHT */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>

        {/* Camera grid + event log */}
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0, overflow: 'hidden' }}>
          <main style={{ flex: 1, overflowY: 'auto', padding: 14 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#30363d', textTransform: 'uppercase', letterSpacing: '0.12em' }}>
                Live feeds
              </span>
              <div style={{ flex: 1, height: 1, background: '#21262d' }} />
              <span style={{ fontFamily: 'IBM Plex Mono', fontSize: 9, color: '#30363d' }}>
                {cameras.filter(c => c.active).length}/{cameras.length} active
              </span>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: 10,
            }}>
              {cameras.map(cam => (
                <CameraCard key={cam.id} camera={cam}
                  onStart={startCamera} onStop={stopCamera}
                  availableMethods={availableMethods} />
              ))}
            </div>
          </main>

          <div style={{ height: 120, flexShrink: 0 }}>
            <AlertLog cameras={cameras} />
          </div>
        </div>

        {/* Sidebar — RIGHT */}
        <Sidebar
          cameras={cameras}
          backendOnline={backendOnline}
          onStartAll={handleStartAll}
          onStopAll={handleStopAll}
          onAddCamera={() => setShowAdd(true)}
          onCameraClick={cam => setAlertCam(cam)}
        />
      </div>
    </div>
  );
}
