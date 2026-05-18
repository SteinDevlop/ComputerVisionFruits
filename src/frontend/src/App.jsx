import { useState, useEffect, useRef } from 'react'
import CameraMode from './components/CameraMode.jsx'
import VideoMode from './components/VideoMode.jsx'
import GalleryMode from './components/GalleryMode.jsx'

const API = ''  // vacío → vite proxy en dev, nginx en docker

export default function App() {
  const [tab, setTab] = useState('camera')
  const [systemStatus, setSystemStatus] = useState({ dot: 'offline', text: 'Verificando…' })
  const [toasts, setToasts] = useState([])

  // Health check cada 15s
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API}/api/health`)
        const d = await r.json()
        const ok = d.classifier_api === 'ok'
        setSystemStatus({
          dot: ok ? 'online' : 'offline',
          text: ok
            ? `Sistema operativo · Detector: ${d.detector_loaded ? 'cargado' : 'no cargado'}`
            : 'Classifier no disponible',
        })
      } catch {
        setSystemStatus({ dot: 'offline', text: 'Backend no disponible (puerto 8080)' })
      }
    }
    check()
    const t = setInterval(check, 15000)
    return () => clearInterval(t)
  }, [])

  // Toast helper global
  const addToast = (msg, type = 'info') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, msg, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000)
  }

  const TABS = [
    { id: 'camera', label: '📷 Cámara en Vivo' },
    { id: 'video',  label: '🎬 Subir Video' },
    { id: 'gallery',label: '📂 Historial' },
  ]

  return (
    <div className="app">
      {/* Navbar */}
      <nav className="navbar">
        <div className="nav-logo">
          <span className="icon">🍎</span>
          <span>Fruit<span className="accent">Vision</span> AI</span>
        </div>
        <div className="nav-status">
          <div className={`status-dot ${systemStatus.dot}`} />
          <span>{systemStatus.text}</span>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero">
        <div className="hero-badge">⚡ YOLO + EfficientNet-B0 · Tiempo Real</div>
        <h1>
          Detección Inteligente de{' '}
          <span className="gradient">Frutas y Verduras</span>
        </h1>
        <p>Conecta tu cámara o sube un video para detectar y clasificar frutas con IA.</p>
      </section>

      {/* Tabs */}
      <div className="tabs-row">
        <div className="tabs">
          {TABS.map(t => (
            <button
              key={t.id}
              className={`tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <main className="main">
        {tab === 'camera'  && <CameraMode api={API} toast={addToast} />}
        {tab === 'video'   && <VideoMode  api={API} toast={addToast} />}
        {tab === 'gallery' && <GalleryMode api={API} toast={addToast} />}
      </main>

      {/* Toasts */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast ${t.type}`}>
            <span>{t.type === 'success' ? '✅' : t.type === 'error' ? '❌' : 'ℹ️'}</span>
            <span>{t.msg}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
