import { useState, useRef, useEffect } from 'react'
import DetectionPanel from './DetectionPanel.jsx'

const FRUIT_EMOJI = {
  Apple:'🍎',Banana:'🍌',Orange:'🍊',Mango:'🥭',Strawberry:'🍓',
  Grape:'🍇',Pineapple:'🍍',Watermelon:'🍉',Lemon:'🍋',Cherry:'🍒',
  Peach:'🍑',Pear:'🍐',Avocado:'🥑',Blueberry:'🫐',Tomato:'🍅',
  Carrot:'🥕',Potato:'🥔',Corn:'🌽',Pepper:'🫑',Cucumber:'🥒',
}
const emoji = (n) => FRUIT_EMOJI[n] || '🍑'
const fmt   = (n) => Number(n).toLocaleString('es-CO')
const fmtMB = (b) => b < 1048576 ? `${(b/1024).toFixed(0)} KB` : `${(b/1048576).toFixed(1)} MB`

const STATUS_LABELS = {
  pending:    '⏳ En cola…',
  loading:    '🔄 Cargando modelos de IA…',
  processing: '🔍 Detectando y clasificando frutas…',
  converting: '🎬 Convirtiendo a H.264 para el navegador…',
  done:       '✅ Procesamiento completado',
  error:      '❌ Error en el procesamiento',
}

export default function VideoMode({ api, toast }) {
  const [file,       setFile]       = useState(null)
  const [jobId,      setJobId]      = useState(null)
  const [job,        setJob]        = useState(null)
  const [processing, setProcessing] = useState(false)
  const [done,       setDone]       = useState(false)
  const [dragging,   setDragging]   = useState(false)
  const wsRef       = useRef(null)
  const fileInputRef = useRef(null)

  const handleFile = (f) => {
    if (!f) return
    const ext = f.name.split('.').pop().toLowerCase()
    if (!['mp4','avi','mov','mkv','webm'].includes(ext)) {
      toast(`Formato no soportado: .${ext}`, 'error')
      return
    }
    setFile(f)
    setJob(null)
    setDone(false)
    setJobId(null)
  }

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const processVideo = async () => {
    if (!file) return
    setProcessing(true)
    setDone(false)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${api}/api/process-video`, { method: 'POST', body: form })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail) }
      const { job_id } = await res.json()
      setJobId(job_id)
      startWebSocket(job_id)
      toast('Video enviado — procesando…', 'info')
    } catch (e) {
      toast(`Error: ${e.message}`, 'error')
      setProcessing(false)
    }
  }

  const startWebSocket = (jid) => {
    // Construir URL WS correctamente tanto en dev (Vite proxy) como en Docker (nginx)
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${proto}//${window.location.host}/ws/${jid}`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws
    ws.onmessage = (ev) => {
      const j = JSON.parse(ev.data)
      setJob(j)
      if (j.status === 'done')  { setDone(true); setProcessing(false); toast('¡Video procesado! 🎉', 'success'); ws.close() }
      if (j.status === 'error') { setProcessing(false); toast(`Error: ${j.error || 'Error desconocido'}`, 'error'); ws.close() }
    }
    ws.onerror = () => { ws.close(); startPolling(jid) }
    ws.onclose = (e) => {
      // Si se cerró antes de done/error, cambiar a polling
      if (!done && processing) startPolling(jid)
    }
  }

  const startPolling = (jid) => {
    const t = setInterval(async () => {
      const r = await fetch(`${api}/api/job/${jid}`)
      const j = await r.json()
      setJob(j)
      if (j.status === 'done')  { setDone(true); setProcessing(false); clearInterval(t); toast('¡Video procesado! 🎉','success') }
      if (j.status === 'error') { setProcessing(false); clearInterval(t); toast(`Error: ${j.error}`, 'error') }
    }, 800)
  }

  useEffect(() => () => wsRef.current?.close(), [])

  const pct   = job?.progress || 0
  const dets  = job?.detections || []
  const avgConf = dets.length ? (dets.reduce((s,d)=>s+d.confianza,0)/dets.length*100).toFixed(0)+'%' : '—'
  const total$ = dets.reduce((s,d)=>s+(d.precio||0),0)

  return (
    <div className="video-layout">
      <div>
        {/* Drop zone */}
        {!done && (
          <div
            className={`dropzone ${dragging ? 'over' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => !processing && fileInputRef.current?.click()}
          >
            <div className="dz-icon">🎥</div>
            <h3>Arrastra tu video aquí</h3>
            <p>MP4, AVI, MOV, MKV, WebM</p>
            <button className="btn btn-primary" type="button" onClick={e => { e.stopPropagation(); fileInputRef.current?.click() }}>
              Seleccionar archivo
            </button>
            <input ref={fileInputRef} type="file" accept=".mp4,.avi,.mov,.mkv,.webm" style={{display:'none'}} onChange={e => handleFile(e.target.files[0])} />
            {file && (
              <div className="file-badge">
                <span>📄</span>
                <div>
                  <div className="fname">{file.name}</div>
                  <div className="fsize">{fmtMB(file.size)}</div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Botón procesar */}
        {file && !done && (
          <div style={{ textAlign: 'center', marginTop: '1rem' }}>
            <button className="btn btn-primary" onClick={processVideo} disabled={processing}>
              {processing ? <><span className="spinner" />Procesando…</> : '🚀 Procesar Video'}
            </button>
          </div>
        )}

        {/* Progreso */}
        {processing && job && (
          <div className="progress-box">
            <div className="progress-header">
              <span className="progress-label">{STATUS_LABELS[job.status] || job.status}</span>
              <span className="progress-pct">{pct}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="progress-meta">
              <span>Frame: <strong>{job.current_frame || 0}</strong></span>
              <span>Total: <strong>{job.total_frames || '—'}</strong></span>
              <span>Detectadas: <strong>{job.objects_classified || 0}</strong></span>
            </div>
          </div>
        )}

        {/* Video resultado */}
        {done && job?.job_id && (
          <>
            <div className="video-player">
              <div className="video-player-header">
                <h3>🎬 Video Procesado</h3>
                <a
                  className="btn btn-ghost btn-sm"
                  href={`${api}/api/job/${job.job_id}/download`}
                  download
                >
                  ⬇ Descargar
                </a>
              </div>
              <video
                key={job.job_id}
                controls
                autoPlay
                src={`${api}/api/job/${job.job_id}/stream`}
              />
            </div>

            {/* Stats */}
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-icon">🍎</div>
                <div className="stat-value">{dets.length}</div>
                <div className="stat-label">Frutas únicas</div>
              </div>
              <div className="stat-card">
                <div className="stat-icon">🎯</div>
                <div className="stat-value">{avgConf}</div>
                <div className="stat-label">Confianza prom.</div>
              </div>
              <div className="stat-card">
                <div className="stat-icon">💰</div>
                <div className="stat-value">${fmt(total$)}</div>
                <div className="stat-label">Precio total COP</div>
              </div>
              <div className="stat-card">
                <div className="stat-icon">📊</div>
                <div className="stat-value">{job.total_frames || 0}</div>
                <div className="stat-label">Frames proc.</div>
              </div>
            </div>

            {/* Botón nuevo video */}
            <div style={{ marginTop: '1rem' }}>
              <button className="btn btn-ghost" onClick={() => { setFile(null); setDone(false); setJob(null); setJobId(null) }}>
                + Procesar otro video
              </button>
            </div>
          </>
        )}
      </div>

      {/* Panel de detecciones */}
      <div>
        <DetectionPanel
          detections={dets}
          title={done ? 'Frutas detectadas' : 'Esperando resultados…'}
        />
      </div>
    </div>
  )
}
