import { useState, useEffect, useRef, useCallback } from 'react'
import DetectionPanel from './DetectionPanel.jsx'

const BBOX_COLORS = [
  '#00c896', '#ff6b35', '#6366f1', '#ec4899',
  '#eab308', '#06b6d4', '#ef4444', '#a855f7',
]
const colorFor = (id) => BBOX_COLORS[id % BBOX_COLORS.length]

const FRUIT_EMOJI = {
  Apple:'🍎',Banana:'🍌',Orange:'🍊',Mango:'🥭',Strawberry:'🍓',
  Grape:'🍇',Pineapple:'🍍',Watermelon:'🍉',Lemon:'🍋',Cherry:'🍒',
  Peach:'🍑',Pear:'🍐',Avocado:'🥑',Blueberry:'🫐',Tomato:'🍅',
  Carrot:'🥕',Potato:'🥔',Corn:'🌽',Pepper:'🫑',Cucumber:'🥒',
  Raspberry:'🫐',Eggplant:'🍆',
}
const emoji = (name) => FRUIT_EMOJI[name] || '🍑'

const INTERVAL_MS = 250  // ms entre llamadas al backend

export default function CameraMode({ api, toast }) {
  const videoRef   = useRef(null)
  const canvasRef  = useRef(null)
  const streamRef  = useRef(null)
  const timerRef   = useRef(null)
  const processingRef = useRef(false)

  const [cameras, setCameras]       = useState([])
  const [selectedCam, setSelectedCam] = useState('')
  const [running, setRunning]       = useState(false)
  const [detections, setDetections] = useState([])
  const [liveResult, setLiveResult] = useState(null)
  const [history, setHistory]       = useState([])
  const [preloading, setPreloading] = useState(false)

  // Enumerar cámaras
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then(devs => {
      const cams = devs.filter(d => d.kind === 'videoinput')
      setCameras(cams)
      if (cams.length) setSelectedCam(cams[0].deviceId)
    }).catch(() => {})
  }, [])

  // Dibujar frame + bboxes en canvas
  const drawFrame = useCallback((dets) => {
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    const ctx = canvas.getContext('2d')
    canvas.width  = video.videoWidth  || 640
    canvas.height = video.videoHeight || 480
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    dets.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox
      const noFruta = det.fruta === 'No es fruta' || !det.clasificado

      // Para objetos no-fruta: solo una línea gris tenue, sin etiqueta
      if (det.fruta === 'No es fruta') {
        ctx.save()
        ctx.strokeStyle = 'rgba(120,130,150,0.35)'
        ctx.lineWidth = 1.5
        ctx.setLineDash([6, 4])
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
        ctx.restore()
        return
      }

      const color = colorFor(det.id)
      const label = det.clasificado
        ? `[${det.id}] ${det.fruta}  ${(det.confianza * 100).toFixed(0)}%`
        : `[${det.id}] Detectando...`
      const price = det.precio ? `$${Number(det.precio).toLocaleString('es-CO')} COP` : ''

      // Bounding box
      ctx.strokeStyle = color
      ctx.lineWidth = 2.5
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

      // Esquinas decorativas
      const cs = 14
      ctx.lineWidth = 4
      ;[[x1,y1,1,1],[x2,y1,-1,1],[x1,y2,1,-1],[x2,y2,-1,-1]].forEach(([cx,cy,dx,dy]) => {
        ctx.beginPath(); ctx.moveTo(cx, cy + dy*cs); ctx.lineTo(cx, cy); ctx.lineTo(cx + dx*cs, cy); ctx.stroke()
      })

      // Label background
      ctx.font = 'bold 13px Inter, sans-serif'
      const tw = ctx.measureText(label).width
      const bh = price ? 42 : 24
      const by = y1 > bh + 4 ? y1 - bh - 2 : y2 + 4

      ctx.fillStyle = color
      ctx.globalAlpha = 0.92
      ctx.beginPath()
      ctx.roundRect(x1, by, Math.max(tw + 18, price ? ctx.measureText(price).width + 18 : 0), bh, 5)
      ctx.fill()
      ctx.globalAlpha = 1

      // Label text
      ctx.fillStyle = '#000'
      ctx.fillText(label, x1 + 8, by + 16)
      if (price) {
        ctx.font = '11px Inter, sans-serif'
        ctx.fillText(price, x1 + 8, by + 33)
      }
    })
  }, [])

  // Loop de captura + inferencia
  const startInference = useCallback(() => {
    timerRef.current = setInterval(async () => {
      if (processingRef.current) return
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || video.readyState < 2) return

      processingRef.current = true
      try {
        // Capturar frame
        const off = document.createElement('canvas')
        off.width  = video.videoWidth  || 640
        off.height = video.videoHeight || 480
        off.getContext('2d').drawImage(video, 0, 0, off.width, off.height)
        const b64 = off.toDataURL('image/jpeg', 0.75).split(',')[1]

        const res = await fetch(`${api}/api/detect-frame`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ imagen: b64 }),
        })
        if (!res.ok) return
        const data = await res.json()
        const dets = data.detections || []

        // Dibujar (incluye "No es fruta" pero con estilo diferente)
        drawFrame(dets)

        // Solo las clasificadas como fruta real para el panel y live result
        const classified = dets.filter(d => d.clasificado && d.fruta !== 'No es fruta' && d.fruta !== 'Detectando...')
        setDetections(dets.filter(d => d.fruta !== 'No es fruta'))

        if (classified.length) {
          const best = classified.reduce((a, b) => b.confianza > a.confianza ? b : a)
          setLiveResult(best)
          setHistory(h => {
            if (h.length && h[0].fruta === best.fruta) return h
            return [best, ...h].slice(0, 12)
          })
        }
      } catch { /* silencioso */ } finally {
        processingRef.current = false
      }
    }, INTERVAL_MS)
  }, [api, drawFrame])

  // Iniciar cámara
  const startCamera = async () => {
    try {
      // Precargar detector si no está cargado
      setPreloading(true)
      try { await fetch(`${api}/api/camera/preload`, { method: 'POST' }) } catch {}
      setPreloading(false)

      // Reset tracker
      await fetch(`${api}/api/camera/reset`, { method: 'POST' }).catch(() => {})

      const constraints = { video: selectedCam ? { deviceId: { exact: selectedCam } } : true }
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      videoRef.current.srcObject = stream
      await videoRef.current.play()
      setRunning(true)
      setDetections([])
      setHistory([])
      startInference()
      toast('📷 Cámara iniciada', 'success')
    } catch (e) {
      setPreloading(false)
      toast(`Error: ${e.message}`, 'error')
    }
  }

  // Detener cámara
  const stopCamera = () => {
    clearInterval(timerRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    setRunning(false)
    setLiveResult(null)
    // Limpiar canvas
    const ctx = canvasRef.current?.getContext('2d')
    if (ctx && canvasRef.current) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
  }

  useEffect(() => () => { clearInterval(timerRef.current); streamRef.current?.getTracks().forEach(t => t.stop()) }, [])

  return (
    <div className="cam-layout">
      {/* Viewport */}
      <div>
        <div className="cam-viewport">
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={canvasRef} />
          {!running && (
            <div className="cam-placeholder">
              <div className="icon">📷</div>
              <p>Selecciona una cámara y presiona Iniciar</p>
            </div>
          )}
        </div>
        <div className="cam-controls">
          <select
            className="cam-select"
            value={selectedCam}
            onChange={e => setSelectedCam(e.target.value)}
            disabled={running}
          >
            {cameras.length === 0 && <option value="">Sin cámaras detectadas</option>}
            {cameras.map((c, i) => (
              <option key={c.deviceId} value={c.deviceId}>
                {c.label || `Cámara ${i + 1}`}
              </option>
            ))}
          </select>
          {!running ? (
            <button className="btn btn-primary" onClick={startCamera} disabled={preloading}>
              {preloading ? <><span className="spinner" />Cargando modelo…</> : '▶ Iniciar'}
            </button>
          ) : (
            <button className="btn btn-danger" onClick={stopCamera}>⏹ Detener</button>
          )}
        </div>
      </div>

      {/* Panel derecho */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {/* Resultado en vivo */}
        <div className="card live-card">
          <h3>Clasificación en Vivo</h3>
          {liveResult ? (
            <>
              <div className="live-fruit" style={{ color: colorFor(liveResult.id) }}>
                {emoji(liveResult.fruta)} {liveResult.fruta}
              </div>
              <div className="live-conf-text">
                Confianza: {(liveResult.confianza * 100).toFixed(1)}%
                &nbsp;·&nbsp; ID: {liveResult.id}
              </div>
              <div className="live-bar-track">
                <div className="live-bar-fill" style={{ width: `${liveResult.confianza * 100}%` }} />
              </div>
              <div className="live-price">
                {liveResult.precio ? `$${Number(liveResult.precio).toLocaleString('es-CO')} COP` : '—'}
              </div>
            </>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '.85rem' }}>
              {running ? 'Buscando frutas…' : 'Inicia la cámara para clasificar'}
            </div>
          )}
          <div className="live-dot-row">
            <div className={`live-dot ${running ? 'active' : ''}`} />
            <span>{running ? `${detections.length} objeto(s) en pantalla` : 'Inactivo'}</span>
          </div>

          {/* Historial */}
          {history.length > 0 && (
            <div className="live-history">
              <div className="live-history-title">Historial</div>
              <div className="history-list">
                {history.map((h, i) => (
                  <div key={i} className="history-item">
                    <span>{emoji(h.fruta)}</span>
                    <span className="hfruit">{h.fruta}</span>
                    <span className="hconf">{(h.confianza * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Panel de detecciones activas */}
        <DetectionPanel detections={detections} title="Objetos en pantalla" />
      </div>
    </div>
  )
}
