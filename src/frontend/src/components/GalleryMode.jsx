import { useState, useEffect } from 'react'

const fmtMB = (b) => `${(b / 1048576).toFixed(1)} MB`
const fmtDate = (ts) => new Date(ts * 1000).toLocaleString('es-CO', {
  day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
})

export default function GalleryMode({ api, toast }) {
  const [videos, setVideos]   = useState([])
  const [loading, setLoading] = useState(true)
  const [modal, setModal]     = useState(null)  // { src, name }

  const load = async () => {
    setLoading(true)
    try {
      const r = await fetch(`${api}/api/videos`)
      const d = await r.json()
      setVideos(d.videos || [])
    } catch {
      toast('Error cargando galería', 'error')
    } finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  return (
    <>
      <div>
        <div className="section-header">
          <div className="section-title">🎬 Videos Procesados</div>
          <button className="btn btn-ghost btn-sm" onClick={load}>↺ Actualizar</button>
        </div>

        {loading ? (
          <div className="empty"><div className="spinner" style={{margin:'0 auto 1rem'}} /><p>Cargando…</p></div>
        ) : videos.length === 0 ? (
          <div className="empty">
            <div className="ei">📭</div>
            <p>No hay videos procesados todavía.<br/>Sube un video en la pestaña "Subir Video".</p>
          </div>
        ) : (
          <div className="gallery-grid">
            {videos.map(v => (
              <div key={v.filename} className="gallery-item" onClick={() => setModal({ src: `${api}/api/videos/${v.filename}`, name: v.filename })}>
                <div className="gallery-thumb">🎬</div>
                <div className="gallery-meta">
                  <div className="gallery-name">{v.filename}</div>
                  <div>{fmtMB(v.size_mb * 1048576)} · {fmtDate(v.created_at)}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Modal */}
      {modal && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{modal.name}</h3>
              <button className="modal-close" onClick={() => setModal(null)}>✕</button>
            </div>
            <video key={modal.src} controls autoPlay src={modal.src} />
          </div>
        </div>
      )}
    </>
  )
}
