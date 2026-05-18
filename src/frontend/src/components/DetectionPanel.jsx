const FRUIT_EMOJI = {
  Apple:'🍎',Banana:'🍌',Orange:'🍊',Mango:'🥭',Strawberry:'🍓',
  Grape:'🍇',Pineapple:'🍍',Watermelon:'🍉',Lemon:'🍋',Cherry:'🍒',
  Peach:'🍑',Pear:'🍐',Avocado:'🥑',Blueberry:'🫐',Tomato:'🍅',
  Carrot:'🥕',Potato:'🥔',Corn:'🌽',Pepper:'🫑',Cucumber:'🥒',
  Raspberry:'🫐',Eggplant:'🍆',
}
const emoji = (n) => FRUIT_EMOJI[n] || '🍑'

const BBOX_COLORS = [
  '#00c896','#ff6b35','#6366f1','#ec4899',
  '#eab308','#06b6d4','#ef4444','#a855f7',
]
const colorFor = (id) => BBOX_COLORS[typeof id === 'number' ? id % BBOX_COLORS.length : 0]

export default function DetectionPanel({ detections = [], title = 'Detecciones' }) {
  if (!detections.length) {
    return (
      <div className="det-panel">
        <div className="det-panel-title">{title}</div>
        <div className="empty">
          <div className="ei">🔍</div>
          <p>Las frutas detectadas aparecerán aquí</p>
        </div>
      </div>
    )
  }

  // Unificar: para cámara son objetos con {id, bbox, fruta, confianza, precio, clasificado}
  // Para video son {fruta, confianza, precio}
  const items = detections.map((d, i) => ({
    id:         d.id ?? i,
    fruta:      d.fruta || '—',
    confianza:  d.confianza || 0,
    precio:     d.precio || 0,
    clasificado: d.clasificado !== undefined ? d.clasificado : true,
  }))

  return (
    <div className="det-panel">
      <div className="det-panel-title">{title} ({items.length})</div>
      {items.map((item) => (
        <div key={item.id} className="det-item">
          <div className="det-emoji">{emoji(item.fruta)}</div>
          <div className="det-info">
            <div className="det-name" style={{ color: colorFor(item.id) }}>
              {item.fruta}
            </div>
            <div className="det-badges">
              <span className="badge badge-green">
                {(item.confianza * 100).toFixed(1)}%
              </span>
              {item.precio > 0 && (
                <span className="badge badge-orange">
                  ${Number(item.precio).toLocaleString('es-CO')} COP
                </span>
              )}
              {item.id !== undefined && (
                <span className="badge badge-gray">ID {item.id}</span>
              )}
              {!item.clasificado && (
                <span className="badge badge-gray">Detectando…</span>
              )}
            </div>
            <div className="det-bar">
              <div className="det-bar-fill" style={{ width: `${item.confianza * 100}%` }} />
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
