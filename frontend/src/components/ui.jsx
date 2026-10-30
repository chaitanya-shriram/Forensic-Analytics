import { useRef } from 'react'
import { Link } from 'react-router-dom'
import { BANK_COLORS, bankColor, riskColor } from '../api'

/** Glass panel with a subtle 3D tilt + light-follow sheen on hover. */
export function Glass({ children, className = '', tilt = 3, style, ...rest }) {
  const ref = useRef(null)
  const onMove = (e) => {
    if (!tilt || !ref.current) return
    const r = ref.current.getBoundingClientRect()
    const x = (e.clientX - r.left) / r.width
    const y = (e.clientY - r.top) / r.height
    ref.current.style.transform = `perspective(1200px) rotateX(${(0.5 - y) * tilt}deg) rotateY(${(x - 0.5) * tilt}deg)`
    ref.current.style.setProperty('--mx', `${x * 100}%`)
    ref.current.style.setProperty('--my', `${y * 100}%`)
  }
  const onLeave = () => { if (ref.current) ref.current.style.transform = '' }
  return (
    <div ref={ref} className={`glass ${tilt ? 'tilt' : ''} ${className}`} style={style} onMouseMove={onMove} onMouseLeave={onLeave} {...rest}>
      {tilt ? <div className="shine" /> : null}
      {children}
    </div>
  )
}

export function Card({ title, extra, children, className = '', pad = true, scroll = false, tilt = 0 }) {
  return (
    <Glass className={`card ${scroll ? 'scroll' : ''} ${className}`} tilt={tilt}>
      {(title || extra) && (
        <header className="card-head">
          <h3>{title}</h3>
          <div className="card-extra">{extra}</div>
        </header>
      )}
      <div className={`${pad ? 'card-body' : ''} ${scroll ? 'scroll-area' : ''}`}>{children}</div>
    </Glass>
  )
}

export function Kpi({ label, value, sub, tone }) {
  return (
    <Glass className={`kpi ${tone || ''}`} tilt={6}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </Glass>
  )
}

/** Risk bar. With `threshold`, a tick marks the flag line and the bar turns red only above it. */
export function Risk({ score, wide, threshold }) {
  if (score == null) return <span className="muted small">n/a</span>
  const over = threshold != null ? score >= threshold : null
  const c = over == null ? riskColor(score) : over ? '#ff5c7a' : score >= threshold - 0.05 ? '#ffbf5e' : '#6aa0ff'
  return (
    <span className={`risk ${wide ? 'wide' : ''}`} title={`risk ${(score * 100).toFixed(2)}${threshold != null ? ` · flag line ${(threshold * 100).toFixed(1)}` : ''}`}>
      <span className="risk-bar" style={{ color: c }}>
        <span style={{ width: `${Math.max(2, score * 100)}%`, background: c }} />
        {threshold != null && <i className="risk-tick" style={{ left: `${threshold * 100}%` }} />}
      </span>
      <span className={`mono ${over ? 'red' : ''}`}>{(score * 100).toFixed(1)}</span>
    </span>
  )
}

export function Step({ n, title, children, lead }) {
  return (
    <Glass className="step" tilt={0}>
      <div className="step-head"><span className="step-n">{n}</span><div><h3>{title}</h3>{lead && <p className="soft small">{lead}</p>}</div></div>
      <div className="step-body">{children}</div>
    </Glass>
  )
}

export function ScoreOrb({ score, label = 'risk' }) {
  const c = riskColor(score)
  const deg = Math.round(score * 360)
  return (
    <div className="score-orb" style={{ background: `conic-gradient(${c} ${deg}deg, rgba(255,255,255,.07) ${deg}deg)`, boxShadow: `0 0 60px -10px ${c}` }}>
      <div style={{ position: 'absolute', inset: 9, borderRadius: '50%', background: 'rgba(8,10,24,.92)' }} />
      <div style={{ position: 'relative', textAlign: 'center' }}>
        <div className="val" style={{ color: c }}>{(score * 100).toFixed(1)}</div>
        <div className="lbl">{label}</div>
      </div>
    </div>
  )
}

export function Bank({ id, name }) {
  return <span className="bank-chip" style={{ '--c': bankColor(id) }}>{name ?? `Bank ${id}`}</span>
}

export function BankDots({ banks }) {
  return <span>{banks.map((b) => <span key={b} className="bank-dot" style={{ background: BANK_COLORS[b % BANK_COLORS.length], color: BANK_COLORS[b % BANK_COLORS.length] }} />)}</span>
}

export function AccountLink({ id, name }) {
  return <Link className="mono link" to={`/accounts/${id}`}>{name}</Link>
}

export function Truth({ v }) {
  return v ? <span className="tag tag-red" title="ground-truth label from the dataset">laundering</span> : <span className="tag">clean</span>
}

export function Loading({ text = 'Loading…' }) {
  return <div className="loading"><div className="spinner" />{text}</div>
}

export function ErrorBox({ error }) {
  if (!error) return null
  return <div className="error">{String(error.message || error)}</div>
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>
}

export function PageHead({ eyebrow, title, children, right }) {
  return (
    <div className="page-head">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {children && <p>{children}</p>}
      </div>
      {right && <div className="row gap wrap">{right}</div>}
    </div>
  )
}

export const chartTip = {
  contentStyle: { background: 'rgba(12,14,32,.9)', border: '1px solid rgba(255,255,255,.14)', borderRadius: 12, fontSize: 12, backdropFilter: 'blur(12px)', boxShadow: '0 10px 30px rgba(0,0,0,.5)' },
  labelStyle: { color: '#b3bad3' },
  cursor: { fill: 'rgba(255,255,255,.04)' },
}
export const axis = { tick: { fill: '#7d86a6', fontSize: 11 }, axisLine: false, tickLine: false }
