import { useEffect, useState } from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import { api, useLive } from './api'
import { Glass } from './components/ui'
import Overview from './pages/Overview'
import Live from './pages/Live'
import Rings from './pages/Rings'
import Accounts from './pages/Accounts'
import Account from './pages/Account'
import Models from './pages/Models'
import Privacy from './pages/Privacy'
import Dataset from './pages/Dataset'
import NestedGnn from './pages/NestedGnn'

const NAV = [
  ['/', 'Triage', 'M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z'],
  ['/live', 'Live monitor', 'M12 2a10 10 0 100 20 10 10 0 000-20zm1 5v5.4l4.2 2.5-.8 1.3L11 13V7h2z'],
  ['/rings', 'Fraud rings', 'M12 2l3 6 6 .9-4.5 4.3 1 6.3L12 16.6 6.5 19.5l1-6.3L3 8.9 9 8z'],
  ['/accounts', 'Accounts', 'M12 12a5 5 0 100-10 5 5 0 000 10zm0 2c-4 0-9 2-9 6v2h18v-2c0-4-5-6-9-6z'],
  ['/gnn', 'How scoring works', 'M12 2a3 3 0 110 6 3 3 0 010-6zM5 14a3 3 0 110 6 3 3 0 010-6zm14 0a3 3 0 110 6 3 3 0 010-6zM11 8.8V11L6.4 14.3l1.2 1.6L12 12.7l4.4 3.2 1.2-1.6L13 11V8.8h-2z'],
  ['/models', 'Models', 'M4 4h4v4H4zm6 0h4v4h-4zm6 0h4v4h-4zM4 10h4v4H4zm6 0h4v4h-4zm6 0h4v4h-4zM4 16h4v4H4zm6 0h4v4h-4zm6 0h4v4h-4z'],
  ['/privacy', 'Privacy & federation', 'M12 1L3 5v6c0 5.5 3.8 10.7 9 12 5.2-1.3 9-6.5 9-12V5l-9-4z'],
  ['/dataset', 'Dataset', 'M12 3C7 3 3 4.8 3 7v10c0 2.2 4 4 9 4s9-1.8 9-4V7c0-2.2-4-4-9-4zm0 2c4.4 0 7 1.5 7 2s-2.6 2-7 2-7-1.5-7-2 2.6-2 7-2zm7 12c0 .5-2.6 2-7 2s-7-1.5-7-2v-2.3c1.7.9 4.2 1.3 7 1.3s5.3-.4 7-1.3V17zm0-5c0 .5-2.6 2-7 2s-7-1.5-7-2V9.7c1.7.9 4.2 1.3 7 1.3s5.3-.4 7-1.3V12z'],
]

export function Aurora() {
  return (
    <div className="aurora" aria-hidden>
      <div className="grid" />
      <div className="blob b1" /><div className="blob b2" /><div className="blob b3" />
      <div className="noise" />
    </div>
  )
}

function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark" />
      <div><div className="brand-name">RingSight</div><div className="brand-sub">fraud ring intelligence</div></div>
    </div>
  )
}

function Boot({ status }) {
  return (
    <div className="boot">
      <Glass className="boot-card" tilt={4}>
        <Brand />
        <h1 className="grad-text" style={{ fontSize: 34 }}>Assembling the consortium</h1>
        <p className="soft">Privacy-preserving cross-bank fraud ring detection with a nested federated heterogeneous GNN.</p>
        <div className="progress"><div style={{ width: `${Math.round((status?.progress || 0) * 100)}%` }} /></div>
        <p className="mono small">{status?.message || 'Connecting to backend…'}</p>
        {status?.state === 'error' && <p className="error">Backend error: {status.message}</p>}
        <p className="muted small">The first start downloads and cleans the IBM AML dataset, then trains five models. Later starts load cached weights in seconds.</p>
      </Glass>
    </div>
  )
}

export default function App() {
  const [status, setStatus] = useState(null)
  const [toast, setToast] = useState(null)
  const live = useLive((ev) => {
    if (ev.type === 'status' || ev.type === 'training') setStatus((s) => ({ ...s, ...(ev.type === 'status' ? ev : { state: 'training' }) }))
    if (ev.type === 'ring_detected') setToast(`Live ring #${ev.ring.ring} (${ev.ring.pattern.replaceAll('_', ' ')}) detected ${ev.ring.minutes_to_detect.toFixed(0)} min after it started`)
    if (ev.type === 'ring_activated') setToast(`New ${ev.ring.pattern.replaceAll('_', ' ')} ring activated across ${ev.ring.banks.length} banks`)
    if (ev.type === 'replay_done') setToast('Replay finished: every held-out real transaction has been scored')
  })
  useEffect(() => {
    let t
    const poll = () => api('/api/status').then((s) => { setStatus(s); if (!s.ready) t = setTimeout(poll, 1000) })
      .catch(() => { t = setTimeout(poll, 1500) })
    poll()
    return () => clearTimeout(t)
  }, [])
  useEffect(() => { if (toast) { const t = setTimeout(() => setToast(null), 6500); return () => clearTimeout(t) } }, [toast])

  return (
    <>
      <Aurora />
      {!status?.ready ? <Boot status={status} /> : (
        <div className="shell">
          <aside className="side">
            <div className="side-inner">
              <Brand />
              <nav>
                {NAV.map(([to, label, d]) => (
                  <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `nav ${isActive ? 'active' : ''}`}>
                    <svg viewBox="0 0 24 24" width="17" height="17"><path d={d} fill="currentColor" /></svg>{label}
                  </NavLink>
                ))}
              </nav>
              <div className="side-foot">
                <div><span className={`dot ${live.connected ? 'on' : ''}`} />{live.connected ? 'Live link connected' : 'Live link offline'}</div>
                <div>compute · <span className="mono">{status.device}</span></div>
                {status.state === 'training' && <div className="amber">{status.message}</div>}
              </div>
            </div>
          </aside>
          <main className="main">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/live" element={<Live />} />
              <Route path="/rings" element={<Rings />} />
              <Route path="/rings/:id" element={<Rings />} />
              <Route path="/accounts" element={<Accounts />} />
              <Route path="/accounts/:id" element={<Account />} />
              <Route path="/models" element={<Models />} />
              <Route path="/privacy" element={<Privacy />} />
              <Route path="/dataset" element={<Dataset />} />
              <Route path="/gnn" element={<NestedGnn />} />
            </Routes>
          </main>
        </div>
      )}
      {toast && <Glass className="toast" tilt={0} onClick={() => setToast(null)}>{toast}</Glass>}
    </>
  )
}
