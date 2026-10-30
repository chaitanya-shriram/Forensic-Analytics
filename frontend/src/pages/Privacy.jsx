import { useState } from 'react'
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { api, useApi, useLive } from '../api'
import { axis, Card, chartTip, Loading } from '../components/ui'

const fmtEps = (e) => (e == null ? 'n/a' : e > 1e4 ? '≫ 10⁴' : e.toFixed(1))

export default function Privacy() {
  const { data: p } = useApi('/api/privacy')
  const { data: exps, setData: setExps } = useApi('/api/experiments')
  const [cfg, setCfg] = useState({ rounds: 15, local_epochs: 4, dp_noise: 0.0, emb_noise: 0.05, exchange: true })
  const [run, setRun] = useState(null)
  const [err, setErr] = useState(null)
  useLive((ev) => {
    if (ev.type === 'training' && ev.model?.startsWith('nested')) setRun({ progress: ev.progress, rec: ev.record })
    if (ev.type === 'experiment') { setRun(null); setExps((x) => [...(x || []), ev.experiment]) }
  })
  const start = () => {
    setErr(null)
    api('/api/experiments', { method: 'POST', body: cfg }).then(() => setRun({ progress: 0 })).catch((e) => setErr(e.message))
  }
  if (!p) return <Loading />
  return (
    <div className="page">
      <div className="page-head">
        <div><h1>Privacy &amp; federation</h1><p className="muted">What crosses bank boundaries, how it is protected, and what it costs in accuracy.</p></div>
      </div>
      <div className="grid-2">
        <Card title="Shared between banks (protected)">
          {p.shared.map((s) => <div key={s.what} className="priv"><b>{s.what}</b><p className="small muted">{s.how}</p></div>)}
        </Card>
        <Card title="Never leaves a bank">
          <ul className="never">{p.never_shared.map((s) => <li key={s}>{s}</li>)}</ul>
          <div className="kpis tight">
            <div><div className="kpi-label">Secure aggregation</div><div className="kpi-value sm green">{p.secure_aggregation ? 'on' : 'off'}</div></div>
            <div><div className="kpi-label">Update DP noise σ</div><div className="kpi-value sm">{p.dp_noise_multiplier}</div></div>
            <div><div className="kpi-label">ε (updates)</div><div className="kpi-value sm">{fmtEps(p.epsilon_updates)}</div></div>
            <div><div className="kpi-label">ε (embeddings)</div><div className="kpi-value sm">{fmtEps(p.epsilon_embeddings)}</div></div>
            <div><div className="kpi-label">Communication</div><div className="kpi-value sm">{p.communication_mb?.toFixed(1)} MB</div></div>
          </div>
          <p className="muted small">{p.note}</p>
        </Card>
      </div>

      <Card title="Privacy / utility experiment (retrains the federated model in the background)">
        <div className="form-row">
          <label>Rounds<input type="number" min="1" max="60" value={cfg.rounds} onChange={(e) => setCfg({ ...cfg, rounds: +e.target.value })} /></label>
          <label>Local epochs<input type="number" min="1" max="10" value={cfg.local_epochs} onChange={(e) => setCfg({ ...cfg, local_epochs: +e.target.value })} /></label>
          <label>Update DP noise σ<input type="number" step="0.05" min="0" value={cfg.dp_noise} onChange={(e) => setCfg({ ...cfg, dp_noise: +e.target.value })} /></label>
          <label>Embedding noise std<input type="number" step="0.05" min="0" value={cfg.emb_noise} onChange={(e) => setCfg({ ...cfg, emb_noise: +e.target.value })} /></label>
          <label className="check"><input type="checkbox" checked={cfg.exchange} onChange={(e) => setCfg({ ...cfg, exchange: e.target.checked })} /> Cross-bank embedding exchange</label>
          <button className="btn primary" disabled={!!run} onClick={start}>{run ? 'Training…' : 'Run experiment'}</button>
        </div>
        {err && <p className="error">{err}</p>}
        {run && <div><div className="progress"><div style={{ width: `${(run.progress || 0) * 100}%` }} /></div>
          <p className="small mono">{run.rec ? `round ${run.rec.step}: loss ${run.rec.loss.toFixed(3)} · val AP ${run.rec.val_ap.toFixed(3)} · ${run.rec.comm_mb.toFixed(1)} MB` : 'starting…'}</p></div>}
      </Card>

      <div className="grid-2">
        <Card title="Experiments" pad={false}>
          {!exps?.length ? <p className="muted small pad">No experiments yet. Try σ = 0.5 with and without the exchange.</p> : (
            <table className="table compact">
              <thead><tr><th>#</th><th>Exchange</th><th>σ upd.</th><th>Emb. noise</th><th>Rounds</th><th>AUC</th><th>AP</th><th>F1</th><th>ε upd.</th></tr></thead>
              <tbody>{exps.map((e) => (
                <tr key={e.id}><td>{e.id}</td><td>{e.exchange ? 'yes' : 'no'}</td><td>{e.dp_noise}</td><td>{e.emb_noise}</td><td>{e.rounds}</td>
                  <td className="mono">{e.auc.toFixed(3)}</td><td className="mono">{e.ap.toFixed(3)}</td><td className="mono">{e.f1.toFixed(3)}</td><td className="mono">{fmtEps(e.privacy?.epsilon_updates)}</td></tr>
              ))}</tbody>
            </table>
          )}
        </Card>
        <Card title="Noise vs. average precision">
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart>
              <CartesianGrid stroke="rgba(255,255,255,.05)" />
              <XAxis type="number" dataKey="dp_noise" name="update σ" {...axis} />
              <YAxis type="number" dataKey="ap" name="AP" domain={['auto', 1]} {...axis} width={40} />
              <ZAxis range={[80, 80]} />
              <Tooltip {...chartTip} formatter={(v) => (+v).toFixed(3)} />
              <Scatter data={(exps || []).filter((e) => e.exchange)} fill="#3ecf8e" name="with exchange" />
              <Scatter data={(exps || []).filter((e) => !e.exchange)} fill="#f2a541" name="bank-local" />
            </ScatterChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  )
}
