import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmtNum, useApi } from '../api'
import { axis, Card, chartTip, Loading } from '../components/ui'

const COLORS = { mlp: '#7d86a6', flat_gnn: '#6aa0ff', nested_central: '#9b7bff', nested_local: '#ffbf5e', nested_federated: '#4ef0b0' }
const SHORT = { mlp: 'Feature MLP', flat_gnn: 'Flat GNN', nested_central: 'Nested · central', nested_local: 'Nested · fed. local', nested_federated: 'Nested · fed. + exchange' }

function Architecture() {
  return (
    <svg viewBox="0 0 900 250" className="arch">
      <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="#7b87a1" /></marker></defs>
      {[0, 1, 2].map((b) => (
        <g key={b} transform={`translate(10, ${15 + b * 78})`}>
          <rect width="190" height="64" rx="10" className="arch-box" />
          <text x="12" y="22" className="arch-t">Bank {'ABC'[b]} local graph</text>
          <text x="12" y="42" className="arch-s">accounts · ghosts · devices · merchants</text>
          <line x1="190" y1="32" x2="230" y2="32" markerEnd="url(#ar)" className="arch-l" />
        </g>
      ))}
      <g transform="translate(240,15)">
        <rect width="200" height="220" rx="10" className="arch-box hi" />
        <text x="12" y="24" className="arch-t">Inner GNN (nested)</text>
        <text x="12" y="46" className="arch-s">rooted 2-hop subgraph / account</text>
        <text x="12" y="66" className="arch-s">+ hop-distance encoding</text>
        <text x="12" y="86" className="arch-s">2 × relation-aware attention</text>
        <text x="12" y="106" className="arch-s">(6 relation types, edge feats)</text>
        <text x="12" y="126" className="arch-s">pool: root ⊕ mean ⊕ attention</text>
        <text x="12" y="160" className="arch-t">→ subgraph embedding z</text>
      </g>
      <line x1="440" y1="125" x2="480" y2="125" markerEnd="url(#ar)" className="arch-l" />
      <g transform="translate(490,15)">
        <rect width="190" height="100" rx="10" className="arch-box" />
        <text x="12" y="24" className="arch-t">Cross-bank exchange</text>
        <text x="12" y="46" className="arch-s">clip ‖z‖≤1 + Gaussian noise</text>
        <text x="12" y="66" className="arch-s">keyed by HMAC pseudonym</text>
        <text x="12" y="86" className="arch-s">→ fills ghost nodes</text>
      </g>
      <g transform="translate(490,135)">
        <rect width="190" height="100" rx="10" className="arch-box hi" />
        <text x="12" y="24" className="arch-t">Outer GNN</text>
        <text x="12" y="46" className="arch-s">account graph: transfers +</text>
        <text x="12" y="66" className="arch-s">shared devices, 2 attn layers</text>
        <text x="12" y="86" className="arch-s">head on [h ⊕ z] → risk</text>
      </g>
      <line x1="585" y1="115" x2="585" y2="133" markerEnd="url(#ar)" className="arch-l" />
      <line x1="680" y1="185" x2="715" y2="185" markerEnd="url(#ar)" className="arch-l" />
      <g transform="translate(720,60)">
        <rect width="170" height="150" rx="10" className="arch-box" />
        <text x="12" y="24" className="arch-t">Federated server</text>
        <text x="12" y="46" className="arch-s">FedAvg of clipped updates</text>
        <text x="12" y="66" className="arch-s">secure aggregation masks</text>
        <text x="12" y="86" className="arch-s">optional DP noise</text>
        <text x="12" y="112" className="arch-t">Ring coordinator</text>
        <text x="12" y="132" className="arch-s">links flagged pseudonyms</text>
      </g>
    </svg>
  )
}

export default function Models() {
  const { data } = useApi('/api/metrics')
  if (!data) return <Loading />
  const models = data.models
  const base = models[0]
  const bars = ['auc', 'ap', 'f1', 'tx_auc', 'tx_ap'].map((k) => ({ metric: k.replace('tx_', 'TX ').toUpperCase(), ...Object.fromEntries(models.map((m) => [m.key, m[k]])) }))
  const central = models.filter((m) => !m.key.startsWith('nested_') || m.key === 'nested_central')
  const fed = models.filter((m) => m.key === 'nested_local' || m.key === 'nested_federated')
  const merge = (ms) => {
    const steps = {}
    ms.forEach((m) => m.history.forEach((h) => { steps[h.step] = { step: h.step, ...steps[h.step], [m.key]: h.val_ap } }))
    return Object.values(steps).sort((a, b) => a.step - b.step)
  }
  return (
    <div className="page">
      <div className="page-head">
        <div><h1>Models &amp; evaluation</h1>
          <div className="eyebrow">evaluation</div><p className="muted">Held-out test split is ring-disjoint: test rings never appear in training. Threshold chosen on validation F1. Active model has {fmtNum(data.n_params)} parameters.</p></div>
      </div>
      <Card title="Architecture: nested federated heterogeneous GNN"><Architecture /></Card>
      <Card title="Test-set comparison" pad={false}>
        <table className="table">
          <thead><tr><th>Model</th><th>Acct AUC</th><th>Acct AP</th><th>F1</th><th>Precision</th><th>Recall</th><th>Tx AUC</th><th>Tx AP</th><th>Ring recall</th><th>Train</th></tr></thead>
          <tbody>{models.map((m) => (
            <tr key={m.key} className={m.key === data.active ? 'active-row' : ''}>
              <td><span className="bank-dot" style={{ background: COLORS[m.key] }} />{m.label}</td>
              <td className="mono">{m.auc.toFixed(3)}</td><td className="mono">{m.ap.toFixed(3)}</td><td className="mono">{m.f1.toFixed(3)}</td>
              <td className="mono">{m.precision.toFixed(3)}</td><td className="mono">{m.recall.toFixed(3)}</td>
              <td className="mono">{m.tx_auc == null ? '–' : m.tx_auc.toFixed(3)}</td><td className="mono">{m.tx_ap == null ? '–' : m.tx_ap.toFixed(3)}</td>
              <td className="mono">{m.ring_recall == null ? '–' : m.ring_recall.toFixed(2)}</td><td className="mono">{m.train_s.toFixed(0)}s</td>
            </tr>
          ))}</tbody>
        </table>
      </Card>
      <Card title="Reading these numbers">
        <p className="soft small">Laundering is rare: {((base.base_rate || 0) * 100).toFixed(2)}% of test accounts and {((base.tx_base_rate || 0) * 100).toFixed(3)}% of test transactions. A random ranker scores average precision equal to that base rate, so an account AP of {base.ap.toFixed(3)} is about {(base.ap / Math.max(base.base_rate || 1e-9, 1e-9)).toFixed(0)}× chance. The test split is ring-disjoint and small ({base.test_positives} laundering accounts), so differences of ±0.02 are within noise; see the multi-seed benchmark in the README.</p>
      </Card>
      <Card title="Metric comparison">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={bars}>
            <CartesianGrid stroke="rgba(255,255,255,.05)" vertical={false} />
            <XAxis dataKey="metric" {...axis} />
            <YAxis domain={[0, 1]} {...axis} width={40} allowDataOverflow />
            <Tooltip {...chartTip} formatter={(v) => (+v).toFixed(3)} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {models.map((m) => <Bar key={m.key} dataKey={m.key} name={SHORT[m.key]} fill={COLORS[m.key]} radius={[3, 3, 0, 0]} />)}
          </BarChart>
        </ResponsiveContainer>
      </Card>
      <div className="grid-2">
        <Card title="Centralised training: validation AP per epoch">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={merge(central)}>
              <CartesianGrid stroke="rgba(255,255,255,.05)" vertical={false} />
              <XAxis dataKey="step" {...axis} />
              <YAxis domain={['auto', 1]} {...axis} width={40} />
              <Tooltip {...chartTip} formatter={(v) => (+v).toFixed(3)} /><Legend wrapperStyle={{ fontSize: 12 }} />
              {central.map((m) => <Line key={m.key} dataKey={m.key} name={SHORT[m.key]} stroke={COLORS[m.key]} dot={false} />)}
            </LineChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Federated training: validation AP per round">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={merge(fed)}>
              <CartesianGrid stroke="rgba(255,255,255,.05)" vertical={false} />
              <XAxis dataKey="step" {...axis} />
              <YAxis domain={['auto', 1]} {...axis} width={40} />
              <Tooltip {...chartTip} formatter={(v) => (+v).toFixed(3)} /><Legend wrapperStyle={{ fontSize: 12 }} />
              {fed.map((m) => <Line key={m.key} dataKey={m.key} name={SHORT[m.key]} stroke={COLORS[m.key]} dot={false} />)}
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>
      <Card title="Pipeline timings">
        <p className="small muted">Feature extraction {data.timings.features_s.toFixed(2)}s · graph views {data.timings.views_s.toFixed(2)}s · nested subgraph extraction {data.timings.subgraphs_s.toFixed(2)}s · split train/val/test {data.split.train}/{data.split.val}/{data.split.test} accounts.</p>
      </Card>
    </div>
  )
}
