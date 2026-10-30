import { lazy, Suspense } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmtMoney, useApi, useMeta } from '../api'
import { AccountLink, axis, Bank, Card, chartTip, ErrorBox, Glass, Loading, Risk, ScoreOrb, Truth } from '../components/ui'

const Graph3D = lazy(() => import('../components/Graph3D'))
const MODEL_NAMES = {
  nested_federated: 'Nested GNN · federated + exchange',
  nested_local: 'Nested GNN · federated, bank-local',
  nested_central: 'Nested GNN · centralised',
  flat_gnn: 'Flat GNN · centralised',
  mlp: 'Feature MLP',
}

function NodeName({ n }) {
  if (n.kind === 'account') return <><AccountLink id={n.id} name={n.name} />{n.external && <span className="tag tag-blue" title={n.member ? 'customer of another member bank, seen through its exchanged embedding' : 'account at a non-member institution'}>{n.member ? 'x-bank' : 'external'}</span>}</>
  if (n.kind === 'merchant') return <span>{n.name}{n.high_risk && <span className="tag tag-red">high-risk</span>}</span>
  return <span className="mono small">{n.name}</span>
}

export default function Account() {
  const { id } = useParams()
  const nav = useNavigate()
  const meta = useMeta()
  const { data: a, error } = useApi(`/api/accounts/${id}`, [id])
  if (error) return <ErrorBox error={error} />
  if (!a) return <Loading text="Running explanation (attention + gradients)…" />
  const ex = a.explanation
  const feats = ex ? ex.features.slice(0, 10).map((f) => ({ ...f, c: f.contribution })) : []
  const ms = Object.entries(a.model_scores).map(([k, v]) => ({ model: MODEL_NAMES[k] || k, score: v }))
  return (
    <div className="page">
      <Glass className="pad" tilt={0}>
        <div className="row between wrap gap-lg">
          <div>
            <div className="eyebrow">account investigation</div>
            <h1 className="mono" style={{ fontSize: 26 }}>{a.name}</h1>
            <div className="row gap wrap small soft">
              <Bank id={a.bank} name={a.bank_name} /><span>pseudonym <span className="mono">{a.pseudonym}</span></span>
              <Truth v={a.truth} /><span className="tag">{a.split}</span>
              {a.rings.map((r) => <Link key={r} className="tag tag-violet" to={`/rings/${r}`}>ring #{r}</Link>)}
            </div>
          </div>
          {a.member ? <div className="row gap-lg"><Link className="btn" to={`/gnn?a=${a.id}`}>Step through the GNN →</Link><ScoreOrb score={a.score} /></div> : <span className="pill">external institution · not scored by the consortium</span>}
        </div>
      </Glass>

      {ex && (
        <div className="grid-2-1">
          <Card title="Why this score?">
            {ex.reasons.length ? <ul className="reasons big">{ex.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul> : <p className="muted">No behavioural red flag stands out; the score comes mostly from network context.</p>}
            <h4>Feature attribution · gradient × input</h4>
            <ResponsiveContainer width="100%" height={270}>
              <BarChart data={feats} layout="vertical" margin={{ left: 10 }}>
                <XAxis type="number" {...axis} />
                <YAxis type="category" dataKey="label" width={200} tick={{ fill: '#b3bad3', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip {...chartTip} formatter={(v, n, p) => [`${(+v).toFixed(3)} (raw ${p.payload.value.toFixed(2)})`, 'contribution']} />
                <Bar dataKey="c" radius={4}>{feats.map((f, i) => <Cell key={i} fill={f.c > 0 ? '#ff5c9a' : '#4ef0b0'} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="muted tiny">Pink pushes risk up, green pulls it down.</p>
          </Card>
          <Card title="Score by model">
            {ms.map((m) => <div key={m.model} className="model-score"><span className="small soft">{m.model}</span><Risk score={m.score} wide /></div>)}
            <p className="muted small" style={{ marginTop: 10 }}>The bank-local model cannot see counterparties' behaviour at other banks; the gap to the federated model is what the cross-bank embedding exchange adds.</p>
          </Card>
        </div>
      )}

      {ex && (
        <div className="grid-2">
          <Card title="Outer GNN · cross-bank attention" extra={<span className="tiny muted">neighbours that drove the score</span>} pad={false} scroll>
            <table className="table compact">
              <thead><tr><th>Neighbour</th><th>Bank</th><th>Relation</th><th>Their risk</th><th>Attention</th></tr></thead>
              <tbody>{ex.outer_attention.map((n, i) => (
                <tr key={i}><td><NodeName n={n} /></td><td><Bank id={n.bank} name={n.bank_name} /></td><td className="small">{n.relations.join(', ')}</td>
                  <td>{n.member ? <Risk score={n.score} /> : <span className="muted tiny">n/a</span>}</td><td><div className="att"><div style={{ width: `${n.weight * 100}%` }} /></div></td></tr>
              ))}</tbody>
            </table>
          </Card>
          <Card title={`Inner GNN · nested subgraph (${ex.subgraph_size} nodes)`} extra={<span className="tiny muted">attention-pooling weight</span>} pad={false} scroll>
            <table className="table compact">
              <thead><tr><th>Node</th><th>Type</th><th>Hop</th><th>Weight</th></tr></thead>
              <tbody>{ex.subgraph_pooling.map((n, i) => (
                <tr key={i}><td><NodeName n={n} /></td><td className="small">{n.kind}</td><td className="mono">{n.hop}</td><td><div className="att"><div style={{ width: `${Math.min(1, n.weight * 3) * 100}%` }} /></div></td></tr>
              ))}</tbody>
            </table>
            <div className="pad tiny muted">Root's last-layer attention: {ex.inner_attention.slice(0, 4).map((n, i) => <span key={i}>{i > 0 && ', '}{n.relation} <b className="soft">{n.name}</b> ({(n.weight * 100).toFixed(0)}%)</span>)}</div>
          </Card>
        </div>
      )}

      <Card title="Network neighbourhood · 3D" extra={<span className="tiny muted">white halo = this account · click to navigate</span>}>
        <Suspense fallback={<Loading text="Rendering 3D graph…" />}>
          <Graph3D nodes={a.graph.nodes} edges={a.graph.edges} focus={`a${a.id}`} nBanks={meta?.n_banks ?? 8} onSelect={(n) => n.account != null && nav(`/accounts/${n.account}`)} height={460} />
        </Suspense>
      </Card>

      <div className="grid-2">
        <Card title="Recent transactions" pad={false} scroll>
          <table className="table compact">
            <thead><tr><th>Time</th><th>Dir</th><th>Counterparty</th><th>Rail</th><th>Amount</th></tr></thead>
            <tbody>{a.transactions.map((t) => {
              const out = t.src === a.id
              return (
                <tr key={t.index}><td className="mono small">{t.time.slice(5)}</td><td className={out ? 'amber' : 'green'}>{out ? 'out' : 'in'}</td>
                  <td>{t.kind === 'merchant' ? <span className="small">{t.merchant}</span> : <AccountLink id={out ? t.dst : t.src} name={out ? t.dst_name : t.src_name} />}{t.cross_bank && <span className="tag tag-blue">x-bank</span>}</td>
                  <td className="small muted">{t.format || '–'}</td><td className="mono">{fmtMoney(t.amount)}</td></tr>
              )
            })}</tbody>
          </table>
        </Card>
        <Card title="Behavioural features (raw)" pad={false} scroll>
          <table className="table compact">
            <tbody>{Object.entries(a.features_raw).map(([k, v]) => <tr key={k}><td className="small soft">{k}</td><td className="mono">{v.toFixed(3)}</td></tr>)}</tbody>
          </table>
        </Card>
      </div>
    </div>
  )
}
