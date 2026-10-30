import { lazy, Suspense } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { fmtMoney, fmtPct, useApi, useMeta } from '../api'
import { AccountLink, Bank, BankDots, Card, Empty, Glass, Loading, PageHead, Risk, Truth } from '../components/ui'

const Graph3D = lazy(() => import('../components/Graph3D'))
const PRETTY = (p) => p?.replaceAll('_', ' ')

function RingDetail({ id }) {
  const { data: r, error } = useApi(`/api/rings/${id}`, [id])
  const meta = useMeta()
  const nav = useNavigate()
  if (error) return <Empty>Ring not found (rings are re-assembled as scores change).</Empty>
  if (!r) return <Loading />
  return (
    <>
      <Card title={<>Ring #{r.id} · {r.pattern_label}</>} extra={<span className="mono tiny muted">pseudonym {r.pseudonym}</span>}>
        <div className="stat-row">
          <div><div className="kpi-label">Members</div><div className="kpi-value sm">{r.size}</div></div>
          <div><div className="kpi-label">Member banks</div><div className="kpi-value sm">{r.bank_names.length}</div></div>
          <div><div className="kpi-label">Internal flow</div><div className="kpi-value sm">{fmtMoney(r.total_flow)}</div></div>
          <div><div className="kpi-label">Links</div><div className="kpi-value sm">{r.n_links}</div></div>
          <div><div className="kpi-label">Mean risk</div><div className="kpi-value sm">{fmtPct(r.mean_score)}</div></div>
          <div title="share of the ring's links that the dataset labels as laundering (evaluation only)"><div className="kpi-label">{r.basis === 'transactions' ? 'Labelled laundering tx' : 'Labelled members'}</div><div className="kpi-value sm">{fmtPct(r.precision, 0)}</div></div>
          {r.n_tx != null && <div><div className="kpi-label">Flagged transactions</div><div className="kpi-value sm">{r.n_tx}</div></div>}
          {r.truth && <div title="overlap with the dataset's labelled laundering attempt"><div className="kpi-label">Matches labelled</div><div className="kpi-value sm">{PRETTY(r.truth.pattern)} · {fmtPct(r.truth.recall, 0)}</div></div>}
        </div>
        <div className="row gap wrap">{r.banks.map((b, i) => <Bank key={b} id={b} name={r.bank_names[i]} />)}</div>
      </Card>
      <Card title="Ring network · 3D" extra={<span className="tiny muted">colour = bank · size = risk · white ring = member · amber = inflow from outside · particles = money direction</span>}>
        <Suspense fallback={<Loading text="Rendering 3D graph…" />}>
          <Graph3D nodes={r.graph.nodes} edges={r.graph.edges} nBanks={meta?.n_banks ?? 8} onSelect={(n) => n.account != null && nav(`/accounts/${n.account}`)} height={520} />
        </Suspense>
      </Card>
      <div className="grid-2">
        <Card title="Members" pad={false} scroll>
          <table className="table compact">
            <thead><tr><th>Account</th><th>Bank</th><th>Risk</th><th>Label</th></tr></thead>
            <tbody>{r.members.map((m) => (
              <tr key={m.id}><td><AccountLink id={m.id} name={m.name} /></td><td><Bank id={m.bank} name={m.bank_name} /></td><td><Risk score={m.score} /></td><td><Truth v={m.truth} /></td></tr>
            ))}</tbody>
          </table>
        </Card>
        <Card title="Money movement timeline (within ring)" pad={false} scroll>
          <table className="table compact">
            <thead><tr><th>Time</th><th>From</th><th>To</th><th>Amount</th></tr></thead>
            <tbody>{r.timeline.map((t) => (
              <tr key={t.index}><td className="mono small">{t.time.slice(5)}</td><td><AccountLink id={t.src} name={t.src_name} /></td>
                <td><AccountLink id={t.dst} name={t.dst_name} />{t.cross_bank && <span className="tag tag-blue">x-bank</span>}</td><td className="mono">{fmtMoney(t.amount)}</td></tr>
            ))}</tbody>
          </table>
          {r.timeline.length === 0 && <p className="muted small pad">Members are linked through a shared intermediary rather than direct transfers.</p>}
        </Card>
      </div>
    </>
  )
}

export default function Rings() {
  const { id } = useParams()
  const nav = useNavigate()
  const { data: rings } = useApi('/api/rings', [], { interval: 10000 })
  if (!rings) return <Loading />
  const sel = id ?? rings[0]?.id
  return (
    <div className="page">
      <PageHead eyebrow="consortium coordinator" title="Fraud rings">
        On real data, rings are connected groups of the riskiest transactions (top 0.05%), and accounts at non-member institutions appear as intermediaries. On synthetic data, they are flagged accounts linked by money flow or shared devices. Only pseudonyms reach the coordinator.
      </PageHead>
      <div className="split">
        <div className="ring-list">
          {rings.length === 0 && <Empty>No rings detected.</Empty>}
          {rings.map((r) => (
            <button key={r.id} className={`ring-item ${String(r.id) === String(sel) ? 'active' : ''}`} onClick={() => nav(`/rings/${r.id}`)}>
              <div className="row between"><b className="display">#{r.id}</b>{r.live && <span className="tag tag-red">live</span>}<Risk score={r.mean_score} /></div>
              <div className="small soft">{r.pattern_label}</div>
              <div className="row between tiny muted"><span>{r.size} accounts{r.n_tx != null ? ` · ${r.n_tx} tx` : ''} · {fmtMoney(r.total_flow)}</span><BankDots banks={r.banks} /></div>
            </button>
          ))}
        </div>
        <div className="col">{sel != null ? <RingDetail id={sel} /> : <Glass className="pad">Select a ring.</Glass>}</div>
      </div>
    </div>
  )
}
