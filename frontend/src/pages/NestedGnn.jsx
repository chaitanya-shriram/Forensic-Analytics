import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api, bankColor, fmtMoney, useApi, useMeta } from '../api'
import { AccountLink, Bank, ErrorBox, Glass, Loading, PageHead, Risk, Step, Truth } from '../components/ui'

const MONEY = new Set(['log_in_amount', 'log_out_amount', 'log_max_amount'])

/** Raw feature values are partly log-scaled; show them the way an analyst thinks about them. */
function human(f) {
  const v = f.value
  if (f.feature === 'log_holding_minutes') {
    const m = Math.expm1(v)
    return m >= 10000 ? 'no forwarding seen' : m >= 120 ? `${(m / 60).toFixed(1)} h` : `${m.toFixed(0)} min`
  }
  if (MONEY.has(f.feature)) return fmtMoney(Math.expm1(v))
  if (f.feature.startsWith('log_') || f.feature.endsWith('_burst')) return Math.round(Math.expm1(v)).toLocaleString()
  if (f.feature.includes('ratio') || f.feature.startsWith('share_')) return `${(v * 100).toFixed(0)}%`
  return v.toFixed(2)
}

const short = (name) => (name.length > 12 ? '…' + name.slice(-6) : name)

/** Radial drawing of the real rooted subgraph: root in the middle, hop-1 ring, hop-2 ring. */
function Subgraph({ sg, nBanks }) {
  const W = 640, H = 520, cx = W / 2, cy = H / 2, R1 = 140, R2 = 235
  const layout = useMemo(() => {
    const pos = {}
    const nodes = sg.nodes.map((n, i) => ({ ...n, i }))
    pos[0] = [cx, cy]
    const h1 = nodes.filter((n) => n.hop === 1)
    h1.forEach((n, k) => { const a = (2 * Math.PI * k) / Math.max(h1.length, 1) - Math.PI / 2; pos[n.i] = [cx + R1 * Math.cos(a), cy + R1 * Math.sin(a)]; n.angle = a })
    const angleOf = Object.fromEntries(h1.map((n) => [n.i, n.angle]))
    const h2 = nodes.filter((n) => n.hop === 2).map((n) => {
      const e = sg.edges.find((e) => (e.source === n.i && angleOf[e.target] != null) || (e.target === n.i && angleOf[e.source] != null))
      return { ...n, parentAngle: e ? angleOf[angleOf[e.target] != null ? e.target : e.source] : 0 }
    }).sort((a, b) => a.parentAngle - b.parentAngle)
    h2.forEach((n, k) => { const a = (2 * Math.PI * k) / Math.max(h2.length, 1) - Math.PI / 2 + 0.12; pos[n.i] = [cx + R2 * Math.cos(a), cy + R2 * Math.sin(a)] })
    return { pos, nodes }
  }, [sg])
  const maxPool = Math.max(...sg.nodes.map((n) => n.pool), 1e-9)
  const ranked = [...sg.nodes.keys()].sort((a, b) => sg.nodes[b].pool - sg.nodes[a].pool).slice(0, 6)
  const color = (n) => (n.kind !== 'account' ? '#9aa3c0' : bankColor(n.bank, nBanks))
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="subgraph-svg">
      <defs><marker id="arr" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="#8a93b8" /></marker></defs>
      <circle cx={cx} cy={cy} r={R1} fill="none" stroke="rgba(255,255,255,.06)" strokeDasharray="3 5" />
      <circle cx={cx} cy={cy} r={R2} fill="none" stroke="rgba(255,255,255,.06)" strokeDasharray="3 5" />
      <text x={cx + R1 + 6} y={cy - 4}>hop 1</text><text x={cx + R2 + 6} y={cy - 4}>hop 2</text>
      {sg.edges.map((e, k) => {
        const [x1, y1] = layout.pos[e.source] || [0, 0]
        const [x2, y2] = layout.pos[e.target] || [0, 0]
        const toRoot = e.target === 0 || e.source === 0
        const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1
        const r2 = 6 + 18 * Math.sqrt(sg.nodes[e.target].pool / maxPool)
        return <line key={k} x1={x1} y1={y1} x2={x2 - (dx / L) * (r2 + 3)} y2={y2 - (dy / L) * (r2 + 3)}
          stroke={toRoot ? '#5ee7ff' : '#8a93b8'} strokeOpacity={toRoot ? 0.35 + 0.65 * Math.min(1, e.attention * 2) : 0.35}
          strokeWidth={toRoot ? 1 + 3 * e.attention : 1} strokeDasharray={e.kind === 'transfer' ? '' : '4 4'}
          markerEnd={e.kind === 'transfer' ? 'url(#arr)' : undefined} />
      })}
      {layout.nodes.map((n) => {
        const [x, y] = layout.pos[n.i] || [0, 0]
        const r = 6 + 18 * Math.sqrt(n.pool / maxPool)
        return (
          <g key={n.i}>
            <title>{`${n.name}${n.kind === 'account' ? (n.external ? (n.member ? ' · other member bank' : ' · external institution') : ' · this bank') : ' · ' + n.kind}\nhop ${n.hop} · pooling weight ${(n.pool * 100).toFixed(1)}%${n.score != null ? `\naccount risk ${(n.score * 100).toFixed(1)}` : ''}${n.label ? '\nlabelled laundering' : ''}`}</title>
            {n.label ? <circle cx={x} cy={y} r={r + 4} fill="none" stroke="#ff5c7a" strokeWidth="2" /> : null}
            <circle cx={x} cy={y} r={r} fill={color(n)} fillOpacity={n.external && !n.member ? 0.45 : 0.9} stroke={n.i === 0 ? '#fff' : 'rgba(0,0,0,.4)'} strokeWidth={n.i === 0 ? 3 : 1} />
            {(n.i === 0 || ranked.includes(n.i)) && <text x={x} y={y + r + 12} textAnchor="middle">{n.i === 0 ? 'THIS ACCOUNT' : short(n.name)}</text>}
          </g>
        )
      })}
    </svg>
  )
}

function ZStrip({ z }) {
  const m = Math.max(...z.map(Math.abs), 1e-9)
  return (
    <div className="zstrip" title="the 64 numbers the inner GNN produced for this account">
      {z.map((v, i) => <i key={i} style={{ background: v >= 0 ? `rgba(94,231,255,${Math.abs(v) / m})` : `rgba(255,92,154,${Math.abs(v) / m})` }} />)}
    </div>
  )
}

export default function NestedGnn() {
  const [sp, setSp] = useSearchParams()
  const meta = useMeta()
  const [input, setInput] = useState('')
  const a = sp.get('a')
  useEffect(() => { if (!a) api('/api/nested/sample').then((r) => setSp({ a: String(r.account) }, { replace: true })) }, [a])
  const { data: d, error } = useApi(a ? `/api/nested/${a}` : null, [a])
  const { data: flagged } = useApi('/api/accounts?flagged=true&limit=8')
  const nBanks = meta?.n_banks ?? 8
  const go = (id) => id !== '' && setSp({ a: String(id) })

  const ex = d?.explanation
  const sg = ex?.subgraph
  const thr = d?.thresholds
  const stats = useMemo(() => {
    if (!sg) return null
    const n = sg.nodes
    return { total: n.length, h1: n.filter((x) => x.hop === 1).length, h2: n.filter((x) => x.hop === 2).length,
      ext: n.filter((x) => x.external && !x.member).length, other: n.filter((x) => x.external && x.member).length,
      laund: n.filter((x) => x.label).length - (n[0]?.label ? 1 : 0) }
  }, [sg])

  return (
    <div className="page">
      <PageHead eyebrow="model explained on real data" title="How the nested GNN scores an account"
        right={<form className="row gap" onSubmit={(e) => { e.preventDefault(); go(input) }}>
          <input placeholder="account id" value={input} onChange={(e) => setInput(e.target.value)} style={{ width: 130 }} />
          <button className="btn" type="submit">Explain</button>
          <button className="btn primary" type="button" onClick={() => api('/api/nested/sample').then((r) => go(r.account))}>Sample</button>
        </form>}>
        Follow one real account through all six stages, from the bank's ledger to the alert. Pick a flagged account below, or enter an id.
      </PageHead>

      <div className="row gap wrap">
        {(flagged?.items || []).map((x) => (
          <button key={x.id} className={`pill ${String(x.id) === a ? 'link' : ''}`} style={{ cursor: 'pointer', border: String(x.id) === a ? '1px solid var(--cyan)' : undefined }} onClick={() => go(x.id)}>
            <span className="mono">{x.name}</span>{x.truth ? <span className="red">●</span> : null}
          </button>
        ))}
      </div>

      <Glass className="pad" tilt={0}>
        <div className="pipeline">
          {[['1 Ledger', 'its own transfers'], ['2 Features', '28 behaviour numbers'], ['3 Neighbourhood', '2-hop subgraph'], ['4 Inner GNN', 'shape, 64 numbers'], ['5 Outer GNN', 'neighbours, incl. other banks'], ['6 Decision', 'risk vs flag line']].map(([t, s], i, arr) => (
            <div key={t} className="row" style={{ flex: '1 1 140px', gap: 8 }}><div className="pp"><b>{t}</b>{s}</div>{i < arr.length - 1 && <span className="arrow">›</span>}</div>
          ))}
        </div>
      </Glass>

      {error && <ErrorBox error={error} />}
      {!d && !error && <Loading text="Running the account through the model…" />}
      {d && ex && (
        <>
          <Glass className="pad" tilt={0}>
            <div className="row between wrap gap-lg">
              <div className="row gap wrap">
                <AccountLink id={d.id} name={d.name} /><Bank id={d.bank} name={d.bank_name} /><Truth v={d.truth} /><span className="tag">{d.split}</span>
                {d.rings.map((r) => <Link key={r} className="tag tag-violet" to={`/rings/${r}`}>ring #{r}</Link>)}
              </div>
              <div className="row gap"><span className="small soft">final risk</span><Risk score={d.score} threshold={thr.account} wide /></div>
            </div>
          </Glass>

          <Step n="1" title="Start from the bank's own ledger"
            lead={`The bank sees every transfer that touches its customer: ${d.n_transactions} in the history window. Each transfer also gets its own risk from the model (step 6). The riskiest are shown.`}>
            <table className="table compact">
              <thead><tr><th>Time</th><th>Dir</th><th>Counterparty</th><th>Rail</th><th>Amount</th><th>Transfer risk</th></tr></thead>
              <tbody>{d.transactions.map((t) => {
                const out = t.src === d.id
                return (
                  <tr key={t.index}>
                    <td className="mono small">{t.time.slice(5)}</td><td className={out ? 'amber' : 'green'}>{out ? 'out' : 'in'}</td>
                    <td><AccountLink id={out ? t.dst : t.src} name={out ? t.dst_name : t.src_name} />{t.cross_bank && <span className="tag tag-blue">x-bank</span>}</td>
                    <td className="small muted">{t.format}</td><td className="mono">{fmtMoney(t.amount)}</td>
                    <td><Risk score={t.risk} threshold={thr.tx} />{t.truth ? <span className="tag tag-red">L</span> : null}</td>
                  </tr>
                )
              })}</tbody>
            </table>
          </Step>

          <Step n="2" title="Turn behaviour into numbers"
            lead="From that ledger the bank computes 28 behaviour features: money in and out, how many counterparties, how fast money is forwarded, cross-bank and outside-consortium share, payment rails, one-day bursts. These are the 8 that moved this account's score most.">
            <table className="table compact">
              <thead><tr><th>Feature</th><th>This account</th><th>vs. all member accounts</th><th>Effect on score</th></tr></thead>
              <tbody>{ex.features.map((f) => (
                <tr key={f.feature}>
                  <td className="soft">{f.label}</td><td className="mono">{human(f)}</td>
                  <td><span className="pct"><span style={{ width: `${f.percentile}%` }} /></span><span className="tiny muted">higher than {f.percentile.toFixed(0)}%</span></td>
                  <td className={f.contribution > 0 ? 'red mono' : 'green mono'}>{f.contribution > 0 ? '▲ raises' : '▼ lowers'} {Math.abs(f.contribution).toFixed(2)}</td>
                </tr>
              ))}</tbody>
            </table>
            {ex.reasons.length > 0 && <><h4>Red flags in plain words</h4><ul className="reasons">{ex.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul></>}
          </Step>

          <Step n="3" title="Cut out the neighbourhood: the “nested” part"
            lead="A normal GNN only averages direct neighbours. Then a chain, a fan-out and a funnel can look alike. The nested GNN first cuts out a small graph around the account (up to 2 hops, 36 nodes) and studies it as a whole, so the shape of the money flow becomes visible.">
            <div className="grid-2-1">
              <Subgraph sg={sg} nBanks={nBanks} />
              <div>
                <div className="verdict" style={{ gridTemplateColumns: '1fr 1fr' }}>
                  <div className="v"><div className="k">nodes</div><div className="val">{stats.total}</div><div className="tiny muted">{stats.h1} at hop 1 · {stats.h2} at hop 2</div></div>
                  <div className="v"><div className="k">outside this bank</div><div className="val">{stats.other + stats.ext}</div><div className="tiny muted">{stats.other} other members · {stats.ext} external</div></div>
                  <div className={`v ${stats.laund ? 'hit' : ''}`}><div className="k">labelled launderers nearby</div><div className="val">{stats.laund}</div><div className="tiny muted">evaluation only, not used to score</div></div>
                </div>
                <div className="legend" style={{ flexDirection: 'column', gap: 6, marginTop: 14 }}>
                  <span><i style={{ background: '#fff' }} />centre: this account</span>
                  <span>colour: bank · faded = external institution</span>
                  <span>size: how much the node counted (pooling weight)</span>
                  <span><i style={{ background: '#5ee7ff' }} />cyan links into the centre: thicker = more attention</span>
                  <span><i style={{ background: 'transparent', border: '2px solid #ff5c7a' }} />red ring: labelled laundering</span>
                  <span>arrow = money direction · dashed = device / merchant</span>
                </div>
              </div>
            </div>
          </Step>

          <Step n="4" title="Read the shape: inner GNN"
            lead="Inside that small graph, every node passes a message to its neighbours twice (2 layers). Each message is weighted by attention: how relevant the sender is. The whole graph is then summarised into 64 numbers.">
            <div className="formula">message = W<sub>relation</sub>·neighbour + edge(amount, count, cross-bank, rail) &nbsp;·&nbsp; weight = softmax(attention)</div><br />
            <div className="formula">summary z = MLP[ this account ‖ average of all nodes ‖ attention-weighted pool ]</div>
            <div className="grid-2" style={{ marginTop: 6 }}>
              <div>
                <h4>Nodes that counted most in the summary</h4>
                {sg.nodes.map((n, i) => ({ ...n, i })).sort((x, y) => y.pool - x.pool).slice(0, 6).map((n) => (
                  <div key={n.i} className="model-score">
                    <span className="small">{n.i === 0 ? <b>this account</b> : n.kind === 'account' ? <AccountLink id={n.id} name={n.name} /> : n.name} <span className="tiny muted">hop {n.hop}{n.external ? (n.member ? ' · other bank' : ' · external') : ''}</span></span>
                    <span className="att"><span style={{ display: 'block', height: '100%', width: `${Math.min(100, n.pool * 300)}%`, background: 'linear-gradient(90deg,#9b7bff,#5ee7ff)', borderRadius: 6 }} /></span>
                  </div>
                ))}
              </div>
              <div>
                <h4>The 64-number summary (z)</h4>
                <ZStrip z={ex.z} />
                <p className="tiny muted" style={{ marginTop: 8 }}>Cyan = positive, pink = negative. Not readable by eye; it is the account's "shape fingerprint" that the next steps use.</p>
              </div>
            </div>
          </Step>

          <Step n="5" title="Ask the neighbours: outer GNN (across banks)"
            lead="Now the model looks at the account's direct trading partners, using their own summaries. A partner at another member bank sends a noised summary under a pseudonym, so its risk is shared without sharing its data. Attention shows whom the model listened to.">
            <table className="table compact">
              <thead><tr><th>Partner</th><th>Bank</th><th>Relation</th><th>Their risk</th><th>Attention</th></tr></thead>
              <tbody>{ex.outer_attention.map((n, i) => (
                <tr key={i}>
                  <td><AccountLink id={n.id} name={n.name} /></td><td><Bank id={n.bank} name={n.bank_name} /></td>
                  <td className="small">{n.relations.join(', ')}</td>
                  <td>{n.member ? <Risk score={n.score} threshold={thr.account} /> : <span className="tiny muted">external · not scored</span>}</td>
                  <td><div className="att"><div style={{ width: `${n.weight * 100}%` }} /></div><span className="tiny muted">{(n.weight * 100).toFixed(0)}%</span></td>
                </tr>
              ))}</tbody>
            </table>
          </Step>

          <Step n="6" title="Score and decide"
            lead={`Two small output networks turn the summaries into probabilities. Each transfer is scored from both parties' summaries plus its own amount, rail, time and cross-bank flag. The account's risk is ${thr.combine === 'txmax' ? 'its riskiest transfer' : thr.combine === 'mean' ? 'the average of its own probability and its riskiest transfer' : 'its own probability'} (the rule that tested best on validation). Flag lines were chosen on validation data for the best balance of catches against false alarms.`}>
            <div className="verdict">
              <div className="v"><div className="k">account network output</div><div className="val">{(ex.p_acc * 100).toFixed(1)}</div><div className="tiny muted">from the account's own summaries</div></div>
              <div className={`v ${ex.tx_max >= thr.tx ? 'hit' : ''}`}><div className="k">riskiest transfer</div><div className="val">{(ex.tx_max * 100).toFixed(1)}</div><div className="tiny muted">flag line for transfers {(thr.tx * 100).toFixed(1)}</div></div>
              <div className={`v ${d.score >= thr.account ? 'hit' : 'ok'}`}><div className="k">final account risk</div><div className="val">{(d.score * 100).toFixed(1)}</div>
                <div className="small">{d.score >= thr.account ? <b className="red">FLAGGED: above {(thr.account * 100).toFixed(1)}</b> : <b className="green">not flagged: below {(thr.account * 100).toFixed(1)}</b>}</div></div>
            </div>
            <p className="tiny muted" style={{ marginTop: 12 }}>Read the score as a ranking: half of all transfers score 0.0, and only about 1 in 1,000 scores above 99. A score in the high 90s means "among the riskiest", not "98% certain". <Link className="link" to={`/accounts/${d.id}`}>Open the full investigation page →</Link></p>
          </Step>
        </>
      )}
    </div>
  )
}
