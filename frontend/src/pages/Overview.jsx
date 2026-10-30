import { lazy, Suspense } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { fmtMoney, fmtNum, useApi } from '../api'
import { AccountLink, Bank, BankDots, Card, ErrorBox, Kpi, Loading, PageHead, Risk, Truth } from '../components/ui'

const ConsortiumScene = lazy(() => import('../components/ConsortiumScene'))

function Party({ id, name, bank, score, thr }) {
  if (id == null || id < 0) return <span className="muted small">—</span>
  return (
    <div className="party">
      <AccountLink id={id} name={name} />
      <span className="tiny">{bank}{score != null && <> · acct {(score * 100).toFixed(1)}{score >= thr && <b className="red"> ●</b>}</>}</span>
    </div>
  )
}

export default function Overview() {
  const nav = useNavigate()
  const { data: o, error } = useApi('/api/overview', [], { interval: 8000 })
  const { data: q } = useApi('/api/review?limit=25', [], { interval: 8000 })
  const { data: accts } = useApi('/api/accounts?flagged=true&limit=10', [], { interval: 10000 })
  const { data: rings } = useApi('/api/rings', [], { interval: 12000 })
  const { data: flows } = useApi('/api/flows')
  if (error) return <ErrorBox error={error} />
  if (!o) return <Loading />
  const thr = o.thresholds || {}
  const bankName = (b) => o.banks[b]?.name ?? 'External'
  return (
    <div className="page">
      <PageHead eyebrow="triage" title="What needs review"
        right={<div className="thr-pills">
          <span className="pill" title="transactions at or above this risk are flagged">tx flag ≥ <b>{(thr.tx * 100).toFixed(1)}</b></span>
          <span className="pill" title="accounts at or above this risk are flagged">account flag ≥ <b>{(thr.account * 100).toFixed(1)}</b></span>
          <Link className="pill link" to="/gnn">how scores are made →</Link>
        </div>}>
        Riskiest transfers first. The white tick on each bar is the flag line, and red means over it. Click any row to investigate the account.
      </PageHead>

      <div className="kpis">
        <Kpi label="Flagged transactions" value={fmtNum(q?.total_flagged)} sub={`of ${fmtNum(o.transfers)} transfers`} tone="red" />
        <Kpi label="Flagged accounts" value={fmtNum(o.flagged)} sub={`of ${fmtNum(o.accounts)} member accounts`} tone="amber" />
        <Kpi label="Rings" value={o.rings} sub={`${fmtMoney(o.suspicious_flow)} moved inside`} tone="violet" />
        <Kpi label="Live alerts" value={fmtNum(o.alerts)} sub={o.stream?.on ? 'stream running' : <Link className="link" to="/live">start live replay →</Link>} />
      </div>

      <div className="grid-2-1">
        <Card title="Review queue · riskiest transfers" pad={false} scroll className="queue-card" extra={<span className="tiny muted">labels shown for evaluation only</span>}>
          {!q ? <Loading /> : (
            <table className="table compact queue">
              <thead><tr><th>Risk</th><th>From</th><th>To</th><th>Amount</th><th>Why</th></tr></thead>
              <tbody>
                {q.items.map((t) => (
                  <tr key={t.index} onClick={() => nav(`/accounts/${t.src_score != null ? t.src : t.dst}`)}>
                    <td><Risk score={t.risk} threshold={thr.tx} />{t.truth ? <span className="tag tag-red" title="labelled laundering">L</span> : null}</td>
                    <td><Party id={t.src} name={t.src_name} bank={bankName(t.src_bank)} score={t.src_score} thr={thr.account} /></td>
                    <td><Party id={t.dst} name={t.dst_name} bank={bankName(t.dst_bank)} score={t.dst_score} thr={thr.account} /></td>
                    <td><div className="mono">{fmtMoney(t.amount)}</div><div className="tiny muted">{t.format} · {t.time.slice(5)}{t.cross_bank ? ' · x-bank' : ''}</div></td>
                    <td className="why">{(t.src_reasons || t.dst_reasons || [])[0] || <span className="muted">network pattern (see account)</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
        <div className="col">
          <Card title="Flagged accounts" pad={false} extra={<Link to="/accounts" className="link tiny">all →</Link>}>
            <div className="mini-list">
              {(accts?.items || []).map((a) => (
                <Link key={a.id} to={`/accounts/${a.id}`} className="mini-row">
                  <div><div className="mono small">{a.name}</div><div className="tiny muted">{a.bank_name}</div></div>
                  <div className="row gap"><Truth v={a.truth} /><Risk score={a.score} threshold={thr.account} /></div>
                </Link>
              ))}
            </div>
          </Card>
          <Card title="Rings" pad={false} extra={<Link to="/rings" className="link tiny">all →</Link>}>
            <div className="mini-list">
              {(rings || []).slice(0, 6).map((r) => (
                <Link key={r.id} to={`/rings/${r.id}`} className="mini-row">
                  <div><div className="small"><b className="display">#{r.id}</b> {r.pattern_label}</div><div className="tiny muted">{r.size} accounts · {r.n_tx ?? r.n_links} tx · {fmtMoney(r.total_flow)}</div></div>
                  <BankDots banks={r.banks} />
                </Link>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <div className="grid-1-2">
        <Card title="Cross-bank money flow" pad={false}>
          <div style={{ height: 320, position: 'relative' }}>
            {flows ? <Suspense fallback={<Loading />}><ConsortiumScene flows={flows} onBank={(b) => !b.external && nav(`/accounts?bank=${b.id}`)} /></Suspense> : <Loading />}
          </div>
          <div className="legend pad"><span><i style={{ background: '#5ee7ff' }} />clean flow</span><span><i style={{ background: '#ff5c7a' }} />touches flagged accounts</span><span><i style={{ background: '#4b5577' }} />external institutions</span></div>
        </Card>
        <Card title="Member banks" pad={false}>
          <table className="table compact">
            <thead><tr><th>Bank</th><th>Accounts</th><th>Flagged</th><th>Share flagged</th><th>External links</th></tr></thead>
            <tbody>
              {o.banks.map((b) => (
                <tr key={b.id} onClick={() => nav(`/accounts?bank=${b.id}`)} style={{ cursor: 'pointer' }}>
                  <td><Bank id={b.id} name={b.name} /></td>
                  <td className="mono">{fmtNum(b.accounts)}</td>
                  <td className="mono amber">{b.flagged}</td>
                  <td className="mono">{((100 * b.flagged) / b.accounts).toFixed(1)}%</td>
                  <td className="mono">{fmtNum(b.external_counterparties)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="tiny muted pad">Model: {o.active_model.label}. Test account AUC {o.active_model.auc.toFixed(3)}.</p>
        </Card>
      </div>
    </div>
  )
}
