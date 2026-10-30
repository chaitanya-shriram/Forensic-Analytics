import { useEffect, useState } from 'react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, fmtMoney, useApi, useLive } from '../api'
import { AccountLink, axis, Bank, Card, chartTip, Kpi, PageHead, Risk, Truth } from '../components/ui'

export default function Live() {
  const { data: init } = useApi('/api/stream')
  const { data: initAlerts } = useApi('/api/alerts?limit=60')
  const { data: initTx } = useApi('/api/transactions/recent?limit=60')
  const [stats, setStats] = useState(null)
  const [txs, setTxs] = useState([])
  const [alerts, setAlerts] = useState([])
  const [lat, setLat] = useState([])
  const [cfg, setCfg] = useState(null)
  const [msg, setMsg] = useState(null)
  const [form, setForm] = useState({ src: '', dst: '', amount: '9500' })
  const [txResult, setTxResult] = useState(null)

  useEffect(() => {
    if (!init) return
    setStats(init)
    setLat(init.latencies.map((l, i) => ({ i, ...l })))
    setCfg({ speed: init.speed, rate: init.rate, ring_every: init.ring_every })
  }, [init])
  useEffect(() => { if (initAlerts) setAlerts(initAlerts) }, [initAlerts])
  useEffect(() => { if (initTx) setTxs(initTx) }, [initTx])

  useLive((ev) => {
    if (ev.type === 'batch') {
      setStats(ev.stats)
      setTxs((t) => [...ev.tx.slice().reverse(), ...t].slice(0, 90))
      if (ev.alerts?.length) setAlerts((a) => [...ev.alerts, ...a].slice(0, 90))
      setLat((l) => [...l, { i: (l.at(-1)?.i ?? 0) + 1, ...ev.latency }].slice(-90))
    }
    if (ev.type === 'error') setMsg(ev.message)
  })

  const real = stats?.source === 'ibm_aml'
  const start = () => api('/api/stream/start', { method: 'POST', body: cfg }).then(setStats).catch((e) => setMsg(e.message))
  const stop = () => api('/api/stream/stop', { method: 'POST' }).then(setStats)
  const inject = () => api('/api/stream/inject-ring', { method: 'POST' })
    .then((r) => setMsg(`Injected ${r.pattern.replaceAll('_', ' ')} ring #${r.id}: ${r.size} accounts across ${r.banks.length} banks.`))
    .catch((e) => setMsg(e.message))
  const submit = (e) => {
    e.preventDefault()
    const body = { src: +form.src, amount: +form.amount, ...(form.dst ? { dst: +form.dst } : { merchant: 0 }) }
    api('/api/transactions', { method: 'POST', body }).then(setTxResult).catch((er) => setTxResult({ error: er.message }))
  }

  const s = stats || {}
  const rp = s.replay
  const lp = s.live_positives
  return (
    <div className="page">
      <PageHead eyebrow={real ? 'replaying held-out real transactions' : 'simulated stream'} title="Live monitor"
        right={<>
          {s.on ? <button className="btn danger" onClick={stop}>■ Stop</button> : <button className="btn primary" onClick={start} disabled={rp && rp.done >= rp.total}>▶ {real ? 'Replay' : 'Start'} stream</button>}
          {!real && <button className="btn" onClick={inject} disabled={!s.sleepers_left}>⚑ Inject ring ({s.sleepers_left ?? 0})</button>}
        </>}>
        {real ? 'Transactions after the training cutoff arrive in time order, in micro-batches. Only accounts whose 2-hop subgraph changed are re-encoded by the inner GNN, then the outer GNN rescores the affected banks.'
          : 'Micro-batches of simulated traffic; only accounts whose 2-hop subgraph changed are re-encoded.'}
      </PageHead>
      {msg && <div className="note" onClick={() => setMsg(null)}>{msg}</div>}

      <div className="kpis">
        <Kpi label="Stream clock" value={s.time ? s.time.slice(5) : '–'} sub={s.on ? `${(s.speed / 60).toFixed(0)} min per second` : 'paused'} />
        <Kpi label={real ? 'Replayed' : 'Transactions'} value={real && rp ? `${((100 * rp.done) / rp.total).toFixed(1)}%` : (s.n_tx || 0).toLocaleString()} sub={real && rp ? `${rp.done.toLocaleString()} / ${rp.total.toLocaleString()} real tx` : `${s.ticks || 0} batches`} tone="violet" />
        <Kpi label="Latency p50 / p95" value={s.latency_ms_p50 ? `${s.latency_ms_p50.toFixed(0)} / ${s.latency_ms_p95.toFixed(0)}` : '–'} sub="ms per micro-batch" tone="green" />
        <Kpi label="Alerts" value={s.alerts ?? 0} sub={`${s.rings ?? 0} rings tracked`} tone="amber" />
        {s.live_tx && real && <Kpi label="Laundering tx flagged" value={`${s.live_tx.flagged_laundering} / ${s.live_tx.laundering}`} sub={`${s.live_tx.flagged.toLocaleString()} of ${s.live_tx.seen.toLocaleString()} replayed tx flagged`} tone="violet" />}
        {lp && <Kpi label="Future launderers caught" value={`${lp.flagged} / ${lp.active}`} sub={`active so far · ${lp.total} total in the held-out period`} tone="red" />}
      </div>
      {real && rp && <div className="progress" style={{ margin: 0 }}><div style={{ width: `${(100 * rp.done) / rp.total}%` }} /></div>}

      <div className="grid-2-1">
        <Card title="Per-batch latency">
          <ResponsiveContainer width="100%" height={210}>
            <AreaChart data={lat} margin={{ top: 10 }}>
              <defs><linearGradient id="lt" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#4ef0b0" stopOpacity={0.4} /><stop offset="1" stopColor="#4ef0b0" stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid stroke="rgba(255,255,255,.05)" vertical={false} />
              <XAxis dataKey="i" {...axis} />
              <YAxis {...axis} width={44} />
              <Tooltip {...chartTip} formatter={(v) => `${(+v).toFixed(0)} ms`} />
              <Area dataKey="total_ms" name="total" stroke="#4ef0b0" strokeWidth={2} fill="url(#lt)" />
              <Area dataKey="scoring_ms" name="GNN rescoring" stroke="#5ee7ff" fill="none" />
              <Area dataKey="features_ms" name="features" stroke="#ffbf5e" fill="none" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
        <Card title={real ? 'Labelled rings active after the cutoff' : 'Live rings · time to detection'} pad={true} scroll>
          {(s.live_rings || []).length === 0 && <p className="muted small">{real ? 'Start the replay; rings appear as their transactions arrive.' : 'Inject a ring or wait for the auto-injector.'}</p>}
          {(s.live_rings || []).map((r) => (
            <div key={r.ring} className="live-ring">
              <div className="row between"><span><b className="display">#{r.ring}</b> <span className="soft">{r.pattern.replaceAll('_', ' ')}</span></span><span className="tiny muted">{r.banks.length} bank(s)</span></div>
              <div className="progress thin"><div style={{ width: `${(100 * r.flagged) / r.members.length}%` }} /></div>
              <div className="tiny">{r.flagged}/{r.members.length} member accounts flagged · {r.detected ? <span className="green">detected after {r.minutes_to_detect.toFixed(0)} min</span> : <span className="amber">forming…</span>}</div>
            </div>
          ))}
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Transaction feed" pad={false} scroll>
          <table className="table compact">
            <thead><tr><th>Time</th><th>From</th><th>To</th><th>Rail</th><th>Amount</th><th>Risk</th></tr></thead>
            <tbody>
              {txs.map((t) => (
                <tr key={t.index} className={t.flagged ? 'hot' : ''}>
                  <td className="mono small">{t.time.slice(5)}</td>
                  <td><AccountLink id={t.src} name={t.src_name} /></td>
                  <td>{t.dst_name ? <><AccountLink id={t.dst} name={t.dst_name} />{t.cross_bank && <span className="tag tag-blue">x-bank</span>}</> : <span className="muted small">{t.merchant}</span>}</td>
                  <td className="tiny muted">{t.format || '–'}</td>
                  <td className="mono">{fmtMoney(t.amount)}</td>
                  <td><Risk score={t.risk} />{t.truth ? <span className="tag tag-red" title="labelled laundering in the dataset">L</span> : null}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card title="Alerts" pad={false} scroll>
          {alerts.length === 0 && <p className="muted small pad">No alerts yet.</p>}
          {alerts.map((a) => (
            <div key={a.id + a.ts} className="alert">
              <div className="row between">
                <div className="row gap wrap"><AccountLink id={a.account} name={a.name} /><Bank id={a.bank} name={a.bank_name} /><Truth v={a.truth} /></div>
                <Risk score={a.score} />
              </div>
              <div className="muted tiny">{a.time}{a.ring != null && <> · ring #{a.ring}</>}</div>
              <ul className="reasons">{a.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
            </div>
          ))}
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Stream settings">
          {cfg && <div className="form-row">
            <label>Speed (stream-seconds per second)<input type="number" value={cfg.speed} onChange={(e) => setCfg({ ...cfg, speed: +e.target.value })} /></label>
            {!real && <label>Tx per batch (mean)<input type="number" value={cfg.rate} onChange={(e) => setCfg({ ...cfg, rate: +e.target.value })} /></label>}
            {!real && <label>Auto-inject ring every N batches<input type="number" value={cfg.ring_every} onChange={(e) => setCfg({ ...cfg, ring_every: +e.target.value })} /></label>}
          </div>}
          <p className="muted tiny" style={{ marginTop: 8 }}>Applies when the stream starts. {real && 'At most 400 transactions per batch; faster speeds queue the rest.'}</p>
        </Card>
        <Card title="Score a transaction now">
          <form onSubmit={submit} className="form-row">
            <label>From account id<input required value={form.src} onChange={(e) => setForm({ ...form, src: e.target.value })} placeholder="e.g. 12" /></label>
            <label>To account id<input value={form.dst} onChange={(e) => setForm({ ...form, dst: e.target.value })} placeholder="e.g. 1400" /></label>
            <label>Amount (USD)<input required value={form.amount} onChange={(e) => setForm({ ...form, amount: e.target.value })} /></label>
            <button className="btn primary" type="submit">Score</button>
          </form>
          {txResult?.error && <p className="error" style={{ marginTop: 10 }}>{txResult.error}</p>}
          {txResult?.latency && (
            <div className="small" style={{ marginTop: 12 }}>
              <p>Processed in <b>{txResult.latency.total_ms.toFixed(0)} ms</b>, {txResult.latency.rescored} subgraphs re-encoded.</p>
              {Object.entries(txResult.before).map(([a, b]) => (
                <p key={a}><AccountLink id={a} name={`account ${a}`} />: {(b * 100).toFixed(1)} → <b>{(txResult.after[a] * 100).toFixed(1)}</b></p>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
