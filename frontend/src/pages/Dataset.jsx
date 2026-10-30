import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { BANK_COLORS, fmtNum, useApi } from '../api'
import { axis, Card, chartTip, Kpi, Loading, PageHead } from '../components/ui'

const PAT = { fan_in: 'Fan-in', fan_out: 'Fan-out', cycle: 'Cycle', scatter_gather: 'Scatter-gather', gather_scatter: 'Gather-scatter', stack: 'Stack', bipartite: 'Bipartite', random: 'Random' }

export default function Dataset() {
  const { data: d } = useApi('/api/dataset')
  if (!d) return <Loading />
  const r = d.report || {}
  const real = d.source === 'ibm_aml'
  const pats = Object.entries(d.patterns || {}).map(([k, v]) => ({ p: PAT[k] || k, n: v }))
  const rates = Object.entries(r.rates_usd || {}).sort((a, b) => b[1] - a[1])
  return (
    <div className="page">
      <PageHead eyebrow={real ? 'IBM AMLworld · HI-Small' : 'synthetic generator'} title="Dataset & cleaning">
        {real ? d.citation : 'Synthetic multi-bank world with injected rings (used for unit tests and demos).'}
      </PageHead>
      <div className="kpis">
        <Kpi label="Raw rows" value={fmtNum(r.rows)} sub={`${fmtNum(r.dropped_duplicates)} exact duplicates dropped`} />
        <Kpi label="Member accounts" value={fmtNum(d.accounts)} sub={`${d.n_banks} consortium banks`} tone="violet" />
        <Kpi label="External accounts" value={fmtNum(d.external_accounts)} sub="other institutions (ghost nodes)" />
        <Kpi label="History / held-out tx" value={`${fmtNum(d.history_tx)} / ${fmtNum(d.future_tx)}`} sub={real ? `cutoff day ${r.cutoff_day}` : ''} tone="green" />
        <Kpi label="Laundering accounts" value={fmtNum(d.positives)} sub={real ? `+${r.live_positives} only after the cutoff` : ''} tone="red" />
        <Kpi label="Labelled rings" value={fmtNum(d.rings)} sub="touching member banks" tone="amber" />
      </div>
      {real && (
        <div className="grid-2">
          <Card title="Cleaning pipeline">
            <ol className="steps">
              <li><b>Deduplicate.</b> {fmtNum(r.dropped_duplicates)} exact duplicate rows removed from {fmtNum(r.rows)}.</li>
              <li><b>Consortium.</b> The {d.n_banks} institutions with the most accounts ({(r.member_codes || []).join(', ')}) become federated clients; {fmtNum(d.external_accounts)} counterparties elsewhere become ghost nodes.</li>
              <li><b>Currency.</b> Every amount converted to USD using {rates.length} exchange rates inferred from the dataset's own cross-currency transfers.</li>
              <li><b>Self-transfers.</b> {fmtNum(r.self_transfers)} same-account reinvestments removed from the graph and kept as a feature.</li>
              <li><b>Temporal split.</b> Transactions before day {r.cutoff_day} build the graph and train the models. The {fmtNum(d.future_tx)} later ones are replayed live, and accounts that only launder after the cutoff are never used for training.</li>
              <li><b>Ring-disjoint evaluation.</b> Labelled laundering attempts that share accounts are grouped, then whole groups go to train, validation or test.</li>
            </ol>
          </Card>
          <Card title="Labelled laundering typologies (touching members)">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={pats} layout="vertical" margin={{ left: 10 }}>
                <XAxis type="number" {...axis} allowDecimals={false} />
                <YAxis type="category" dataKey="p" width={120} tick={{ fill: '#b3bad3', fontSize: 12 }} axisLine={false} tickLine={false} />
                <Tooltip {...chartTip} />
                <Bar dataKey="n" radius={[0, 6, 6, 0]}>{pats.map((_, i) => <Cell key={i} fill={BANK_COLORS[i % BANK_COLORS.length]} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}
      <div className="grid-2">
        <Card title="Account features" >
          <div className="chip-grid">{d.features.map((f) => <span key={f} className="tag" style={{ margin: 0 }}>{f}</span>)}</div>
          <p className="muted small" style={{ marginTop: 12 }}>Each feature is computed by the account's own bank from its own ledger. No raw records cross bank boundaries.</p>
        </Card>
        {real && (
          <Card title="Inferred FX rates (USD per unit)" pad={false} scroll>
            <table className="table compact"><tbody>{rates.map(([c, v]) => <tr key={c}><td className="soft">{c}</td><td className="mono">{v < 0.01 ? v.toExponential(3) : v.toFixed(4)}</td></tr>)}</tbody></table>
          </Card>
        )}
      </div>
    </div>
  )
}
