import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useApi } from '../api'
import { AccountLink, Bank, Card, Loading, PageHead, Risk, Truth } from '../components/ui'

export default function Accounts() {
  const [sp] = useSearchParams()
  const [bank, setBank] = useState(sp.get('bank') ?? '')
  const [q, setQ] = useState('')
  const [flagged, setFlagged] = useState('')
  const [page, setPage] = useState(0)
  const limit = 50
  const params = new URLSearchParams({ limit, offset: page * limit })
  if (bank !== '') params.set('bank', bank)
  if (q) params.set('q', q)
  if (flagged !== '') params.set('flagged', flagged)
  const { data } = useApi(`/api/accounts?${params}`, [])
  const { data: o } = useApi('/api/overview')
  return (
    <div className="page">
      <PageHead eyebrow="member customers" title="Accounts">
        Ranked by federated GNN risk. The label column is the dataset's ground truth, shown for evaluation only.
      </PageHead>
      <Card>
        <div className="row gap wrap" style={{ paddingTop: 14 }}>
          <input className="search" placeholder="Search account id or pseudonym…" value={q} onChange={(e) => { setQ(e.target.value); setPage(0) }} />
          <select value={bank} onChange={(e) => { setBank(e.target.value); setPage(0) }}>
            <option value="">All member banks</option>
            {o?.banks.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}
          </select>
          <select value={flagged} onChange={(e) => { setFlagged(e.target.value); setPage(0) }}>
            <option value="">All accounts</option><option value="true">Flagged</option><option value="false">Not flagged</option>
          </select>
          <span className="pill">{data?.total?.toLocaleString() ?? '…'} results</span>
        </div>
      </Card>
      <Card pad={false}>
        {!data ? <Loading /> : (
          <table className="table">
            <thead><tr><th>Account</th><th>Pseudonym</th><th>Bank</th><th>Risk</th><th>Status</th><th>Label</th><th>Split</th></tr></thead>
            <tbody>
              {data.items.map((a) => (
                <tr key={a.id}>
                  <td><AccountLink id={a.id} name={a.name} /></td>
                  <td className="mono tiny muted">{a.pseudonym}</td>
                  <td><Bank id={a.bank} name={a.bank_name} /></td>
                  <td><Risk score={a.score} /></td>
                  <td>{a.flagged ? <span className="tag tag-red">flagged</span> : <span className="tag">clear</span>}</td>
                  <td><Truth v={a.truth} /></td>
                  <td className="small muted">{a.split}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="row gap pad">
          <button className="btn" disabled={page === 0} onClick={() => setPage(page - 1)}>← Prev</button>
          <span className="small muted">page {page + 1}</span>
          <button className="btn" disabled={!data || (page + 1) * limit >= data.total} onClick={() => setPage(page + 1)}>Next →</button>
        </div>
      </Card>
    </div>
  )
}
