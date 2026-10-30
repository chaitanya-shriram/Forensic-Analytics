import { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react'

export async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw Object.assign(new Error(data?.detail?.message || data?.detail || res.statusText), { status: res.status, data })
  return data
}

export function useApi(path, deps = [], { interval } = {}) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const load = useCallback(() => {
    if (!path) return
    api(path).then((d) => { setData(d); setError(null) }).catch(setError).finally(() => setLoading(false))
  }, [path])
  useEffect(() => {
    setLoading(true)
    load()
    if (!interval) return
    const t = setInterval(load, interval)
    return () => clearInterval(t)
  }, [load, interval, ...deps])
  return { data, error, loading, reload: load, setData }
}

// ---- live event bus over a single websocket
const LiveCtx = createContext(null)

export function LiveProvider({ children }) {
  const [connected, setConnected] = useState(false)
  const subs = useRef(new Set())
  useEffect(() => {
    let ws, stop = false, retry
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/ws`)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => { setConnected(false); if (!stop) retry = setTimeout(connect, 1500) }
      ws.onmessage = (m) => {
        const ev = JSON.parse(m.data)
        subs.current.forEach((f) => f(ev))
      }
    }
    connect()
    return () => { stop = true; clearTimeout(retry); ws && ws.close() }
  }, [])
  const subscribe = useCallback((f) => { subs.current.add(f); return () => subs.current.delete(f) }, [])
  return <LiveCtx.Provider value={{ connected, subscribe }}>{children}</LiveCtx.Provider>
}

export function useLive(handler) {
  const ctx = useContext(LiveCtx)
  const ref = useRef(handler)
  ref.current = handler
  useEffect(() => ctx.subscribe((ev) => ref.current(ev)), [ctx])
  return ctx
}

export const BANK_COLORS = ['#5ee7ff', '#9b7bff', '#4ef0b0', '#ffbf5e', '#ff5c9a', '#6aa0ff', '#f4e46b', '#ff8a5c', '#7d86a6']
export const EXTERNAL_COLOR = '#566080'
export const bankColor = (b, nBanks = 8) => (b >= nBanks ? EXTERNAL_COLOR : BANK_COLORS[b % BANK_COLORS.length])
export const fmtMoney = (v) => v == null ? '–' : v >= 1e6 ? `$${(v / 1e6).toFixed(2)}M` : v >= 1e3 ? `$${(v / 1e3).toFixed(1)}k` : `$${v.toFixed(0)}`
export const fmtPct = (v, d = 1) => v == null ? '–' : `${(v * 100).toFixed(d)}%`
export const fmtNum = (v) => v == null ? '–' : v.toLocaleString()
export const riskColor = (s) => s >= 0.8 ? '#ff5c7a' : s >= 0.5 ? '#ffbf5e' : s >= 0.2 ? '#f4e46b' : '#4ef0b0'

// dataset metadata is static for a session: fetch once, share everywhere
let metaPromise = null
export function useMeta() {
  const [meta, setMeta] = useState(null)
  useEffect(() => {
    metaPromise = metaPromise || api('/api/dataset').catch((e) => { metaPromise = null; throw e })
    metaPromise.then(setMeta).catch(() => {})
  }, [])
  return meta
}
