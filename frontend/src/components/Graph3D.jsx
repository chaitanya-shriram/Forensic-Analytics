import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
import { bankColor, riskColor } from '../api'

let haloTex = null
function halo() {
  if (haloTex) return haloTex
  const c = document.createElement('canvas')
  c.width = c.height = 128
  const g = c.getContext('2d')
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64)
  grd.addColorStop(0, 'rgba(255,255,255,0.9)')
  grd.addColorStop(0.25, 'rgba(255,255,255,0.35)')
  grd.addColorStop(1, 'rgba(255,255,255,0)')
  g.fillStyle = grd
  g.fillRect(0, 0, 128, 128)
  haloTex = new THREE.CanvasTexture(c)
  return haloTex
}

function useSize(ref) {
  const [w, setW] = useState(600)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [ref])
  return w
}

/**
 * 3D money-flow network. Nodes: accounts (colour = bank, size = risk, halo = flagged) and devices.
 * Links carry directional particles whose density follows the amount transferred.
 */
export default function Graph3D({ nodes, edges, focus, onSelect, height = 480, nBanks = 8 }) {
  const wrap = useRef(null)
  const fg = useRef(null)
  const width = useSize(wrap)
  const data = useMemo(() => {
    const maxAmt = Math.max(1, ...edges.map((e) => e.amount || 0))
    return {
      nodes: nodes.map((n) => ({ ...n })),
      links: edges.map((e) => ({ ...e, w: e.kind === 'device' ? 0.4 : 0.6 + 3 * Math.sqrt((e.amount || 0) / maxAmt) })),
    }
  }, [nodes, edges])

  useEffect(() => {
    const g = fg.current
    if (!g) return
    g.d3Force('charge').strength(-90)
    g.d3Force('link').distance(42)
    const ctl = g.controls()
    ctl.autoRotate = true
    ctl.autoRotateSpeed = 0.6
    const t = setTimeout(() => g.zoomToFit(600, 40), 900)
    return () => clearTimeout(t)
  }, [data])

  const nodeObject = (n) => {
    const group = new THREE.Group()
    if (n.kind === 'device') {
      const m = new THREE.Mesh(new THREE.OctahedronGeometry(3.2), new THREE.MeshStandardMaterial({ color: '#9aa3c0', emissive: '#3a4060', roughness: 0.3, metalness: 0.6 }))
      group.add(m)
      return group
    }
    const ext = n.bank >= nBanks
    const color = new THREE.Color(ext ? '#566080' : bankColor(n.bank, nBanks))
    const r = ext ? 2.4 : 3 + 5 * (n.score ?? 0)
    const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.55, roughness: 0.25, metalness: 0.35, transparent: ext, opacity: ext ? 0.7 : 1 })
    group.add(new THREE.Mesh(new THREE.SphereGeometry(r, 24, 18), mat))
    if ((n.score ?? 0) >= 0.5 || n.id === focus) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: halo(), color: n.id === focus ? '#ffffff' : riskColor(n.score), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }))
      sp.scale.set(r * 6, r * 6, 1)
      group.add(sp)
    }
    if (n.member) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(r + 2, 0.35, 8, 40), new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.7 }))
      group.add(ring)
    }
    return group
  }

  return (
    <div ref={wrap} className="graph3d" style={{ height }}>
      <ForceGraph3D
        ref={fg}
        width={width}
        height={height}
        graphData={data}
        backgroundColor="rgba(0,0,0,0)"
        showNavInfo={false}
        nodeThreeObject={nodeObject}
        nodeLabel={(n) => `<div style="background:rgba(10,12,30,.9);border:1px solid rgba(255,255,255,.15);padding:6px 10px;border-radius:10px;font:12px Inter">
          <b>${n.label}</b>${n.kind === 'device' ? '' : `<br/>risk ${((n.score ?? 0) * 100).toFixed(1)}${n.truth ? ' · <span style="color:#ff5c7a">laundering</span>' : ''}`}</div>`}
        linkColor={(l) => (l.kind === 'device' ? 'rgba(180,190,220,0.35)' : l.kind === 'inflow' ? 'rgba(255,191,94,0.6)' : 'rgba(150,170,255,0.55)')}
        linkWidth={(l) => l.w}
        linkOpacity={0.55}
        linkDirectionalParticles={(l) => (l.kind === 'device' ? 0 : Math.min(6, 1 + Math.round(l.w * 1.2)))}
        linkDirectionalParticleWidth={(l) => 1.2 + l.w * 0.5}
        linkDirectionalParticleSpeed={0.006}
        linkDirectionalParticleColor={(l) => (l.kind === 'inflow' ? '#ffbf5e' : '#5ee7ff')}
        linkDirectionalArrowLength={(l) => (l.kind === 'device' ? 0 : 3)}
        linkDirectionalArrowRelPos={1}
        onNodeClick={(n) => onSelect && onSelect(n)}
        enableNodeDrag={false}
        warmupTicks={60}
        cooldownTicks={120}
      />
      <div className="hint">drag to orbit · scroll to zoom · click a node</div>
    </div>
  )
}
