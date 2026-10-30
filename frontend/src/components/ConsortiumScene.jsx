import { useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Float, Html, OrbitControls, Sparkles, Stars } from '@react-three/drei'
import * as THREE from 'three'
import { bankColor } from '../api'

const R = 3.4

function position(i, n, external) {
  if (external) return new THREE.Vector3(0, 0, 0)
  const a = (i / n) * Math.PI * 2
  return new THREE.Vector3(Math.cos(a) * R, Math.sin(a * 2) * 0.35, Math.sin(a) * R)
}

function Flow({ from, to, volume, maxVol, suspicious }) {
  const { curve, geom } = useMemo(() => {
    const mid = from.clone().add(to).multiplyScalar(0.5)
    mid.y += 0.6 + from.distanceTo(to) * 0.25
    const c = new THREE.QuadraticBezierCurve3(from, mid, to)
    return { curve: c, geom: new THREE.TubeGeometry(c, 40, 0.008 + 0.03 * Math.sqrt(volume / maxVol), 6, false) }
  }, [from, to, volume, maxVol])
  const share = Math.min(1, suspicious / Math.max(volume, 1))
  const color = new THREE.Color('#5ee7ff').lerp(new THREE.Color('#ff5c7a'), Math.min(1, share * 4))
  const n = Math.max(1, Math.round(1 + 5 * Math.sqrt(volume / maxVol)))
  const dots = useRef([])
  const offsets = useMemo(() => Array.from({ length: n }, (_, i) => i / n), [n])
  useFrame(({ clock }) => {
    const t = clock.getElapsedTime() * 0.12
    dots.current.forEach((m, i) => { if (m) m.position.copy(curve.getPoint((t + offsets[i]) % 1)) })
  })
  return (
    <group>
      <mesh geometry={geom}><meshBasicMaterial color={color} transparent opacity={0.28} /></mesh>
      {offsets.map((_, i) => (
        <mesh key={i} ref={(m) => (dots.current[i] = m)}>
          <sphereGeometry args={[0.045, 10, 10]} />
          <meshBasicMaterial color={color} toneMapped={false} />
        </mesh>
      ))}
    </group>
  )
}

function BankNode({ node, pos, maxAcc, nBanks, onClick }) {
  const ext = node.external
  const color = ext ? '#4b5577' : bankColor(node.id, nBanks)
  const r = ext ? 0.9 : 0.22 + 0.3 * Math.sqrt(node.accounts / maxAcc)
  const ref = useRef()
  useFrame(({ clock }) => { if (ref.current) ref.current.rotation.y = clock.getElapsedTime() * 0.4 })
  return (
    <group position={pos}>
      <Float speed={ext ? 0.6 : 1.6} floatIntensity={ext ? 0.2 : 0.6} rotationIntensity={0.2}>
        <mesh ref={ref} onClick={onClick}>
          {ext ? <icosahedronGeometry args={[r, 2]} /> : <sphereGeometry args={[r, 40, 30]} />}
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={ext ? 0.25 : 0.9} roughness={0.2} metalness={0.4}
            wireframe={ext} transparent opacity={ext ? 0.35 : 1} />
        </mesh>
        {!ext && (
          <mesh scale={1.9}>
            <sphereGeometry args={[r, 24, 16]} />
            <meshBasicMaterial color={color} transparent opacity={0.08} />
          </mesh>
        )}
        {!ext && node.flagged > 0 && (
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[r * 1.6, 0.012, 8, 64]} />
            <meshBasicMaterial color="#ff5c7a" transparent opacity={0.85} toneMapped={false} />
          </mesh>
        )}
        <Html center distanceFactor={9} position={[0, r + (ext ? 0.35 : 0.42), 0]} style={{ pointerEvents: 'none' }}>
          <div style={{ whiteSpace: 'nowrap', textAlign: 'center', font: '600 12px Space Grotesk, sans-serif', color: '#eef1fb', textShadow: '0 2px 10px #000' }}>
            {node.name}
            <div style={{ font: '500 10px Inter', color: ext ? '#7d86a6' : '#b3bad3' }}>
              {node.accounts.toLocaleString()} acc{!ext && node.flagged ? <span style={{ color: '#ff5c7a' }}> · {node.flagged} flagged</span> : null}
            </div>
          </div>
        </Html>
      </Float>
    </group>
  )
}

/** Animated 3D consortium: member banks orbit the cloud of external institutions; tubes carry
 * money-flow particles (cyan = clean, red = share touching flagged accounts). */
export default function ConsortiumScene({ flows, onBank }) {
  const { nodes, links } = flows
  const nBanks = nodes.filter((n) => !n.external).length
  const pos = useMemo(() => nodes.map((n, i) => position(i, nBanks, n.external)), [nodes, nBanks])
  const maxAcc = Math.max(...nodes.filter((n) => !n.external).map((n) => n.accounts))
  const pairs = useMemo(() => {
    const m = new Map()
    links.forEach((l) => {
      const k = [Math.min(l.source, l.target), Math.max(l.source, l.target)].join('-')
      const e = m.get(k) || { a: Math.min(l.source, l.target), b: Math.max(l.source, l.target), volume: 0, suspicious: 0 }
      e.volume += l.volume; e.suspicious += l.suspicious
      m.set(k, e)
    })
    return [...m.values()].sort((x, y) => y.volume - x.volume).slice(0, 28)
  }, [links])
  const maxVol = Math.max(1, ...pairs.map((p) => p.volume))
  return (
    <Canvas camera={{ position: [0, 3.6, 8.2], fov: 45 }} dpr={[1, 1.8]} gl={{ antialias: true, alpha: true }}>
      <ambientLight intensity={0.35} />
      <pointLight position={[6, 8, 6]} intensity={60} color="#9b7bff" />
      <pointLight position={[-6, -3, -4]} intensity={40} color="#5ee7ff" />
      <Stars radius={60} depth={30} count={1500} factor={3} fade speed={0.6} />
      <Sparkles count={70} scale={[10, 4, 10]} size={1.6} speed={0.35} color="#5ee7ff" opacity={0.5} />
      {pairs.map((p) => <Flow key={`${p.a}-${p.b}`} from={pos[p.a]} to={pos[p.b]} volume={p.volume} maxVol={maxVol} suspicious={p.suspicious} />)}
      {nodes.map((n, i) => <BankNode key={n.id} node={n} pos={pos[i]} maxAcc={maxAcc} nBanks={nBanks} onClick={() => onBank && onBank(n)} />)}
      <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.45} minPolarAngle={0.6} maxPolarAngle={1.6} />
    </Canvas>
  )
}
