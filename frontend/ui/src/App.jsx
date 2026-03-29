// src/App.jsx
import { useState, useEffect, useCallback, useRef } from 'react'

import Header          from './components/Header.jsx'
import ControlPanel    from './components/ControlPanel.jsx'
import AlertFeed       from './components/AlertFeed.jsx'
import PredictionPanel from './components/PredictionPanel.jsx'
import LogTable        from './components/LogTable.jsx'
import MetricChart     from './components/MetricChart.jsx'

import {
  fetchModelStatus,
  fetchHistory,
  fetchAlerts,
  openStream,
} from './api.js'

const MAX_HISTORY = 200
const MAX_ALERTS  = 100

// ── Rule-based alert generator ────────────────────────────────────────────────
// Returns an alert object if any metric exceeds its user-set limit,
// or null if everything is within limits.
function checkLimits(data, limits) {
  if (!data || data.status !== 'ok') return null

  const exceeded = []
  if (limits.cpu    != null && data.cpu    > limits.cpu)    exceeded.push(`CPU ${data.cpu.toFixed(1)}% > ${limits.cpu}%`)
  if (limits.memory != null && data.memory > limits.memory) exceeded.push(`MEM ${data.memory.toFixed(1)}% > ${limits.memory}%`)
  if (limits.disk   != null && data.disk   > limits.disk)   exceeded.push(`DISK ${data.disk.toFixed(1)}% > ${limits.disk}%`)

  if (exceeded.length === 0) return null

  return {
    source:    'rule',
    severity:  'CRITICAL',
    timestamp: data.timestamp,
    cpu:       data.cpu,
    memory:    data.memory,
    disk:      data.disk,
    exceeded,
    error:     data.error,
    error_ratio: data.error_ratio,
  }
}

export default function App() {
  const [modelStatus,  setModelStatus]  = useState(null)
  const [history,      setHistory]      = useState([])
  const [alerts,       setAlerts]       = useState([])
  const [latest,       setLatest]       = useState(null)
  const [streamStatus, setStreamStatus] = useState('connecting')

  // Per-metric user-defined limits (null = not set)
  const [limits, setLimits] = useState({ cpu: null, memory: null, disk: null })

  // Keep limits in a ref so the SSE callback always reads latest value
  const limitsRef = useRef(limits)
  useEffect(() => { limitsRef.current = limits }, [limits])

  // ── Bootstrap ──────────────────────────────────────────────────────────────
  useEffect(() => {
    async function init() {
      try {
        const [status, hist, alts] = await Promise.all([
          fetchModelStatus(),
          fetchHistory(MAX_HISTORY),
          fetchAlerts(50),
        ])
        setModelStatus(status)
        setHistory(hist)
        // Mark backend alerts with source=lstm
        setAlerts(alts.map(a => ({ ...a, source: 'lstm' })))
        if (hist.length) setLatest(hist[hist.length - 1])
      } catch (e) {
        console.error('Init failed:', e)
      }
    }
    init()
  }, [])

  // ── SSE stream ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const close = openStream(
      (data) => {
        setStreamStatus('connected')
        if (data.status !== 'ok') return

        setLatest(data)

        // Add to history
        setHistory(prev => {
          const next = [...prev, data]
          return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next
        })

        const newAlerts = []

        // ── Rule-based alert (user metric limits) ──
        const ruleAlert = checkLimits(data, limitsRef.current)
        if (ruleAlert) newAlerts.push(ruleAlert)

        // ── LSTM anomaly alert ──
        if (data.is_anomaly) {
          newAlerts.push({ ...data, source: 'lstm' })
        }

        if (newAlerts.length > 0) {
          setAlerts(prev => {
            const next = [...prev, ...newAlerts]
            return next.length > MAX_ALERTS ? next.slice(-MAX_ALERTS) : next
          })
        }
      },
      () => setStreamStatus('error')
    )
    return close
  }, [])

  // ── Callbacks ──────────────────────────────────────────────────────────────
  const handleModelLoaded = useCallback(async () => {
    const s = await fetchModelStatus()
    setModelStatus(s)
  }, [])

  const handleLimitsChange = useCallback((newLimits) => {
    setLimits(newLimits)
    limitsRef.current = newLimits
  }, [])

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{
      minHeight:     '100vh',
      background:    'var(--bg)',
      display:       'flex',
      flexDirection: 'column',
    }}>

      {/* Header */}
      <Header
        modelStatus={modelStatus}
        onModelLoaded={handleModelLoaded}
      />

      {/* Stream status bar */}
      {streamStatus !== 'connected' && (
        <div style={{
          padding:      '5px 20px',
          background:   'rgba(250,204,21,.07)',
          borderBottom: '1px solid rgba(250,204,21,.18)',
          fontFamily:   'var(--font-mono)',
          fontSize:     '10px',
          color:        'var(--warning)',
        }}>
          {streamStatus === 'connecting'
            ? '⟳ Connecting to inference stream...'
            : '⚠ Stream disconnected — is the backend running?'}
        </div>
      )}

      {/* Main layout */}
      <main style={{
        flex:    1,
        padding: '16px 20px',
        display: 'flex',
        gap:     '14px',
      }}>

        {/* Left column */}
        <div style={{
          flex:          1,
          display:       'flex',
          flexDirection: 'column',
          gap:           '14px',
          minWidth:      0,
        }}>

          {/* Zone 2: metric cards + per-metric limit inputs */}
          <ControlPanel
            latest={latest}
            modelStatus={modelStatus}
            limits={limits}
            onLimitsChange={handleLimitsChange}
          />

          {/* Zone 3: live DB log table */}
          <LogTable history={history} limits={limits} />

          {/* Zone 4: three individual metric charts */}
          <div style={{
            display:             'grid',
            gridTemplateColumns: '1fr 1fr 1fr',
            gap:                 '14px',
            minHeight:           '260px',
          }}>
            <MetricChart
              history={history}
              dataKey="memory"
              label="Realtime Memory Usage"
              color="var(--mem-color)"
              warningLine={limits.memory ?? 85}
            />
            <MetricChart
              history={history}
              dataKey="cpu"
              label="Realtime CPU Usage"
              color="var(--cpu-color)"
              warningLine={limits.cpu ?? 85}
            />
            <MetricChart
              history={history}
              dataKey="disk"
              label="Realtime Disk Usage"
              color="var(--disk-color)"
              warningLine={limits.disk ?? 90}
            />
          </div>
        </div>

        {/* Right column: sticky alert feed + prediction */}
        <div style={{
          width:         '290px',
          flexShrink:    0,
          display:       'flex',
          flexDirection: 'column',
          gap:           '14px',
          position:      'sticky',
          top:           '57px',
          height:        'calc(100vh - 73px)',
          alignSelf:     'flex-start',
        }}>
          <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <AlertFeed alerts={alerts} />
          </div>
          <div style={{ height: '280px', flexShrink: 0 }}>
            <PredictionPanel history={history} />
          </div>
        </div>

      </main>

      {/* Footer */}
      <footer style={{
        padding:        '7px 20px',
        borderTop:      '1px solid var(--border)',
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'space-between',
        fontFamily:     'var(--font-mono)',
        fontSize:       '10px',
        color:          'var(--text-dimmer)',
      }}>
        <span>AIOps · Incident Detection · LSTM Autoencoder</span>
        <span style={{ display:'flex', alignItems:'center', gap:'6px' }}>
          <span
            className={`dot ${streamStatus === 'connected' ? 'normal' : 'warning'}`}
            style={{ margin: 0 }}
          />
          {streamStatus === 'connected' ? 'Stream live' : 'Stream offline'}
          {latest?.timestamp && (
            <span style={{ marginLeft:'10px' }}>
              last: {new Date(latest.timestamp).toLocaleTimeString('en-GB', { hour12:false })}
            </span>
          )}
        </span>
      </footer>
    </div>
  )
}