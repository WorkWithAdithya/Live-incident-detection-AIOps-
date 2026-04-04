// src/App.jsx
import { useState, useEffect, useCallback, useRef } from 'react'

import Header          from './components/Header.jsx'
import ControlPanel    from './components/ControlPanel.jsx'
import AlertFeed       from './components/AlertFeed.jsx'
import PredictionPanel from './components/PredictionPanel.jsx'
import LogTable        from './components/LogTable.jsx'
import MetricChart     from './components/MetricChart.jsx'

import { fetchModelStatus, fetchLogs, fetchAlerts, openStream, sendRuleAlert, fetchEmailStatus } from './api.js'

const MAX_HISTORY = 300
const MAX_ALERTS  = 100

const DEFAULT_LIMITS = {
  cpu_warning: null,    cpu_critical: null,
  memory_warning: null, memory_critical: null,
  disk_warning: null,   disk_critical: null,
}

// ── Severity helpers ──────────────────────────────────────────────────────────

function getSeverity(row, limits) {
  // Returns 'CRITICAL' | 'WARNING' | 'NORMAL'
  let worst = 'NORMAL'
  for (const key of ['cpu', 'memory', 'disk']) {
    const val = row[key]
    if (val == null) continue
    const c = limits[`${key}_critical`]
    const w = limits[`${key}_warning`]
    if (c != null && val > c) return 'CRITICAL'   // short-circuit
    if (w != null && val > w) worst = 'WARNING'
  }
  return worst
}

function getExceeded(row, limits) {
  const exceeded = []
  for (const [key, label] of [['cpu','CPU'],['memory','Memory'],['disk','Disk']]) {
    const val = row[key]
    if (val == null) continue
    const c = limits[`${key}_critical`]
    const w = limits[`${key}_warning`]
    if (c != null && val > c) exceeded.push(`${label} ${val.toFixed(1)}% > ${c}% (critical)`)
    else if (w != null && val > w) exceeded.push(`${label} ${val.toFixed(1)}% > ${w}% (warning)`)
  }
  return exceeded
}

function buildRuleAlert(row, limits) {
  const severity = getSeverity(row, limits)
  if (severity === 'NORMAL') return null
  return {
    source:      'rule',
    severity,
    timestamp:   row.timestamp,
    cpu:         row.cpu,
    memory:      row.memory,
    disk:        row.disk,
    exceeded:    getExceeded(row, limits),
    error:       row.error,
    error_ratio: row.error_ratio,
  }
}

// rowSeverity used by LogTable — reads live limits state
export function rowSeverity(row, limits) {
  return getSeverity(row, limits)
}

// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  const [modelStatus,  setModelStatus]  = useState(null)
  const [history,      setHistory]      = useState([])
  const [alerts,       setAlerts]       = useState([])
  const [latest,       setLatest]       = useState(null)
  const [streamStatus, setStreamStatus] = useState('connecting')
  const [limits,       setLimits]       = useState(DEFAULT_LIMITS)
  const [emailEnabled, setEmailEnabled] = useState(false)

  // Always-current refs — safe to read inside SSE callback
  const limitsRef  = useRef(limits)
  const historyRef = useRef(history)
  useEffect(() => { limitsRef.current  = limits  }, [limits])
  useEffect(() => { historyRef.current = history }, [history])

  // ── Bootstrap ──────────────────────────────────────────────────────────────
  useEffect(() => {
    async function init() {
      try {
        const [status, logs, alts, emailSt] = await Promise.all([
          fetchModelStatus(),
          fetchLogs(200),
          fetchAlerts(50),
          fetchEmailStatus(),
        ])
        setEmailEnabled(emailSt?.enabled ?? false)
        setModelStatus(status)
        const enriched = logs.map(r => ({
          ...r,
          error: null, threshold: status?.threshold ?? null,
          error_ratio: null, is_anomaly: false,
          warming_up: false, actual_rows: 0, flagged: [],
        }))
        setHistory(enriched)
        historyRef.current = enriched
        if (enriched.length) setLatest(enriched[enriched.length - 1])
        // Only keep WARNING/CRITICAL LSTM alerts from backend
        setAlerts(alts
          .filter(a => a.severity === 'WARNING' || a.severity === 'CRITICAL')
          .map(a => ({ ...a, source: 'lstm' }))
        )
      } catch (e) { console.error('Init failed:', e) }
    }
    init()
  }, [])

  // ── SSE: each event = one new DB row ──────────────────────────────────────
  useEffect(() => {
    const close = openStream(
      (data) => {
        setStreamStatus('connected')
        if (data.status === 'no_new_rows' || data.status === 'model_not_loaded') return
        if (data.status !== 'ok') return

        setLatest(data)

        setHistory(prev => {
          if (data.id && prev.some(r => r.id === data.id)) return prev
          const next = [...prev, data]
          return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next
        })

        const newAlerts = []

        // Rule-based: check new row against current limits
        const ruleAlert = buildRuleAlert(data, limitsRef.current)
        if (ruleAlert) {
          newAlerts.push(ruleAlert)
          // Email on new SSE rule breach
          const tInfo = {}
          const lim = limitsRef.current
          for (const [key, label] of [['cpu','CPU'],['memory','Memory'],['disk','Disk']]) {
            if (lim[`${key}_warning`]  != null) tInfo[`${label} Warning`]  = lim[`${key}_warning`]
            if (lim[`${key}_critical`] != null) tInfo[`${label} Critical`] = lim[`${key}_critical`]
          }
          sendRuleAlert({
            severity: ruleAlert.severity,
            cpu: data.cpu, memory: data.memory, disk: data.disk,
            exceeded: ruleAlert.exceeded,
            threshold_info: tInfo,
          })
        }

        // LSTM-based: only WARNING/CRITICAL
        if (data.is_anomaly && (data.severity === 'WARNING' || data.severity === 'CRITICAL')) {
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

  // ── When limits change: immediately scan ALL history rows ─────────────────
  // This is the key fix — if user sets limits after data is already loaded,
  // we generate alerts for existing rows right away without waiting for new SSE events.
  const handleLimitsChange = useCallback((newLimits) => {
    setLimits(newLimits)
    limitsRef.current = newLimits

    // Scan current history and build rule alerts for any exceeding row
    const ruleAlerts = historyRef.current
      .map(row => buildRuleAlert(row, newLimits))
      .filter(Boolean)

    if (ruleAlerts.length > 0) {
      // Send email for worst severity found
      const worstAlert = ruleAlerts.reduce((a, b) =>
        (b.severity === 'CRITICAL' ? b : a.severity === 'CRITICAL' ? a : b), ruleAlerts[0])

      // Build threshold_info for email
      const tInfo = {}
      for (const [key, label] of [['cpu','CPU'],['memory','Memory'],['disk','Disk']]) {
        if (newLimits[`${key}_warning`]  != null) tInfo[`${label} Warning`]  = newLimits[`${key}_warning`]
        if (newLimits[`${key}_critical`] != null) tInfo[`${label} Critical`] = newLimits[`${key}_critical`]
      }
      sendRuleAlert({
        severity:       worstAlert.severity,
        cpu:            worstAlert.cpu,
        memory:         worstAlert.memory,
        disk:           worstAlert.disk,
        exceeded:       worstAlert.exceeded,
        threshold_info: tInfo,
      })

      setAlerts(prev => {
        // Remove previous rule alerts (they'll be regenerated fresh)
        const lstmOnly = prev.filter(a => a.source !== 'rule')
        const next = [...lstmOnly, ...ruleAlerts]
        return next.length > MAX_ALERTS ? next.slice(-MAX_ALERTS) : next
      })
    } else {
      // No rows exceed the new limits — clear old rule alerts
      setAlerts(prev => prev.filter(a => a.source !== 'rule'))
    }
  }, [])

  const handleModelLoaded = useCallback(async () => {
    const s = await fetchModelStatus()
    setModelStatus(s)
  }, [])

  return (
    <div style={{ minHeight:'100vh', background:'var(--bg)', display:'flex', flexDirection:'column' }}>

      <Header modelStatus={modelStatus} onModelLoaded={handleModelLoaded} />

      {streamStatus !== 'connected' && (
        <div style={{
          padding:'5px 20px', background:'rgba(250,204,21,.07)',
          borderBottom:'1px solid rgba(250,204,21,.18)',
          fontFamily:'var(--font-mono)', fontSize:'10px', color:'var(--warning)',
        }}>
          {streamStatus === 'connecting' ? '⟳ Connecting...' : '⚠ Stream disconnected'}
        </div>
      )}

      <main style={{ flex:1, padding:'16px 20px', display:'flex', gap:'14px' }}>

        {/* Left column */}
        <div style={{ flex:1, display:'flex', flexDirection:'column', gap:'14px', minWidth:0 }}>

          <ControlPanel
            modelStatus={modelStatus}
            limits={limits}
            onLimitsChange={handleLimitsChange}
          />

          <LogTable history={history} limits={limits} rowSeverity={getSeverity} />

          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:'14px', minHeight:'260px' }}>
            <MetricChart
              history={history} dataKey="memory" label="Realtime Memory Usage"
              color="var(--mem-color)"
              warningLine={limits.memory_warning}
              criticalLine={limits.memory_critical}
            />
            <MetricChart
              history={history} dataKey="cpu" label="Realtime CPU Usage"
              color="var(--cpu-color)"
              warningLine={limits.cpu_warning}
              criticalLine={limits.cpu_critical}
            />
            <MetricChart
              history={history} dataKey="disk" label="Realtime Disk Usage"
              color="var(--disk-color)"
              warningLine={limits.disk_warning}
              criticalLine={limits.disk_critical}
            />
          </div>
        </div>

        {/* Right column — sticky */}
        <div style={{
          width:'290px', flexShrink:0, display:'flex', flexDirection:'column', gap:'14px',
          position:'sticky', top:'57px', height:'calc(100vh - 73px)', alignSelf:'flex-start',
        }}>
          <div style={{ flex:1, minHeight:0, overflow:'hidden' }}>
            <AlertFeed alerts={alerts} />
          </div>
          <div style={{ height:'280px', flexShrink:0 }}>
            <PredictionPanel history={history} />
          </div>
        </div>

      </main>

      <footer style={{
        padding:'7px 20px', borderTop:'1px solid var(--border)',
        display:'flex', alignItems:'center', justifyContent:'space-between',
        fontFamily:'var(--font-mono)', fontSize:'10px', color:'var(--text-dimmer)',
      }}>
        <span style={{ display:'flex', alignItems:'center', gap:'10px' }}>
          <span>AIOps · Incident Detection · LSTM Autoencoder</span>
          <span style={{
            fontFamily:'var(--font-mono)', fontSize:'9px',
            color: emailEnabled ? 'var(--normal)' : 'var(--text-dimmer)',
            background: emailEnabled ? 'rgba(74,222,128,.07)' : 'transparent',
            border: emailEnabled ? '1px solid rgba(74,222,128,.2)' : '1px solid var(--border)',
            borderRadius: 'var(--radius)', padding: '2px 8px',
          }}>
            {emailEnabled ? '📧 Email alerts on' : '📧 Email alerts off'}
          </span>
        </span>
        <span style={{ display:'flex', alignItems:'center', gap:'6px' }}>
          <span className={`dot ${streamStatus === 'connected' ? 'normal' : 'warning'}`} style={{ margin:0 }} />
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