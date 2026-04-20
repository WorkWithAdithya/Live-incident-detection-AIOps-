// src/components/PredictionPanel.jsx
// - Only WARNING severity (no critical)
// - Predictions persist until predicted_at timestamp passes
// - Also checks current metrics against warning limits (live detection)

import { useState, useEffect, useRef } from 'react'

const LOG_INTERVAL_SEC = 1

function formatTs(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('en-GB', {
    hour12:false, day:'2-digit', month:'2-digit',
    hour:'2-digit', minute:'2-digit', second:'2-digit',
  })
}

function formatEta(seconds) {
  if (seconds <= 0)  return 'now'
  if (seconds < 60)  return `~${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return s > 0 ? `~${m}m ${s}s` : `~${m}m`
}

// ── Breach card ───────────────────────────────────────────────────────────────
function BreachCard({ breach }) {
  const metricColor = { cpu:'var(--cpu-color)', memory:'var(--mem-color)', disk:'var(--disk-color)' }
  const mColor     = metricColor[breach.metric] ?? 'var(--text)'
  const isLive     = breach.source === 'current'

  return (
    <div style={{
      padding:'10px 12px', borderRadius:'var(--radius)',
      border:'1px solid rgba(250,204,21,.3)',
      background:'rgba(250,204,21,.04)',
      marginBottom:'8px',
    }}>
      {/* Header */}
      <div style={{ display:'flex', alignItems:'center', gap:'7px', marginBottom:'8px' }}>
        <span className="badge WARNING">WARNING</span>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:'11px', fontWeight:'600', color:mColor }}>
          {breach.label}
        </span>
        <span style={{
          marginLeft:'auto', fontFamily:'var(--font-mono)', fontSize:'9px',
          color: isLive ? 'var(--warning)' : 'var(--text-dimmer)',
          background: isLive ? 'rgba(250,204,21,.08)' : 'rgba(255,255,255,.04)',
          border:`1px solid ${isLive ? 'rgba(250,204,21,.25)' : 'var(--border)'}`,
          borderRadius:'var(--radius)', padding:'1px 6px',
        }}>
          {isLive ? 'LIVE' : 'LSTM FORECAST'}
        </span>
      </div>

      {/* Three numbers */}
      <div style={{ display:'flex', gap:'10px', marginBottom:'8px' }}>
        {[
          [isLive ? 'CURRENT' : 'PREDICTED', `${breach.predicted_value.toFixed(1)}%`, 'var(--warning)'],
          ['LIMIT',  `${breach.limit}%`,  'var(--warning)'],
          ['ETA',    isLive ? 'now' : formatEta(breach.seconds_ahead), 'var(--text)'],
        ].map(([label, val, color]) => (
          <div key={label}>
            <div style={{
              fontFamily:'var(--font-mono)', fontSize:'9px',
              color:'var(--text-dimmer)', marginBottom:'2px',
            }}>
              {label}
            </div>
            <div style={{
              fontFamily:'var(--font-mono)', fontSize:'16px',
              fontWeight:'600', color,
            }}>
              {val}
            </div>
          </div>
        ))}
      </div>

      {/* Criteria */}
      <div style={{
        fontFamily:'var(--font-mono)', fontSize:'10px',
        color:'var(--text-dimmer)', marginBottom:'5px',
      }}>
        Criteria:&nbsp;
        <span style={{ color:'var(--text)' }}>{breach.criteria}</span>
      </div>

      {/* Breach timestamp */}
      <div style={{
        fontFamily:'var(--font-mono)', fontSize:'10px', color:'var(--warning)',
        background:'rgba(250,204,21,.06)',
        border:'1px solid rgba(250,204,21,.15)',
        borderRadius:'var(--radius)', padding:'5px 8px',
      }}>
        {isLive
          ? <>Breach happening <strong>now</strong></>
          : <>Predicted breach at: <strong>{formatTs(breach.predicted_at)}</strong></>
        }
      </div>
    </div>
  )
}

// ── Build "current" breaches from live values vs warning limits ───────────────
function getCurrentBreaches(latest, limits) {
  if (!latest || !limits) return []
  const breaches = []
  const metrics = [
    { key:'cpu',    val: latest.cpu,    label:'CPU Usage' },
    { key:'memory', val: latest.memory, label:'Memory Usage' },
    { key:'disk',   val: latest.disk,   label:'Disk Usage' },
  ]
  for (const { key, val, label } of metrics) {
    if (val == null) continue
    const wLim = limits[`${key}_warning`]
    if (wLim != null && val > wLim) {
      breaches.push({
        metric: key, label, severity: 'WARNING',
        predicted_value: val, limit: wLim, seconds_ahead: 0,
        predicted_at: latest.timestamp,
        criteria: `${label} > ${wLim}% (warning)`,
        source: 'current',
      })
    }
  }
  return breaches
}

// ── Main panel ────────────────────────────────────────────────────────────────
export default function PredictionPanel({ latest, limits, forecasterReady: forecasterReadyProp }) {
  const forecast         = latest?.forecast          ?? []
  const forecastBreaches = (latest?.forecast_breaches ?? []).filter(b => b.severity === 'WARNING')
  const forecasterReady  = forecasterReadyProp ?? latest?.forecaster_ready ?? false

  const hasLimits = limits && Object.values(limits).some(v => v != null)

  // ── Persist predictions until their predicted_at time passes ────────────
  const [stickyBreaches, setStickyBreaches] = useState([])
  const timerRef = useRef(null)

  useEffect(() => {
    if (forecastBreaches.length > 0) {
      setStickyBreaches(prev => {
        const map = new Map()
        for (const b of prev) map.set(b.metric, b)
        for (const b of forecastBreaches) map.set(b.metric, { ...b, source: 'forecast' })
        return Array.from(map.values())
      })
    }
  }, [forecastBreaches])

  // Clean up expired predictions every second
  useEffect(() => {
    timerRef.current = setInterval(() => {
      const now = Date.now()
      setStickyBreaches(prev => {
        const filtered = prev.filter(b => {
          if (!b.predicted_at) return false
          return new Date(b.predicted_at).getTime() > now
        })
        return filtered.length !== prev.length ? filtered : prev
      })
    }, 1000)
    return () => clearInterval(timerRef.current)
  }, [])

  // ── Current breaches (live values exceeding warning limits NOW) ─────────
  const currentBreaches = getCurrentBreaches(latest, limits)

  // ── Merge: current breaches take priority over forecast for same metric ─
  const mergedMap = new Map()
  for (const b of stickyBreaches) mergedMap.set(b.metric, b)
  for (const b of currentBreaches) mergedMap.set(b.metric, b)

  const allBreaches = Array.from(mergedMap.values())
    .sort((a, b) => a.seconds_ahead - b.seconds_ahead)

  const hasBreaches = allBreaches.length > 0

  return (
    <div className="panel" style={{
      display:'flex', flexDirection:'column', height:'100%', overflow:'hidden',
    }}>

      {/* Header */}
      <div style={{
        display:'flex', alignItems:'center', justifyContent:'space-between',
        marginBottom:'10px', flexShrink:0,
      }}>
        <div className="section-label" style={{ marginBottom:0 }}>Prediction</div>
        <div style={{ display:'flex', alignItems:'center', gap:'6px' }}>
          {hasBreaches && (
            <span className="badge WARNING">WARNING</span>
          )}
          <span style={{
            fontFamily:'var(--font-mono)', fontSize:'9px',
            color: forecasterReady ? 'var(--normal)' : 'var(--text-dimmer)',
            background: forecasterReady ? 'rgba(74,222,128,.07)' : 'transparent',
            border:`1px solid ${forecasterReady ? 'rgba(74,222,128,.2)' : 'var(--border)'}`,
            borderRadius:'var(--radius)', padding:'1px 7px',
          }}>
            {forecasterReady ? 'LSTM ON' : 'LSTM OFF'}
          </span>
        </div>
      </div>

      {/* ── Forecaster not trained ── */}
      {!forecasterReady && (
        <div style={{
          flex:1, display:'flex', flexDirection:'column',
          alignItems:'center', justifyContent:'center',
          gap:'10px', textAlign:'center',
        }}>
          <div style={{ fontSize:'26px' }}>⏳</div>
          <div style={{
            fontFamily:'var(--font-mono)', fontSize:'11px', color:'var(--text-dim)',
          }}>
            Forecaster not trained
          </div>
          <div style={{
            fontFamily:'var(--font-mono)', fontSize:'10px',
            color:'var(--text-dimmer)', maxWidth:'190px', lineHeight:1.8,
          }}>
            Run in <code style={{ color:'var(--cpu-color)' }}>ai_model/</code>:
            <br />
            <code style={{ color:'var(--mem-color)' }}>
              python -m model.train_forecaster
            </code>
          </div>
        </div>
      )}

      {/* ── No limits set ── */}
      {forecasterReady && !hasLimits && (
        <div style={{
          flex:1, display:'flex', alignItems:'center', justifyContent:'center',
          textAlign:'center',
        }}>
          <div style={{
            fontFamily:'var(--font-mono)', fontSize:'10px',
            color:'var(--text-dimmer)', lineHeight:1.6,
          }}>
            Set Warning limits above to enable breach predictions.
          </div>
        </div>
      )}

      {/* ── No breaches ── */}
      {forecasterReady && hasLimits && !hasBreaches && (
        <div style={{
          flex:1, display:'flex', alignItems:'center', justifyContent:'center',
        }}>
          <div style={{ display:'flex', alignItems:'center', gap:'10px' }}>
            <div style={{
              width:'36px', height:'36px', borderRadius:'50%',
              border:'2px solid var(--normal)', background:'rgba(74,222,128,.08)',
              display:'flex', alignItems:'center', justifyContent:'center',
              fontSize:'16px', flexShrink:0,
            }}>
              ✓
            </div>
            <div>
              <div style={{
                fontFamily:'var(--font-mono)', fontSize:'12px',
                fontWeight:'600', color:'var(--normal)',
              }}>
                No Anomaly Predicted
              </div>
              <div style={{
                fontFamily:'var(--font-mono)', fontSize:'10px',
                color:'var(--text-dimmer)',
              }}>
                All metrics within limits
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Breaches (live + predicted) ── */}
      {forecasterReady && hasLimits && hasBreaches && (
        <div style={{ flex:1, overflowY:'auto', minHeight:0 }}>
          <div style={{
            fontFamily:'var(--font-mono)', fontSize:'10px',
            color:'var(--text-dimmer)', marginBottom:'8px',
          }}>
            <span style={{ color:'var(--warning)' }}>
              {allBreaches.length} anomal{allBreaches.length > 1 ? 'ies' : 'y'}
            </span>
            &nbsp;detected / predicted:
          </div>

          {allBreaches.map((b, i) => (
            <BreachCard
              key={`${b.metric}-${b.source}`}
              breach={b}
            />
          ))}
        </div>
      )}
    </div>
  )
}