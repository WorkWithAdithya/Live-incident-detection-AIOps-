// src/components/PredictionPanel.jsx
// Predicts which metric will breach a user-set limit and when.
// Uses linear regression slope over the last 20 readings per metric.

import { useMemo } from 'react'

const LOG_INTERVAL_SEC = 5   // must match LOG_INTERVAL_SECONDS in .env

// ── Linear regression slope ────────────────────────────────────────────────
function slope(values) {
  const n = values.length
  if (n < 2) return 0
  const xMean = (n - 1) / 2
  const yMean = values.reduce((a, b) => a + b, 0) / n
  let num = 0, den = 0
  values.forEach((y, x) => {
    num += (x - xMean) * (y - yMean)
    den += (x - xMean) ** 2
  })
  return den === 0 ? 0 : num / den
}

// Steps until value reaches target given current value and slope.
// Returns null if not reachable (slope going wrong way or already there).
function stepsToTarget(current, target, s) {
  if (s <= 0) return null
  const diff = target - current
  if (diff <= 0) return 0
  return Math.ceil(diff / s)
}

function addSeconds(baseTs, seconds) {
  const d = new Date(baseTs)
  d.setSeconds(d.getSeconds() + seconds)
  return d
}

function formatTs(d) {
  if (!d) return '—'
  return d.toLocaleString('en-GB', {
    hour12: false, day: '2-digit', month: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

// ── Core prediction engine ──────────────────────────────────────────────────
function computePredictions(history, limits) {
  if (!history || history.length < 3) return []

  const recent   = history.slice(-20)
  const latestTs = recent[recent.length - 1]?.timestamp
  if (!latestTs) return []

  const metrics = [
    { key: 'cpu',    label: 'CPU Usage',    color: 'var(--cpu-color)'  },
    { key: 'memory', label: 'Memory Usage', color: 'var(--mem-color)'  },
    { key: 'disk',   label: 'Disk Usage',   color: 'var(--disk-color)' },
  ]

  const predictions = []

  for (const { key, label, color } of metrics) {
    const values  = recent.map(r => r[key]).filter(v => v != null)
    if (values.length < 2) continue

    const current = values[values.length - 1]
    const s       = slope(values)
    const wLimit  = limits[`${key}_warning`]
    const cLimit  = limits[`${key}_critical`]

    // Check Critical first (more severe), then Warning
    for (const [limit, severity] of [[cLimit, 'CRITICAL'], [wLimit, 'WARNING']]) {
      if (limit == null) continue

      if (current > limit) {
        // Already breached right now
        predictions.push({
          metric:   key,
          label,
          color,
          severity,
          limit,
          current,
          slope:    s,
          status:   'breached',
          eta:      null,
          etaTs:    null,
          criteria: `${label} > ${limit}%`,
          stepsAway: 0,
        })
        break
      }

      const steps = stepsToTarget(current, limit, s)
      if (steps !== null && steps <= 120) {
        // Will breach within 120 steps (~10 min at 5s)
        const secsAway = steps * LOG_INTERVAL_SEC
        const etaTs    = addSeconds(latestTs, secsAway)
        predictions.push({
          metric:    key,
          label,
          color,
          severity,
          limit,
          current,
          slope:     s,
          status:    'approaching',
          eta:       secsAway,
          etaTs,
          criteria:  `${label} > ${limit}%`,
          stepsAway: steps,
        })
        break // report only worst severity per metric
      }
    }
  }

  // Sort: breached first, then nearest ETA first
  predictions.sort((a, b) => {
    if (a.status === 'breached' && b.status !== 'breached') return -1
    if (b.status === 'breached' && a.status !== 'breached') return  1
    return (a.eta ?? 9999) - (b.eta ?? 9999)
  })

  return predictions
}

// ── Prediction row card ─────────────────────────────────────────────────────
function PredictionRow({ pred }) {
  const isBreach = pred.status === 'breached'
  const sevColor = pred.severity === 'CRITICAL' ? 'var(--critical)' : 'var(--warning)'
  const sevBg    = pred.severity === 'CRITICAL'
    ? 'rgba(248,113,113,.05)' : 'rgba(250,204,21,.04)'
  const sevBorder= pred.severity === 'CRITICAL'
    ? 'rgba(248,113,113,.35)' : 'rgba(250,204,21,.3)'

  return (
    <div style={{
      padding:      '10px 12px',
      borderRadius: 'var(--radius)',
      border:       `1px solid ${isBreach ? sevBorder : 'var(--border)'}`,
      background:   isBreach ? sevBg : 'var(--bg-panel2)',
      marginBottom: '8px',
    }}>

      {/* Header */}
      <div style={{ display:'flex', alignItems:'center', gap:'8px', marginBottom:'8px' }}>
        <span className={`badge ${pred.severity}`}>{pred.severity}</span>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:'11px', fontWeight:'600', color: pred.color }}>
          {pred.label}
        </span>
        {isBreach && (
          <span style={{
            fontFamily:'var(--font-mono)', fontSize:'9px',
            color:'var(--critical)', marginLeft:'auto',
            animation:'pulse 1.2s infinite',
          }}>
            ● ACTIVE NOW
          </span>
        )}
      </div>

      {/* Values grid */}
      <div style={{ display:'flex', gap:'12px', marginBottom:'8px' }}>
        <div>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:'9px', color:'var(--text-dimmer)', marginBottom:'2px' }}>
            CURRENT
          </div>
          <div style={{
            fontFamily:'var(--font-mono)', fontSize:'17px', fontWeight:'600',
            color: isBreach ? sevColor : pred.color,
          }}>
            {pred.current.toFixed(1)}%
          </div>
        </div>

        <div>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:'9px', color:'var(--text-dimmer)', marginBottom:'2px' }}>
            {pred.severity} LIMIT
          </div>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:'17px', fontWeight:'600', color: sevColor }}>
            {pred.limit}%
          </div>
        </div>

        {!isBreach && pred.eta != null && (
          <div>
            <div style={{ fontFamily:'var(--font-mono)', fontSize:'9px', color:'var(--text-dimmer)', marginBottom:'2px' }}>
              ETA
            </div>
            <div style={{ fontFamily:'var(--font-mono)', fontSize:'17px', fontWeight:'600', color:'var(--text)' }}>
              {pred.eta < 60 ? `~${pred.eta}s` : `~${Math.round(pred.eta / 60)}m`}
            </div>
          </div>
        )}
      </div>

      {/* Criteria */}
      <div style={{
        fontFamily:'var(--font-mono)', fontSize:'10px',
        color:'var(--text-dimmer)', marginBottom:'4px',
      }}>
        Criteria:&nbsp;
        <span style={{ color:'var(--text)' }}>{pred.criteria}</span>
      </div>

      {/* Predicted breach timestamp */}
      {!isBreach && pred.etaTs && (
        <div style={{
          fontFamily:'var(--font-mono)', fontSize:'10px', color: sevColor,
          background: pred.severity === 'CRITICAL'
            ? 'rgba(248,113,113,.07)' : 'rgba(250,204,21,.06)',
          border: `1px solid ${pred.severity === 'CRITICAL'
            ? 'rgba(248,113,113,.2)' : 'rgba(250,204,21,.15)'}`,
          borderRadius:'var(--radius)', padding:'5px 8px', marginTop:'6px',
        }}>
          Predicted breach at:&nbsp;
          <strong>{formatTs(pred.etaTs)}</strong>
        </div>
      )}

      {isBreach && (
        <div style={{
          fontFamily:'var(--font-mono)', fontSize:'10px', color: sevColor,
          background: sevBg, border:`1px solid ${sevBorder}`,
          borderRadius:'var(--radius)', padding:'5px 8px', marginTop:'6px',
        }}>
          {pred.severity} threshold exceeded — immediate action required
        </div>
      )}

      {/* Trend indicator */}
      <div style={{
        fontFamily:'var(--font-mono)', fontSize:'9px',
        color:'var(--text-dimmer)', marginTop:'5px',
      }}>
        Trend:&nbsp;
        {pred.slope > 0.001
          ? <span style={{ color:'var(--critical)' }}>↑ rising +{pred.slope.toFixed(3)}%/reading</span>
          : pred.slope < -0.001
          ? <span style={{ color:'var(--normal)' }}>↓ falling {pred.slope.toFixed(3)}%/reading</span>
          : <span style={{ color:'var(--text-dimmer)' }}>→ stable</span>
        }
      </div>
    </div>
  )
}

// ── Panel ───────────────────────────────────────────────────────────────────
export default function PredictionPanel({ history, limits }) {
  const predictions = useMemo(
    () => computePredictions(history, limits ?? {}),
    [history, limits]
  )

  const hasLimits = limits && Object.values(limits).some(v => v != null)
  const hasPreds  = predictions.length > 0
  const anyBreach = predictions.some(p => p.status === 'breached')
  const worstSev  = predictions.find(p => p.severity === 'CRITICAL')?.severity
                 ?? predictions.find(p => p.severity === 'WARNING')?.severity
  const latest    = history?.slice(-1)[0]

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
        {hasPreds && worstSev && (
          <span className={`badge ${worstSev}`}>{worstSev}</span>
        )}
      </div>

      {/* No limits configured */}
      {!hasLimits && (
        <div style={{
          flex:1, display:'flex', flexDirection:'column',
          alignItems:'center', justifyContent:'center',
          gap:'8px', textAlign:'center',
        }}>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:'24px', color:'var(--text-dimmer)' }}>—</div>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:'11px', color:'var(--text-dim)' }}>
            No thresholds configured
          </div>
          <div style={{
            fontFamily:'var(--font-mono)', fontSize:'10px',
            color:'var(--text-dimmer)', maxWidth:'190px', lineHeight:1.7,
          }}>
            Set Warning and Critical % limits in the Detection Thresholds panel above to enable anomaly prediction
          </div>
        </div>
      )}

      {/* Limits set, no predicted anomalies */}
      {hasLimits && !hasPreds && (
        <div style={{
          flex:1, display:'flex', flexDirection:'column',
          alignItems:'center', justifyContent:'center',
          gap:'10px', textAlign:'center',
        }}>
          <div style={{
            width:'48px', height:'48px', borderRadius:'50%',
            border:'2px solid var(--normal)', background:'rgba(74,222,128,.08)',
            display:'flex', alignItems:'center', justifyContent:'center', fontSize:'20px',
          }}>✓</div>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:'13px', fontWeight:'600', color:'var(--normal)' }}>
            System Stable
          </div>
          <div style={{
            fontFamily:'var(--font-mono)', fontSize:'10px',
            color:'var(--text-dimmer)', maxWidth:'190px', lineHeight:1.6,
          }}>
            No metric trending toward a threshold breach in the next 10 minutes
          </div>

          {/* Current values vs limits */}
          {latest && (
            <div style={{
              width:'100%', borderTop:'1px solid var(--border)',
              paddingTop:'10px', marginTop:'2px',
            }}>
              {[
                ['CPU',  latest.cpu,    limits.cpu_warning,    limits.cpu_critical,    'var(--cpu-color)'],
                ['MEM',  latest.memory, limits.memory_warning, limits.memory_critical, 'var(--mem-color)'],
                ['DISK', latest.disk,   limits.disk_warning,   limits.disk_critical,   'var(--disk-color)'],
              ].map(([label, val, warn, crit, color]) =>
                val != null && (warn != null || crit != null) ? (
                  <div key={label} style={{
                    display:'flex', justifyContent:'space-between', alignItems:'center',
                    fontFamily:'var(--font-mono)', fontSize:'10px', marginBottom:'5px',
                  }}>
                    <span style={{ color, width:'36px' }}>{label}</span>
                    <span style={{ color:'var(--text)', fontWeight:'600' }}>{val.toFixed(1)}%</span>
                    {warn != null && <span style={{ color:'var(--warning)' }}>W:{warn}%</span>}
                    {crit != null && <span style={{ color:'var(--critical)' }}>C:{crit}%</span>}
                  </div>
                ) : null
              )}
            </div>
          )}
        </div>
      )}

      {/* Predictions found */}
      {hasLimits && hasPreds && (
        <div style={{ flex:1, overflowY:'auto', minHeight:0 }}>
          {anyBreach && (
            <div style={{
              fontFamily:'var(--font-mono)', fontSize:'10px', color:'var(--critical)',
              background:'rgba(248,113,113,.07)', border:'1px solid rgba(248,113,113,.25)',
              borderRadius:'var(--radius)', padding:'6px 10px', marginBottom:'10px',
            }}>
              ⚠ One or more thresholds are currently breached
            </div>
          )}

          {predictions.map((pred, i) => (
            <PredictionRow key={`${pred.metric}-${i}`} pred={pred} />
          ))}

          <div style={{
            fontFamily:'var(--font-mono)', fontSize:'9px',
            color:'var(--text-dimmer)', marginTop:'6px', lineHeight:1.6,
          }}>
            Based on linear trend · last {Math.min(history?.length ?? 0, 20)} readings · {LOG_INTERVAL_SEC}s interval
          </div>
        </div>
      )}
    </div>
  )
}