// src/components/PredictionPanel.jsx
// Phase 3: Uses real LSTM Forecaster output carried in every SSE event.
// Shows predicted future metric values, breach timestamps, and which
// metric will cause the anomaly — all driven by the trained forecaster.

const LOG_INTERVAL_SEC = 5
const HORIZON          = 12

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

// ── Tiny SVG spark line ───────────────────────────────────────────────────────
function SparkLine({ values, warnLimit, critLimit, color, width = 150, height = 32 }) {
  if (!values || values.length < 2) return null

  const allVals = [...values]
  if (warnLimit != null) allVals.push(warnLimit)
  if (critLimit != null) allVals.push(critLimit)

  const minV  = Math.min(...allVals) * 0.96
  const maxV  = Math.max(...allVals) * 1.04 || 1
  const range = maxV - minV || 1

  const toY = v => height - ((v - minV) / range) * height

  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width
    return `${x.toFixed(1)},${toY(v).toFixed(1)}`
  }).join(' ')

  return (
    <svg width={width} height={height} style={{ display:'block', overflow:'visible' }}>
      {/* Warning limit line */}
      {warnLimit != null && (
        <line x1={0} y1={toY(warnLimit)} x2={width} y2={toY(warnLimit)}
          stroke="var(--warning)" strokeWidth={1} strokeDasharray="4 3" opacity={0.7} />
      )}
      {/* Critical limit line */}
      {critLimit != null && (
        <line x1={0} y1={toY(critLimit)} x2={width} y2={toY(critLimit)}
          stroke="var(--critical)" strokeWidth={1} strokeDasharray="4 3" opacity={0.7} />
      )}
      {/* Forecast trajectory */}
      <polyline points={pts} fill="none" stroke={color}
        strokeWidth={1.8} strokeLinejoin="round" />
      {/* First and last dots */}
      {(() => {
        const first = pts.split(' ')[0].split(',')
        const last  = pts.split(' ').pop().split(',')
        return <>
          <circle cx={parseFloat(first[0])} cy={parseFloat(first[1])}
            r={2.5} fill={color} opacity={0.6} />
          <circle cx={parseFloat(last[0])}  cy={parseFloat(last[1])}
            r={3}   fill={color} />
        </>
      })()}
    </svg>
  )
}

// ── Forecast sparklines for all 3 metrics ────────────────────────────────────
function ForecastSparklines({ forecast, limits }) {
  if (!forecast || forecast.length === 0) return null

  const metrics = [
    { key:'cpu',    label:'CPU',  color:'var(--cpu-color)',
      wk:'cpu_warning',    ck:'cpu_critical'    },
    { key:'memory', label:'MEM',  color:'var(--mem-color)',
      wk:'memory_warning', ck:'memory_critical' },
    { key:'disk',   label:'DISK', color:'var(--disk-color)',
      wk:'disk_warning',   ck:'disk_critical'   },
  ]

  return (
    <div style={{ borderTop:'1px solid var(--border)', paddingTop:'10px', marginTop:'6px' }}>
      <div style={{
        fontFamily:'var(--font-mono)', fontSize:'9px', color:'var(--text-dimmer)',
        marginBottom:'8px', letterSpacing:'0.06em',
      }}>
        LSTM FORECAST — NEXT {HORIZON * LOG_INTERVAL_SEC}s TRAJECTORY
      </div>
      {metrics.map(({ key, label, color, wk, ck }) => {
        const values = forecast.map(f => f[key])
        const last   = values[values.length - 1]
        const first  = values[0]
        const trend  = last - first
        return (
          <div key={key} style={{
            display:'flex', alignItems:'center', gap:'8px', marginBottom:'9px',
          }}>
            <span style={{
              fontFamily:'var(--font-mono)', fontSize:'10px',
              color, width:'34px', flexShrink:0,
            }}>
              {label}
            </span>
            <SparkLine
              values={values}
              warnLimit={limits?.[wk]}
              critLimit={limits?.[ck]}
              color={color}
              width={140} height={28}
            />
            <div style={{ flexShrink:0, textAlign:'right', minWidth:'42px' }}>
              <div style={{
                fontFamily:'var(--font-mono)', fontSize:'12px',
                fontWeight:'600', color,
              }}>
                {last.toFixed(1)}%
              </div>
              <div style={{
                fontFamily:'var(--font-mono)', fontSize:'9px',
                color: trend > 1 ? 'var(--critical)'
                     : trend < -1 ? 'var(--normal)'
                     : 'var(--text-dimmer)',
              }}>
                {trend > 0.1 ? `↑+${trend.toFixed(1)}` : trend < -0.1 ? `↓${trend.toFixed(1)}` : '→ stable'}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ── Breach prediction card ────────────────────────────────────────────────────
function BreachCard({ breach }) {
  const isCrit     = breach.severity === 'CRITICAL'
  const sevColor   = isCrit ? 'var(--critical)' : 'var(--warning)'
  const metricColor = { cpu:'var(--cpu-color)', memory:'var(--mem-color)', disk:'var(--disk-color)' }
  const mColor     = metricColor[breach.metric] ?? 'var(--text)'

  return (
    <div style={{
      padding:'10px 12px', borderRadius:'var(--radius)',
      border:`1px solid ${isCrit ? 'rgba(248,113,113,.35)' : 'rgba(250,204,21,.3)'}`,
      background: isCrit ? 'rgba(248,113,113,.05)' : 'rgba(250,204,21,.04)',
      marginBottom:'8px',
    }}>
      {/* Header */}
      <div style={{ display:'flex', alignItems:'center', gap:'7px', marginBottom:'8px' }}>
        <span className={`badge ${breach.severity}`}>{breach.severity}</span>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:'11px', fontWeight:'600', color:mColor }}>
          {breach.label}
        </span>
        <span style={{
          marginLeft:'auto', fontFamily:'var(--font-mono)', fontSize:'9px',
          color:'var(--text-dimmer)', background:'rgba(255,255,255,.04)',
          border:'1px solid var(--border)', borderRadius:'var(--radius)', padding:'1px 6px',
        }}>
          LSTM FORECAST
        </span>
      </div>

      {/* Three numbers */}
      <div style={{ display:'flex', gap:'10px', marginBottom:'8px' }}>
        {[
          ['PREDICTED', `${breach.predicted_value.toFixed(1)}%`, sevColor],
          ['LIMIT',     `${breach.limit}%`,                      sevColor],
          ['ETA',       formatEta(breach.seconds_ahead),         'var(--text)'],
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
        fontFamily:'var(--font-mono)', fontSize:'10px', color:sevColor,
        background: isCrit ? 'rgba(248,113,113,.07)' : 'rgba(250,204,21,.06)',
        border:`1px solid ${isCrit ? 'rgba(248,113,113,.2)' : 'rgba(250,204,21,.15)'}`,
        borderRadius:'var(--radius)', padding:'5px 8px',
      }}>
        Predicted breach at: <strong>{formatTs(breach.predicted_at)}</strong>
      </div>
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────
export default function PredictionPanel({ latest, limits, forecasterReady: forecasterReadyProp }) {
  const forecast         = latest?.forecast          ?? []
  const forecastBreaches = latest?.forecast_breaches ?? []
  // Use direct prop if available (set from modelStatus on load),
  // fall back to latest SSE value. This fixes the "LSTM OFF" flash on startup.
  const forecasterReady  = forecasterReadyProp ?? latest?.forecaster_ready ?? false

  const hasLimits   = limits && Object.values(limits).some(v => v != null)
  const hasBreaches = forecastBreaches.length > 0
  const worstSev    = forecastBreaches.find(b => b.severity === 'CRITICAL')?.severity
                   ?? forecastBreaches.find(b => b.severity === 'WARNING')?.severity

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
          {hasBreaches && worstSev && (
            <span className={`badge ${worstSev}`}>{worstSev}</span>
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

      {/* ── State 1: Forecaster not trained ── */}
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

      {/* ── State 2: Forecaster ready, no limits set ── */}
      {forecasterReady && !hasLimits && (
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>
          <div style={{
            fontFamily:'var(--font-mono)', fontSize:'10px',
            color:'var(--text-dimmer)', marginBottom:'6px', lineHeight:1.6,
          }}>
            Set Warning/Critical limits to see breach predictions.
          </div>
          <div style={{ flex:1, overflowY:'auto', minHeight:0 }}>
            <ForecastSparklines forecast={forecast} limits={limits} />
          </div>
        </div>
      )}

      {/* ── State 3: Forecaster ready, limits set, no predicted breaches ── */}
      {forecasterReady && hasLimits && !hasBreaches && (
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>
          <div style={{
            display:'flex', alignItems:'center', gap:'10px',
            marginBottom:'8px', flexShrink:0,
          }}>
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
                No Breach Predicted
              </div>
              <div style={{
                fontFamily:'var(--font-mono)', fontSize:'10px',
                color:'var(--text-dimmer)',
              }}>
                All metrics stable for next {HORIZON * LOG_INTERVAL_SEC}s
              </div>
            </div>
          </div>
          <div style={{ flex:1, overflowY:'auto', minHeight:0 }}>
            <ForecastSparklines forecast={forecast} limits={limits} />
          </div>
        </div>
      )}

      {/* ── State 4: Breaches predicted ── */}
      {forecasterReady && hasLimits && hasBreaches && (
        <div style={{ flex:1, overflowY:'auto', minHeight:0 }}>
          <div style={{
            fontFamily:'var(--font-mono)', fontSize:'10px',
            color:'var(--text-dimmer)', marginBottom:'8px',
          }}>
            LSTM predicts&nbsp;
            <span style={{ color: worstSev === 'CRITICAL' ? 'var(--critical)' : 'var(--warning)' }}>
              {forecastBreaches.length} breach{forecastBreaches.length > 1 ? 'es' : ''}
            </span>
            &nbsp;in the next {HORIZON * LOG_INTERVAL_SEC}s:
          </div>

          {forecastBreaches.map((b, i) => (
            <BreachCard key={`${b.metric}-${b.severity}-${i}`} breach={b} />
          ))}

          <ForecastSparklines forecast={forecast} limits={limits} />
        </div>
      )}
    </div>
  )
}