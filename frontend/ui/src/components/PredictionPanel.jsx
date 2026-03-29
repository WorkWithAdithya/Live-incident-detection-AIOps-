// src/components/PredictionPanel.jsx
// Predicts whether an anomaly is likely in the near future
// based on the rolling trend of reconstruction error

import { useMemo } from 'react'

function linearTrend(values) {
  // Simple least-squares slope over last N values
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

function extrapolate(values, steps, slope) {
  if (!values.length) return null
  const last = values[values.length - 1]
  return last + slope * steps
}

export default function PredictionPanel({ history }) {
  const prediction = useMemo(() => {
    if (!history || history.length < 5) {
      return { status: 'waiting', label: 'Collecting data...', detail: '' }
    }

    const recent    = history.slice(-20)
    const errors    = recent.map(d => d.error)
    const threshold = recent[recent.length - 1]?.threshold ?? 0.002
    const slope     = linearTrend(errors)

    // Project 6 steps ahead (~30 seconds at 5s interval)
    const projected = extrapolate(errors, 6, slope)
    const currentErr = errors[errors.length - 1]

    // How many steps until projected error crosses threshold?
    let stepsToThreshold = null
    if (slope > 0 && currentErr < threshold) {
      stepsToThreshold = Math.ceil((threshold - currentErr) / slope)
    }

    const isRising    = slope > 0.000005
    const isFalling   = slope < -0.000005
    const nearThresh  = projected != null && projected >= threshold * 0.85
    const overThresh  = currentErr >= threshold

    let status, label, detail, color

    if (overThresh) {
      status = 'anomaly'
      label  = 'Anomaly Detected'
      detail = `Reconstruction error ${currentErr.toFixed(6)} is above threshold ${threshold.toFixed(6)}`
      color  = 'var(--critical)'
    } else if (nearThresh && isRising) {
      status = 'warning'
      label  = 'Anomaly Likely Soon'
      detail = stepsToThreshold != null
        ? `Error trending up — threshold may be crossed in ~${stepsToThreshold * 5}s`
        : `Error is approaching threshold (${(projected / threshold * 100).toFixed(0)}% of threshold)`
      color  = 'var(--warning)'
    } else if (isRising) {
      status = 'rising'
      label  = 'Error Trending Up'
      detail = `Gradual increase detected — monitoring closely`
      color  = 'var(--warning)'
    } else if (isFalling) {
      status = 'stable'
      label  = 'System Stabilising'
      detail = `Error is decreasing — anomaly risk reducing`
      color  = 'var(--normal)'
    } else {
      status = 'stable'
      label  = 'System Stable'
      detail = `No anomaly predicted — error well below threshold`
      color  = 'var(--normal)'
    }

    return {
      status, label, detail, color,
      currentErr, threshold, slope,
      projected, stepsToThreshold,
    }
  }, [history])

  const isAnomaly = prediction.status === 'anomaly'
  const isWarning = prediction.status === 'warning' || prediction.status === 'rising'

  return (
    <div className="panel" style={{
      display:       'flex',
      flexDirection: 'column',
      height:        '100%',
      position:      'relative',
      overflow:      'hidden',
    }}>
      <div className="section-label">Prediction</div>

      {/* Main status */}
      <div style={{
        flex:           1,
        display:        'flex',
        flexDirection:  'column',
        alignItems:     'center',
        justifyContent: 'center',
        gap:            '10px',
        textAlign:      'center',
      }}>
        {/* Indicator circle */}
        <div style={{
          width:        '48px',
          height:       '48px',
          borderRadius: '50%',
          border:       `2px solid ${prediction.color ?? 'var(--text-dimmer)'}`,
          background:   prediction.color
            ? `${prediction.color}15`
            : 'transparent',
          display:        'flex',
          alignItems:     'center',
          justifyContent: 'center',
          animation:      isAnomaly ? 'pulse 1.2s infinite' : 'none',
        }}>
          <span style={{ fontSize: '20px' }}>
            {isAnomaly ? '⚠' : isWarning ? '↑' : '✓'}
          </span>
        </div>

        {/* Label */}
        <div style={{
          fontFamily:  'var(--font-mono)',
          fontSize:    '13px',
          fontWeight:  '600',
          color:       prediction.color ?? 'var(--text)',
          lineHeight:  1.3,
        }}>
          {prediction.label}
        </div>

        {/* Detail */}
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize:   '10px',
          color:      'var(--text-dimmer)',
          maxWidth:   '200px',
          lineHeight: 1.5,
        }}>
          {prediction.detail}
        </div>
      </div>

      {/* Stats row */}
      {prediction.currentErr != null && (
        <div style={{
          borderTop:   '1px solid var(--border)',
          paddingTop:  '10px',
          display:     'grid',
          gridTemplate:'1fr 1fr / 1fr 1fr',
          gap:         '6px',
          flexShrink:  0,
        }}>
          {[
            ['Current Err', prediction.currentErr?.toFixed(6)],
            ['Threshold',   prediction.threshold?.toFixed(6)],
            ['Trend Slope', prediction.slope != null
              ? (prediction.slope > 0 ? '+' : '') + prediction.slope.toFixed(7)
              : '—'],
            ['Projected',   prediction.projected?.toFixed(6) ?? '—'],
          ].map(([k, v]) => (
            <div key={k}>
              <div style={{ fontFamily:'var(--font-mono)', fontSize:'9px', color:'var(--text-dimmer)' }}>{k}</div>
              <div style={{ fontFamily:'var(--font-mono)', fontSize:'11px', color:'var(--text)', marginTop:'1px' }}>{v}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}