// src/components/ControlPanel.jsx
// Per-metric threshold control: user sets CPU / Memory / Disk limits.
// When any live value exceeds its limit → alert is triggered.

import { useState } from 'react'

function MetricCard({ label, value, color, limit, onLimitChange, onLimitSubmit }) {
  const exceeded = limit != null && value != null && value > limit

  return (
    <div style={{
      background:   'var(--bg-panel2)',
      border:       `1px solid ${exceeded ? 'rgba(248,113,113,.45)' : 'var(--border)'}`,
      borderRadius: 'var(--radius-lg)',
      padding:      '12px 14px',
      flex:         1,
      transition:   'border-color .2s',
    }}>
      {/* Label + exceeded badge */}
      <div style={{
        display:       'flex',
        alignItems:    'center',
        justifyContent:'space-between',
        marginBottom:  '8px',
      }}>
        <span style={{
          fontFamily:    'var(--font-mono)',
          fontSize:      '10px',
          color:         'var(--text-dimmer)',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}>
          {label}
        </span>
        {exceeded && (
          <span style={{
            fontFamily:   'var(--font-mono)',
            fontSize:     '9px',
            color:        'var(--critical)',
            background:   'rgba(248,113,113,.1)',
            border:       '1px solid rgba(248,113,113,.3)',
            borderRadius: 'var(--radius)',
            padding:      '1px 6px',
          }}>
            EXCEEDED
          </span>
        )}
      </div>

      {/* Live value */}
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize:   '26px',
        fontWeight: '600',
        color:      exceeded ? 'var(--critical)' : (color ?? 'var(--text)'),
        lineHeight: 1,
        transition: 'color .2s',
      }}>
        {value != null ? `${value.toFixed(1)}%` : '—'}
      </div>

      {/* Progress bar */}
      <div style={{
        marginTop:    '8px',
        height:       '3px',
        background:   'var(--border)',
        borderRadius: '2px',
        overflow:     'hidden',
        position:     'relative',
      }}>
        {/* Value bar */}
        <div style={{
          position:   'absolute',
          left:       0, top: 0, bottom: 0,
          width:      `${Math.min(value ?? 0, 100)}%`,
          background: exceeded ? 'var(--critical)' : color,
          borderRadius:'2px',
          transition: 'width .5s ease, background .2s',
        }} />
        {/* Threshold marker */}
        {limit != null && (
          <div style={{
            position:  'absolute',
            top:       '-2px',
            bottom:    '-2px',
            left:      `${Math.min(limit, 100)}%`,
            width:     '2px',
            background:'var(--warning)',
            borderRadius:'1px',
          }} />
        )}
      </div>

      {/* Threshold input */}
      <div style={{
        marginTop:  '10px',
        display:    'flex',
        alignItems: 'center',
        gap:        '6px',
      }}>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize:   '10px',
          color:      'var(--text-dimmer)',
          whiteSpace: 'nowrap',
        }}>
          Limit
        </span>
        <input
          type="number"
          min="0"
          max="100"
          step="1"
          placeholder="e.g. 85"
          value={limit ?? ''}
          onChange={e => onLimitChange(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && onLimitSubmit()}
          style={{
            width:     '70px',
            padding:   '4px 7px',
            fontSize:  '11px',
          }}
        />
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize:   '10px',
          color:      'var(--text-dimmer)',
        }}>
          %
        </span>
        {limit != null && (
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize:   '10px',
            color:      'var(--warning)',
            marginLeft: '2px',
          }}>
            → {limit}%
          </span>
        )}
      </div>
    </div>
  )
}

export default function ControlPanel({ latest, modelStatus, limits, onLimitsChange }) {
  // Local draft state for inputs before submit
  const [drafts, setDrafts] = useState({
    cpu:    limits.cpu    != null ? String(limits.cpu)    : '',
    memory: limits.memory != null ? String(limits.memory) : '',
    disk:   limits.disk   != null ? String(limits.disk)   : '',
  })
  const [msg, setMsg] = useState('')

  function updateDraft(metric, val) {
    setDrafts(prev => ({ ...prev, [metric]: val }))
  }

  function submitAll() {
    const parsed = {
      cpu:    drafts.cpu    !== '' ? parseFloat(drafts.cpu)    : null,
      memory: drafts.memory !== '' ? parseFloat(drafts.memory) : null,
      disk:   drafts.disk   !== '' ? parseFloat(drafts.disk)   : null,
    }
    // Validate
    for (const [k, v] of Object.entries(parsed)) {
      if (v !== null && (isNaN(v) || v < 0 || v > 100)) {
        setMsg(`✗ ${k} limit must be 0–100`)
        setTimeout(() => setMsg(''), 3000)
        return
      }
    }
    onLimitsChange(parsed)
    setMsg('✓ Limits applied')
    setTimeout(() => setMsg(''), 3000)
  }

  function resetAll() {
    setDrafts({ cpu: '', memory: '', disk: '' })
    onLimitsChange({ cpu: null, memory: null, disk: null })
    setMsg('✓ Limits cleared')
    setTimeout(() => setMsg(''), 3000)
  }

  const metrics = [
    { key: 'cpu',    label: 'CPU Usage',    color: 'var(--cpu-color)' },
    { key: 'memory', label: 'Memory Usage', color: 'var(--mem-color)' },
    { key: 'disk',   label: 'Disk Usage',   color: 'var(--disk-color)' },
  ]

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

      {/* Metric cards — each with its own limit input */}
      <div style={{ display: 'flex', gap: '10px' }}>
        {metrics.map(({ key, label, color }) => (
          <MetricCard
            key={key}
            label={label}
            value={latest?.[key]}
            color={color}
            limit={limits[key]}
            onLimitChange={val => updateDraft(key, val)}
            onLimitSubmit={submitAll}
          />
        ))}
      </div>

      {/* Action row */}
      <div style={{
        display:      'flex',
        alignItems:   'center',
        gap:          '10px',
        borderTop:    '1px solid var(--border)',
        paddingTop:   '12px',
        flexWrap:     'wrap',
      }}>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize:   '11px',
          color:      'var(--text-dim)',
        }}>
          Set threshold of all the values mentioned above
        </span>

        <button className="primary" onClick={submitAll}>
          Set Value
        </button>

        <button onClick={resetAll}>
          Clear All
        </button>

        {msg && (
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize:   '10px',
            color:      msg.startsWith('✓') ? 'var(--normal)' : 'var(--critical)',
          }}>
            {msg}
          </span>
        )}

        {/* LSTM threshold info (read-only) */}
        {modelStatus?.threshold && (
          <span style={{
            fontFamily:  'var(--font-mono)',
            fontSize:    '10px',
            color:       'var(--text-dimmer)',
            marginLeft:  'auto',
            borderLeft:  '1px solid var(--border)',
            paddingLeft: '10px',
          }}>
            LSTM thresh: {modelStatus.threshold.toFixed(6)}
          </span>
        )}
      </div>

      {/* Warmup notice */}
      {latest?.warming_up && (
        <div style={{
          fontFamily:   'var(--font-mono)',
          fontSize:     '10px',
          color:        'var(--warning)',
          background:   'rgba(250,204,21,.06)',
          border:       '1px solid rgba(250,204,21,.2)',
          borderRadius: 'var(--radius)',
          padding:      '5px 10px',
        }}>
          ⏳ Warming up — {latest.actual_rows ?? 0}/60 rows in window
        </div>
      )}
    </div>
  )
}