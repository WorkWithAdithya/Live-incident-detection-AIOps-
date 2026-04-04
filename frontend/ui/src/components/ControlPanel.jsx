// src/components/ControlPanel.jsx
// Clean threshold setter — no metric value display.
// User sets Warning and Critical % for CPU, Memory, Disk.

import { useState } from 'react'

const METRICS = [
  { key: 'cpu',    label: 'CPU Usage',    color: 'var(--cpu-color)'  },
  { key: 'memory', label: 'Memory Usage', color: 'var(--mem-color)'  },
  { key: 'disk',   label: 'Disk Usage',   color: 'var(--disk-color)' },
]

function ThresholdRow({ metricLabel, metricColor, draftWarning, draftCritical,
                        appliedWarning, appliedCritical,
                        onWarningChange, onCriticalChange }) {
  return (
    <div style={{
      display:      'flex',
      alignItems:   'center',
      gap:          '16px',
      padding:      '10px 14px',
      background:   'var(--bg-panel2)',
      border:       '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      flexWrap:     'wrap',
    }}>

      {/* Metric name */}
      <span style={{
        fontFamily:    'var(--font-mono)',
        fontSize:      '11px',
        fontWeight:    '500',
        color:         metricColor,
        width:         '110px',
        flexShrink:    0,
        letterSpacing: '0.04em',
      }}>
        {metricLabel}
      </span>

      {/* Divider */}
      <div style={{ width: '1px', height: '28px', background: 'var(--border)', flexShrink: 0 }} />

      {/* Warning input */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
        <div style={{ width: '9px', height: '9px', borderRadius: '2px', background: 'var(--warning)', flexShrink: 0 }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-dimmer)', width: '52px' }}>
          Warning
        </span>
        <input
          type="number"
          min="0" max="100" step="1"
          placeholder="—"
          value={draftWarning}
          onChange={e => onWarningChange(e.target.value)}
          style={{ width: '72px' }}
        />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-dimmer)' }}>%</span>
        {appliedWarning != null && (
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--warning)',
            background: 'rgba(250,204,21,.08)', border: '1px solid rgba(250,204,21,.2)',
            borderRadius: 'var(--radius)', padding: '1px 6px',
          }}>
            active: {appliedWarning}%
          </span>
        )}
      </div>

      {/* Divider */}
      <div style={{ width: '1px', height: '28px', background: 'var(--border)', flexShrink: 0 }} />

      {/* Critical input */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
        <div style={{ width: '9px', height: '9px', borderRadius: '2px', background: 'var(--critical)', flexShrink: 0 }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-dimmer)', width: '52px' }}>
          Critical
        </span>
        <input
          type="number"
          min="0" max="100" step="1"
          placeholder="—"
          value={draftCritical}
          onChange={e => onCriticalChange(e.target.value)}
          style={{ width: '72px' }}
        />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-dimmer)' }}>%</span>
        {appliedCritical != null && (
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--critical)',
            background: 'rgba(248,113,113,.08)', border: '1px solid rgba(248,113,113,.2)',
            borderRadius: 'var(--radius)', padding: '1px 6px',
          }}>
            active: {appliedCritical}%
          </span>
        )}
      </div>

      {/* Severity legend for this row */}
      <div style={{ marginLeft: 'auto', display: 'flex', gap: '10px', alignItems: 'center', flexShrink: 0 }}>
        {[
          ['NORMAL',   'var(--normal)',   appliedWarning  != null ? `< ${appliedWarning}%`  : 'not set'],
          ['WARNING',  'var(--warning)',  appliedWarning  != null && appliedCritical != null
                                            ? `${appliedWarning}% – ${appliedCritical}%`
                                            : appliedWarning != null ? `> ${appliedWarning}%` : 'not set'],
          ['CRITICAL', 'var(--critical)', appliedCritical != null ? `> ${appliedCritical}%` : 'not set'],
        ].map(([sev, color, range]) => (
          <div key={sev} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{ width: '7px', height: '7px', borderRadius: '1px', background: color }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--text-dimmer)' }}>
              {sev}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color }}>
              {range}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ControlPanel({ modelStatus, limits, onLimitsChange }) {

  const [drafts, setDrafts] = useState({
    cpu_warning: '', cpu_critical: '',
    memory_warning: '', memory_critical: '',
    disk_warning: '', disk_critical: '',
  })
  const [msg, setMsg] = useState('')

  function updateDraft(key, val) {
    setDrafts(prev => ({ ...prev, [key]: val }))
  }

  function flash(text) {
    setMsg(text)
    setTimeout(() => setMsg(''), 3500)
  }

  function submitAll() {
    const next = { ...limits }

    for (const { key } of METRICS) {
      const wKey = `${key}_warning`
      const cKey = `${key}_critical`
      const wVal = drafts[wKey] !== '' ? parseFloat(drafts[wKey]) : null
      const cVal = drafts[cKey] !== '' ? parseFloat(drafts[cKey]) : null

      if (wVal !== null && (isNaN(wVal) || wVal < 0 || wVal > 100)) {
        flash(`✗ ${key.toUpperCase()} Warning must be between 0 and 100`)
        return
      }
      if (cVal !== null && (isNaN(cVal) || cVal < 0 || cVal > 100)) {
        flash(`✗ ${key.toUpperCase()} Critical must be between 0 and 100`)
        return
      }

      const effectiveW = wVal ?? next[wKey]
      const effectiveC = cVal ?? next[cKey]
      if (effectiveW != null && effectiveC != null && effectiveW >= effectiveC) {
        flash(`✗ ${key.toUpperCase()}: Warning (${effectiveW}%) must be less than Critical (${effectiveC}%)`)
        return
      }

      if (wVal !== null) next[wKey] = wVal
      if (cVal !== null) next[cKey] = cVal
    }

    onLimitsChange(next)
    setDrafts({
      cpu_warning: '', cpu_critical: '',
      memory_warning: '', memory_critical: '',
      disk_warning: '', disk_critical: '',
    })
    flash('✓ Thresholds applied')
  }

  function clearAll() {
    setDrafts({
      cpu_warning: '', cpu_critical: '',
      memory_warning: '', memory_critical: '',
      disk_warning: '', disk_critical: '',
    })
    onLimitsChange({
      cpu_warning: null, cpu_critical: null,
      memory_warning: null, memory_critical: null,
      disk_warning: null, disk_critical: null,
    })
    flash('✓ All thresholds cleared')
  }

  const hasActive = Object.values(limits).some(v => v != null)

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2px' }}>
        <div className="section-label" style={{ marginBottom: 0 }}>Detection Thresholds</div>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-dimmer)' }}>
          Set Warning and Critical % limits per metric — leave blank to skip
        </span>
      </div>

      {/* One row per metric */}
      {METRICS.map(({ key, label, color }) => (
        <ThresholdRow
          key={key}
          metricLabel={label}
          metricColor={color}
          draftWarning={drafts[`${key}_warning`]}
          draftCritical={drafts[`${key}_critical`]}
          appliedWarning={limits[`${key}_warning`]}
          appliedCritical={limits[`${key}_critical`]}
          onWarningChange={val => updateDraft(`${key}_warning`, val)}
          onCriticalChange={val => updateDraft(`${key}_critical`, val)}
        />
      ))}

      {/* Action row */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '10px',
        borderTop: '1px solid var(--border)', paddingTop: '10px', flexWrap: 'wrap',
      }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-dim)' }}>
          Set threshold of all the values mentioned above
        </span>

        <button className="primary" onClick={submitAll}>Set Value</button>
        <button onClick={clearAll} disabled={!hasActive}>Clear All</button>

        {msg && (
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: '10px',
            color: msg.startsWith('✓') ? 'var(--normal)' : 'var(--critical)',
          }}>
            {msg}
          </span>
        )}

        {modelStatus?.threshold && (
          <span style={{
            marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: '10px',
            color: 'var(--text-dimmer)', borderLeft: '1px solid var(--border)', paddingLeft: '10px',
          }}>
            LSTM thresh: {modelStatus.threshold.toFixed(6)}
          </span>
        )}
      </div>
    </div>
  )
}