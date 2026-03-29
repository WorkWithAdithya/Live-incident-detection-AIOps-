// src/components/LogTable.jsx
// Real-time DB log table. Cells that exceed the user-set limit are highlighted.

import { useEffect, useRef } from 'react'

function formatTs(ts) {
  if (!ts) return '—'
  return new Date(ts).toLocaleString('en-GB', {
    hour12: false, year:'numeric', month:'2-digit', day:'2-digit',
    hour:'2-digit', minute:'2-digit', second:'2-digit',
  })
}

const COL = {
  ts:     { label: 'Timestamp',     width: '190px', align: 'left'   },
  cpu:    { label: 'CPU Usage',     width: '100px', align: 'right'  },
  memory: { label: 'Memory Usage',  width: '110px', align: 'right'  },
  disk:   { label: 'Disk Usage',    width: '100px', align: 'right'  },
  error:  { label: 'Anomaly Score', width: '120px', align: 'right'  },
  status: { label: 'Status',        width: '95px',  align: 'center' },
}

const METRIC_COLOR = {
  cpu:    'var(--cpu-color)',
  memory: 'var(--mem-color)',
  disk:   'var(--disk-color)',
}

export default function LogTable({ history, limits = {} }) {
  const tbodyRef = useRef(null)
  const prevLen  = useRef(0)

  useEffect(() => {
    if (history.length > prevLen.current && tbodyRef.current) {
      const wrapper = tbodyRef.current.closest('[data-scroll]')
      if (wrapper) wrapper.scrollTop = wrapper.scrollHeight
    }
    prevLen.current = history.length
  }, [history])

  const thStyle = (col) => ({
    padding:       '7px 10px',
    fontFamily:    'var(--font-mono)',
    fontSize:      '10px',
    fontWeight:    '500',
    color:         'var(--text-dimmer)',
    textAlign:     col.align,
    width:         col.width,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    borderBottom:  '1px solid var(--border)',
    position:      'sticky',
    top:           0,
    background:    'var(--bg-panel)',
    whiteSpace:    'nowrap',
  })

  // Returns style for a metric cell — red if exceeding user limit
  function metricTd(col, value, limitKey) {
    const exceeded = limits[limitKey] != null && value != null && value > limits[limitKey]
    return {
      padding:     '6px 10px',
      fontFamily:  'var(--font-mono)',
      fontSize:    '11px',
      textAlign:   col.align,
      whiteSpace:  'nowrap',
      borderBottom:'1px solid var(--border)',
      color:       exceeded ? 'var(--critical)' : METRIC_COLOR[limitKey],
      fontWeight:  exceeded ? '600' : '400',
      background:  exceeded ? 'rgba(248,113,113,.05)' : 'transparent',
    }
  }

  return (
    <div className="panel" style={{ display:'flex', flexDirection:'column' }}>
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'8px' }}>
        <div className="section-label" style={{ marginBottom:0 }}>
          Real-time DB Logs — Timestamp · CPU Usage · Memory Usage · Disk Usage
        </div>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:'10px', color:'var(--text-dimmer)' }}>
          {history.length} rows
        </span>
      </div>

      {/* Limit legend */}
      {(limits.cpu != null || limits.memory != null || limits.disk != null) && (
        <div style={{
          display:      'flex',
          gap:          '12px',
          marginBottom: '8px',
          fontFamily:   'var(--font-mono)',
          fontSize:     '10px',
          color:        'var(--text-dimmer)',
        }}>
          <span>Limits:</span>
          {limits.cpu    != null && <span style={{ color:'var(--cpu-color)'  }}>CPU &gt; {limits.cpu}%</span>}
          {limits.memory != null && <span style={{ color:'var(--mem-color)'  }}>MEM &gt; {limits.memory}%</span>}
          {limits.disk   != null && <span style={{ color:'var(--disk-color)' }}>DISK &gt; {limits.disk}%</span>}
          <span style={{ color:'var(--critical)' }}>← highlighted in red</span>
        </div>
      )}

      <div
        data-scroll
        style={{
          overflowY:    'auto',
          maxHeight:    '200px',
          border:       '1px solid var(--border)',
          borderRadius: 'var(--radius)',
        }}
      >
        <table style={{ width:'100%', borderCollapse:'collapse', tableLayout:'fixed' }}>
          <thead>
            <tr>
              {Object.values(COL).map(col => (
                <th key={col.label} style={thStyle(col)}>{col.label}</th>
              ))}
            </tr>
          </thead>
          <tbody ref={tbodyRef}>
            {history.length === 0 ? (
              <tr>
                <td colSpan={6} style={{
                  padding:'24px', textAlign:'center',
                  color:'var(--text-dimmer)', fontFamily:'var(--font-mono)', fontSize:'11px',
                }}>
                  Waiting for logs...
                </td>
              </tr>
            ) : (
              [...history].reverse().map((row, i) => {
                const sev = row.severity ?? 'NORMAL'
                return (
                  <tr key={i}>
                    {/* Timestamp */}
                    <td style={{
                      padding:'6px 10px', fontFamily:'var(--font-mono)',
                      fontSize:'10px', color:'var(--text-dim)',
                      borderBottom:'1px solid var(--border)', whiteSpace:'nowrap',
                    }}>
                      {formatTs(row.timestamp)}
                    </td>

                    {/* CPU */}
                    <td style={metricTd(COL.cpu, row.cpu, 'cpu')}>
                      {row.cpu?.toFixed(2)}%
                      {limits.cpu != null && row.cpu > limits.cpu && ' ▲'}
                    </td>

                    {/* Memory */}
                    <td style={metricTd(COL.memory, row.memory, 'memory')}>
                      {row.memory?.toFixed(2)}%
                      {limits.memory != null && row.memory > limits.memory && ' ▲'}
                    </td>

                    {/* Disk */}
                    <td style={metricTd(COL.disk, row.disk, 'disk')}>
                      {row.disk?.toFixed(2)}%
                      {limits.disk != null && row.disk > limits.disk && ' ▲'}
                    </td>

                    {/* LSTM anomaly score */}
                    <td style={{
                      padding:'6px 10px', fontFamily:'var(--font-mono)',
                      fontSize:'11px', textAlign:'right', whiteSpace:'nowrap',
                      borderBottom:'1px solid var(--border)',
                      color: row.is_anomaly ? 'var(--critical)' : 'var(--text-dimmer)',
                    }}>
                      {row.error?.toFixed(6)}
                    </td>

                    {/* Status badge */}
                    <td style={{
                      padding:'6px 10px', textAlign:'center',
                      borderBottom:'1px solid var(--border)',
                    }}>
                      <span className={`badge ${sev}`}>{sev}</span>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}