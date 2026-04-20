// src/components/LogTable.jsx
// Shows last 10 logs — oldest at top, newest at bottom.

import { useEffect, useRef } from 'react'

const VISIBLE_ROWS = 10

function formatTs(ts) {
  if (!ts) return '—'
  return new Date(ts).toLocaleString('en-GB', {
    hour12:false, year:'numeric', month:'2-digit', day:'2-digit',
    hour:'2-digit', minute:'2-digit', second:'2-digit',
  })
}

const COL = {
  ts:     { label:'Timestamp',     width:'190px', align:'left'   },
  cpu:    { label:'CPU Usage',     width:'100px', align:'right'  },
  memory: { label:'Memory Usage',  width:'110px', align:'right'  },
  disk:   { label:'Disk Usage',    width:'100px', align:'right'  },
  status: { label:'Status',        width:'95px',  align:'center' },
}

export default function LogTable({ history, limits = {}, rowSeverity }) {
  const scrollRef = useRef(null)
  const prevLen   = useRef(0)

  // Auto-scroll to bottom when new rows arrive
  useEffect(() => {
    if (history.length > prevLen.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
    prevLen.current = history.length
  }, [history])

  const thStyle = col => ({
    padding:'7px 10px', fontFamily:'var(--font-mono)', fontSize:'10px', fontWeight:'500',
    color:'var(--text-dimmer)', textAlign:col.align, width:col.width,
    letterSpacing:'0.06em', textTransform:'uppercase',
    borderBottom:'1px solid var(--border)',
    position:'sticky', top:0, background:'var(--bg-panel)', whiteSpace:'nowrap',
  })

  function metricStyle(col, value, warnKey, critKey) {
    const w = limits[warnKey], c = limits[critKey]
    const overCrit = c != null && value > c
    const overWarn = w != null && value > w
    return {
      padding:'6px 10px', fontFamily:'var(--font-mono)', fontSize:'11px',
      textAlign:col.align, whiteSpace:'nowrap', borderBottom:'1px solid var(--border)',
      color: overCrit ? 'var(--critical)' : overWarn ? 'var(--warning)'
           : col === COL.cpu ? 'var(--cpu-color)' : col === COL.memory ? 'var(--mem-color)' : 'var(--disk-color)',
      fontWeight: (overCrit || overWarn) ? '600' : '400',
      background: overCrit ? 'rgba(248,113,113,.05)' : overWarn ? 'rgba(250,204,21,.04)' : 'transparent',
    }
  }

  // Only show the last VISIBLE_ROWS — oldest first, newest last
  const visible = history.slice(-VISIBLE_ROWS)

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

      {/* Active limits legend */}
      {Object.values(limits).some(v => v != null) && (
        <div style={{ display:'flex', gap:'12px', marginBottom:'8px', fontFamily:'var(--font-mono)', fontSize:'10px', flexWrap:'wrap' }}>
          {[
            ['cpu_warning',    'CPU warn',    'var(--warning)' ],
            ['cpu_critical',   'CPU crit',    'var(--critical)'],
            ['memory_warning', 'MEM warn',    'var(--warning)' ],
            ['memory_critical','MEM crit',    'var(--critical)'],
            ['disk_warning',   'DISK warn',   'var(--warning)' ],
            ['disk_critical',  'DISK crit',   'var(--critical)'],
          ].filter(([k]) => limits[k] != null).map(([k, label, color]) => (
            <span key={k} style={{ color }}>
              {label} &gt; {limits[k]}%
            </span>
          ))}
        </div>
      )}

      <div ref={scrollRef} data-scroll style={{ overflowY:'auto', maxHeight:'200px', border:'1px solid var(--border)', borderRadius:'var(--radius)' }}>
        <table style={{ width:'100%', borderCollapse:'collapse', tableLayout:'fixed' }}>
          <thead>
            <tr>{Object.values(COL).map(col => <th key={col.label} style={thStyle(col)}>{col.label}</th>)}</tr>
          </thead>
          <tbody>
            {visible.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ padding:'24px', textAlign:'center', color:'var(--text-dimmer)', fontFamily:'var(--font-mono)', fontSize:'11px' }}>
                  Waiting for logs...
                </td>
              </tr>
            ) : (
              visible.map((row, i) => {
                const sev = rowSeverity ? rowSeverity(row, limits) : (row.severity ?? 'NORMAL')
                return (
                  <tr key={row.id ?? i}>
                    <td style={{ padding:'6px 10px', fontFamily:'var(--font-mono)', fontSize:'10px', color:'var(--text-dim)', borderBottom:'1px solid var(--border)', whiteSpace:'nowrap' }}>
                      {formatTs(row.timestamp)}
                    </td>
                    <td style={metricStyle(COL.cpu,    row.cpu,    'cpu_warning',    'cpu_critical')}>
                      {row.cpu?.toFixed(2)}%
                    </td>
                    <td style={metricStyle(COL.memory, row.memory, 'memory_warning', 'memory_critical')}>
                      {row.memory?.toFixed(2)}%
                    </td>
                    <td style={metricStyle(COL.disk,   row.disk,   'disk_warning',   'disk_critical')}>
                      {row.disk?.toFixed(2)}%
                    </td>
                    <td style={{ padding:'6px 10px', textAlign:'center', borderBottom:'1px solid var(--border)' }}>
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