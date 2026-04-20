// src/components/AlertFeed.jsx
// Only shows WARNING alerts — NORMAL rows are never displayed here.

import { useEffect, useRef } from 'react'

function formatTime(ts) {
  if (!ts) return '—'
  return new Date(ts).toLocaleTimeString('en-GB', { hour12: false })
}

function AlertRow({ alert }) {
  const source = alert.source ?? 'lstm'

  return (
    <div style={{
      padding:      '8px 0',
      borderBottom: '1px solid var(--border)',
    }}>
      {/* Top row */}
      <div style={{ display:'flex', alignItems:'center', gap:'7px', marginBottom:'3px' }}>
        <span className="dot warning" />
        <span className="badge WARNING">WARNING</span>

        {/* Source tag */}
        <span style={{
          fontFamily:   'var(--font-mono)',
          fontSize:     '9px',
          color:        source === 'rule' ? 'var(--warning)' : 'var(--text-dimmer)',
          background:   source === 'rule' ? 'rgba(250,204,21,.08)' : 'rgba(255,255,255,.04)',
          border:       `1px solid ${source === 'rule' ? 'rgba(250,204,21,.2)' : 'var(--border)'}`,
          borderRadius: 'var(--radius)',
          padding:      '1px 5px',
        }}>
          {source === 'rule' ? 'RULE' : 'LSTM'}
        </span>

        <span style={{
          fontFamily: 'var(--font-mono)', fontSize:'10px',
          color:'var(--text-dimmer)', marginLeft:'auto',
        }}>
          {formatTime(alert.timestamp)}
        </span>
      </div>

      {/* Metric values */}
      <div style={{
        fontFamily:'var(--font-mono)', fontSize:'10px',
        color:'var(--text-dim)', paddingLeft:'14px',
      }}>
        CPU {alert.cpu?.toFixed(1)}%
        &nbsp;·&nbsp;MEM {alert.memory?.toFixed(1)}%
        &nbsp;·&nbsp;DISK {alert.disk?.toFixed(1)}%
      </div>

      {/* Which metrics exceeded limit (rule alerts) */}
      {alert.exceeded?.length > 0 && (
        <div style={{
          fontFamily:'var(--font-mono)', fontSize:'10px',
          color:'var(--warning)', paddingLeft:'14px', marginTop:'2px',
        }}>
          ↳ {alert.exceeded.join(' · ')}
        </div>
      )}

      {/* LSTM flagged metrics */}
      {alert.flagged?.length > 0 && source === 'lstm' && (
        <div style={{
          fontFamily:'var(--font-mono)', fontSize:'10px',
          color:'var(--warning)', paddingLeft:'14px', marginTop:'2px',
        }}>
          ↳ {alert.flagged.join(' · ')}
        </div>
      )}

      {/* LSTM score */}
      {alert.error != null && source === 'lstm' && (
        <div style={{
          fontFamily:'var(--font-mono)', fontSize:'9px',
          color:'var(--text-dimmer)', paddingLeft:'14px', marginTop:'1px',
        }}>
          err {alert.error.toFixed(6)} · {alert.error_ratio?.toFixed(2)}× thresh
        </div>
      )}
    </div>
  )
}

export default function AlertFeed({ alerts }) {
  const listRef = useRef(null)
  const prevLen = useRef(0)

  // Only show WARNING
  const visible = alerts.filter(a => a.severity === 'WARNING')

  useEffect(() => {
    if (visible.length > prevLen.current && listRef.current) {
      listRef.current.scrollTop = 0
    }
    prevLen.current = visible.length
  }, [visible])

  return (
    <div className="panel" style={{
      display:'flex', flexDirection:'column',
      height:'100%', overflow:'hidden',
    }}>

      {/* Header */}
      <div style={{
        display:'flex', alignItems:'center', justifyContent:'space-between',
        marginBottom:'10px', flexShrink:0,
      }}>
        <div className="section-label" style={{ marginBottom:0 }}>Alert Detected</div>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:'10px', color:'var(--text-dimmer)' }}>
          {visible.length} total
        </span>
      </div>

      {/* Count tile — WARNING only */}
      <div style={{ display:'flex', gap:'8px', marginBottom:'10px', flexShrink:0 }}>
        <div style={{
          flex:1, background:'var(--bg-panel2)', border:'1px solid var(--border)',
          borderRadius:'var(--radius)', padding:'6px 8px', textAlign:'center',
        }}>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:'9px', color:'var(--text-dimmer)', marginBottom:'2px' }}>
            WARNING
          </div>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:'16px', fontWeight:'600', color:'var(--warning)' }}>
            {visible.length}
          </div>
        </div>
      </div>

      {/* Alert list */}
      <div ref={listRef} style={{ flex:1, overflowY:'auto', minHeight:0 }}>
        {visible.length === 0 ? (
          <div style={{
            display:'flex', alignItems:'center', justifyContent:'center',
            height:'60px', color:'var(--text-dimmer)',
            fontFamily:'var(--font-mono)', fontSize:'11px',
          }}>
            No warning alerts yet
          </div>
        ) : (
          [...visible].reverse().map((a, i) => (
            <AlertRow key={i} alert={a} />
          ))
        )}
      </div>
    </div>
  )
}