// src/components/AlertFeed.jsx
// Shows rule-based alerts (metric > user limit) and LSTM anomaly alerts.

import { useEffect, useRef } from 'react'

function formatTime(ts) {
  if (!ts) return '—'
  return new Date(ts).toLocaleTimeString('en-GB', { hour12: false })
}

function AlertRow({ alert }) {
  const sev    = alert.severity ?? 'NORMAL'
  const source = alert.source   ?? 'lstm'   // 'rule' | 'lstm'

  return (
    <div style={{
      padding:      '8px 0',
      borderBottom: '1px solid var(--border)',
    }}>
      {/* Top row: dot + badge + source tag + time */}
      <div style={{
        display:    'flex',
        alignItems: 'center',
        gap:        '7px',
        marginBottom:'3px',
      }}>
        <span className={`dot ${sev.toLowerCase()}`} />
        <span className={`badge ${sev}`}>{sev}</span>
        <span style={{
          fontFamily:   'var(--font-mono)',
          fontSize:     '9px',
          color:        source === 'rule' ? 'var(--warning)' : 'var(--text-dimmer)',
          background:   source === 'rule' ? 'rgba(250,204,21,.08)' : 'transparent',
          border:       source === 'rule' ? '1px solid rgba(250,204,21,.2)' : 'none',
          borderRadius: 'var(--radius)',
          padding:      source === 'rule' ? '1px 5px' : '0',
        }}>
          {source === 'rule' ? 'RULE' : 'LSTM'}
        </span>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize:   '10px',
          color:      'var(--text-dimmer)',
          marginLeft: 'auto',
        }}>
          {formatTime(alert.timestamp)}
        </span>
      </div>

      {/* Metric values */}
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize:   '10px',
        color:      'var(--text-dim)',
        paddingLeft:'14px',
      }}>
        CPU {alert.cpu?.toFixed(1)}%
        &nbsp;·&nbsp;MEM {alert.memory?.toFixed(1)}%
        &nbsp;·&nbsp;DISK {alert.disk?.toFixed(1)}%
      </div>

      {/* Which metrics exceeded the user-set limit */}
      {alert.exceeded?.length > 0 && (
        <div style={{
          fontFamily:  'var(--font-mono)',
          fontSize:    '10px',
          color:       'var(--critical)',
          paddingLeft: '14px',
          marginTop:   '2px',
        }}>
          ↳ Exceeded limit: {alert.exceeded.join(' · ')}
        </div>
      )}

      {/* LSTM flagged metrics */}
      {alert.flagged?.length > 0 && source === 'lstm' && (
        <div style={{
          fontFamily:  'var(--font-mono)',
          fontSize:    '10px',
          color:       'var(--critical)',
          paddingLeft: '14px',
          marginTop:   '2px',
        }}>
          ↳ LSTM: {alert.flagged.join(' · ')}
        </div>
      )}

      {/* LSTM error score */}
      {alert.error != null && source === 'lstm' && (
        <div style={{
          fontFamily:  'var(--font-mono)',
          fontSize:    '9px',
          color:       'var(--text-dimmer)',
          paddingLeft: '14px',
          marginTop:   '1px',
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

  useEffect(() => {
    if (alerts.length > prevLen.current && listRef.current) {
      listRef.current.scrollTop = 0
    }
    prevLen.current = alerts.length
  }, [alerts])

  const counts = {
    NORMAL:   alerts.filter(a => a.severity === 'NORMAL').length,
    WARNING:  alerts.filter(a => a.severity === 'WARNING').length,
    CRITICAL: alerts.filter(a => a.severity === 'CRITICAL').length,
  }

  return (
    <div className="panel" style={{
      display:       'flex',
      flexDirection: 'column',
      height:        '100%',
      overflow:      'hidden',
    }}>

      {/* Header */}
      <div style={{
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'space-between',
        marginBottom:   '10px',
        flexShrink:     0,
      }}>
        <div className="section-label" style={{ marginBottom: 0 }}>Alert Detected</div>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:'10px', color:'var(--text-dimmer)' }}>
          {alerts.length} total
        </span>
      </div>

      {/* Count tiles */}
      <div style={{ display:'flex', gap:'8px', marginBottom:'10px', flexShrink:0 }}>
        {[
          ['NORMAL',   counts.NORMAL,   'var(--normal)'],
          ['WARNING',  counts.WARNING,  'var(--warning)'],
          ['CRITICAL', counts.CRITICAL, 'var(--critical)'],
        ].map(([sev, count, color]) => (
          <div key={sev} style={{
            flex:1, background:'var(--bg-panel2)',
            border:'1px solid var(--border)', borderRadius:'var(--radius)',
            padding:'6px 8px', textAlign:'center',
          }}>
            <div style={{ fontFamily:'var(--font-mono)', fontSize:'9px', color:'var(--text-dimmer)', marginBottom:'2px' }}>
              {sev}
            </div>
            <div style={{ fontFamily:'var(--font-mono)', fontSize:'16px', fontWeight:'600', color }}>
              {count}
            </div>
          </div>
        ))}
      </div>

      {/* Scrollable alert list */}
      <div ref={listRef} style={{ flex:1, overflowY:'auto', minHeight:0 }}>
        {alerts.length === 0 ? (
          <div style={{
            display:'flex', alignItems:'center', justifyContent:'center',
            height:'60px', color:'var(--text-dimmer)',
            fontFamily:'var(--font-mono)', fontSize:'11px',
          }}>
            No alerts yet
          </div>
        ) : (
          [...alerts].reverse().map((a, i) => (
            <AlertRow key={i} alert={a} />
          ))
        )}
      </div>
    </div>
  )
}