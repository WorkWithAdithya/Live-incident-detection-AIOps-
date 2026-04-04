// src/components/MetricChart.jsx
import React, { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts'

const CustomTooltip = ({ active, payload, label, color, unit }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background:'var(--bg-panel)', border:'1px solid var(--border-2)',
      borderRadius:'var(--radius)', padding:'7px 11px',
      fontFamily:'var(--font-mono)', fontSize:'11px', pointerEvents:'none',
    }}>
      <div style={{ color:'var(--text-dimmer)', marginBottom:'3px', fontSize:'10px' }}>
        {label ? new Date(label).toLocaleTimeString('en-GB', { hour12:false }) : ''}
      </div>
      <div style={{ color }}>{payload[0]?.value?.toFixed(2)}{unit}</div>
    </div>
  )
}

function AlertDot({ cx, cy, payload, warningLine, criticalLine }) {
  if (cx == null || cy == null || payload?.value == null) return null
  const overCrit = criticalLine != null && payload.value > criticalLine
  const overWarn = warningLine  != null && payload.value > warningLine
  if (!overCrit && !overWarn) return null
  return (
    <circle cx={cx} cy={cy} r={3}
      fill={overCrit ? 'var(--critical)' : 'var(--warning)'}
      stroke="var(--bg)" strokeWidth={1}
    />
  )
}

function MetricChart({ history, dataKey, label, color, warningLine, criticalLine, unit = '%' }) {

  const data = useMemo(() => (
    history.slice(-80).map(d => ({ ts: d.timestamp, value: d[dataKey] ?? null }))
  ), [history, dataKey])

  const vals   = data.map(d => d.value).filter(v => v != null)
  const latest = vals.length ? vals[vals.length - 1] : null
  const minVal = vals.length ? Math.min(...vals) : 0
  const maxVal = vals.length ? Math.max(...vals) : 100
  const avgVal = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0

  const yMin = Math.max(0,   Math.floor(minVal - 5))
  const yMax = Math.min(100, Math.ceil(maxVal  + 5))

  const isCritical = criticalLine != null && latest != null && latest > criticalLine
  const isWarning  = !isCritical && warningLine != null && latest != null && latest > warningLine
  const dispColor  = isCritical ? 'var(--critical)' : isWarning ? 'var(--warning)' : color

  return (
    <div className="panel" style={{ height:'100%', display:'flex', flexDirection:'column' }}>

      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'10px', flexShrink:0 }}>
        <div className="section-label" style={{ marginBottom:0 }}>{label}</div>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:'20px', fontWeight:'600', color:dispColor, transition:'color .2s' }}>
          {latest != null ? `${latest.toFixed(1)}${unit}` : '—'}
        </span>
      </div>

      <div style={{ flex:1, minHeight:0 }}>
        {data.length === 0 ? (
          <div style={{ height:'100%', display:'flex', alignItems:'center', justifyContent:'center',
            color:'var(--text-dimmer)', fontFamily:'var(--font-mono)', fontSize:'11px' }}>
            Waiting for stream...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top:4, right:8, left:-24, bottom:0 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="ts"
                tickFormatter={ts => ts ? new Date(ts).toLocaleTimeString('en-GB', {
                  hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit'
                }) : ''}
                tick={{ fill:'var(--text-dimmer)', fontSize:8, fontFamily:'var(--font-mono)' }}
                interval="preserveStartEnd" tickLine={false}
                axisLine={{ stroke:'var(--border)' }}
              />
              <YAxis
                domain={[yMin, yMax]}
                tick={{ fill:'var(--text-dimmer)', fontSize:8, fontFamily:'var(--font-mono)' }}
                tickLine={false} axisLine={false} width={32}
              />
              <Tooltip content={<CustomTooltip color={dispColor} unit={unit} />} isAnimationActive={false} />

              {/* Warning reference line */}
              {warningLine != null && (
                <ReferenceLine y={warningLine}
                  stroke="var(--warning)" strokeDasharray="5 3" strokeWidth={1.2}
                  label={{ value:`warn ${warningLine}%`, position:'insideTopRight',
                    fill:'var(--warning)', fontSize:8, fontFamily:'var(--font-mono)' }}
                />
              )}

              {/* Critical reference line */}
              {criticalLine != null && (
                <ReferenceLine y={criticalLine}
                  stroke="var(--critical)" strokeDasharray="5 3" strokeWidth={1.2}
                  label={{ value:`crit ${criticalLine}%`, position:'insideTopRight',
                    fill:'var(--critical)', fontSize:8, fontFamily:'var(--font-mono)' }}
                />
              )}

              <Line
                type="monotone" dataKey="value"
                stroke={dispColor} strokeWidth={1.5}
                dot={(props) => <AlertDot {...props} warningLine={warningLine} criticalLine={criticalLine} />}
                activeDot={{ r:4, fill:color, stroke:'var(--bg)', strokeWidth:1 }}
                isAnimationActive={false} connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {vals.length > 0 && (
        <div style={{ display:'flex', gap:'16px', marginTop:'8px', paddingTop:'8px', borderTop:'1px solid var(--border)', flexShrink:0 }}>
          {[['Min', minVal], ['Avg', avgVal], ['Max', maxVal]].map(([k, v]) => (
            <div key={k}>
              <div style={{ fontFamily:'var(--font-mono)', fontSize:'9px', color:'var(--text-dimmer)' }}>{k}</div>
              <div style={{ fontFamily:'var(--font-mono)', fontSize:'12px', color: k === 'Max' && criticalLine != null && v > criticalLine ? 'var(--critical)' : color }}>
                {v.toFixed(1)}{unit}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default React.memo(MetricChart)