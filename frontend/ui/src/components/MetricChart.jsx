// src/components/MetricChart.jsx
// Reusable single-metric live chart used for CPU, Memory, Disk separately

import {
  AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer,
} from 'recharts'

function formatTick(ts) {
  if (!ts) return ''
  return new Date(ts).toLocaleTimeString('en-GB', {
    hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

const CustomTooltip = ({ active, payload, label, unit, color }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background:   'var(--bg-panel)',
      border:       '1px solid var(--border-2)',
      borderRadius: 'var(--radius)',
      padding:      '7px 11px',
      fontFamily:   'var(--font-mono)',
      fontSize:     '11px',
    }}>
      <div style={{ color: 'var(--text-dimmer)', marginBottom: '3px' }}>{formatTick(label)}</div>
      <div style={{ color }}>
        {payload[0]?.value?.toFixed(2)}{unit}
      </div>
    </div>
  )
}

export default function MetricChart({
  history,
  dataKey,
  label,
  color,
  warningLine = 85,
  unit = '%',
}) {
  const data = history.slice(-80).map(d => ({
    ts:    d.timestamp,
    value: d[dataKey],
  }))

  const latest = data.length ? data[data.length - 1]?.value : null

  return (
    <div className="panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{
        display:        'flex',
        alignItems:     'center',
        justifyContent: 'space-between',
        marginBottom:   '10px',
        flexShrink:     0,
      }}>
        <div className="section-label" style={{ marginBottom: 0 }}>
          {label}
        </div>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize:   '18px',
          fontWeight: '600',
          color,
        }}>
          {latest != null ? `${latest.toFixed(1)}${unit}` : '—'}
        </span>
      </div>

      {/* Chart */}
      {data.length === 0 ? (
        <div style={{
          flex:           1,
          display:        'flex',
          alignItems:     'center',
          justifyContent: 'center',
          color:          'var(--text-dimmer)',
          fontFamily:     'var(--font-mono)',
          fontSize:       '11px',
        }}>
          Waiting for stream...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={160}>
          <AreaChart data={data} margin={{ top: 4, right: 6, left: -24, bottom: 0 }}>
            <defs>
              <linearGradient id={`grad-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={color} stopOpacity={0.15} />
                <stop offset="95%" stopColor={color} stopOpacity={0.01} />
              </linearGradient>
            </defs>

            <CartesianGrid
              stroke="var(--border)"
              strokeDasharray="3 3"
              vertical={false}
            />
            <XAxis
              dataKey="ts"
              tickFormatter={formatTick}
              tick={{ fill: 'var(--text-dimmer)', fontSize: 8, fontFamily: 'var(--font-mono)' }}
              interval="preserveStartEnd"
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
            />
            <YAxis
              domain={[0, 100]}
              tickFormatter={v => `${v}`}
              tick={{ fill: 'var(--text-dimmer)', fontSize: 8, fontFamily: 'var(--font-mono)' }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              content={<CustomTooltip unit={unit} color={color} />}
            />

            {/* Warning reference line */}
            <ReferenceLine
              y={warningLine}
              stroke="var(--border-2)"
              strokeDasharray="4 4"
              strokeWidth={1}
              label={{
                value:      `${warningLine}%`,
                position:   'insideTopRight',
                fill:       'var(--text-dimmer)',
                fontSize:   8,
                fontFamily: 'var(--font-mono)',
              }}
            />

            <Area
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={1.5}
              fill={`url(#grad-${dataKey})`}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}

      {/* Min / Avg / Max */}
      {data.length > 0 && (() => {
        const vals = data.map(d => d.value).filter(v => v != null)
        const min  = Math.min(...vals)
        const max  = Math.max(...vals)
        const avg  = vals.reduce((a, b) => a + b, 0) / vals.length
        return (
          <div style={{
            display:    'flex',
            gap:        '14px',
            marginTop:  '8px',
            paddingTop: '8px',
            borderTop:  '1px solid var(--border)',
            flexShrink: 0,
          }}>
            {[['Min', min], ['Avg', avg], ['Max', max]].map(([k, v]) => (
              <div key={k}>
                <div style={{ fontFamily:'var(--font-mono)', fontSize:'9px', color:'var(--text-dimmer)' }}>{k}</div>
                <div style={{ fontFamily:'var(--font-mono)', fontSize:'12px', color }}>
                  {v.toFixed(1)}{unit}
                </div>
              </div>
            ))}
          </div>
        )
      })()}
    </div>
  )
}