// src/components/AnomalyScoreChart.jsx
// Bottom right: live anomaly score (reconstruction error) vs threshold
// Only NORMAL and WARNING severity levels.

import {
  ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer,
} from 'recharts'

function formatTick(ts) {
  if (!ts) return ''
  const d = new Date(ts)
  return d.toLocaleTimeString('en-GB', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  return (
    <div style={{
      background:   'var(--bg-panel)',
      border:       '1px solid var(--border-2)',
      borderRadius: 'var(--radius)',
      padding:      '8px 12px',
      fontFamily:   'var(--font-mono)',
      fontSize:     '11px',
    }}>
      <div style={{ color:'var(--text-dimmer)', marginBottom:'4px' }}>{formatTick(label)}</div>
      <div style={{ color: d?.is_anomaly ? 'var(--warning)' : 'var(--text)' }}>
        Error: {d?.error?.toFixed(6)}
      </div>
      <div style={{ color:'var(--thresh-color)' }}>
        Thresh: {d?.threshold?.toFixed(6)}
      </div>
      <div style={{ color:'var(--text-dimmer)', marginTop:'2px' }}>
        {d?.error_ratio?.toFixed(2)}× threshold
      </div>
      {d?.severity && (
        <div style={{ marginTop:'4px' }}>
          <span className={`badge ${d.severity}`}>{d.severity}</span>
        </div>
      )}
    </div>
  )
}

export default function AnomalyScoreChart({ history }) {
  const data = history.slice(-80).map(d => ({
    ts:          d.timestamp,
    error:       d.error,
    threshold:   d.threshold,
    error_ratio: d.error_ratio,
    is_anomaly:  d.is_anomaly,
    severity:    d.severity,
  }))

  // Dynamic y-axis max — at least 2× current threshold
  const maxErr    = data.length ? Math.max(...data.map(d => d.error)) : 0.005
  const thresh    = data.length ? data[data.length - 1]?.threshold : null
  const yMax      = Math.max(maxErr * 1.2, (thresh ?? 0.002) * 2)

  return (
    <div className="panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'10px' }}>
        <div className="section-label" style={{ marginBottom: 0 }}>
          Anomaly Score vs Threshold
        </div>
        {thresh && (
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize:   '10px',
            color:      'var(--thresh-color)',
          }}>
            thresh {thresh.toFixed(6)}
          </span>
        )}
      </div>

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
        <ResponsiveContainer width="100%" height={155}>
          <ComposedChart data={data} margin={{ top: 4, right: 6, left: -22, bottom: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="ts"
              tickFormatter={formatTick}
              tick={{ fill:'var(--text-dimmer)', fontSize:9, fontFamily:'var(--font-mono)' }}
              interval="preserveStartEnd"
              tickLine={false}
              axisLine={{ stroke:'var(--border)' }}
            />
            <YAxis
              domain={[0, yMax]}
              tickFormatter={v => v.toFixed(4)}
              tick={{ fill:'var(--text-dimmer)', fontSize:9, fontFamily:'var(--font-mono)' }}
              tickLine={false} axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Threshold line */}
            {thresh && (
              <ReferenceLine
                y={thresh}
                stroke="var(--thresh-color)"
                strokeDasharray="5 3"
                strokeWidth={1.5}
                label={{
                  value:    'threshold',
                  position: 'insideTopRight',
                  fill:     'var(--thresh-color)',
                  fontSize: 9,
                  fontFamily: 'var(--font-mono)',
                }}
              />
            )}

            {/* Error area */}
            <Area
              type="monotone"
              dataKey="error"
              fill="rgba(250,204,21,0.06)"
              stroke="none"
              isAnimationActive={false}
            />

            {/* Error line — warning dots above threshold */}
            <Line
              type="monotone"
              dataKey="error"
              stroke="var(--err-color)"
              strokeWidth={1.5}
              dot={(props) => {
                const { cx, cy, payload } = props
                if (!payload.is_anomaly) return null
                return (
                  <circle
                    key={`dot-${cx}-${cy}`}
                    cx={cx} cy={cy} r={3}
                    fill='var(--warning)'
                    stroke="none"
                  />
                )
              }}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}

      {/* Live count */}
      {data.length > 0 && (
        <div style={{
          display:    'flex',
          gap:        '14px',
          marginTop:  '8px',
          paddingTop: '8px',
          borderTop:  '1px solid var(--border)',
        }}>
          {['NORMAL','WARNING'].map(sev => {
            const count = data.filter(d => d.severity === sev).length
            return (
              <div key={sev} style={{ textAlign:'center' }}>
                <div style={{ fontFamily:'var(--font-mono)', fontSize:'9px', color:'var(--text-dimmer)' }}>
                  {sev}
                </div>
                <div style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize:   '14px',
                  fontWeight: '600',
                  color: sev === 'NORMAL' ? 'var(--normal)' : 'var(--warning)',
                }}>
                  {count}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}