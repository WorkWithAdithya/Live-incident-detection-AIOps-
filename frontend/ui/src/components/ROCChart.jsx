// src/components/ROCChart.jsx
// Bottom middle: ROC curve from evaluation data

import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { runEvaluation } from '../api.js'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  return (
    <div style={{
      background: 'var(--bg-panel)',
      border:     '1px solid var(--border-2)',
      borderRadius:'var(--radius)',
      padding:    '6px 10px',
      fontFamily: 'var(--font-mono)',
      fontSize:   '10px',
    }}>
      <div style={{ color:'var(--text-dim)' }}>FPR: {d?.fpr?.toFixed(3)}</div>
      <div style={{ color:'var(--text)' }}>TPR: {d?.tpr?.toFixed(3)}</div>
    </div>
  )
}

export default function ROCChart({ evalData, onEvalComplete }) {
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  async function handleEvaluate() {
    setLoading(true)
    setError(null)
    try {
      const data = await runEvaluation()
      onEvalComplete(data)
    } catch (e) {
      setError('Evaluation failed — check backend logs')
    } finally {
      setLoading(false)
    }
  }

  const rocData = evalData?.roc_curve
    ? evalData.roc_curve.fpr.map((f, i) => ({
        fpr: f,
        tpr: evalData.roc_curve.tpr[i],
      }))
    : null

  return (
    <div className="panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
        <div className="section-label" style={{ marginBottom: 0 }}>
          ROC Curve
          {evalData?.roc_curve && (
            <span style={{ color: 'var(--text-dim)', marginLeft: '6px' }}>
              AUC={evalData.roc_curve.auc}
            </span>
          )}
        </div>
        <button
          onClick={handleEvaluate}
          disabled={loading}
          style={{ padding: '3px 10px', fontSize: '10px' }}
        >
          {loading ? 'Running...' : 'Run Eval'}
        </button>
      </div>

      {error && (
        <div style={{ fontFamily:'var(--font-mono)', fontSize:'10px', color:'var(--critical)', marginBottom:'8px' }}>
          {error}
        </div>
      )}

      {!rocData ? (
        <div style={{
          flex:           1,
          display:        'flex',
          flexDirection:  'column',
          alignItems:     'center',
          justifyContent: 'center',
          color:          'var(--text-dimmer)',
          fontFamily:     'var(--font-mono)',
          fontSize:       '11px',
          gap:            '10px',
        }}>
          <span>Click Run Eval to compute</span>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={rocData} margin={{ top: 4, right: 6, left: -22, bottom: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
            <XAxis
              dataKey="fpr"
              type="number" domain={[0,1]}
              tickFormatter={v => v.toFixed(1)}
              tick={{ fill:'var(--text-dimmer)', fontSize:9, fontFamily:'var(--font-mono)' }}
              tickLine={false}
              label={{ value:'FPR', position:'insideBottom', offset:-1, fill:'var(--text-dimmer)', fontSize:9, fontFamily:'var(--font-mono)' }}
            />
            <YAxis
              domain={[0,1]}
              tickFormatter={v => v.toFixed(1)}
              tick={{ fill:'var(--text-dimmer)', fontSize:9, fontFamily:'var(--font-mono)' }}
              tickLine={false} axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              segment={[{x:0,y:0},{x:1,y:1}]}
              stroke="var(--border-2)" strokeDasharray="4 4"
            />
            <Line
              type="monotone" dataKey="tpr"
              stroke="var(--text)" strokeWidth={1.5}
              dot={false} isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}

      {evalData?.metrics && (
        <div style={{
          display:    'flex',
          gap:        '10px',
          flexWrap:   'wrap',
          marginTop:  '8px',
          paddingTop: '8px',
          borderTop:  '1px solid var(--border)',
        }}>
          {[
            ['F1',    evalData.metrics.f1],
            ['Prec',  evalData.metrics.precision],
            ['Rec',   evalData.metrics.recall],
            ['Acc',   evalData.metrics.accuracy],
          ].map(([k, v]) => (
            <div key={k} style={{ textAlign: 'center' }}>
              <div style={{ fontFamily:'var(--font-mono)', fontSize:'9px', color:'var(--text-dimmer)' }}>{k}</div>
              <div style={{ fontFamily:'var(--font-mono)', fontSize:'13px', fontWeight:'600' }}>
                {v != null ? (v * 100).toFixed(1) + '%' : '—'}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}