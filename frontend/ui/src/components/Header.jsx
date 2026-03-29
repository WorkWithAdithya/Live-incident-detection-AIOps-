// src/components/Header.jsx
import { useState } from 'react'
import { loadModel } from '../api.js'

export default function Header({ modelStatus, onModelLoaded }) {
  const [loading, setLoading] = useState(false)

  async function handleLoad() {
    setLoading(true)
    try {
      const res = await loadModel()
      if (res.success) onModelLoaded(res)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const loaded = modelStatus?.loaded

  return (
    <header style={{
      display:        'flex',
      alignItems:     'center',
      justifyContent: 'space-between',
      padding:        '14px 20px',
      borderBottom:   '1px solid var(--border)',
      background:     'var(--bg)',
      position:       'sticky',
      top:            0,
      zIndex:         100,
    }}>
      {/* Title */}
      <div>
        <div style={{
          fontFamily:    'var(--font-mono)',
          fontSize:      '15px',
          fontWeight:    '600',
          letterSpacing: '0.02em',
          color:         'var(--text)',
        }}>
          Incident Detection · Log Analysis
        </div>
        <div style={{ fontSize: '11px', color: 'var(--text-dimmer)', marginTop: '2px' }}>
          LSTM Autoencoder · Real-time AIOps
        </div>
      </div>

      {/* Model status card */}
      <div style={{
        display:      'flex',
        alignItems:   'center',
        gap:          '12px',
        background:   'var(--bg-panel)',
        border:       '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        padding:      '8px 14px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span className={`dot ${loaded ? 'normal' : 'off'}`} />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-dim)' }}>
            {loaded ? 'Model Loaded' : 'Model Not Loaded'}
          </span>
        </div>

        {loaded && modelStatus?.threshold != null && (
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize:   '10px',
            color:      'var(--text-dimmer)',
            borderLeft: '1px solid var(--border)',
            paddingLeft:'10px',
          }}>
            thresh {modelStatus.threshold.toFixed(6)}
          </span>
        )}

        {!loaded && (
          <button
            className="primary"
            onClick={handleLoad}
            disabled={loading}
            style={{ padding: '4px 12px', fontSize: '10px' }}
          >
            {loading ? 'Loading...' : 'Load Model'}
          </button>
        )}
      </div>
    </header>
  )
}