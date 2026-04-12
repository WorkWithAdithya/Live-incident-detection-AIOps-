// src/api.js
const BASE = '/api'

export async function fetchModelStatus() {
  const r = await fetch(`${BASE}/model/status`)
  return r.json()
}

export async function loadModel() {
  const r = await fetch(`${BASE}/model/load`, { method: 'POST' })
  return r.json()
}

export async function setThreshold(value) {
  const r = await fetch(`${BASE}/threshold`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ threshold: parseFloat(value) }),
  })
  return r.json()
}

export async function fetchLogs(n = 200) {
  const r = await fetch(`${BASE}/logs?n=${n}`)
  const d = await r.json()
  return d.logs ?? []
}

export async function fetchAlerts(n = 50) {
  const r = await fetch(`${BASE}/alerts?n=${n}`)
  const d = await r.json()
  return d.alerts ?? []
}

export async function fetchSessionMetrics() {
  const r = await fetch(`${BASE}/session/metrics`)
  return r.json()
}

export async function runEvaluation() {
  const r = await fetch(`${BASE}/evaluate`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

/**
 * Sends the user-set metric limits to the backend.
 * Backend stores them and passes to engine.run() on every SSE cycle.
 * Call this every time the user clicks "Set Value".
 */
export async function syncLimits(limits) {
  try {
    await fetch(`${BASE}/limits`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(limits),
    })
  } catch (e) {
    console.warn('Failed to sync limits to backend:', e)
  }
}

/**
 * Fetches the latest LSTM forecast snapshot from the backend.
 * Returns { forecast: [...12 steps], forecaster_ready: bool }
 */
export async function fetchForecast() {
  const r = await fetch(`${BASE}/forecast`)
  return r.json()
}

export async function sendRuleAlert({
  severity, cpu, memory, disk, exceeded, threshold_info
}) {
  try {
    await fetch(`${BASE}/alert/rule`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        severity, cpu, memory, disk, exceeded, threshold_info
      }),
    })
  } catch (e) {
    console.warn('Failed to send rule alert email:', e)
  }
}

export async function fetchEmailStatus() {
  const r = await fetch(`${BASE}/alert/email-status`)
  return r.json()
}

export function openStream(onMessage, onError) {
  const es = new EventSource('/stream')
  es.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)) }
    catch (err) { console.warn('SSE parse error:', err) }
  }
  es.onerror = (e) => {
    console.error('SSE error:', e)
    if (onError) onError(e)
  }
  return () => es.close()
}

export async function reloadForecaster() {
  const r = await fetch(`${BASE}/model/reload-forecaster`, { method: 'POST' })
  return r.json()
}

export async function fetchDebugPaths() {
  const r = await fetch(`${BASE}/model/debug-paths`)
  return r.json()
}