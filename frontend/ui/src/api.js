// src/api.js
// All communication with the FastAPI backend

const BASE = '/api'

// ── REST helpers ──────────────────────────────────────────────────────────────

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

export async function fetchHistory(n = 120) {
  const r = await fetch(`${BASE}/history?n=${n}`)
  const d = await r.json()
  return d.history ?? []
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

// ── SSE stream ────────────────────────────────────────────────────────────────

/**
 * Opens an SSE connection to /stream.
 * Calls onMessage(parsedData) for each event.
 * Returns a cleanup function to close the connection.
 */
export function openStream(onMessage, onError) {
  const es = new EventSource('/stream')

  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data)
      onMessage(data)
    } catch (err) {
      console.warn('SSE parse error:', err)
    }
  }

  es.onerror = (e) => {
    console.error('SSE error:', e)
    if (onError) onError(e)
  }

  return () => es.close()
}