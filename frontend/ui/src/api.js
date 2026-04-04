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
  /**
   * Fetches rows DIRECTLY from NeonDB via backend.
   * Each row: { id, timestamp, cpu, memory, disk }
   * Used to hydrate the charts on page load — reflects real DB data.
   */
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

export async function sendRuleAlert({ severity, cpu, memory, disk, exceeded, threshold_info }) {
  try {
    await fetch(`${BASE}/alert/rule`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ severity, cpu, memory, disk, exceeded, threshold_info }),
    })
  } catch (e) {
    console.warn('Failed to send rule alert email:', e)
  }
}

export async function fetchEmailStatus() {
  const r = await fetch(`${BASE}/alert/email-status`)
  return r.json()
}