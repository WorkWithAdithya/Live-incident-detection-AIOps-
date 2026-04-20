// src/App.jsx — Warning-only (no critical level)

import { useState, useEffect, useCallback, useRef } from 'react'
import Header          from './components/Header.jsx'
import ControlPanel    from './components/ControlPanel.jsx'
import AlertFeed       from './components/AlertFeed.jsx'
import PredictionPanel from './components/PredictionPanel.jsx'
import LogTable        from './components/LogTable.jsx'
import MetricChart     from './components/MetricChart.jsx'
import { fetchModelStatus, fetchLogs, fetchAlerts, fetchEmailStatus, openStream, sendRuleAlert, syncLimits } from './api.js'

const MAX_HISTORY = 300
const MAX_ALERTS  = 100
const DEFAULT_LIMITS = {
  cpu_warning:null, cpu_critical:null,
  memory_warning:null, memory_critical:null,
  disk_warning:null, disk_critical:null,
}

function getSeverity(row, limits) {
  for (const key of ['cpu','memory','disk']) {
    const val=row[key]; if(val==null) continue
    const w=limits[`${key}_warning`]
    if(w!=null&&val>w) return 'WARNING'
  }
  return 'NORMAL'
}

function getExceeded(row, limits) {
  const exceeded=[]
  for(const[key,label]of[['cpu','CPU'],['memory','Memory'],['disk','Disk']]) {
    const val=row[key]; if(val==null) continue
    const w=limits[`${key}_warning`]
    if(w!=null&&val>w) exceeded.push(`${label} ${val.toFixed(1)}% > ${w}% (warning)`)
  }
  return exceeded
}

function buildRuleAlert(row, limits) {
  const sev=getSeverity(row,limits)
  if(sev==='NORMAL') return null
  return { source:'rule', severity:sev, timestamp:row.timestamp,
    cpu:row.cpu, memory:row.memory, disk:row.disk,
    exceeded:getExceeded(row,limits), error:row.error, error_ratio:row.error_ratio }
}

export default function App() {
  const [modelStatus, setModelStatus]   = useState(null)
  const [history,     setHistory]       = useState([])
  const [alerts,      setAlerts]        = useState([])
  const [latest,      setLatest]        = useState(null)
  const [streamStatus,setStreamStatus]  = useState('connecting')
  const [limits,      setLimits]        = useState(DEFAULT_LIMITS)
  const [emailEnabled,setEmailEnabled]  = useState(false)
  const limitsRef=useRef(limits), historyRef=useRef(history)
  useEffect(()=>{limitsRef.current=limits},[limits])
  useEffect(()=>{historyRef.current=history},[history])

  useEffect(()=>{
    async function init(){
      try{
        const[status,logs,alts,emailSt]=await Promise.all([
          fetchModelStatus(),fetchLogs(200),fetchAlerts(50),fetchEmailStatus()])
        setModelStatus(status); setEmailEnabled(emailSt?.enabled??false)
        const enriched=logs.map(r=>({...r,error:null,threshold:status?.threshold??null,
          error_ratio:null,is_anomaly:false,warming_up:false,actual_rows:0,flagged:[],
          forecast:[],forecast_breaches:[],forecaster_ready:status?.forecaster_ready??false}))
        setHistory(enriched); historyRef.current=enriched
        if(enriched.length) setLatest(enriched[enriched.length-1])
        setAlerts(alts.filter(a=>a.severity==='WARNING').map(a=>({...a,source:'lstm'})))
      }catch(e){console.error('Init failed:',e)}
    }
    init()
  },[])

  useEffect(()=>{
    const close=openStream((data)=>{
      setStreamStatus('connected')
      if(data.status==='no_new_rows'||data.status==='model_not_loaded') return
      if(data.status!=='ok') return
      setLatest(data)
      setHistory(prev=>{
        if(data.id&&prev.some(r=>r.id===data.id)) return prev
        const next=[...prev,data]
        const trimmed=next.length>MAX_HISTORY?next.slice(-MAX_HISTORY):next
        historyRef.current=trimmed; return trimmed
      })
      const newAlerts=[]
      const ruleAlert=buildRuleAlert(data,limitsRef.current)
      if(ruleAlert){
        newAlerts.push(ruleAlert)
        const tInfo={},lim=limitsRef.current
        for(const[key,label]of[['cpu','CPU'],['memory','Memory'],['disk','Disk']]){
          if(lim[`${key}_warning`]!=null) tInfo[`${label} Warning`]=lim[`${key}_warning`]
        }
        sendRuleAlert({severity:ruleAlert.severity,cpu:data.cpu,memory:data.memory,
          disk:data.disk,exceeded:ruleAlert.exceeded,threshold_info:tInfo})
      }
      if(data.is_anomaly&&data.severity==='WARNING')
        newAlerts.push({...data,source:'lstm'})
      if(newAlerts.length>0)
        setAlerts(prev=>{const next=[...prev,...newAlerts];return next.length>MAX_ALERTS?next.slice(-MAX_ALERTS):next})
    },()=>setStreamStatus('error'))
    return close
  },[])

  const handleLimitsChange=useCallback((newLimits)=>{
    setLimits(newLimits); limitsRef.current=newLimits
    syncLimits(newLimits)
    const ruleAlerts=historyRef.current.map(r=>buildRuleAlert(r,newLimits)).filter(Boolean)
    if(ruleAlerts.length>0){
      const worst=ruleAlerts[ruleAlerts.length-1]
      const tInfo={}
      for(const[key,label]of[['cpu','CPU'],['memory','Memory'],['disk','Disk']]){
        if(newLimits[`${key}_warning`]!=null) tInfo[`${label} Warning`]=newLimits[`${key}_warning`]
      }
      sendRuleAlert({severity:worst.severity,cpu:worst.cpu,memory:worst.memory,
        disk:worst.disk,exceeded:worst.exceeded,threshold_info:tInfo})
      setAlerts(prev=>{const lstmOnly=prev.filter(a=>a.source!=='rule');
        const next=[...lstmOnly,...ruleAlerts];return next.length>MAX_ALERTS?next.slice(-MAX_ALERTS):next})
    } else {
      setAlerts(prev=>prev.filter(a=>a.source!=='rule'))
    }
  },[])

  const handleModelLoaded=useCallback(async()=>{
    const s=await fetchModelStatus();setModelStatus(s)
  },[])

  return (
    <div style={{minHeight:'100vh',background:'var(--bg)',display:'flex',flexDirection:'column'}}>
      <Header modelStatus={modelStatus} onModelLoaded={handleModelLoaded}/>
      {streamStatus!=='connected'&&(
        <div style={{padding:'5px 20px',background:'rgba(250,204,21,.07)',
          borderBottom:'1px solid rgba(250,204,21,.18)',fontFamily:'var(--font-mono)',
          fontSize:'10px',color:'var(--warning)'}}>
          {streamStatus==='connecting'?'⟳ Connecting...':'⚠ Stream disconnected'}
        </div>
      )}
      <main style={{flex:1,padding:'16px 20px',display:'flex',gap:'14px'}}>
        {/* Left */}
        <div style={{flex:1,display:'flex',flexDirection:'column',gap:'14px',minWidth:0}}>
          <ControlPanel modelStatus={modelStatus} limits={limits} onLimitsChange={handleLimitsChange}/>
          <LogTable history={history} limits={limits} rowSeverity={getSeverity}/>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:'14px',minHeight:'260px'}}>
            <MetricChart history={history} dataKey="memory" label="Realtime Memory Usage"
              color="var(--mem-color)" warningLine={limits.memory_warning}/>
            <MetricChart history={history} dataKey="cpu" label="Realtime CPU Usage"
              color="var(--cpu-color)" warningLine={limits.cpu_warning}/>
            <MetricChart history={history} dataKey="disk" label="Realtime Disk Usage"
              color="var(--disk-color)" warningLine={limits.disk_warning}/>
          </div>
        </div>
        {/* Right sticky */}
        <div style={{width:'300px',flexShrink:0,display:'flex',flexDirection:'column',
          gap:'14px',position:'sticky',top:'57px',height:'calc(100vh - 73px)',alignSelf:'flex-start'}}>
          <div style={{flex:'0 0 54%',minHeight:0,overflow:'hidden'}}>
            <AlertFeed alerts={alerts}/>
          </div>
          <div style={{flex:'0 0 43%',minHeight:0,overflow:'hidden'}}>
            <PredictionPanel
              latest={latest}
              limits={limits}
              forecasterReady={modelStatus?.forecaster_ready ?? latest?.forecaster_ready ?? false}
            />
          </div>
        </div>
      </main>
      <footer style={{padding:'7px 20px',borderTop:'1px solid var(--border)',
        display:'flex',alignItems:'center',justifyContent:'space-between',
        fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text-dimmer)'}}>
        <span style={{display:'flex',alignItems:'center',gap:'8px'}}>
          <span>AIOps · LSTM Autoencoder + Forecaster</span>
          {[
            [emailEnabled,'📧 Email'],
            [latest?.forecaster_ready,'🔮 Forecaster'],
          ].map(([on,label])=>(
            <span key={label} style={{fontSize:'9px',
              color:on?'var(--normal)':'var(--text-dimmer)',
              background:on?'rgba(74,222,128,.07)':'transparent',
              border:`1px solid ${on?'rgba(74,222,128,.2)':'var(--border)'}`,
              borderRadius:'var(--radius)',padding:'2px 7px'}}>
              {label} {on?'on':'off'}
            </span>
          ))}
        </span>
        <span style={{display:'flex',alignItems:'center',gap:'6px'}}>
          <span className={`dot ${streamStatus==='connected'?'normal':'warning'}`} style={{margin:0}}/>
          {streamStatus==='connected'?'Stream live':'Stream offline'}
          {latest?.timestamp&&(
            <span style={{marginLeft:'10px'}}>
              last: {new Date(latest.timestamp).toLocaleTimeString('en-GB',{hour12:false})}
            </span>
          )}
        </span>
      </footer>
    </div>
  )
}