"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Upload, Shield, Activity, AlertTriangle, Zap, Crosshair, BarChart3, Database, Cpu, Radio, Maximize2, RefreshCw, Image as ImageIcon } from "lucide-react"
import GlassNav from "@/components/GlassNav"
import Link from "next/link"

const API_BASE = "http://localhost:8000"
const REVIEW_STORE_KEY = "roadguard_accident_review_log_v1"

type IncidentEvent = {
    id?: number
    incident_type: string
    label?: string
    license_plate?: string | null
    timestamp?: string
    camera_id?: string
}

type PastIncident = {
    id: number
    incident_type: string
    license_plate: string | null
    status: string
    timestamp: string
    camera_id: string
    evidence_image: string | null
}

type ReviewRecord = {
    key: string
    incidentId: number | null
    incidentType: string
    isAccident: boolean
    cameraId: string | null
    licensePlate: string | null
    detectedAt: string
    reviewed: boolean
}

export default function LiveDashboard() {
    const [file, setFile] = useState<File | null>(null)
    const [analyzing, setAnalyzing] = useState(false)
    const [progress, setProgress] = useState(0)
    const [incidents, setIncidents] = useState<IncidentEvent[]>([])
    const [currentFrame, setCurrentFrame] = useState<string | null>(null)
    const [stats, setStats] = useState({
        fps: 0,
        objects: 0,
        gpu: 0,
        cpu: 0,
        latency: 0
    })
    const [isCritical, setIsCritical] = useState(false)
    const [reviewLog, setReviewLog] = useState<ReviewRecord[]>([])
    const [pastIncidents, setPastIncidents] = useState<PastIncident[]>([])
    const [loadingPast, setLoadingPast] = useState(true)
    const [refreshingPast, setRefreshingPast] = useState(false)

    const fetchPastIncidents = useCallback(async (showRefresh = false) => {
        if (showRefresh) setRefreshingPast(true)
        try {
            const res = await fetch(`${API_BASE}/incidents/?limit=20`)
            const data = await res.json()
            setPastIncidents(data)
        } catch (e) {
            console.error("Failed to fetch past incidents:", e)
        } finally {
            setLoadingPast(false)
            setRefreshingPast(false)
        }
    }, [])

    useEffect(() => {
        fetchPastIncidents()
    }, [fetchPastIncidents])

    useEffect(() => {
        try {
            const raw = window.localStorage.getItem(REVIEW_STORE_KEY)
            if (!raw) return
            const parsed = JSON.parse(raw)
            if (Array.isArray(parsed)) {
                setReviewLog(parsed)
            }
        } catch {
            // Ignore storage parse errors and start with an empty review log.
        }
    }, [])

    const persistReviewLog = (next: ReviewRecord[]) => {
        setReviewLog(next)
        try {
            window.localStorage.setItem(REVIEW_STORE_KEY, JSON.stringify(next))
        } catch {
            // Ignore localStorage write issues in restricted browser contexts.
        }
    }

    const saveIncidentForReview = (incident: IncidentEvent) => {
        const incidentId = typeof incident.id === "number" ? incident.id : null
        const key = incidentId ? `inc_${incidentId}` : `tmp_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
        const newRecord: ReviewRecord = {
            key,
            incidentId,
            incidentType: incident.incident_type,
            isAccident: incident.incident_type === "accident",
            cameraId: incident.camera_id || null,
            licensePlate: incident.license_plate || null,
            detectedAt: incident.timestamp || new Date().toISOString(),
            reviewed: false,
        }

        const merged = [newRecord, ...reviewLog.filter((r) => r.key !== key)].slice(0, 200)
        persistReviewLog(merged)
    }

    const markReviewed = (key: string) => {
        const next = reviewLog.map((r) => (r.key === key ? { ...r, reviewed: true } : r))
        persistReviewLog(next)
    }
    
    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const uploaded = e.target.files?.[0]
        if (uploaded) setFile(uploaded)
    }

    const startAnalysis = async () => {
        if (!file) return
        setAnalyzing(true)
        setProgress(0)
        setIncidents([])
        setIsCritical(false)

        const formData = new FormData()
        formData.append("file", file)

        try {
            const res = await fetch(`${API_BASE}/video/upload`, {
                method: "POST",
                body: formData,
            })
            const { job_id } = await res.json()

            const sse = new EventSource(`${API_BASE}/video/stream/${job_id}`)
            
            sse.onmessage = (event) => {
                const data = JSON.parse(event.data)
                console.log("SSE DATA:", data)
                
                if (data.type === "frame") {
                    setCurrentFrame(`data:image/jpeg;base64,${data.frame}`)
                    setProgress(data.progress || 0)
                    setStats({
                        fps: data.fps || 0,
                        objects: data.objects_count || 0,
                        gpu: data.gpu_load || 0,
                        cpu: 30 + Math.floor(Math.random() * 10), // CPU still simulated as it's less critical
                        latency: data.latency_ms || 0
                    })
                } else if (data.type === "incident") {
                    setIncidents(prev => [data.incident, ...prev].slice(0, 10))
                    saveIncidentForReview(data.incident as IncidentEvent)
                    if (data.incident.incident_type === 'accident') {
                        setIsCritical(true)
                        setTimeout(() => setIsCritical(false), 2000)
                    }
                } else if (data.type === "status" && data.state === "done") {
                    sse.close()
                    setAnalyzing(false)
                    fetchPastIncidents()
                }
            }
        } catch (err) {
            console.error("Analysis failed:", err)
            setAnalyzing(false)
        }
    }

    return (
        <div className="min-h-screen bg-background text-white/90 selection:bg-roadguard-indigo/30">
            <GlassNav />
            
            <main className="max-w-[1700px] mx-auto px-6 pt-32 pb-20">
                {/* Header Title Section */}
                <div className="mb-12 flex flex-col md:flex-row md:items-end justify-between gap-6">
                    <div>
                        <motion.div 
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="flex items-center gap-3 text-roadguard-indigo mb-3"
                        >
                            <Radio size={18} />
                            <span className="text-[10px] font-black uppercase tracking-[0.4em]">Live Intelligence Stream</span>
                        </motion.div>
                        <h1 className="text-4xl md:text-5xl font-black tracking-tighter">
                            Tactical <span className="text-roadguard-indigo/40">Command</span>
                        </h1>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="px-4 py-2 glass-card rounded-xl text-[10px] font-mono text-white/40">
                            STATION_ID: RG-DELTA-01
                        </div>
                        <div className="px-4 py-2 glass-card rounded-xl text-[10px] font-mono text-roadguard-indigo">
                            SIGNAL: ENCRYPTED
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-12 gap-8">
                    
                    {/* Left Panel: Controls & Telemetry */}
                    <div className="col-span-12 xl:col-span-3 space-y-8">
                        {/* Control Module */}
                        <motion.div 
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="glass-card p-8 rounded-[2.5rem] relative overflow-hidden group"
                        >
                            <div className="absolute -top-10 -right-10 w-40 h-40 bg-roadguard-cyan/5 blur-[80px] group-hover:bg-roadguard-cyan/10 transition-colors" />
                            
                            <h2 className="text-[10px] font-black uppercase tracking-[0.3em] text-roadguard-indigo mb-8 flex items-center gap-3">
                                <Activity size={14} /> Neural Ingestion
                            </h2>
                            
                            {!file ? (
                                <label className="flex flex-col items-center justify-center aspect-square glass-card border-dashed border-white/10 rounded-[2rem] hover:border-roadguard-cyan/50 hover:bg-white/[0.05] transition-all cursor-pointer group/upload">
                                    <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4 group-hover/upload:scale-110 transition-transform">
                                        <Upload className="text-white/20 group-hover/upload:text-roadguard-cyan transition-colors" size={28} />
                                    </div>
                                    <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-white/40">Drop Visual Data</span>
                                    <input type="file" className="hidden" onChange={handleUpload} accept="video/*" />
                                </label>
                            ) : (
                                <div className="space-y-6">
                                    <div className="p-6 glass-card rounded-2xl border-white/10">
                                        <div className="text-[9px] font-black uppercase tracking-[0.2em] text-white/20 mb-2">Payload Source</div>
                                        <div className="font-bold truncate text-sm text-roadguard-cyan">{file.name}</div>
                                    </div>
                                    <button 
                                        onClick={startAnalysis}
                                        disabled={analyzing}
                                        className="w-full py-5 bg-roadguard-indigo text-white font-black text-[10px] uppercase tracking-[0.3em] rounded-2xl hover:scale-[1.02] active:scale-95 transition-all shadow-[0_20px_40px_-10px_rgba(99,102,241,0.3)] disabled:opacity-50"
                                    >
                                        {analyzing ? "Pipeline Online" : "Ignite Analysis"}
                                    </button>
                                    {analyzing && (
                                        <button onClick={() => window.location.reload()} className="w-full py-4 text-white/30 font-bold text-[9px] uppercase tracking-widest hover:text-white transition-colors">
                                            Abort Operation
                                        </button>
                                    )}
                                </div>
                            )}
                        </motion.div>

                        {/* Telemetry Grid */}
                        <div className="grid grid-cols-2 gap-4">
                            {[
                                { label: "Inference", value: `${stats.fps} FPS`, icon: Zap, color: "text-premium-gold" },
                                { label: "Entities", value: stats.objects, icon: Crosshair, color: "text-roadguard-indigo" },
                                { label: "GPU Load", value: `${stats.gpu}%`, icon: Cpu, color: "text-roadguard-blue" },
                                { label: "Latency", value: `${stats.latency}ms`, icon: Radio, color: "text-roadguard-indigo/40" },
                            ].map((m, i) => (
                                <motion.div 
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: i * 0.1 }}
                                    key={m.label} 
                                    className="glass-card p-5 rounded-3xl border-white/5"
                                >
                                    <m.icon size={14} className={`${m.color} mb-3`} />
                                    <div className="text-xl font-black tracking-tight tabular-nums">{m.value}</div>
                                    <div className="text-[9px] font-bold uppercase tracking-widest text-white/20">{m.label}</div>
                                </motion.div>
                            ))}
                        </div>
                    </div>

                    {/* Center Panel: Primary Viewport */}
                    <div className="col-span-12 xl:col-span-6 space-y-8">
                        <motion.div 
                            initial={{ opacity: 0, scale: 0.98 }}
                            animate={{ 
                                opacity: 1, 
                                scale: 1,
                                boxShadow: isCritical ? "0 0 100px -20px rgba(239,68,68,0.4)" : "0 40px 100px -40px rgba(0,0,0,0.5)"
                            }}
                            className={`aspect-video glass-card rounded-[3.5rem] border-2 ${isCritical ? 'border-alert-red' : 'border-white/[0.08]'} overflow-hidden relative group transition-all duration-700`}
                        >
                            {currentFrame ? (
                                <img src={currentFrame} className="w-full h-full object-cover" alt="Tactical View" />
                            ) : (
                                <div className="w-full h-full flex flex-col items-center justify-center bg-slate-900/40">
                                    <div className="relative mb-10 opacity-20">
                                        <Shield size={80} className="text-white" />
                                    </div>
                                    <h3 className="text-2xl font-black tracking-tighter mb-2 text-white/60">Awaiting Visual Input</h3>
                                    <p className="text-white/10 text-[10px] font-black uppercase tracking-[0.3em]">Neural link calibrated & standby</p>
                                </div>
                            )}

                            {/* HUD Overlays */}
                            <div className="absolute inset-0 pointer-events-none p-12 flex flex-col justify-between">
                                <div className="flex justify-between items-start">
                                    <div className="px-5 py-2.5 glass-card bg-black/40 rounded-xl text-[10px] font-mono tracking-tighter text-roadguard-indigo flex items-center gap-3">
                                        <div className="w-1.5 h-1.5 rounded-full bg-roadguard-indigo shadow-[0_0_8px_rgba(129,140,248,1)]" />
                                        PRIMARY_VISUAL_ARRAY // RG-V3.2
                                    </div>
                                    <div className="flex gap-2">
                                        <div className="p-2.5 glass-card bg-black/40 rounded-xl text-white/40">
                                            <Maximize2 size={14} />
                                        </div>
                                    </div>
                                </div>

                                <div className="space-y-6">
                                    <div className="flex justify-between items-end">
                                        <div className="flex flex-col">
                                            <div className="text-[10px] font-black uppercase tracking-[0.3em] text-white/30 mb-2">Analysis Vector</div>
                                            <div className="text-5xl font-black tracking-tighter tabular-nums leading-none">
                                                {Math.round(progress)}<span className="text-xl align-top ml-1 text-white/20">%</span>
                                            </div>
                                        </div>
                                        <div className="flex flex-col items-end">
                                            <div className="text-[10px] font-mono text-roadguard-indigo mb-2">X: 124.592 | Y: 89.231</div>
                                            <div className="text-[10px] font-black uppercase tracking-[0.2em] text-white/20">Coordinate Sync // LOCK</div>
                                        </div>
                                    </div>
                                    <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden p-[1px] border border-white/5">
                                        <motion.div 
                                            className="h-full bg-gradient-to-r from-roadguard-blue via-roadguard-indigo to-white rounded-full shadow-[0_0_20px_rgba(99,102,241,0.5)]"
                                            initial={{ width: 0 }}
                                            animate={{ width: `${progress}%` }}
                                            transition={{ ease: "circOut", duration: 0.5 }}
                                        />
                                    </div>
                                </div>
                            </div>
                        </motion.div>

                        <div className="grid grid-cols-2 gap-6">
                            <motion.div 
                                whileHover={{ y: -5 }}
                                className="glass-card p-8 rounded-[2.5rem] glass-card-hover"
                            >
                                <h4 className="text-[9px] font-black uppercase tracking-[0.3em] text-white/20 mb-4">Vision Enhancement</h4>
                                <div className="flex items-center gap-5">
                                    <div className="w-12 h-12 rounded-2xl bg-roadguard-cyan/10 flex items-center justify-center">
                                        <Zap size={22} className="text-roadguard-cyan" />
                                    </div>
                                    <div>
                                        <div className="text-sm font-bold tracking-tight">DeblurGAN-v2 Active</div>
                                        <div className="text-[9px] font-medium text-white/30">Frame Reconstruction Active</div>
                                    </div>
                                </div>
                            </motion.div>
                            <motion.div 
                                whileHover={{ y: -5 }}
                                className="glass-card p-8 rounded-[2.5rem] glass-card-hover"
                            >
                                <h4 className="text-[9px] font-black uppercase tracking-[0.3em] text-white/20 mb-4">Neural Architecture</h4>
                                <div className="flex items-center gap-5">
                                    <div className="w-12 h-12 rounded-2xl bg-roadguard-blue/10 flex items-center justify-center">
                                        <Shield size={22} className="text-roadguard-blue" />
                                    </div>
                                    <div>
                                        <div className="text-sm font-bold tracking-tight">YOLO11x + DeepSORT</div>
                                        <div className="text-[9px] font-medium text-white/30">Ensemble Detection Logic</div>
                                    </div>
                                </div>
                            </motion.div>
                        </div>
                    </div>

                    {/* Right Panel: Incident Stream */}
                    <div className="col-span-12 xl:col-span-3 space-y-8 h-full">
                        <motion.div 
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="glass-card p-8 rounded-[2.5rem] flex flex-col h-full relative overflow-hidden group"
                        >
                            <div className="absolute -top-10 -right-10 w-40 h-40 bg-alert-red/5 blur-[80px] group-hover:bg-alert-red/10 transition-colors" />
                            
                            <div className="flex items-center justify-between mb-10">
                                <h2 className="text-[10px] font-black uppercase tracking-[0.3em] text-alert-red flex items-center gap-3">
                                    <AlertTriangle size={14} /> Incident Feed
                                </h2>
                                <div className="px-3 py-1 bg-white/5 rounded-lg text-[9px] font-mono text-white/30">
                                    {incidents.length} EVENTS
                                </div>
                            </div>
                            
                            <div className="space-y-4 overflow-y-auto max-h-[700px] pr-2 custom-scrollbar">
                                <AnimatePresence mode="popLayout">
                                    {incidents.length > 0 ? incidents.map((inc, i) => (
                                        <motion.div
                                            key={i}
                                            initial={{ opacity: 0, y: 20, scale: 0.95 }}
                                            animate={{ opacity: 1, y: 0, scale: 1 }}
                                            exit={{ opacity: 0, scale: 0.9 }}
                                            className={`p-6 rounded-3xl border glass-card-hover ${
                                                inc.incident_type === 'accident' ? 'bg-alert-red/10 border-alert-red/20' : 'bg-white/5 border-white/10'
                                            }`}
                                        >
                                            <div className="flex justify-between items-start mb-4">
                                                <span className={`px-2.5 py-1 rounded-lg text-[8px] font-black uppercase tracking-widest ${
                                                    inc.incident_type === 'accident' ? 'bg-alert-red text-white' : 'bg-white/10 text-white/60'
                                                }`}>
                                                    {inc.incident_type}
                                                </span>
                                                <span className="text-[9px] font-mono text-white/20">{new Date().toLocaleTimeString()}</span>
                                            </div>
                                            <div className="font-bold text-sm mb-3 tracking-tight">{inc.label || "Entity Identified"}</div>
                                            <div className="flex items-center gap-3">
                                                <div className="flex-grow h-[1px] bg-white/10" />
                                                <div className="font-mono text-[10px] text-roadguard-cyan font-black tracking-widest">{inc.license_plate || "SCANNING..."}</div>
                                            </div>
                                        </motion.div>
                                    )) : (
                                        <div className="h-full flex flex-col items-center justify-center py-32 space-y-6 opacity-30">
                                            <div className="w-20 h-20 rounded-full border border-dashed border-white/20 flex items-center justify-center animate-spin-slow">
                                                <Crosshair size={32} />
                                            </div>
                                            <span className="text-[10px] font-black uppercase tracking-[0.3em]">Awaiting Signal...</span>
                                        </div>
                                    )}
                                </AnimatePresence>
                            </div>
                        </motion.div>
                    </div>

                </div>

                {/* ═══ Live Incident Feed — Visual Image Grid ═══ */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="mt-16"
                >
                    <div className="flex items-center justify-between mb-8">
                        <div className="flex items-center gap-5">
                            <div className="w-3 h-3 rounded-full bg-alert-red animate-pulse shadow-[0_0_12px_rgba(239,68,68,0.6)]" />
                            <h2 className="text-3xl md:text-4xl font-black tracking-tighter">
                                Live Incident <span className="text-roadguard-indigo/20">Feed</span>
                            </h2>
                            <div className="hidden md:block px-4 py-1.5 glass-card rounded-xl text-[9px] font-mono text-white/30">
                                Global Model Cache: ONLINE • Latest 20 incidents
                            </div>
                        </div>
                        <button
                            onClick={() => fetchPastIncidents(true)}
                            disabled={refreshingPast}
                            className="flex items-center gap-3 px-6 py-3 glass-card rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] text-white/50 hover:text-roadguard-indigo hover:border-roadguard-indigo/30 transition-all duration-500 group"
                        >
                            <RefreshCw size={14} className={`group-hover:text-roadguard-indigo transition-colors ${refreshingPast ? 'animate-spin' : ''}`} />
                            Refresh
                        </button>
                    </div>

                    {loadingPast ? (
                        <div className="py-20 text-center flex flex-col items-center">
                            <div className="w-12 h-12 border-2 border-white/5 border-t-roadguard-cyan rounded-full animate-spin mb-6" />
                            <p className="text-white/20 font-black uppercase tracking-[0.4em] text-[9px]">Loading incident archive...</p>
                        </div>
                    ) : pastIncidents.length === 0 ? (
                        <div className="py-20 text-center flex flex-col items-center glass-card rounded-[3rem]">
                            <ImageIcon size={48} className="text-white/5 mb-6" />
                            <p className="text-white/20 font-black uppercase tracking-[0.4em] text-[10px]">No incidents recorded yet</p>
                            <p className="text-white/10 text-xs mt-2">Upload and analyze a video to populate the feed</p>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                            <AnimatePresence mode="popLayout">
                                {pastIncidents.map((inc, i) => {
                                    const evidenceUrl = inc.evidence_image
                                        ? `${API_BASE}/evidence/${inc.evidence_image.replace(/\\/g, '/')}`
                                        : null
                                    const isAccident = inc.incident_type === 'accident'
                                    const isHitRun = inc.incident_type === 'hit_and_run'
                                    return (
                                        <motion.div
                                            key={inc.id}
                                            initial={{ opacity: 0, y: 20, scale: 0.95 }}
                                            animate={{ opacity: 1, y: 0, scale: 1 }}
                                            exit={{ opacity: 0, scale: 0.9 }}
                                            transition={{ delay: i * 0.04 }}
                                        >
                                            <Link href={`/incidents/${inc.id}`} className="block group">
                                                <div className={`rounded-[2rem] overflow-hidden border transition-all duration-700 group-hover:-translate-y-2 group-hover:shadow-2xl relative ${
                                                    isAccident ? 'border-alert-red/30 group-hover:border-alert-red/60 group-hover:shadow-alert-red/10' :
                                                    isHitRun ? 'border-premium-gold/30 group-hover:border-premium-gold/60 group-hover:shadow-premium-gold/10' :
                                                    'border-white/[0.08] group-hover:border-white/20'
                                                }`}>
                                                    {/* Image Area */}
                                                    <div className="aspect-[16/10] relative overflow-hidden bg-surface-dark">
                                                        {evidenceUrl ? (
                                                            <img
                                                                src={evidenceUrl}
                                                                alt={`Incident ${inc.id}`}
                                                                className="w-full h-full object-cover brightness-[0.7] group-hover:brightness-90 group-hover:scale-105 transition-all duration-700"
                                                            />
                                                        ) : (
                                                            <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-surface-dark to-background">
                                                                <Shield size={40} className="text-white/5" />
                                                            </div>
                                                        )}

                                                        {/* Gradient overlay */}
                                                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent pointer-events-none" />

                                                        {/* Top badges */}
                                                        <div className="absolute top-4 left-4 right-4 flex items-start justify-between">
                                                            <span className={`px-3 py-1.5 rounded-xl text-[8px] font-black uppercase tracking-widest backdrop-blur-md shadow-lg ${
                                                                isAccident ? 'bg-alert-red/90 text-white shadow-alert-red/30' :
                                                                isHitRun ? 'bg-premium-gold/90 text-black shadow-premium-gold/30' :
                                                                'bg-white/20 text-white/90'
                                                            }`}>
                                                                {inc.incident_type.replace('_', ' ')}
                                                            </span>
                                                            <span className={`px-3 py-1.5 rounded-xl text-[8px] font-black uppercase tracking-widest backdrop-blur-md ${
                                                                inc.status === 'resolved' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
                                                                inc.status === 'investigating' ? 'bg-roadguard-blue/20 text-roadguard-blue border border-roadguard-blue/30' :
                                                                'bg-premium-gold/10 text-premium-gold border border-premium-gold/30'
                                                            }`}>
                                                                {inc.status}
                                                            </span>
                                                        </div>

                                                        {/* Bottom info overlay */}
                                                        <div className="absolute bottom-0 left-0 right-0 p-4">
                                                            <div className="flex items-end justify-between">
                                                                <div>
                                                                    <div className="font-mono text-[11px] font-black text-roadguard-indigo tracking-widest mb-1">
                                                                        {inc.license_plate || "NO PLATE"}
                                                                    </div>
                                                                    <div className="text-[9px] font-medium text-white/40">
                                                                        {inc.camera_id} • {new Date(inc.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                                                    </div>
                                                                </div>
                                                                <div className="text-[9px] font-mono text-white/20">
                                                                    #{inc.id.toString().padStart(3, '0')}
                                                                </div>
                                                            </div>
                                                        </div>

                                                        {/* Live indicator dot */}
                                                        {i < 2 && (
                                                            <div className="absolute top-4 right-4 hidden">
                                                                <div className="w-2 h-2 rounded-full bg-alert-red animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]" />
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            </Link>
                                        </motion.div>
                                    )
                                })}
                            </AnimatePresence>
                        </div>
                    )}
                </motion.div>

            </main>

            <style jsx global>{`
                .custom-scrollbar::-webkit-scrollbar {
                    width: 4px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: transparent;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                }
                @keyframes spin-slow {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
                .animate-spin-slow {
                    animation: spin-slow 8s linear infinite;
                }
            `}</style>
        </div>
    )
}
