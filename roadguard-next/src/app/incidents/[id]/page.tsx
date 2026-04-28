"use client"

import { useEffect, useState, use } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ArrowLeft, Clock, MapPin, Shield, User, CheckCircle, AlertCircle, Fingerprint, Activity, FileText, Zap, Radio, Maximize2, Camera, AlertTriangle, Trash2, Download } from "lucide-react"
import GlassNav from "@/components/GlassNav"
import Link from "next/link"
import { useRouter } from "next/navigation"

const API_BASE = "http://localhost:8000"

export default function IncidentDetailPage({ params }: { params: Promise<{ id: string }> }) {
    const { id } = use(params)
    const [incident, setIncident] = useState<any>(null)
    const [loading, setLoading] = useState(true)
    const [updating, setUpdating] = useState(false)
    const router = useRouter()

    useEffect(() => {
        const fetchIncident = async () => {
            try {
                const res = await fetch(`${API_BASE}/incidents/${id}`)
                const data = await res.json()
                setIncident(data)
            } catch (error) {
                console.error("Failed to fetch incident:", error)
            } finally {
                setLoading(false)
            }
        }
        fetchIncident()
    }, [id])

    const updateStatus = async (newStatus: string) => {
        setUpdating(true)
        try {
            await fetch(`${API_BASE}/incidents/${id}/status?status=${newStatus}`, { method: 'PUT' })
            setIncident({ ...incident, status: newStatus })
        } catch (error) {
            console.error("Failed to update status:", error)
        } finally {
            setUpdating(false)
        }
    }

    if (loading) return (
        <div className="min-h-screen bg-background flex flex-col items-center justify-center space-y-8">
            <div className="w-20 h-20 border-2 border-white/5 border-t-roadguard-indigo rounded-full animate-spin shadow-[0_0_40px_rgba(99,102,241,0.2)]" />
            <p className="text-white/20 font-black uppercase tracking-[0.5em] text-[10px]">Retrieving Secure Records...</p>
        </div>
    )

    if (!incident) return (
        <div className="min-h-screen bg-background flex flex-col items-center justify-center">
            <h2 className="text-4xl font-black tracking-tighter mb-4 text-white/20">CASE_NOT_FOUND</h2>
            <Link href="/incidents" className="text-roadguard-indigo font-black uppercase tracking-widest text-xs">Return to Archive</Link>
        </div>
    )

    const evidenceUrl = incident.evidence_image 
        ? `${API_BASE}/evidence/${incident.evidence_image.replace(/\\/g, '/')}`
        : null

    return (
        <div className="min-h-screen bg-background text-white/90 selection:bg-roadguard-indigo/30 overflow-x-hidden">
            <GlassNav />
            
            <main className="max-w-[1600px] mx-auto px-6 pt-32 pb-20 relative">
                
                {/* Back Link with Animation */}
                <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}>
                    <Link href="/incidents" className="inline-flex items-center gap-4 text-white/30 hover:text-roadguard-indigo font-black text-[10px] uppercase tracking-[0.4em] mb-12 group transition-all">
                        <div className="w-10 h-10 rounded-xl glass-card flex items-center justify-center group-hover:bg-roadguard-indigo group-hover:text-black transition-all">
                            <ArrowLeft size={16} />
                        </div>
                        SECURE_DATABASE_BACK
                    </Link>
                </motion.div>

                <div className="grid grid-cols-12 gap-12">
                    
                    {/* Primary Evidence Viewport (8 cols) */}
                    <div className="col-span-12 xl:col-span-8 space-y-10">
                        {/* Main Image Frame */}
                        <motion.div 
                            initial={{ opacity: 0, scale: 0.98 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="glass-card p-4 rounded-[3.5rem] relative overflow-hidden group shadow-2xl"
                        >
                            <div className="aspect-video rounded-[2.5rem] overflow-hidden relative">
                                {evidenceUrl ? (
                                    <img 
                                        src={evidenceUrl} 
                                        alt="Tactical Evidence" 
                                        className="w-full h-full object-cover brightness-75 group-hover:brightness-100 transition-all duration-1000"
                                    />
                                ) : (
                                    <div className="w-full h-full flex flex-col items-center justify-center bg-surface-dark/50">
                                        <AlertCircle size={80} className="text-white/5 mb-6" />
                                        <span className="text-[10px] font-black uppercase tracking-[0.6em] text-white/10 italic">Evidence Data Corrupted</span>
                                    </div>
                                )}
                                
                                {/* Gradient Vignette */}
                                <div className="absolute inset-0 bg-gradient-to-t from-background/90 via-transparent to-transparent pointer-events-none" />
                                
                                {/* Evidence HUD Overlay */}
                                <div className="absolute inset-0 pointer-events-none p-12 flex flex-col justify-between">
                                    <div className="flex justify-between items-start">
                                        <div className="space-y-3">
                                            <div className="px-5 py-2.5 glass-card bg-black/40 rounded-xl text-[10px] font-mono font-black text-roadguard-indigo flex items-center gap-3">
                                                <div className="w-1.5 h-1.5 rounded-full bg-roadguard-indigo animate-pulse shadow-[0_0_8px_rgba(129,140,248,1)]" />
                                                FRAME_CAPTURE // {incident.id}
                                            </div>
                                            <div className="px-5 py-2.5 glass-card bg-black/40 rounded-xl text-[10px] font-mono font-black text-white/40">
                                                CAM_SOURCE // SECTOR_{incident.camera_id}
                                            </div>
                                        </div>
                                        <div className="p-3 glass-card bg-black/40 rounded-xl text-white/20">
                                            <Maximize2 size={16} />
                                        </div>
                                    </div>

                                    <div className="flex justify-between items-end">
                                        <div className="flex flex-col gap-2">
                                            <div className="text-[10px] font-black uppercase tracking-[0.4em] text-white/20">Scene Metadata</div>
                                            <div className="text-lg font-mono font-black text-white italic tracking-tighter">
                                                LAT: 51.5074° N | LON: 0.1278° W
                                            </div>
                                        </div>
                                        <div className="text-[8px] font-black uppercase tracking-[0.5em] text-white/10">Neural Vision Array v3.2</div>
                                    </div>
                                </div>
                            </div>
                        </motion.div>

                        {/* Identification & Confidence Bento */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <motion.div 
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="glass-card p-10 rounded-[3rem] relative overflow-hidden group"
                            >
                                <div className="absolute -top-10 -right-10 w-40 h-40 bg-roadguard-indigo/5 blur-[60px] group-hover:bg-roadguard-indigo/10 transition-colors" />
                                <div className="flex items-center gap-8 relative z-10">
                                    <div className="w-20 h-20 rounded-[2rem] glass-card flex items-center justify-center text-roadguard-indigo group-hover:scale-110 transition-transform duration-500">
                                        <Fingerprint size={40} />
                                    </div>
                                    <div className="flex-grow flex items-center justify-between">
                                        <div>
                                            <div className="text-[10px] font-black uppercase tracking-[0.4em] text-white/20 mb-3 italic">Autonomous Identification</div>
                                            {incident.license_plate ? (
                                                <div className="text-5xl font-mono font-black tracking-[0.2em] text-white tabular-nums">{incident.license_plate}</div>
                                            ) : (
                                                <div className="flex flex-col gap-1">
                                                    <div className="text-2xl font-black tracking-tighter text-alert-red flex items-center gap-2 animate-pulse">
                                                        <AlertTriangle size={24} /> 
                                                        PLATE_NOT_DETECTED
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                        {evidenceUrl && incident.license_plate && (
                                            <div className="w-32 h-16 rounded-2xl overflow-hidden border border-white/10 bg-slate-900/50 shadow-2xl">
                                                <img src={evidenceUrl} alt="Plate Crop" className="w-full h-full object-cover scale-150" />
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </motion.div>

                            <motion.div 
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.1 }}
                                className="glass-card p-10 rounded-[3rem] flex items-center justify-between group overflow-hidden"
                            >
                                <div className="absolute -top-10 -right-10 w-40 h-40 bg-roadguard-blue/5 blur-[60px] group-hover:bg-roadguard-blue/10 transition-colors" />
                                <div className="flex flex-col gap-3 relative z-10">
                                    <div className="text-[10px] font-black uppercase tracking-[0.4em] text-white/20 italic">Statistical Reliability</div>
                                    <div className="text-6xl font-black tracking-tighter text-roadguard-blue tabular-nums leading-none">
                                        {Math.round(incident.confidence * 100)}<span className="text-xl align-top ml-1 text-white/10">%</span>
                                    </div>
                                </div>
                                <div className="relative z-10 opacity-10 group-hover:opacity-30 group-hover:scale-110 transition-all duration-700">
                                    <Activity size={80} className="text-roadguard-blue" />
                                </div>
                            </motion.div>
                        </div>
                    </div>

                    {/* Dossier Sidebar (4 cols) */}
                    <div className="col-span-12 xl:col-span-4 space-y-10">
                        <motion.div 
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="glass-card p-12 rounded-[3.5rem] h-full relative overflow-hidden flex flex-col group"
                        >
                            <div className="absolute top-0 right-0 p-12 opacity-[0.02] pointer-events-none italic font-black text-[15rem] leading-none select-none">
                                {incident.id}
                            </div>
                            
                            <div className="relative z-10 flex flex-col h-full">
                                <div className="flex items-center gap-4 mb-12">
                                    <span className={`px-4 py-2 rounded-xl text-[9px] font-black uppercase tracking-widest ${
                                        incident.incident_type === 'accident' ? 'bg-alert-red text-white shadow-[0_10px_30px_-5px_rgba(239,68,68,0.4)]' : 'bg-premium-gold text-black shadow-[0_10px_30px_-5px_rgba(251,191,36,0.4)]'
                                    }`}>
                                        {incident.incident_type.replace('_', ' ')}
                                    </span>
                                    <div className="w-1.5 h-1.5 rounded-full bg-white/10" />
                                    <span className="text-[10px] font-black uppercase tracking-[0.3em] text-white/30">DOSSIER_RG_{incident.id}</span>
                                </div>

                                <h2 className="text-5xl font-black tracking-tighter mb-16 leading-[0.9]">INCIDENT <br/><span className="text-white/10 italic">MANIFEST</span></h2>

                                <div className="space-y-10 mb-20">
                                    {[
                                        { label: "Temporal Stamp", val: new Date(incident.timestamp).toLocaleString(), icon: Clock, color: "text-roadguard-indigo" },
                                        { label: "Deployment Sector", val: `Grid Sector ${incident.camera_id} // HUB-14`, icon: MapPin, color: "text-roadguard-blue" },
                                        { label: "Analysis Engine", val: "Neural Ensemble-X // Turbopack", icon: Shield, color: "text-premium-gold" },
                                        { label: "Capture Device", val: "HEVC-4K High-Dynamic CCTV", icon: Camera, color: "text-white/40" },
                                    ].map(item => (
                                        <div key={item.label} className="flex items-start gap-6 group/item">
                                            <div className={`mt-1 p-3 glass-card rounded-xl ${item.color} group-hover/item:scale-110 transition-transform duration-500`}>
                                                <item.icon size={16} />
                                            </div>
                                            <div>
                                                <div className="text-[10px] font-black uppercase tracking-[0.3em] text-white/20 mb-1.5 italic">{item.label}</div>
                                                <div className="text-sm font-bold tracking-tight text-white/70 group-hover/item:text-white transition-colors">{item.val}</div>
                                            </div>
                                        </div>
                                    ))}
                                </div>

                                {/* Dynamic Status Control Block */}
                                <div className="p-10 glass-card bg-surface-dark/40 rounded-[3rem] mb-12 border-white/[0.05]">
                                    <div className="text-[10px] font-black uppercase tracking-[0.4em] text-roadguard-indigo mb-10 flex items-center gap-3">
                                        <Zap size={14} className="animate-pulse" /> Classification State
                                    </div>
                                    <div className="flex flex-col gap-4">
                                        {[
                                            { label: 'Pending', icon: Radio },
                                            { label: 'Investigating', icon: Activity },
                                            { label: 'Resolved', icon: CheckCircle },
                                        ].map((status) => (
                                            <button
                                                key={status.label}
                                                disabled={updating}
                                                onClick={() => updateStatus(status.label.toLowerCase())}
                                                className={`flex items-center justify-between px-8 py-5 rounded-[1.5rem] border transition-all duration-700 font-black text-[10px] uppercase tracking-[0.3em] group/btn ${
                                                    incident.status === status.label.toLowerCase()
                                                    ? 'bg-roadguard-indigo border-roadguard-indigo text-white shadow-[0_15px_30px_-10px_rgba(99,102,241,0.4)]'
                                                    : 'bg-white/[0.02] border-white/5 text-white/30 hover:bg-white/[0.05] hover:text-white'
                                                }`}
                                            >
                                                <span className="flex items-center gap-4">
                                                    <status.icon size={16} className={incident.status === status.label.toLowerCase() ? 'animate-pulse' : ''} />
                                                    {status.label}
                                                </span>
                                                {incident.status === status.label.toLowerCase() && <motion.div layoutId="status-check"><CheckCircle size={16} /></motion.div>}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Personnel Attachment */}
                                {incident.owner && (
                                    <motion.div 
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        className="mt-auto pt-10 border-t border-white/[0.05]"
                                    >
                                        <div className="flex gap-4 mb-8">
                                            <button 
                                                onClick={() => evidenceUrl && window.open(evidenceUrl)}
                                                className="flex-1 py-4 glass-card rounded-2xl flex items-center justify-center gap-3 text-[10px] font-black uppercase tracking-widest hover:bg-white hover:text-black transition-all"
                                            >
                                                <Download size={14} /> Download Evidence
                                            </button>
                                            <button 
                                                onClick={async () => {
                                                    if (confirm("Purge this record?")) {
                                                        await fetch(`${API_BASE}/incidents/${id}`, { method: 'DELETE' })
                                                        router.push('/incidents')
                                                    }
                                                }}
                                                className="p-4 glass-card rounded-2xl flex items-center justify-center text-alert-red hover:bg-alert-red hover:text-white transition-all"
                                            >
                                                <Trash2 size={18} />
                                            </button>
                                        </div>
                                        <div className="text-[10px] font-black uppercase tracking-[0.4em] text-white/20 mb-8 italic text-center">Linked Personnel Profile</div>
                                        <div className="flex items-center gap-6 glass-card p-6 rounded-3xl">
                                            <div className="w-16 h-16 rounded-2xl glass-card flex items-center justify-center text-white/20 group-hover:text-roadguard-indigo transition-colors">
                                                <User size={28} />
                                            </div>
                                            <div>
                                                <div className="text-lg font-black tracking-tighter text-white/90">{incident.owner.name}</div>
                                                <div className="text-[10px] font-black uppercase tracking-widest text-roadguard-indigo/50">Subject ID // {incident.owner.id}</div>
                                                <div className="text-[10px] font-medium text-white/20 tracking-wider mt-1">{incident.owner.vehicle}</div>
                                            </div>
                                        </div>
                                    </motion.div>
                                )}
                            </div>
                        </motion.div>
                    </div>
                </div>
            </main>
        </div>
    )
}
