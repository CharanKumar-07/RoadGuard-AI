"use client"

import { useEffect, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Search, MapPin, Clock, Shield, ChevronRight, Eye, Filter, FileText, Activity, Database, AlertCircle, Trash2, Download, Image as ImageIcon, RefreshCw, MoreHorizontal, ShieldAlert } from "lucide-react"
import GlassNav from "@/components/GlassNav"
import Link from "next/link"

const API_BASE = "http://localhost:8000"

export default function IncidentsPage() {
    const [incidents, setIncidents] = useState<any[]>([])
    const [loading, setLoading] = useState(true)
    const [searchTerm, setSearchTerm] = useState("")
    const [filterType, setFilterType] = useState("All")
    const [startDate, setStartDate] = useState("")
    const [endDate, setEndDate] = useState("")
    const [minCase, setMinCase] = useState("")
    const [maxCase, setMaxCase] = useState("")
    const [showFilters, setShowFilters] = useState(false)
    const [activeMenu, setActiveMenu] = useState<number | null>(null)
    const [purgingCaseId, setPurgingCaseId] = useState<number | null>(null)
    const [appliedFilters, setAppliedFilters] = useState({
        type: "All",
        start: "",
        end: "",
        min: "",
        max: ""
    })

    const fetchIncidents = async () => {
        try {
            const res = await fetch(`${API_BASE}/incidents/?limit=100`)
            const data = await res.json()
            setIncidents(data)
        } catch (error) {
            console.error("Failed to fetch incidents:", error)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchIncidents()
    }, [])

    const executePurge = async () => {
        if (!purgingCaseId) return
        
        try {
            const res = await fetch(`${API_BASE}/incidents/${purgingCaseId}`, { method: "DELETE" })
            if (res.ok) {
                setIncidents(prev => prev.filter(inc => inc.id !== purgingCaseId))
                setPurgingCaseId(null)
            }
        } catch (error) {
            console.error("Purge failed:", error)
        }
    }

    const deleteIncident = (e: React.MouseEvent, id: number) => {
        e.preventDefault()
        e.stopPropagation()
        setPurgingCaseId(id)
    }

    const downloadEvidence = (e: React.MouseEvent, url: string, filename: string) => {
        e.preventDefault()
        e.stopPropagation()
        const link = document.createElement("a")
        link.href = url
        link.download = filename
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
    }

    const filtered = incidents.filter(inc => {
        const matchesSearch = (
            inc.license_plate?.toLowerCase().includes(searchTerm.toLowerCase()) || 
            inc.incident_type?.toLowerCase().includes(searchTerm.toLowerCase()) ||
            inc.id.toString().includes(searchTerm)
        )
        const matchesType = appliedFilters.type === "All" || inc.incident_type === appliedFilters.type.toLowerCase()
        
        const incDate = new Date(inc.timestamp)
        const matchesStart = !appliedFilters.start || incDate >= new Date(appliedFilters.start)
        const matchesEnd = !appliedFilters.end || incDate <= new Date(appliedFilters.end + "T23:59:59")
        
        const caseId = parseInt(inc.id)
        const matchesMin = !appliedFilters.min || caseId >= parseInt(appliedFilters.min)
        const matchesMax = !appliedFilters.max || caseId <= parseInt(appliedFilters.max)

        return matchesSearch && matchesType && matchesStart && matchesEnd && matchesMin && matchesMax
    })

    const applyFilters = () => {
        setAppliedFilters({
            type: filterType,
            start: startDate,
            end: endDate,
            min: minCase,
            max: maxCase
        })
        setShowFilters(false)
    }

    const resetFilters = () => {
        setSearchTerm("")
        setFilterType("All")
        setStartDate("")
        setEndDate("")
        setMinCase("")
        setMaxCase("")
        setAppliedFilters({
            type: "All",
            start: "",
            end: "",
            min: "",
            max: ""
        })
    }

    return (
        <div className="min-h-screen bg-background text-white/90 selection:bg-roadguard-indigo/30">
            <GlassNav />
            
            <main className="max-w-[1500px] mx-auto px-6 pt-36 pb-20">
                
                {/* Premium Tactical Header */}
                <div className="mb-20">
                    <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-12 border-b border-white/[0.05] pb-16">
                        <div className="max-w-2xl">
                            <motion.div 
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                className="flex items-center gap-4 mb-6"
                            >
                                <div className="w-10 h-[1px] bg-roadguard-indigo/50" />
                                <span className="text-[10px] font-black uppercase tracking-[0.5em] text-roadguard-indigo">Global Incident Intelligence</span>
                            </motion.div>
                            <motion.h1 
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tighter mb-8 leading-none whitespace-nowrap"
                            >
                                <span className="text-emerald-500">DATA</span> <span className="text-white">ARCHIVE</span>
                            </motion.h1>
                            <p className="text-white/40 text-xl font-medium leading-relaxed max-w-lg">
                                Accessing high-fidelity records of autonomous detections and historical safety interventions.
                            </p>
                        </div>

                        <div className="flex items-center gap-4 relative">
                            <div className="relative group">
                                <Search className="absolute left-6 top-1/2 -translate-y-1/2 text-white/20 group-focus-within:text-roadguard-indigo transition-colors" size={20} />
                                <input 
                                    type="text"
                                    placeholder="SEARCH ARCHIVE..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    className="pl-16 pr-8 py-5 glass-card bg-surface-dark/40 rounded-2xl w-full xl:w-[400px] outline-none focus:border-roadguard-indigo/30 focus:bg-surface-dark/60 transition-all font-mono text-lg tracking-widest uppercase"
                                />
                            </div>

                            <div className="flex items-center gap-3">
                                <button 
                                    onClick={() => setShowFilters(!showFilters)}
                                    className={`p-5 rounded-2xl transition-all flex items-center gap-3 font-black text-[10px] uppercase tracking-widest ${showFilters ? 'bg-roadguard-indigo text-white shadow-[0_0_20px_rgba(99,102,241,0.3)]' : 'glass-card text-white/40 hover:text-white'}`}
                                >
                                    <Filter size={18} />
                                    <span>Filters</span>
                                </button>
                                
                                <button 
                                    onClick={resetFilters}
                                    className="p-5 glass-card bg-white/[0.03] hover:bg-white/[0.1] rounded-2xl transition-all group"
                                    title="Reset All"
                                >
                                    <RefreshCw size={18} className="text-white/20 group-hover:text-white transition-all group-hover:rotate-180 duration-700" />
                                </button>
                            </div>

                            {/* Filter Popover */}
                            <AnimatePresence>
                                {showFilters && (
                                    <motion.div 
                                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                                        animate={{ opacity: 1, y: 0, scale: 1 }}
                                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                                        className="absolute top-full right-0 mt-4 w-[450px] glass-card bg-surface-dark/95 border-white/10 p-8 rounded-[2rem] shadow-2xl z-50 premium-blur"
                                    >
                                        <div className="space-y-8">
                                            {/* Section: Type */}
                                            <div>
                                                <label className="text-[10px] font-black uppercase tracking-[0.2em] text-white/20 mb-4 block">Incident Category</label>
                                                <div className="grid grid-cols-3 gap-3">
                                                    {["All", "Accident", "Hit_and_run"].map((t) => (
                                                        <button 
                                                            key={t}
                                                            onClick={() => setFilterType(t)}
                                                            className={`py-3 rounded-xl text-[10px] font-black uppercase tracking-widest border transition-all ${filterType === t ? 'bg-roadguard-indigo/20 border-roadguard-indigo text-roadguard-indigo' : 'bg-white/[0.02] border-white/5 text-white/40 hover:border-white/10'}`}
                                                        >
                                                            {t.replace('_', ' ')}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>

                                            {/* Section: Date Range */}
                                            <div>
                                                <label className="text-[10px] font-black uppercase tracking-[0.2em] text-white/20 mb-4 block">Temporal Range</label>
                                                <div className="flex items-center gap-3 bg-black/20 p-3 rounded-2xl border border-white/5">
                                                    <input 
                                                        type="date"
                                                        value={startDate}
                                                        onChange={(e) => setStartDate(e.target.value)}
                                                        className="bg-transparent outline-none text-xs font-bold text-white uppercase flex-1"
                                                    />
                                                    <span className="text-white/10">→</span>
                                                    <input 
                                                        type="date"
                                                        value={endDate}
                                                        onChange={(e) => setEndDate(e.target.value)}
                                                        className="bg-transparent outline-none text-xs font-bold text-white uppercase flex-1"
                                                    />
                                                </div>
                                            </div>

                                            {/* Section: Case ID Range */}
                                            <div>
                                                <label className="text-[10px] font-black uppercase tracking-[0.2em] text-white/20 mb-4 block">Numeric Case Range</label>
                                                <div className="flex items-center gap-4">
                                                    <input 
                                                        type="number"
                                                        placeholder="MIN ID"
                                                        value={minCase}
                                                        onChange={(e) => setMinCase(e.target.value)}
                                                        className="bg-black/20 p-4 rounded-xl border border-white/5 outline-none text-sm font-mono text-roadguard-indigo w-full focus:border-roadguard-indigo/50"
                                                    />
                                                    <input 
                                                        type="number"
                                                        placeholder="MAX ID"
                                                        value={maxCase}
                                                        onChange={(e) => setMaxCase(e.target.value)}
                                                        className="bg-black/20 p-4 rounded-xl border border-white/5 outline-none text-sm font-mono text-roadguard-indigo w-full focus:border-roadguard-indigo/50"
                                                    />
                                                </div>
                                            </div>

                                            <button 
                                                onClick={applyFilters}
                                                className="w-full py-5 bg-roadguard-indigo text-white rounded-2xl font-black text-xs uppercase tracking-[0.3em] hover:bg-roadguard-indigo/80 transition-all shadow-[0_10px_30px_rgba(99,102,241,0.3)]"
                                            >
                                                OK
                                            </button>
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </div>
                </div>

                {loading ? (
                    <div className="py-40 text-center flex flex-col items-center">
                        <div className="w-16 h-16 border-2 border-white/5 border-t-roadguard-indigo rounded-full animate-spin mb-8 shadow-[0_0_20px_rgba(99,102,241,0.2)]" />
                        <p className="text-white/20 font-black uppercase tracking-[0.5em] text-[10px]">Reconstructing Archive Packets...</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                        <AnimatePresence mode="popLayout">
                            {filtered.map((inc, i) => (
                                <motion.div
                                    key={inc.id}
                                    initial={{ opacity: 0, y: 30 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: i * 0.05, type: "spring", stiffness: 100 }}
                                    className="group"
                                >
                                    <Link href={`/incidents/${inc.id}`} className="block">
                                        <div className="glass-card p-8 rounded-[2.5rem] transition-all duration-700 glass-card-hover relative overflow-hidden group-hover:-translate-y-2">
                                            {/* Accent Light */}
                                            <div className={`absolute -top-24 -right-24 w-48 h-48 blur-[80px] opacity-10 transition-colors duration-700 ${inc.incident_type === 'accident' ? 'bg-alert-red' : 'bg-roadguard-indigo'}`} />
                                            
                                            {/* Status Header */}
                                            <div className="flex justify-between items-start mb-10">
                                                <div className="flex flex-col gap-1.5">
                                                    <div className="flex items-center gap-2 mb-1">
                                                        <span className="text-[10px] font-black uppercase tracking-[0.3em] text-white/20">Case: {inc.id}</span>
                                                    </div>
                                                    <h3 className={`text-2xl font-black tracking-tighter ${inc.incident_type === 'accident' ? 'text-alert-red' : 'text-roadguard-indigo'}`}>
                                                        {inc.incident_type.toUpperCase().replace('_', ' ')}
                                                    </h3>
                                                </div>
                                                <div className="w-12 h-12 glass-card rounded-2xl flex items-center justify-center text-white/20 group-hover:bg-white group-hover:text-black transition-all duration-500">
                                                    <Eye size={20} />
                                                </div>
                                            </div>

                                            {/* Plate Section */}
                                            <div className="mb-8 p-6 glass-card bg-black/20 rounded-3xl group-hover:border-white/20 transition-all duration-700 flex items-center justify-between">
                                                <div>
                                                    <div className="text-[9px] font-black uppercase tracking-widest text-white/20 mb-3">Neural Identification</div>
                                                    <div className="text-3xl font-mono font-black text-white tracking-[0.2em]">
                                                        {inc.license_plate || (inc.status === 'pending' ? "SCANNING" : "UNIDENTIFIED")}
                                                    </div>
                                                </div>
                                                {inc.evidence_image && (
                                                    <div className="w-24 h-12 rounded-xl overflow-hidden border border-white/5 bg-slate-900 shadow-lg">
                                                        <img 
                                                            src={`${API_BASE}/evidence/${inc.evidence_image}`} 
                                                            alt="Plate Crop"
                                                            className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity"
                                                        />
                                                    </div>
                                                )}
                                            </div>

                                            {/* Metadata Info */}
                                            <div className="flex items-center justify-between gap-4 mb-8">
                                                <div className="flex items-center gap-2.5 text-white/40">
                                                    <div className="w-8 h-8 rounded-xl glass-card flex items-center justify-center">
                                                        <Clock size={12} className="text-white/30" />
                                                    </div>
                                                    <span className="text-[11px] font-bold tracking-tight">
                                                        {new Date(inc.timestamp).toLocaleDateString()} — {new Date(inc.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-2.5 text-white/40">
                                                    <div className="w-8 h-8 rounded-xl glass-card flex items-center justify-center">
                                                        <MapPin size={12} className="text-white/30" />
                                                    </div>
                                                    <span className="text-[11px] font-bold tracking-tight">{inc.camera_id}</span>
                                                </div>
                                            </div>

                                            {/* Footer Status */}
                                            <div className="pt-8 border-t border-white/[0.05] flex items-center justify-between">
                                                <div className="flex items-center gap-3">
                                                    <div className={`w-1.5 h-1.5 rounded-full ${inc.status === 'Resolved' ? 'bg-green-500' : 'bg-premium-gold'} shadow-[0_0_8px_currentColor]`} />
                                                    <span className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40">{inc.status}</span>
                                                </div>
                                                <div className="flex items-center gap-2 relative">
                                                    <button 
                                                        onClick={(e) => {
                                                            e.preventDefault()
                                                            e.stopPropagation()
                                                            setActiveMenu(activeMenu === inc.id ? null : inc.id)
                                                        }}
                                                        className={`p-2.5 rounded-xl transition-all duration-300 ${activeMenu === inc.id ? 'bg-white text-black' : 'glass-card text-white/40 hover:text-white'}`}
                                                        title="Manage Record"
                                                    >
                                                        <MoreHorizontal size={18} />
                                                    </button>

                                                    <AnimatePresence>
                                                        {activeMenu === inc.id && (
                                                            <motion.div 
                                                                initial={{ opacity: 0, scale: 0.9, y: 10 }}
                                                                animate={{ opacity: 1, scale: 1, y: 0 }}
                                                                exit={{ opacity: 0, scale: 0.9, y: 10 }}
                                                                className="absolute bottom-full right-0 mb-4 w-60 bg-[#0f1117] border border-white/10 p-2.5 rounded-2xl shadow-[0_25px_60px_rgba(0,0,0,0.8)] z-50"
                                                                onClick={(e) => e.stopPropagation()}
                                                            >
                                                                <button 
                                                                    onClick={(e) => {
                                                                        downloadEvidence(e, `${API_BASE}/evidence/${inc.evidence_image}`, `case_${inc.id}.jpg`)
                                                                        setActiveMenu(null)
                                                                    }}
                                                                    className="w-full flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-white/[0.08] transition-colors text-left group/item"
                                                                >
                                                                    <Download size={14} className="text-white/20 group-hover/item:text-roadguard-indigo" />
                                                                    <span className="text-[10px] font-black uppercase tracking-widest text-white/60">Download Data</span>
                                                                </button>
                                                                <div className="h-[1px] bg-white/5 my-1" />
                                                                <button 
                                                                    onClick={(e) => {
                                                                        deleteIncident(e, inc.id)
                                                                        setActiveMenu(null)
                                                                    }}
                                                                    className="w-full flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-alert-red/10 transition-colors text-left group/item"
                                                                >
                                                                    <ShieldAlert size={14} className="text-white/20 group-hover/item:text-alert-red" />
                                                                    <span className="text-[10px] font-black uppercase tracking-widest text-alert-red opacity-80">Purge Case</span>
                                                                </button>
                                                            </motion.div>
                                                        )}
                                                    </AnimatePresence>

                                                    <ChevronRight size={16} className="text-white/20 group-hover:text-roadguard-indigo transition-transform duration-500 group-hover:translate-x-1" />
                                                </div>
                                            </div>

                                            {/* Ghost ID Background */}
                                            <div className="absolute -bottom-6 -right-6 text-[80px] font-black text-white/[0.02] pointer-events-none select-none italic tracking-tighter">
                                                {inc.id.toString().padStart(3, '0')}
                                            </div>
                                        </div>
                                    </Link>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </div>
                )}

                {/* Purge Confirmation Modal */}
                <AnimatePresence>
                    {purgingCaseId && (
                        <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
                            <motion.div 
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                onClick={() => setPurgingCaseId(null)}
                                className="absolute inset-0 bg-black/60 backdrop-blur-md"
                            />
                            <motion.div 
                                initial={{ opacity: 0, scale: 0.9, y: 20 }}
                                animate={{ opacity: 1, scale: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.9, y: 20 }}
                                className="relative w-full max-w-md glass-card bg-surface-dark border-alert-red/20 p-10 rounded-[3rem] shadow-2xl overflow-hidden"
                            >
                                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-alert-red/40 to-transparent" />
                                
                                <div className="w-16 h-16 rounded-2xl bg-alert-red/10 flex items-center justify-center text-alert-red mb-8">
                                    <ShieldAlert size={32} />
                                </div>
                                
                                <h3 className="text-2xl font-black tracking-tight mb-4 text-white">Security Intervention</h3>
                                <p className="text-white/40 text-sm leading-relaxed mb-10 font-medium">
                                    Do you confirm the permanent removal of <span className="text-white">Case #{purgingCaseId}</span> from the central registry? This action is irreversible.
                                </p>
                                
                                <div className="flex flex-col gap-3">
                                    <button 
                                        onClick={executePurge}
                                        className="w-full py-5 bg-alert-red text-white rounded-2xl font-black text-xs uppercase tracking-[0.3em] hover:bg-alert-red/80 transition-all shadow-[0_10px_30px_rgba(244,63,94,0.3)]"
                                    >
                                        Confirm Purge
                                    </button>
                                    <button 
                                        onClick={() => setPurgingCaseId(null)}
                                        className="w-full py-5 glass-card hover:bg-white/5 rounded-2xl font-black text-xs uppercase tracking-[0.3em] text-white/40 hover:text-white transition-all"
                                    >
                                        Cancel Request
                                    </button>
                                </div>
                            </motion.div>
                        </div>
                    )}
                </AnimatePresence>
                
                {filtered.length === 0 && !loading && (
                    <motion.div 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="py-40 text-center flex flex-col items-center"
                    >
                        <div className="w-24 h-24 rounded-full glass-card flex items-center justify-center mb-8 text-white/10">
                            <AlertCircle size={48} />
                        </div>
                        <h3 className="text-2xl font-black tracking-tight mb-2">Zero Matches Found</h3>
                        <p className="text-white/20 font-black uppercase tracking-[0.4em] text-[10px]">Refine your parameters or clear filters</p>
                    </motion.div>
                )}
            </main>
        </div>
    )
}
