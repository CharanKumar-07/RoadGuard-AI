"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { Activity, AlertTriangle, Shield, Car, Target, Cpu, Database, Zap, Radio, TrendingUp, BarChart3 } from "lucide-react"
import GlassNav from "@/components/GlassNav"

const API_BASE = "http://localhost:8000"

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        return (
            <div className="glass-card p-6 rounded-2xl shadow-2xl border-white/20 bg-surface-dark/90 premium-blur">
                <p className="text-[10px] font-black text-white/30 uppercase tracking-[0.2em] mb-2">{label}</p>
                <p className="text-xl font-black text-roadguard-cyan tracking-tighter">
                    {payload[0].value} <span className="text-[10px] text-white/20 uppercase tracking-widest ml-1">Incidents</span>
                </p>
            </div>
        )
    }
    return null
}

export default function AnalyticsPage() {
    const [stats, setStats] = useState<any>(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const res = await fetch(`${API_BASE}/incidents/stats/summary`)
                const data = await res.json()
                setStats(data)
            } catch (error) {
                console.error("Failed to fetch stats:", error)
            } finally {
                setLoading(false)
            }
        }
        fetchStats()
    }, [])

    if (loading) {
        return (
            <div className="min-h-screen bg-background flex flex-col items-center justify-center space-y-8">
                <div className="w-20 h-20 border-2 border-white/5 border-t-roadguard-cyan rounded-full animate-spin shadow-[0_0_40px_rgba(6,182,212,0.2)]" />
                <p className="text-white/20 font-black uppercase tracking-[0.5em] text-[10px]">Analyzing Sector Data...</p>
            </div>
        )
    }

    const byType = stats?.by_type || {}
    const byStatus = stats?.by_status || {}
    
    const pieData = Object.entries(byType).map(([name, value]) => ({ 
        name: String(name).toUpperCase().replace('_', ' '), 
        value: Number(value) 
    }))
    const COLORS = ['#3b82f6', '#ef4444', '#fbbf24', '#06b6d4', '#8b5cf6']

    const dailyData = stats?.daily || []

    return (
        <div className="min-h-screen bg-background mesh-gradient text-white/90 selection:bg-roadguard-cyan/30 overflow-x-hidden">
            <GlassNav />
            
            <main className="max-w-[1700px] mx-auto px-6 pt-36 pb-20">
                
                {/* Premium Tactical Header */}
                <div className="mb-20">
                    <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-12 border-b border-white/[0.05] pb-16">
                        <div className="max-w-2xl">
                            <motion.div 
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                className="flex items-center gap-4 mb-6"
                            >
                                <Radio size={16} className="text-roadguard-cyan animate-pulse" />
                                <span className="text-[10px] font-black uppercase tracking-[0.5em] text-roadguard-cyan">Sector Intelligence Diagnostic</span>
                            </motion.div>
                            <motion.h1 
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="text-6xl md:text-8xl font-black tracking-tighter leading-[0.85] mb-8"
                            >
                                MISSION <span className="text-white/10 italic">OVERVIEW</span>
                            </motion.h1>
                            <p className="text-white/40 text-xl font-medium leading-relaxed max-w-lg">
                                Comprehensive analytical breakdown of autonomous surveillance operations and neural detection throughput.
                            </p>
                        </div>

                        <div className="flex flex-wrap gap-4 lg:justify-end">
                            {[
                                { label: "System Health", val: "NOMINAL", icon: Zap, color: "text-green-400" },
                                { label: "Neural Load", val: "42.8%", icon: Cpu, color: "text-roadguard-cyan" },
                                { label: "Data Integrity", val: "99.9%", icon: Shield, color: "text-roadguard-blue" }
                            ].map(s => (
                                <div key={s.label} className="px-6 py-4 glass-card rounded-2xl flex items-center gap-4 group hover:border-white/20 transition-all duration-500">
                                    <s.icon size={16} className={`${s.color} group-hover:scale-110 transition-transform`} />
                                    <div>
                                        <div className="text-[8px] font-black uppercase tracking-widest text-white/30">{s.label}</div>
                                        <div className="text-xs font-black tracking-widest text-white/80">{s.val}</div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Tactical Stats Bento Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-20">
                    {[
                        { label: "Total Intercepts", value: stats?.total || 0, icon: Target, color: "text-roadguard-cyan", bg: "bg-roadguard-cyan/5" },
                        { label: "Critical Anomalies", value: byType.accident || 0, icon: AlertTriangle, color: "text-alert-red", bg: "bg-alert-red/5" },
                        { label: "High-Priority Detections", value: byType.hit_and_run || 0, icon: Car, color: "text-premium-gold", bg: "bg-premium-gold/5" },
                        { label: "Pending Cases", value: byStatus.pending || 0, icon: TrendingUp, color: "text-roadguard-blue", bg: "bg-roadguard-blue/5" },
                    ].map((stat, i) => (
                        <motion.div
                            key={stat.label}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.1 }}
                            className="glass-card p-10 rounded-[3rem] group glass-card-hover relative overflow-hidden"
                        >
                            <div className={`absolute -top-10 -right-10 w-40 h-40 ${stat.bg} blur-[60px] group-hover:blur-[80px] transition-all duration-700`} />
                            
                            <div className="relative z-10">
                                <div className="text-[10px] font-black uppercase tracking-[0.3em] text-white/20 mb-12 group-hover:text-white/40 transition-colors flex items-center gap-3">
                                    <stat.icon size={12} className={stat.color} />
                                    {stat.label}
                                </div>
                                <div className="text-7xl font-black mb-3 tracking-tighter tabular-nums leading-none">{stat.value}</div>
                                <div className="text-[9px] font-bold text-white/20 tracking-[0.2em] uppercase flex items-center gap-2">
                                    <div className="w-1.5 h-1.5 rounded-full bg-roadguard-cyan/20 group-hover:bg-roadguard-cyan transition-colors" />
                                    Active Stream Sync
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>

                <div className="grid grid-cols-12 gap-8">
                    {/* Primary Area Chart */}
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="col-span-12 lg:col-span-8 glass-card p-12 rounded-[3.5rem] relative group"
                    >
                        <div className="flex items-center justify-between mb-16">
                            <div>
                                <h3 className="text-3xl font-black tracking-tighter flex items-center gap-5 mb-2">
                                    <Activity size={28} className="text-roadguard-cyan" />
                                    Temporal Flux
                                </h3>
                                <p className="text-[10px] font-black uppercase tracking-[0.4em] text-white/20">Anomaly detection trend over 30 cycles</p>
                            </div>
                            <div className="px-6 py-3 glass-card rounded-2xl text-[9px] font-black uppercase tracking-[0.3em] text-white/30 group-hover:border-roadguard-cyan/30 transition-all">
                                LOG_INTERVAL: 24H
                            </div>
                        </div>
                        <div className="h-[500px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={dailyData}>
                                    <defs>
                                        <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                                            <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="4 4" stroke="#ffffff05" vertical={false} />
                                    <XAxis 
                                        dataKey="date" 
                                        stroke="#ffffff05" 
                                        fontSize={10} 
                                        tickLine={false} 
                                        axisLine={false}
                                        tick={{ fill: '#ffffff20', fontWeight: 800 }}
                                        dy={20}
                                    />
                                    <YAxis 
                                        stroke="#ffffff05" 
                                        fontSize={10} 
                                        tickLine={false} 
                                        axisLine={false}
                                        tick={{ fill: '#ffffff20', fontWeight: 800 }}
                                        dx={-20}
                                    />
                                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(6, 182, 212, 0.2)', strokeWidth: 2 }} />
                                    <Area 
                                        type="monotone" 
                                        dataKey="count" 
                                        stroke="#06b6d4" 
                                        strokeWidth={3}
                                        fillOpacity={1} 
                                        fill="url(#colorCount)" 
                                        animationDuration={2000}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </motion.div>

                    {/* Composition Pie Chart */}
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.2 }}
                        className="col-span-12 lg:col-span-4 glass-card p-12 rounded-[3.5rem] flex flex-col group"
                    >
                        <h3 className="text-3xl font-black tracking-tighter mb-12 flex items-center gap-4">
                            <BarChart3 size={24} className="text-roadguard-blue" />
                            Spectral Logic
                        </h3>
                        <div className="h-[380px] w-full mb-12 relative">
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={pieData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={100}
                                        outerRadius={140}
                                        paddingAngle={8}
                                        dataKey="value"
                                        stroke="none"
                                        animationBegin={200}
                                        animationDuration={1500}
                                    >
                                        {pieData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                        ))}
                                    </Pie>
                                </PieChart>
                            </ResponsiveContainer>
                            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                                <div className="text-[10px] font-black uppercase tracking-[0.4em] text-white/20">TOTAL</div>
                                <div className="text-4xl font-black tracking-tighter leading-none">{stats?.total || 0}</div>
                            </div>
                        </div>
                        <div className="space-y-6 flex-grow">
                            {pieData.map((item, i) => (
                                <div key={item.name} className="flex items-center justify-between group/item">
                                    <div className="flex items-center gap-5">
                                        <div className="w-1.5 h-1.5 rounded-full shadow-[0_0_8px_currentColor]" style={{ color: COLORS[i % COLORS.length], backgroundColor: 'currentColor' }}></div>
                                        <span className="text-white/30 font-black text-[10px] uppercase tracking-[0.3em] group-hover/item:text-white transition-colors">{item.name}</span>
                                    </div>
                                    <span className="font-black text-2xl tracking-tighter tabular-nums">{item.value}</span>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                </div>
            </main>
        </div>
    )
}
