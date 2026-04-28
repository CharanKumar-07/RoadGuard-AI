"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import Link from "next/link";

export default function GlassNav() {
    const { scrollY } = useScroll();

    // Fade in the navbar background after zooming past the hero section
    const bgOpacity = useTransform(scrollY, [0, 200], [0, 0.75]);
    const backdropBlur = useTransform(scrollY, [0, 200], ["blur(0px)", "blur(12px)"]);
    const borderOpacity = useTransform(scrollY, [0, 200], [0, 0.1]);

    return (
        <motion.nav
            className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 py-4"
            style={{
                backgroundColor: useTransform(bgOpacity, (v) => `rgba(5, 5, 5, ${v})`),
                backdropFilter: backdropBlur,
                borderBottom: useTransform(borderOpacity, (v) => `1px solid rgba(255, 255, 255, ${v})`),
            }}
        >
            <div className="flex items-center gap-2 text-white/90 font-medium tracking-wide">
                <span className="text-xl font-bold tracking-tight">RoadGuard AI</span>
            </div>

            <div className="hidden md:flex items-center gap-8 text-[10px] font-black uppercase tracking-[0.2em] text-white/40">
                <Link href="/" className="hover:text-white transition-colors duration-200">Home</Link>
                <Link href="/live" className="hover:text-[#00D6FF] active:text-[#00D6FF] transition-colors duration-200">Live Feed</Link>
                <Link href="/incidents" className="hover:text-white transition-colors duration-200">Registry</Link>
                <Link href="/analytics" className="hover:text-white transition-colors duration-200">Analytics</Link>
                <a href="#" className="hover:text-white transition-colors duration-200">Contact</a>
            </div>

            <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="px-5 py-2 text-sm font-medium text-white bg-white/5 rounded-full border border-white/10 hover:border-[#00D6FF]/50 hover:bg-[#00D6FF]/10 transition-all duration-300 shadow-[0_0_15px_rgba(0,214,255,0)] hover:shadow-[0_0_15px_rgba(0,214,255,0.2)]"
            >
                Request Demo
            </motion.button>
        </motion.nav>
    );
}
