"use client";

import { motion, Variants } from "framer-motion";

interface StorySectionProps {
    children: React.ReactNode;
    direction?: "up" | "left" | "right";
    delay?: number;
}

export function StorySection({ children, direction = "up", delay = 0 }: StorySectionProps) {

    const variants: Variants = {
        hidden: { 
            opacity: 0, 
            y: direction === "up" ? 50 : 0,
            x: direction === "left" ? -50 : (direction === "right" ? 50 : 0)
        },
        visible: { 
            opacity: 1, 
            y: 0, 
            x: 0,
            transition: { 
                duration: 0.8, 
                ease: "easeOut", 
                delay 
            } 
        }
    };

    return (
        <motion.div
            variants={variants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: false, margin: "-20% 0px -20% 0px" }}
            className="w-full"
        >
            {children}
        </motion.div>
    );
}
