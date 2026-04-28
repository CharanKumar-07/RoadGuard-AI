"use client";

import { useEffect, useRef, useState } from "react";
import { useScroll, useTransform, useMotionValueEvent } from "framer-motion";

interface ImageSequenceCanvasProps {
    frameCount: number;
}

export default function ImageSequenceCanvas({ frameCount }: ImageSequenceCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [images, setImages] = useState<HTMLImageElement[]>([]);
    const [isLoaded, setIsLoaded] = useState(false);
    const { scrollYProgress } = useScroll();
    
    // Map scroll progress to frame index
    const frameIndex = useTransform(scrollYProgress, [0, 1], [0, frameCount - 1]);

    // 1. Preload Images
    useEffect(() => {
        const loadedImages: HTMLImageElement[] = [];
        let loadedCount = 0;

        for (let i = 0; i < frameCount; i++) {
            const img = new Image();
            const pad = (i + 1).toString().padStart(3, "0");
            img.src = `/sequence/ezgif-frame-${pad}.jpg`;
            img.onload = () => {
                loadedCount++;
                if (loadedCount === frameCount) {
                    setIsLoaded(true);
                }
            };
            loadedImages.push(img);
        }
        setImages(loadedImages);
    }, [frameCount]);

    // 2. Render Function
    const renderFrame = (index: number) => {
        const canvas = canvasRef.current;
        if (!canvas || !images[index] || !images[index].complete) return;
        
        const ctx = canvas.getContext("2d", { alpha: false });
        if (!ctx) return;

        const img = images[index];
        
        // Cover scaling logic
        const canvasAspect = canvas.width / canvas.height;
        const imgAspect = img.width / img.height;
        
        let drawW, drawH, drawX, drawY;
        
        if (canvasAspect > imgAspect) {
            drawW = canvas.width;
            drawH = canvas.width / imgAspect;
            drawX = 0;
            drawY = (canvas.height - drawH) / 2;
        } else {
            drawH = canvas.height;
            drawW = canvas.height * imgAspect;
            drawY = 0;
            drawX = (canvas.width - drawW) / 2;
        }

        ctx.drawImage(img, drawX, drawY, drawW, drawH);
    };

    // 3. Handle Resize
    useEffect(() => {
        const handleResize = () => {
            if (canvasRef.current) {
                canvasRef.current.width = window.innerWidth;
                canvasRef.current.height = window.innerHeight;
                renderFrame(Math.round(frameIndex.get()));
            }
        };

        window.addEventListener("resize", handleResize);
        handleResize();
        return () => window.removeEventListener("resize", handleResize);
    }, [isLoaded]);

    // 4. Listen to scroll changes (v12 recommended way)
    useMotionValueEvent(frameIndex, "change", (latest) => {
        renderFrame(Math.round(latest));
    });

    // 5. Initial render when loaded
    useEffect(() => {
        if (isLoaded) {
            renderFrame(0);
        }
    }, [isLoaded]);

    return (
        <div className="fixed inset-0 w-full h-full z-0 pointer-events-none">
            <canvas ref={canvasRef} className="w-full h-full object-cover" />
            {/* Absolute fallback overlay to blend the canvas edges fully, just in case */}
            <div className="absolute inset-0 bg-[#050505]/20 pointer-events-none" />
        </div>
    );
}
