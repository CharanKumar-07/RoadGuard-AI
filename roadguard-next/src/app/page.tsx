import Link from "next/link";
import ImageSequenceCanvas from "@/components/ImageSequenceCanvas";
import GlassNav from "@/components/GlassNav";
import { StorySection } from "@/components/StorySection";

export default function Home() {
  return (
    <main className="relative bg-[#000000] text-white selection:bg-[#00D6FF]/30">
      <GlassNav />

      {/* Fixed Background Canvas */}
      <ImageSequenceCanvas frameCount={240} />

      {/* 
        Scrollable Content Layers 
        Using ~400vh to orchestrate the 5 storytelling beats
      */}
      <div className="relative z-10 w-full">

        {/* Section 1: HERO (0-15% scroll approx) */}
        <section className="h-[100vh] flex flex-col items-center justify-center text-center px-6">
          <StorySection direction="up" delay={0.2}>
            <h1 className="text-7xl md:text-[10rem] font-black tracking-tighter mb-8 text-white drop-shadow-2xl">
              ROAD<span className="text-white/20">GUARD</span>
            </h1>
            <p className="text-xl md:text-3xl font-medium text-white/80 max-w-2xl mx-auto mb-4 drop-shadow-lg">
              Real-time intelligence for safer roads.
            </p>
            <p className="text-base text-white/60 max-w-xl mx-auto">
              Deep learning-powered accident detection, ANPR, and traffic monitoring for modern cities.
            </p>
          </StorySection>
        </section>

        {/* Section 2: ENGINEERING REVEAL (15-40%) */}
        <section className="h-[100vh] flex flex-col items-start justify-center px-8 md:px-24">
          <StorySection direction="left">
            <h2 className="text-4xl md:text-6xl font-bold tracking-tight mb-6 text-white/90 drop-shadow-xl max-w-2xl">
              Precision detection,<br />frame by frame.
            </h2>
            <div className="space-y-4 text-lg text-white/60 max-w-xl">
              <p>
                Multiple deep learning models work in parallel: YOLOv8 for objects, a dedicated ANPR pipeline, and real-time tracking.
              </p>
              <p>
                Every vehicle, person, and license plate is identified with industry-leading accuracy.
              </p>
            </div>
          </StorySection>
        </section>

        {/* Section 3: ANPR & DEBLURRING (40-65%) */}
        <section className="h-[100vh] flex flex-col items-end justify-center text-right px-8 md:px-24">
          <StorySection direction="right">
            <h2 className="text-4xl md:text-6xl font-bold tracking-tight mb-6 text-white/90 drop-shadow-xl max-w-2xl ml-auto">
              See clearly, even<br />when cameras don't.
            </h2>
            <ul className="space-y-4 text-lg text-white/60 max-w-xl ml-auto list-none">
              <li className="flex items-center justify-end gap-3">
                DeblurGAN-v2 enhances low-quality footage.
                <div className="w-1.5 h-1.5 rounded-full bg-[#00D6FF]" />
              </li>
              <li className="flex items-center justify-end gap-3">
                PaddleOCR reads plates from any angle, day or night.
                <div className="w-1.5 h-1.5 rounded-full bg-[#00D6FF]" />
              </li>
              <li className="flex items-center justify-end gap-3">
                Hit-and-run vehicles are identified and logged automatically.
                <div className="w-1.5 h-1.5 rounded-full bg-[#00D6FF]" />
              </li>
            </ul>
          </StorySection>
        </section>

        {/* Section 4: VIOLATION LOGIC (65-85%) */}
        <section className="h-[100vh] flex flex-col items-start justify-center px-8 md:px-24">
          <StorySection direction="left">
            <h2 className="text-4xl md:text-6xl font-bold tracking-tight mb-6 text-white/90 drop-shadow-xl max-w-2xl">
              Enforce traffic laws,<br /><span className="text-gradient">automatically.</span>
            </h2>
            <div className="space-y-6 text-lg text-white/60 max-w-xl">
              <div className="pl-4 border-l-2 border-[#FF3B30]/50">
                <p className="text-white/90 font-medium">Wrong-way detection</p>
                <p className="text-sm">Using advanced trajectory analysis.</p>
              </div>
              <div className="pl-4 border-l-2 border-[#FF3B30]/50">
                <p className="text-white/90 font-medium">Red-light crossing alerts</p>
                <p className="text-sm">With synchronized signal data integration.</p>
              </div>
              <div className="pl-4 border-l-2 border-[#0050FF]/50">
                <p className="text-white/90 font-medium">Speed estimation</p>
                <p className="text-sm">Calculated from standard CCTV feeds.</p>
              </div>
              <div className="pl-4 border-l-2 border-[#0050FF]/50">
                <p className="text-white/90 font-medium">Helmet compliance</p>
                <p className="text-sm">For motorcyclists and multi-passenger tracking.</p>
              </div>
            </div>
          </StorySection>
        </section>

        {/* Section 5: DASHBOARD & ALERTS (85-100%) */}
        <section className="h-[100vh] flex flex-col items-center justify-center text-center px-6 bg-gradient-to-t from-[#050505] to-transparent">
          <StorySection direction="up">
            <h2 className="text-5xl md:text-7xl font-bold tracking-tighter mb-4 text-white/90 drop-shadow-2xl">
              From detection to action,<br />in seconds.
            </h2>
            <p className="text-xl text-[#00D6FF] font-medium mb-12">
              RoadGuard AI. Deployed in cities worldwide.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-8 mb-12">
              <Link href="/live" className="px-10 py-5 text-sm font-black uppercase tracking-[0.2em] text-black bg-[#00D6FF] rounded-2xl hover:bg-white transition-all duration-500 shadow-[0_0_30px_rgba(0,214,255,0.2)]">
                Start Mission
              </Link>
              <Link href="/incidents" className="px-10 py-5 text-sm font-black uppercase tracking-[0.2em] text-white bg-white/5 rounded-2xl border border-white/10 hover:bg-white/10 transition-all duration-500">
                Access Archive
              </Link>
            </div>

            <p className="text-sm text-white/40 uppercase tracking-widest font-medium">
              Built for traffic authorities, smart cities, and highway patrol.
            </p>
          </StorySection>
        </section>

      </div>
    </main>
  );
}
