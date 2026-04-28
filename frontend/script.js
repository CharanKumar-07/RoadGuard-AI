// Register GSAP plugins
gsap.registerPlugin(ScrollTrigger);

// 2. Custom Cursor
const cursor = document.querySelector('.custom-cursor');
let mouseX = 0;
let mouseY = 0;

window.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    gsap.to(cursor, {
        x: mouseX,
        y: mouseY,
        duration: 0.1,
        ease: "power2.out"
    });
});

// Hover effects for cursor
const hoverElements = document.querySelectorAll('a, button');
hoverElements.forEach(el => {
    el.addEventListener('mouseenter', () => cursor.classList.add('hover'));
    el.addEventListener('mouseleave', () => cursor.classList.remove('hover'));
});

// 3. Canvas 3D Sequence
const canvas = document.getElementById("hero-lightpass");
const context = canvas.getContext("2d", { alpha: false });



// Using max 240 frames based on directory inspection
const frameCount = 240;
const currentFrame = index => {
    const pad = (index + 1).toString().padStart(3, '0');
    return `ezgif-4510f3d7cedf6cf6-jpg/ezgif-frame-${pad}.jpg`;
};

const images = [];
const imageSeq = { frame: 0 };

let scaledX = 0, scaledY = 0, scaledW = 0, scaledH = 0;
let isScaleComputed = false;

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    isScaleComputed = false;
    render();
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas(); // initial setup

// Preload sequence
const firstImg = new Image();
firstImg.onload = () => {
    images[0] = firstImg;
    render();
    preloadRestOfImages();
};
firstImg.src = currentFrame(0);

function preloadRestOfImages() {
    for (let i = 1; i < frameCount; i++) {
        const img = new Image();
        img.src = currentFrame(i);
        images[i] = img;
    }
}

function render() {
    const frameIndex = Math.round(imageSeq.frame);
    if (images[frameIndex] && images[frameIndex].complete) {
        const img = images[frameIndex];

        if (img.width === 0) return;

        if (!isScaleComputed) {
            const scale = Math.max(canvas.width / img.width, canvas.height / img.height);
            scaledW = img.width * scale;
            scaledH = img.height * scale;
            scaledX = (canvas.width / 2) - (scaledW / 2);
            scaledY = (canvas.height / 2) - (scaledH / 2);
            isScaleComputed = true;
        }

        context.drawImage(img, scaledX, scaledY, scaledW, scaledH);
    }
}

// 4. GSAP Animations

// Canvas Sequence ScrollTrigger
gsap.to(imageSeq, {
    frame: frameCount - 1,
    snap: "frame",
    ease: "none",
    scrollTrigger: {
        trigger: ".scroll-wrapper",
        start: "top top",
        end: "bottom bottom",
        scrub: 1.5 // Added some scrub smoothing
    },
    onUpdate: render
});

// Hero text animations
gsap.from(".reveal-text span", {
    y: 100,
    opacity: 0,
    duration: 1.5,
    ease: "power4.out",
    delay: 0.2
});

gsap.from(".fade-up", {
    y: 40,
    opacity: 0,
    duration: 1.2,
    ease: "power3.out",
    stagger: 0.2,
    delay: 0.6
});

// Step card animations
gsap.utils.toArray('.glass-card').forEach(card => {
    gsap.fromTo(card,
        { y: 150, opacity: 0 },
        {
            y: 0,
            opacity: 1,
            duration: 1.5,
            ease: "power3.out",
            scrollTrigger: {
                trigger: card,
                start: "top 90%",
                end: "bottom 60%",
                toggleActions: "play reverse play reverse" /* Animate in and out smoothly */
            }
        }
    );
});

// 5. Canvas 2 3D Sequence
const canvas2 = document.getElementById("second-lightpass");
const context2 = canvas2.getContext("2d", { alpha: false });

const frameCount2 = 240;
const currentFrame2 = index => {
    const pad = (index + 1).toString().padStart(3, '0');
    return `ezgif-4510f3d7cedf6cf6-jpg/ezgif-frame-${pad}.jpg`;
};

const images2 = [];
const imageSeq2 = { frame: 0 };

let scaledX2 = 0, scaledY2 = 0, scaledW2 = 0, scaledH2 = 0;
let isScaleComputed2 = false;

function resizeCanvas2() {
    canvas2.width = window.innerWidth;
    canvas2.height = window.innerHeight;
    isScaleComputed2 = false;
    render2();
}
window.addEventListener("resize", resizeCanvas2);
resizeCanvas2();

const firstImg2 = new Image();
firstImg2.onload = () => {
    images2[0] = firstImg2;
    render2();
    preloadRestOfImages2();
};
firstImg2.src = currentFrame2(0);

function preloadRestOfImages2() {
    for (let i = 1; i < frameCount2; i++) {
        const img = new Image();
        img.src = currentFrame2(i);
        images2[i] = img;
    }
}

function render2() {
    const frameIndex = Math.round(imageSeq2.frame);
    if (images2[frameIndex] && images2[frameIndex].complete) {
        const img = images2[frameIndex];

        if (img.width === 0) return;

        if (!isScaleComputed2) {
            const scale = Math.max(canvas2.width / img.width, canvas2.height / img.height);
            scaledW2 = img.width * scale;
            scaledH2 = img.height * scale;
            scaledX2 = (canvas2.width / 2) - (scaledW2 / 2);
            scaledY2 = (canvas2.height / 2) - (scaledH2 / 2);
            isScaleComputed2 = true;
        }

        context2.drawImage(img, scaledX2, scaledY2, scaledW2, scaledH2);
    }
}

// Fade Canvases
gsap.to("#canvas1-wrapper", {
    opacity: 0,
    ease: "none",
    scrollTrigger: {
        trigger: ".scroll-wrapper-2",
        start: "top bottom",
        end: "top center",
        scrub: true
    }
});
gsap.to("#canvas2-wrapper", {
    opacity: 1,
    ease: "none",
    scrollTrigger: {
        trigger: ".scroll-wrapper-2",
        start: "top bottom",
        end: "top center",
        scrub: true
    }
});

// Canvas 2 Sequence ScrollTrigger
gsap.to(imageSeq2, {
    frame: frameCount2 - 1,
    snap: "frame",
    ease: "none",
    scrollTrigger: {
        trigger: ".scroll-wrapper-2",
        start: "top top",
        end: "bottom bottom",
        scrub: 1.5
    },
    onUpdate: render2
});
