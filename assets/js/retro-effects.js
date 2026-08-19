/* =============================================================================
   RETRO EFFECTS - Pixel Particles & Interactive Animations
   ============================================================================= */

(function () {
  "use strict";

  // Respect reduced motion preference
  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  if (prefersReducedMotion) return;

  // -------------------------------------------------------------------------
  // PIXEL PARTICLE CANVAS
  // Floating pixel particles in the hero background
  // -------------------------------------------------------------------------
  function initPixelCanvas() {
    const canvas = document.getElementById("pixel-canvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    let particles = [];
    let animationId;
    let width, height;

    function resize() {
      const hero = canvas.closest(".hero");
      if (!hero) return;
      width = hero.offsetWidth;
      height = hero.offsetHeight;
      canvas.width = width;
      canvas.height = height;
    }

    function createParticles() {
      particles = [];
      const count = Math.min(Math.floor((width * height) / 25000), 60);

      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          size: Math.random() * 3 + 1,
          speedX: (Math.random() - 0.5) * 0.3,
          speedY: (Math.random() - 0.5) * 0.3,
          opacity: Math.random() * 0.4 + 0.1,
          pulse: Math.random() * Math.PI * 2,
          pulseSpeed: Math.random() * 0.02 + 0.005,
        });
      }
    }

    function getParticleColor() {
      const isDark = document.body.classList.contains("dark-mode");
      return isDark
        ? { r: 196, g: 181, b: 253 }
        : { r: 90, g: 31, b: 153 };
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      const color = getParticleColor();

      particles.forEach((p) => {
        p.x += p.speedX;
        p.y += p.speedY;
        p.pulse += p.pulseSpeed;

        // Wrap around edges
        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;

        const currentOpacity =
          p.opacity * (0.5 + 0.5 * Math.sin(p.pulse));

        ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${currentOpacity})`;

        // Draw pixel-style square particles
        const size = Math.round(p.size);
        ctx.fillRect(
          Math.round(p.x),
          Math.round(p.y),
          size,
          size
        );
      });

      // Draw connections between nearby particles
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 120) {
            const lineOpacity = (1 - dist / 120) * 0.08;
            ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${lineOpacity})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(
              Math.round(particles[i].x),
              Math.round(particles[i].y)
            );
            ctx.lineTo(
              Math.round(particles[j].x),
              Math.round(particles[j].y)
            );
            ctx.stroke();
          }
        }
      }

      animationId = requestAnimationFrame(draw);
    }

    // Pause when not visible
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            if (!animationId) draw();
          } else {
            if (animationId) {
              cancelAnimationFrame(animationId);
              animationId = null;
            }
          }
        });
      },
      { threshold: 0.1 }
    );

    resize();
    createParticles();
    observer.observe(canvas);

    window.addEventListener("resize", () => {
      resize();
      createParticles();
    });
  }

  // -------------------------------------------------------------------------
  // TYPED TEXT EFFECT FOR HERO SUBTITLE
  // Adds a subtle typewriter feel to the subtitle
  // -------------------------------------------------------------------------
  function initTypedEffect() {
    const subtitle = document.querySelector(".pixel-subtitle");
    if (!subtitle) return;

    subtitle.style.opacity = "0";
    subtitle.style.transition = "opacity 0.5s ease";

    setTimeout(() => {
      subtitle.style.opacity = "1";
    }, 600);
  }

  // -------------------------------------------------------------------------
  // XP BAR ANIMATION
  // Animate the XP bars when they come into view
  // -------------------------------------------------------------------------
  function initXPBars() {
    const bars = document.querySelectorAll(".retro-xp-bar span");
    if (!bars.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const bar = entry.target;
            const targetWidth = bar.style.width;
            bar.style.width = "0%";

            requestAnimationFrame(() => {
              setTimeout(() => {
                bar.style.width = targetWidth;
              }, 200);
            });

            observer.unobserve(bar);
          }
        });
      },
      { threshold: 0.3 }
    );

    bars.forEach((bar) => observer.observe(bar));
  }

  // -------------------------------------------------------------------------
  // HOVER SOUND EFFECT (Optional visual feedback)
  // Adds a subtle pixel "click" animation on retro buttons
  // -------------------------------------------------------------------------
  function initRetroButtonEffects() {
    const buttons = document.querySelectorAll(".retro-btn");

    buttons.forEach((btn) => {
      btn.addEventListener("mouseenter", () => {
        btn.style.transition = "all 0.1s ease";
      });

      btn.addEventListener("mouseleave", () => {
        btn.style.transition = "all 0.25s ease";
      });
    });
  }

  // -------------------------------------------------------------------------
  // PARALLAX PIXEL GRID
  // Subtle parallax on the pixel grid background
  // -------------------------------------------------------------------------
  function initParallaxGrid() {
    const grid = document.querySelector(".pixel-grid");
    if (!grid) return;

    let ticking = false;

    window.addEventListener("scroll", () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          const scrollY = window.scrollY;
          grid.style.transform = `translateY(${scrollY * 0.15}px)`;
          ticking = false;
        });
        ticking = true;
      }
    });
  }

  // -------------------------------------------------------------------------
  // INITIALIZE ALL EFFECTS
  // -------------------------------------------------------------------------
  function init() {
    initPixelCanvas();
    initTypedEffect();
    initXPBars();
    initRetroButtonEffects();
    initParallaxGrid();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
