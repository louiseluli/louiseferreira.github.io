/**
 * LIGHTBOX - Image Gallery Viewer
 * Click any figure to view full-size with navigation
 */

(function () {
  "use strict";

  let lightbox = null;
  let lightboxImg = null;
  let lightboxCaption = null;
  let currentImages = [];
  let currentIndex = 0;

  /**
   * Create lightbox HTML structure
   */
  function createLightbox() {
    lightbox = document.createElement("div");
    lightbox.className = "lightbox";
    lightbox.innerHTML = `
      <button class="lightbox-close" aria-label="Close lightbox">&times;</button>
      <button class="lightbox-nav lightbox-prev" aria-label="Previous image">&#8249;</button>
      <div class="lightbox-content">
        <img src="" alt="" />
        <div class="lightbox-caption"></div>
      </div>
      <button class="lightbox-nav lightbox-next" aria-label="Next image">&#8250;</button>
    `;
    document.body.appendChild(lightbox);

    lightboxImg = lightbox.querySelector(".lightbox-content img");
    lightboxCaption = lightbox.querySelector(".lightbox-caption");

    // Event listeners
    lightbox.querySelector(".lightbox-close").addEventListener("click", closeLightbox);
    lightbox.querySelector(".lightbox-prev").addEventListener("click", showPrev);
    lightbox.querySelector(".lightbox-next").addEventListener("click", showNext);
    
    // Close on background click
    lightbox.addEventListener("click", function (e) {
      if (e.target === lightbox) {
        closeLightbox();
      }
    });

    // Keyboard navigation
    document.addEventListener("keydown", function (e) {
      if (!lightbox.classList.contains("active")) return;
      
      if (e.key === "Escape") closeLightbox();
      if (e.key === "ArrowLeft") showPrev();
      if (e.key === "ArrowRight") showNext();
    });
  }

  /**
   * Open lightbox with specific image
   */
  function openLightbox(img, caption, images, index) {
    currentImages = images;
    currentIndex = index;
    
    lightboxImg.src = img;
    lightboxImg.alt = caption;
    lightboxCaption.textContent = caption;
    
    lightbox.classList.add("active");
    document.body.style.overflow = "hidden";
    
    // Update nav visibility
    updateNavVisibility();
  }

  /**
   * Close lightbox
   */
  function closeLightbox() {
    lightbox.classList.remove("active");
    document.body.style.overflow = "";
  }

  /**
   * Show previous image
   */
  function showPrev() {
    if (currentIndex > 0) {
      currentIndex--;
      updateLightboxContent();
    }
  }

  /**
   * Show next image
   */
  function showNext() {
    if (currentIndex < currentImages.length - 1) {
      currentIndex++;
      updateLightboxContent();
    }
  }

  /**
   * Update lightbox content
   */
  function updateLightboxContent() {
    const item = currentImages[currentIndex];
    lightboxImg.src = item.src;
    lightboxImg.alt = item.caption;
    lightboxCaption.textContent = item.caption;
    updateNavVisibility();
  }

  /**
   * Update navigation button visibility
   */
  function updateNavVisibility() {
    const prevBtn = lightbox.querySelector(".lightbox-prev");
    const nextBtn = lightbox.querySelector(".lightbox-next");
    
    prevBtn.style.opacity = currentIndex === 0 ? "0.3" : "1";
    prevBtn.style.pointerEvents = currentIndex === 0 ? "none" : "auto";
    
    nextBtn.style.opacity = currentIndex === currentImages.length - 1 ? "0.3" : "1";
    nextBtn.style.pointerEvents = currentIndex === currentImages.length - 1 ? "none" : "auto";
  }

  /**
   * Initialize lightbox on all figures
   */
  function init() {
    createLightbox();

    // Collect all figures with images
    const figures = document.querySelectorAll(".visual-grid figure, .hero-collage figure");
    
    // Build image array
    const allImages = [];
    figures.forEach((figure, index) => {
      const img = figure.querySelector("img");
      const caption = figure.querySelector("figcaption");
      
      if (img) {
        allImages.push({
          src: img.src,
          caption: caption ? caption.textContent : ""
        });

        // Click handler
        figure.addEventListener("click", function () {
          openLightbox(img.src, caption ? caption.textContent : "", allImages, index);
        });
      }
    });
  }

  // Run when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
