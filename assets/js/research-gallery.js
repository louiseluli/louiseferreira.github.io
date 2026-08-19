/* Research Gallery - Interactive figure navigation for AlgoFairness project */

document.addEventListener("DOMContentLoaded", function () {
  const gallery = document.querySelector(".research-gallery");
  if (!gallery) return;

  const thumbs = gallery.querySelectorAll(".thumb");
  const images = gallery.querySelectorAll(".gallery-image");
  const captionText = document.getElementById("gallery-caption-text");

  // Caption mapping
  const captions = {
    "gallery-pareto":
      "Accuracy-Fairness Pareto Frontier: Reweighing and post-processing are the only non-dominated strategies of five tested",
    "gallery-corpus":
      "Representation by Race x Gender: Black, Latina, and Asian women each hold under 3% of the 535K-video corpus",
    "gallery-mitigation":
      "Decision Margins, Reweighing: the best-accuracy mitigation strategy (95.3%), cutting Black Women's EOD by 53%",
    "gallery-fairness":
      "Fairness vs. Threshold: max Equal Opportunity Difference across groups peaks near threshold 0.6",
  };

  thumbs.forEach(thumb => {
    thumb.addEventListener("click", function () {
      const targetId = this.getAttribute("data-target");

      // Update active states
      thumbs.forEach(t => t.classList.remove("active"));
      images.forEach(img => img.classList.remove("active"));

      this.classList.add("active");
      const targetImage = document.getElementById(targetId);
      if (targetImage) {
        targetImage.classList.add("active");
      }

      // Update caption
      if (captionText && captions[targetId]) {
        captionText.textContent = captions[targetId];
      }
    });
  });

  // Auto-rotate gallery every 6 seconds (optional)
  let currentIndex = 0;
  const thumbArray = Array.from(thumbs);

  function autoRotate() {
    currentIndex = (currentIndex + 1) % thumbArray.length;
    thumbArray[currentIndex].click();
  }

  // Uncomment to enable auto-rotation:
  // setInterval(autoRotate, 6000);

  // Keyboard navigation
  document.addEventListener("keydown", function (e) {
    if (!gallery.matches(":hover")) return;

    if (e.key === "ArrowLeft") {
      currentIndex = (currentIndex - 1 + thumbArray.length) % thumbArray.length;
      thumbArray[currentIndex].click();
    } else if (e.key === "ArrowRight") {
      currentIndex = (currentIndex + 1) % thumbArray.length;
      thumbArray[currentIndex].click();
    }
  });

  // Touch swipe support for mobile
  let touchStartX = 0;
  let touchEndX = 0;

  const galleryMain = gallery.querySelector(".gallery-main");

  if (galleryMain) {
    galleryMain.addEventListener(
      "touchstart",
      function (e) {
        touchStartX = e.changedTouches[0].screenX;
      },
      { passive: true }
    );

    galleryMain.addEventListener(
      "touchend",
      function (e) {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
      },
      { passive: true }
    );
  }

  function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;

    if (Math.abs(diff) > swipeThreshold) {
      if (diff > 0) {
        // Swipe left - next image
        currentIndex = (currentIndex + 1) % thumbArray.length;
      } else {
        // Swipe right - previous image
        currentIndex =
          (currentIndex - 1 + thumbArray.length) % thumbArray.length;
      }
      thumbArray[currentIndex].click();
    }
  }
});
