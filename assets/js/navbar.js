/* FILE: assets/js/navbar.js */
/**
 * NAVIGATION SYSTEM
 * Handles mobile menu toggle, scroll effects, and active link highlighting.
 */

(function () {
  "use strict";

  // ============================================================================
  // DOM ELEMENTS
  // ============================================================================
  let navbarToggle = null;
  let navbarMenu = null;
  let navbarOverlay = null;
  let siteHeader = null;

  // ============================================================================
  // MOBILE MENU FUNCTIONS
  // ============================================================================

  /**
   * Toggle mobile menu open/closed state
   */
  function toggleMobileMenu() {
    const isOpen = navbarMenu.classList.contains("active");

    if (isOpen) {
      closeMobileMenu();
    } else {
      openMobileMenu();
    }
  }

  /**
   * Open the mobile menu
   */
  function openMobileMenu() {
    navbarMenu.classList.add("active");
    navbarToggle.classList.add("active");
    navbarToggle.setAttribute("aria-expanded", "true");

    if (navbarOverlay) {
      navbarOverlay.classList.add("active");
    }

    // Prevent body scroll when menu is open
    document.body.classList.add("menu-open");

    console.log("Menu opened");
  }

  /**
   * Close the mobile menu
   */
  function closeMobileMenu() {
    navbarMenu.classList.remove("active");
    navbarToggle.classList.remove("active");
    navbarToggle.setAttribute("aria-expanded", "false");

    if (navbarOverlay) {
      navbarOverlay.classList.remove("active");
    }

    // Re-enable body scroll
    document.body.classList.remove("menu-open");

    console.log("Menu closed");
  }

  // ============================================================================
  // SCROLL EFFECTS
  // ============================================================================

  /**
   * Handle header style changes on scroll
   */
  function handleScroll() {
    if (!siteHeader) return;

    const scrollY = window.scrollY || window.pageYOffset;

    // Add scrolled class after 50px
    if (scrollY > 50) {
      siteHeader.classList.add("scrolled");
    } else {
      siteHeader.classList.remove("scrolled");
    }

    // Update scroll progress indicator
    updateScrollProgress();
  }

  /**
   * Update the scroll progress bar in the header
   */
  function updateScrollProgress() {
    const scrollY = window.scrollY || window.pageYOffset;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrollPercent = docHeight > 0 ? (scrollY / docHeight) * 100 : 0;
    
    if (siteHeader) {
      siteHeader.style.setProperty('--scroll-progress', `${Math.min(scrollPercent, 100)}%`);
    }
  }

  // ============================================================================
  // OVERLAY CREATION
  // ============================================================================

  /**
   * Create overlay element if it doesn't exist
   */
  function ensureOverlayExists() {
    navbarOverlay = document.querySelector(".navbar-overlay");

    if (!navbarOverlay) {
      navbarOverlay = document.createElement("div");
      navbarOverlay.className = "navbar-overlay";
      navbarOverlay.setAttribute("aria-hidden", "true");

      // Insert after header
      const header = document.querySelector(".site-header");
      if (header && header.parentNode) {
        header.parentNode.insertBefore(navbarOverlay, header.nextSibling);
      } else {
        document.body.appendChild(navbarOverlay);
      }

      console.log("Overlay created");
    }

    // Click overlay to close menu
    navbarOverlay.addEventListener("click", closeMobileMenu);
  }

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  /**
   * Initialize all navbar functionality
   */
  function init() {
    // Cache DOM elements
    navbarToggle = document.querySelector(".navbar-toggle");
    navbarMenu = document.querySelector(".navbar-menu");
    siteHeader = document.querySelector(".site-header");

    // Debug: Check if elements exist
    console.log("Navbar Toggle found:", !!navbarToggle);
    console.log("Navbar Menu found:", !!navbarMenu);
    console.log("Site Header found:", !!siteHeader);

    if (!navbarToggle) {
      console.error("ERROR: .navbar-toggle button not found!");
      return;
    }

    if (!navbarMenu) {
      console.error("ERROR: .navbar-menu not found!");
      return;
    }

    // Ensure overlay exists
    ensureOverlayExists();

    // CRITICAL: Attach click event to burger button
    navbarToggle.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      toggleMobileMenu();
    });

    // Close menu when clicking on a link
    const navLinks = navbarMenu.querySelectorAll("a");
    navLinks.forEach(function (link) {
      link.addEventListener("click", function () {
        if (window.innerWidth <= 992) {
          closeMobileMenu();
        }
      });
    });

    // Scroll effects
    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll(); // Run once on load

    // Keyboard: Close on Escape
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && navbarMenu.classList.contains("active")) {
        closeMobileMenu();
      }
    });

    // Handle window resize
    window.addEventListener("resize", function () {
      if (window.innerWidth > 992 && navbarMenu.classList.contains("active")) {
        closeMobileMenu();
      }
    });

    console.log("✅ Navbar initialized successfully");
  }

  // Run when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
