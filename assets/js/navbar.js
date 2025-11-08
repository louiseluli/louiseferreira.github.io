/**
 * NAVIGATION BAR FUNCTIONALITY
 *
 * Features:
 * - Mobile hamburger menu toggle
 * - Active page highlighting
 * - Smooth scroll to sections
 * - Scroll-based navbar style changes
 * - Accessible keyboard navigation
 * - Auto-close menu on link click (mobile)
 * - Auto-close menu on outside click
 */

(function () {
  "use strict";

  // ============================================================================
  // CONFIGURATION
  // ============================================================================
  const SCROLL_THRESHOLD = 50; // Pixels scrolled before adding .scrolled class
  const SMOOTH_SCROLL_DURATION = 800; // Milliseconds

  // ============================================================================
  // DOM ELEMENTS (Lazy loaded in init)
  // ============================================================================
  let navbar = null;
  let navbarToggle = null;
  let navbarMenu = null;
  let navbarOverlay = null;
  let navLinks = null;

  // ============================================================================
  // MOBILE MENU TOGGLE
  // ============================================================================

  /**
   * Toggle mobile menu open/closed
   */
  function toggleMobileMenu() {
    const isActive = navbarMenu.classList.toggle("active");
    navbarToggle.classList.toggle("active");

    // Show/hide overlay
    if (navbarOverlay) {
      navbarOverlay.classList.toggle("active");
    }

    // Prevent body scroll when menu is open
    document.body.classList.toggle("menu-open", isActive);

    // Update aria-expanded for accessibility
    navbarToggle.setAttribute("aria-expanded", isActive);

    // Announce to screen readers
    const announcement = isActive ? "Menu opened" : "Menu closed";
    announceToScreenReader(announcement);
  }

  /**
   * Close mobile menu
   */
  function closeMobileMenu() {
    navbarMenu.classList.remove("active");
    navbarToggle.classList.remove("active");
    if (navbarOverlay) {
      navbarOverlay.classList.remove("active");
    }
    document.body.classList.remove("menu-open");
    navbarToggle.setAttribute("aria-expanded", "false");
  }

  // ============================================================================
  // SCROLL EFFECTS
  // ============================================================================

  /**
   * Add/remove .scrolled class based on scroll position
   */
  function handleScroll() {
    const header = document.querySelector(".site-header");
    if (!header) return;

    if (window.scrollY > SCROLL_THRESHOLD) {
      header.classList.add("scrolled");
    } else {
      header.classList.remove("scrolled");
    }
  }

  // Throttle scroll event for performance
  let scrollTimeout;
  function throttledScroll() {
    if (scrollTimeout) return;

    scrollTimeout = setTimeout(() => {
      handleScroll();
      scrollTimeout = null;
    }, 100);
  }

  // ============================================================================
  // ACTIVE PAGE HIGHLIGHTING
  // ============================================================================

  /**
   * Highlight current page in navigation
   */
  function setActivePage() {
    if (!navLinks) return;

    const currentPath = window.location.pathname;
    const currentPage = currentPath.split("/").pop() || "index.html";

    navLinks.forEach(link => {
      const linkPath = link.getAttribute("href");

      // Remove existing active class
      link.classList.remove("active");

      // Add active class to matching link
      if (
        linkPath === currentPage ||
        (currentPage === "" && linkPath === "index.html") ||
        (currentPage === "index.html" && linkPath === "/")
      ) {
        link.classList.add("active");
        link.setAttribute("aria-current", "page");
      } else {
        link.removeAttribute("aria-current");
      }
    });
  }

  // ============================================================================
  // SMOOTH SCROLL (for same-page anchor links)
  // ============================================================================

  /**
   * Smooth scroll to anchor sections
   * @param {Event} e - Click event
   */
  function smoothScrollToSection(e) {
    const href = e.currentTarget.getAttribute("href");

    // Only handle anchor links (#section)
    if (!href || !href.startsWith("#")) return;

    e.preventDefault();

    const targetId = href.substring(1);
    const targetElement = document.getElementById(targetId);

    if (!targetElement) return;

    // Calculate offset for fixed header
    const headerHeight = navbar ? navbar.offsetHeight : 72;
    const targetPosition = targetElement.offsetTop - headerHeight - 20;

    // Smooth scroll
    window.scrollTo({
      top: targetPosition,
      behavior: "smooth",
    });

    // Close mobile menu if open
    if (window.innerWidth <= 768) {
      closeMobileMenu();
    }

    // Update URL without jumping
    history.pushState(null, null, href);

    // Focus target element for accessibility
    targetElement.setAttribute("tabindex", "-1");
    targetElement.focus();
  }

  // ============================================================================
  // CREATE MOBILE OVERLAY
  // ============================================================================

  /**
   * Create overlay element for mobile menu backdrop
   */
  function createOverlay() {
    const overlay = document.createElement("div");
    overlay.className = "navbar-overlay";
    overlay.setAttribute("aria-hidden", "true");

    // Close menu when clicking overlay
    overlay.addEventListener("click", closeMobileMenu);

    document.body.appendChild(overlay);
    return overlay;
  }

  // ============================================================================
  // KEYBOARD NAVIGATION
  // ============================================================================

  /**
   * Handle keyboard events for accessibility
   * @param {KeyboardEvent} e
   */
  function handleKeyboard(e) {
    // Close mobile menu on Escape key
    if (e.key === "Escape" && navbarMenu.classList.contains("active")) {
      closeMobileMenu();
      navbarToggle.focus(); // Return focus to toggle button
    }
  }

  // ============================================================================
  // ACCESSIBILITY HELPERS
  // ============================================================================

  /**
   * Announce changes to screen readers
   * @param {string} message - Message to announce
   */
  function announceToScreenReader(message) {
    const announcement = document.createElement("div");
    announcement.setAttribute("role", "status");
    announcement.setAttribute("aria-live", "polite");
    announcement.className = "sr-only";
    announcement.textContent = message;

    document.body.appendChild(announcement);

    // Remove after announcement
    setTimeout(() => {
      document.body.removeChild(announcement);
    }, 1000);
  }

  // ============================================================================
  // EVENT LISTENERS
  // ============================================================================

  /**
   * Attach all event listeners
   */
  function attachEventListeners() {
    // Mobile menu toggle
    if (navbarToggle) {
      navbarToggle.addEventListener("click", toggleMobileMenu);
    }

    // Close menu when clicking nav links
    if (navLinks) {
      navLinks.forEach(link => {
        // Smooth scroll for anchor links
        if (link.getAttribute("href")?.startsWith("#")) {
          link.addEventListener("click", smoothScrollToSection);
        }

        // Close mobile menu on any link click
        link.addEventListener("click", () => {
          if (window.innerWidth <= 768) {
            setTimeout(closeMobileMenu, 300);
          }
        });
      });
    }

    // Scroll effects
    window.addEventListener("scroll", throttledScroll, { passive: true });

    // Keyboard navigation
    document.addEventListener("keydown", handleKeyboard);

    // Close menu on window resize (if going from mobile to desktop)
    let resizeTimeout;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        if (window.innerWidth > 768) {
          closeMobileMenu();
        }
      }, 150);
    });
  }

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  /**
   * Initialize navbar functionality
   */
  function init() {
    // Wait for DOM
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", initNavbar);
    } else {
      initNavbar();
    }
  }

  /**
   * Initialize after DOM is ready
   */
  function initNavbar() {
    // Get DOM elements
    navbar = document.querySelector(".site-header");
    navbarToggle = document.querySelector(".navbar-toggle");
    navbarMenu = document.querySelector(".navbar-menu");
    navLinks = document.querySelectorAll(".navbar-menu a");

    // Create overlay for mobile menu
    navbarOverlay = createOverlay();

    // Set initial state
    if (navbarToggle) {
      navbarToggle.setAttribute("aria-expanded", "false");
      navbarToggle.setAttribute("aria-label", "Open navigation menu");
    }

    // Highlight current page
    setActivePage();

    // Attach event listeners
    attachEventListeners();

    // Initial scroll check
    handleScroll();

    console.log("✅ Navbar initialized");
  }

  // ============================================================================
  // RUN
  // ============================================================================
  init();
})();
