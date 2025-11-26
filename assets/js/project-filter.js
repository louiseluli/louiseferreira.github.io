/* FILE: assets/js/project-filter.js (UPDATED FULL VERSION) */
/**
 * PROJECT FILTER SYSTEM
 * Features:
 * - Filter projects by category
 * - Smooth animations
 * - Accessibility announcements
 */

(function () {
  "use strict";

  // ============================================================================
  // DOM ELEMENTS
  // ============================================================================
  let filterButtons = null;
  let projectCards = null;
  let featuredProject = null;

  // ============================================================================
  // FILTER LOGIC
  // ============================================================================

  /**
   * Filter projects based on selected category
   */
  function filterProjects(category) {
    if (category === "all") {
      showAllProjects();
      return;
    }

    // Filter regular project cards
    projectCards.forEach(card => {
      const cardCategories = card.getAttribute("data-category");

      if (!cardCategories) {
        hideCard(card);
        return;
      }

      const categories = cardCategories.toLowerCase().split(" ");

      if (categories.includes(category.toLowerCase())) {
        showCard(card);
      } else {
        hideCard(card);
      }
    });

    // Filter featured project (special handling)
    if (featuredProject) {
      const featuredCategory = featuredProject.getAttribute("data-category");

      if (featuredCategory) {
        const featuredCategories = featuredCategory.toLowerCase().split(" ");
        if (featuredCategories.includes(category.toLowerCase())) {
          showCard(featuredProject);
        } else {
          hideCard(featuredProject);
        }
      }
    }

    announceFilterChange(category);
  }

  /**
   * Show all projects
   */
  function showAllProjects() {
    projectCards.forEach(card => showCard(card));
    if (featuredProject) {
      showCard(featuredProject);
    }
    announceFilterChange("all");
  }

  /**
   * Show a card with animation
   */
  function showCard(card) {
    card.style.display = ""; // Reset display
    card.classList.remove("hidden");

    // Small delay to trigger transition if CSS supports it
    setTimeout(() => {
      card.classList.add("fade-in");
      card.style.opacity = "1";
      card.style.transform = "translateY(0)";
    }, 10);
  }

  /**
   * Hide a card
   */
  function hideCard(card) {
    card.classList.remove("fade-in");
    card.classList.add("hidden");
    // Immediate style override for responsiveness
    card.style.display = "none";
  }

  // ============================================================================
  // BUTTON STATE MANAGEMENT
  // ============================================================================

  /**
   * Update active button state
   */
  function updateButtonStates(activeButton) {
    filterButtons.forEach(btn => {
      btn.classList.remove("active");
      btn.setAttribute("aria-pressed", "false");
    });

    activeButton.classList.add("active");
    activeButton.setAttribute("aria-pressed", "true");
  }

  // ============================================================================
  // ACCESSIBILITY
  // ============================================================================

  /**
   * Announce filter changes to screen readers
   */
  function announceFilterChange(category) {
    const visibleCards = Array.from(projectCards).filter(
      card => !card.classList.contains("hidden")
    ).length;

    const featuredVisible =
      featuredProject && !featuredProject.classList.contains("hidden") ? 1 : 0;

    const totalVisible = visibleCards + featuredVisible;
    const categoryName = category === "all" ? "all categories" : category;
    const message = `Showing ${totalVisible} project${
      totalVisible !== 1 ? "s" : ""
    } in ${categoryName}`;

    announceToScreenReader(message);
  }

  /**
   * Helper to append SR-only text
   */
  function announceToScreenReader(message) {
    const announcement = document.createElement("div");
    announcement.setAttribute("role", "status");
    announcement.setAttribute("aria-live", "polite");
    announcement.className = "sr-only";
    announcement.textContent = message;

    document.body.appendChild(announcement);

    setTimeout(() => {
      if (announcement.parentNode) {
        document.body.removeChild(announcement);
      }
    }, 1000);
  }

  // ============================================================================
  // EVENT LISTENERS
  // ============================================================================

  function attachEventListeners() {
    filterButtons.forEach(button => {
      button.addEventListener("click", () => {
        const category = button.getAttribute("data-filter");
        if (!category) return;

        updateButtonStates(button);
        filterProjects(category);
      });
    });
  }

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  function init() {
    // Get DOM elements
    filterButtons = document.querySelectorAll(".filter-btn");
    projectCards = document.querySelectorAll(".project-card");
    featuredProject = document.querySelector(".featured-project");

    if (!filterButtons.length) return;

    // Set initial ARIA attributes
    filterButtons.forEach(btn => {
      const isActive = btn.classList.contains("active");
      btn.setAttribute("role", "button");
      btn.setAttribute("aria-pressed", isActive ? "true" : "false");
    });

    attachEventListeners();
    console.log("✅ Project filter initialized");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
