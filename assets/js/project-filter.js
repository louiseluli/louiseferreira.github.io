/**
 * PROJECT FILTER SYSTEM - FIXED VERSION
 *
 * Features:
 * - Filter projects by category
 * - Smooth animations
 * - Better error handling
 * - Debug logging
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
    console.log("Filtering by category:", category);

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

    // Filter featured project
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
    console.log("Showing all projects");
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
    card.style.display = "";
    card.classList.remove("hidden");

    // Small delay for animation
    setTimeout(() => {
      card.classList.add("fade-in");
    }, 10);
  }

  /**
   * Hide a card
   */
  function hideCard(card) {
    card.classList.remove("fade-in");
    card.classList.add("hidden");
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

    console.log(message);
    announceToScreenReader(message);
  }

  /**
   * Announce to screen readers
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

  /**
   * Attach click listeners
   */
  function attachEventListeners() {
    console.log(
      "Attaching event listeners to",
      filterButtons.length,
      "buttons"
    );

    filterButtons.forEach((button, index) => {
      button.addEventListener("click", e => {
        console.log("Button clicked:", index);

        const category = button.getAttribute("data-filter");
        console.log("Filter category:", category);

        if (!category) {
          console.error("No data-filter attribute found on button");
          return;
        }

        updateButtonStates(button);
        filterProjects(category);
      });
    });
  }

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  /**
   * Initialize the filter system
   */
  function init() {
    console.log("🎯 Initializing project filter system...");

    // Wait for DOM
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", initFilter);
    } else {
      initFilter();
    }
  }

  /**
   * Initialize after DOM ready
   */
  function initFilter() {
    // Get DOM elements
    filterButtons = document.querySelectorAll(".filter-btn");
    projectCards = document.querySelectorAll(".project-card");
    featuredProject = document.querySelector(".featured-project");

    console.log("Found elements:");
    console.log("- Filter buttons:", filterButtons.length);
    console.log("- Project cards:", projectCards.length);
    console.log("- Featured project:", featuredProject ? "Yes" : "No");

    // Exit if no buttons found
    if (!filterButtons.length) {
      console.error(
        '❌ No filter buttons found! Make sure elements have class "filter-btn"'
      );
      return;
    }

    if (!projectCards.length) {
      console.warn("⚠️ No project cards found");
    }

    // Set initial ARIA attributes
    filterButtons.forEach(btn => {
      const isActive = btn.classList.contains("active");
      btn.setAttribute("role", "button");
      btn.setAttribute("aria-pressed", isActive ? "true" : "false");
    });

    // Attach listeners
    attachEventListeners();

    console.log("✅ Project filter initialized successfully");
  }

  // ============================================================================
  // RUN
  // ============================================================================
  init();
})();
