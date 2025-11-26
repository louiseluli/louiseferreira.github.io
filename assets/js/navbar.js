/* FILE: assets/js/theme-toggle.js (UPDATED FULL VERSION) */
/**
 * THEME TOGGLE SYSTEM
 * Handles Dark/Light mode switching with persistence and accessibility.
 */

(function () {
  "use strict";

  // ============================================================================
  // CONFIGURATION
  // ============================================================================
  const STORAGE_KEY = "louise-portfolio-theme";
  const DARK_MODE_CLASS = "dark-mode";
  const DATA_THEME_ATTR = "data-theme";
  const NO_TRANSITION_CLASS = "no-transition";

  // ============================================================================
  // DOM ELEMENTS
  // ============================================================================
  const body = document.body;
  let toggleButton = null;

  // ============================================================================
  // THEME LOGIC
  // ============================================================================

  /**
   * Determine preferred theme from storage or system settings
   * @returns {string} 'dark' or 'light'
   */
  function getInitialTheme() {
    const savedTheme = localStorage.getItem(STORAGE_KEY);
    if (savedTheme) {
      return savedTheme;
    }

    if (
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
    ) {
      return "dark";
    }

    return "light";
  }

  /**
   * Apply the theme state to the DOM
   * @param {string} theme - 'dark' or 'light'
   * @param {boolean} disableTransition - prevent flash of styles
   */
  function applyTheme(theme, disableTransition = false) {
    if (disableTransition) {
      body.classList.add(NO_TRANSITION_CLASS);
    }

    if (theme === "dark") {
      body.classList.add(DARK_MODE_CLASS);
      body.setAttribute(DATA_THEME_ATTR, "dark");
    } else {
      body.classList.remove(DARK_MODE_CLASS);
      body.setAttribute(DATA_THEME_ATTR, "light");
    }

    if (disableTransition) {
      setTimeout(() => {
        body.classList.remove(NO_TRANSITION_CLASS);
      }, 50);
    }
  }

  /**
   * Toggle the current theme state
   */
  function toggleTheme() {
    const isDark = body.classList.contains(DARK_MODE_CLASS);
    const newTheme = isDark ? "light" : "dark"; // Toggle inverse

    applyTheme(newTheme, false);
    localStorage.setItem(STORAGE_KEY, newTheme);
    updateButtonLabel(newTheme === "dark");

    console.log(`Theme toggled to: ${newTheme}`);
  }

  /**
   * Update button accessibility label
   */
  function updateButtonLabel(isDark) {
    if (toggleButton) {
      const label = isDark ? "Switch to light mode" : "Switch to dark mode";
      toggleButton.setAttribute("aria-label", label);
    }
  }

  // ============================================================================
  // COMPONENT SETUP
  // ============================================================================

  /**
   * Find existing button or create a new one
   */
  function setupToggleButton() {
    // Check if button already exists in HTML (priority)
    toggleButton = document.getElementById("themeToggle");

    if (!toggleButton) {
      createToggleButton();
    } else {
      // Attach listener to existing button
      toggleButton.addEventListener("click", toggleTheme);
    }

    const isDark = body.classList.contains(DARK_MODE_CLASS);
    updateButtonLabel(isDark);
  }

  /**
   * Create button DOM elements if missing (Fallback)
   */
  function createToggleButton() {
    toggleButton = document.createElement("button");
    toggleButton.id = "themeToggle";
    toggleButton.className = "theme-toggle";
    toggleButton.setAttribute("type", "button");

    // SVG Icons (Inline to ensure they exist if JS creates the button)
    toggleButton.innerHTML = `
      <svg class="sun-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
      </svg>
      <svg class="moon-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
      </svg>
    `;

    document.body.appendChild(toggleButton);
    toggleButton.addEventListener("click", toggleTheme);
  }

  // ============================================================================
  // EVENT LISTENERS
  // ============================================================================

  /**
   * System preference listener
   */
  function watchSystemPreference() {
    if (!window.matchMedia) return;

    const darkModeQuery = window.matchMedia("(prefers-color-scheme: dark)");

    darkModeQuery.addEventListener("change", e => {
      // Only override if user hasn't set a specific preference
      if (!localStorage.getItem(STORAGE_KEY)) {
        applyTheme(e.matches ? "dark" : "light", false);
        updateButtonLabel(e.matches);
      }
    });
  }

  /**
   * Keyboard shortcut (Ctrl+Shift+D)
   */
  function enableKeyboardShortcut() {
    document.addEventListener("keydown", e => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "D") {
        e.preventDefault();
        toggleTheme();
      }
    });
  }

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  function init() {
    // 1. Apply theme immediately to avoid flash
    const initialTheme = getInitialTheme();
    applyTheme(initialTheme, true);

    // 2. Setup UI when DOM is ready
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => {
        setupToggleButton();
        watchSystemPreference();
        enableKeyboardShortcut();
      });
    } else {
      setupToggleButton();
      watchSystemPreference();
      enableKeyboardShortcut();
    }
  }

  init();
})();
