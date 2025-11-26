/**
 * THEME TOGGLE - Dark/Light Mode Switcher
 *
 * Features:
 * - Toggles .dark-mode class on <body>
 * - Saves preference to localStorage
 * - Detects system preference on first visit
 * - Prevents transition flash on page load
 * - Accessible keyboard controls
 */

(function () {
  "use strict";

  // ============================================================================
  // CONFIGURATION
  // ============================================================================
  const STORAGE_KEY = "louise-portfolio-theme";
  const DARK_MODE_CLASS = "dark-mode";
  const NO_TRANSITION_CLASS = "no-transition";

  // ============================================================================
  // DOM ELEMENTS
  // ============================================================================
  const body = document.body;
  let toggleButton = null;

  // ============================================================================
  // THEME DETECTION & INITIALIZATION
  // ============================================================================

  /**
   * Get saved theme from localStorage or detect system preference
   * @returns {string} 'dark' or 'light'
   */
  function getInitialTheme() {
    // Check if user has a saved preference
    const savedTheme = localStorage.getItem(STORAGE_KEY);
    if (savedTheme) {
      return savedTheme;
    }

    // Check system preference (respects OS dark mode setting)
    if (
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
    ) {
      return "dark";
    }

    // Default to light theme
    return "light";
  }

  /**
   * Apply theme without transition (for page load)
   */
  function applyInitialTheme() {
    const theme = getInitialTheme();

    // Temporarily disable transitions
    body.classList.add(NO_TRANSITION_CLASS);

    if (theme === "dark") {
      body.classList.add(DARK_MODE_CLASS);
    }

    // Re-enable transitions after a brief delay
    setTimeout(() => {
      body.classList.remove(NO_TRANSITION_CLASS);
    }, 50);
  }

  // ============================================================================
  // THEME TOGGLE FUNCTIONS
  // ============================================================================

  /**
   * Toggle between dark and light mode
   */
  function toggleTheme() {
    const isDark = body.classList.toggle(DARK_MODE_CLASS);
    const newTheme = isDark ? "dark" : "light";

    // Sync attribute for CSS selectors [data-theme="dark"]
    body.setAttribute("data-theme", newTheme);

    localStorage.setItem(STORAGE_KEY, newTheme);
    updateButtonLabel(isDark);

    // Optional: Log for debugging
    console.log(`Theme switched to: ${newTheme}`);
  }

  /**
   * Update toggle button's accessible label
   * @param {boolean} isDark - Current theme state
   */
  function updateButtonLabel(isDark) {
    if (toggleButton) {
      const label = isDark ? "Switch to light mode" : "Switch to dark mode";
      toggleButton.setAttribute("aria-label", label);
    }
  }

  // ============================================================================
  // CREATE TOGGLE BUTTON
  // ============================================================================

  /**
   * Dynamically create the theme toggle button
   * (Can also be added directly in HTML)
   */
  function createToggleButton() {
    toggleButton = document.createElement("button");
    toggleButton.className = "theme-toggle";
    toggleButton.setAttribute("aria-label", "Toggle dark mode");
    toggleButton.setAttribute("type", "button");

    // Icon container
    const iconContainer = document.createElement("span");
    iconContainer.className = "theme-toggle-icon";
    iconContainer.setAttribute("aria-hidden", "true");

    // Sun icon (shown in dark mode)
    const sunIcon = document.createElement("span");
    sunIcon.className = "icon-sun";
    sunIcon.textContent = "☀️"; // Can replace with SVG later

    // Moon icon (shown in light mode)
    const moonIcon = document.createElement("span");
    moonIcon.className = "icon-moon";
    moonIcon.textContent = "🌙"; // Can replace with SVG later

    iconContainer.appendChild(sunIcon);
    iconContainer.appendChild(moonIcon);
    toggleButton.appendChild(iconContainer);

    // Add to page
    body.appendChild(toggleButton);

    // Attach event listener
    toggleButton.addEventListener("click", toggleTheme);

    // Set initial label
    const isDark = body.classList.contains(DARK_MODE_CLASS);
    updateButtonLabel(isDark);
  }

  // ============================================================================
  // SYSTEM PREFERENCE WATCHER
  // ============================================================================

  /**
   * Listen for system theme changes (optional feature)
   */
  function watchSystemPreference() {
    if (!window.matchMedia) return;

    const darkModeQuery = window.matchMedia("(prefers-color-scheme: dark)");

    // Only auto-switch if user hasn't manually set a preference
    darkModeQuery.addEventListener("change", e => {
      if (!localStorage.getItem(STORAGE_KEY)) {
        if (e.matches) {
          body.classList.add(DARK_MODE_CLASS);
        } else {
          body.classList.remove(DARK_MODE_CLASS);
        }
        updateButtonLabel(e.matches);
      }
    });
  }

  // ============================================================================
  // KEYBOARD SHORTCUT (Optional Enhancement)
  // ============================================================================

  /**
   * Enable Ctrl+Shift+D to toggle theme
   */
  function enableKeyboardShortcut() {
    document.addEventListener("keydown", e => {
      // Ctrl+Shift+D or Cmd+Shift+D
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "D") {
        e.preventDefault();
        toggleTheme();
      }
    });
  }

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  /**
   * Initialize theme system when DOM is ready
   */
  function init() {
    // Apply theme immediately (before page renders)
    applyInitialTheme();

    // Wait for DOM to be fully loaded
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => {
        createToggleButton();
        watchSystemPreference();
        enableKeyboardShortcut();
      });
    } else {
      // DOM already loaded
      createToggleButton();
      watchSystemPreference();
      enableKeyboardShortcut();
    }
  }

  // ============================================================================
  // RUN
  // ============================================================================
  init();
})();
