/* FILE: assets/js/cinema-search.js */
/**
 * CINEMA ANALYSIS DEMO SCRIPT
 * Simulates the backend search & recommendation engine using a static subset of data.
 * * Context: The full project processed 140,000+ titles using LightGBM and NetworkX.
 * This script provides a lightweight, client-side "Try It" experience.
 */

(function () {
  "use strict";

  // ============================================================================
  // 1. STATIC DATASET (Subset of the 140k processed films)
  // ============================================================================
  const FILM_DATA = [
    {
      id: "tt0050419",
      title: "The Bridge on the River Kwai",
      year: 1957,
      director: "David Lean",
      genre: ["Drama", "War"],
      centrality: 0.98, // NetworkX Graph Centrality Score
      cluster: "Golden Age Epics",
      keywords: ["war", "bridge", "pow", "thailand"],
    },
    {
      id: "tt0053125",
      title: "North by Northwest",
      year: 1959,
      director: "Alfred Hitchcock",
      genre: ["Adventure", "Mystery", "Thriller"],
      centrality: 0.95,
      cluster: "Hitchcockian Suspense",
      keywords: ["mistaken identity", "spy", "crop duster", "mount rushmore"],
    },
    {
      id: "tt0054215",
      title: "Psycho",
      year: 1960,
      director: "Alfred Hitchcock",
      genre: ["Horror", "Mystery", "Thriller"],
      centrality: 0.97,
      cluster: "Hitchcockian Suspense",
      keywords: ["motel", "shower", "killer", "mother"],
    },
    {
      id: "tt0052520",
      title: "The 400 Blows (Les quatre cents coups)",
      year: 1959,
      director: "François Truffaut",
      genre: ["Crime", "Drama"],
      centrality: 0.88,
      cluster: "French New Wave",
      keywords: ["childhood", "paris", "rebellion", "french"],
    },
    {
      id: "tt0050083",
      title: "12 Angry Men",
      year: 1957,
      director: "Sidney Lumet",
      genre: ["Crime", "Drama"],
      centrality: 0.92,
      cluster: "Courtroom Dramas",
      keywords: ["jury", "justice", "courtroom", "doubt"],
    },
    {
      id: "tt0047478",
      title: "Seven Samurai (Shichinin no samurai)",
      year: 1954,
      director: "Akira Kurosawa",
      genre: ["Action", "Adventure", "Drama"],
      centrality: 0.96,
      cluster: "Japanese Golden Age",
      keywords: ["samurai", "bandits", "village", "japan"],
    },
    {
      id: "tt0034583",
      title: "Casablanca",
      year: 1942,
      director: "Michael Curtiz",
      genre: ["Drama", "Romance", "War"],
      centrality: 0.99,
      cluster: "Hollywood Classics",
      keywords: ["nazi", "morocco", "bar", "resistance"],
    },
    {
      id: "tt0017925",
      title: "Metropolis",
      year: 1927,
      director: "Fritz Lang",
      genre: ["Drama", "Sci-Fi"],
      centrality: 0.85,
      cluster: "German Expressionism",
      keywords: ["robot", "dystopia", "future", "silent"],
    },
    {
      id: "tt0053472",
      title: "The House on Haunted Hill",
      year: 1959,
      director: "William Castle",
      genre: ["Horror", "Mystery"],
      centrality: 0.75,
      cluster: "Cult Horror",
      keywords: ["vincent price", "ghost", "haunted house", "party"],
    },
    {
      id: "tt0033467",
      title: "Citizen Kane",
      year: 1941,
      director: "Orson Welles",
      genre: ["Drama", "Mystery"],
      centrality: 0.94,
      cluster: "Hollywood Classics",
      keywords: ["newspaper", "tycoon", "rosebud", "flashback"],
    },
    {
      id: "tt0032138",
      title: "The Wizard of Oz",
      year: 1939,
      director: "Victor Fleming",
      genre: ["Adventure", "Family", "Fantasy"],
      centrality: 0.93,
      cluster: "Musical Fantasy",
      keywords: ["witch", "tornado", "magic", "songs"],
    },
    {
      id: "tt0043265",
      title: "A Streetcar Named Desire",
      year: 1951,
      director: "Elia Kazan",
      genre: ["Drama"],
      centrality: 0.89,
      cluster: "Method Acting Era",
      keywords: ["new orleans", "madness", "sister", "play adaptation"],
    },
  ];

  // ============================================================================
  // 2. DOM ELEMENTS & INIT
  // ============================================================================
  let searchInput, resultsContainer, statsElement;

  function init() {
    searchInput = document.getElementById("cinemaSearchInput");
    resultsContainer = document.getElementById("cinemaResults");
    statsElement = document.getElementById("demoStats");

    if (!searchInput || !resultsContainer) {
      console.warn("Cinema search elements not found on this page.");
      return;
    }

    // Initial Render
    renderResults(FILM_DATA);
    updateStats(FILM_DATA.length);

    // Event Listener
    searchInput.addEventListener("input", handleSearch);
  }

  // ============================================================================
  // 3. SEARCH LOGIC
  // ============================================================================
  function handleSearch(e) {
    const query = e.target.value.toLowerCase().trim();

    if (!query) {
      renderResults(FILM_DATA);
      updateStats(FILM_DATA.length);
      return;
    }

    const filtered = FILM_DATA.filter(film => {
      // Search across multiple fields (simulating the NLP pipeline)
      return (
        film.title.toLowerCase().includes(query) ||
        film.director.toLowerCase().includes(query) ||
        film.year.toString().includes(query) ||
        film.genre.some(g => g.toLowerCase().includes(query)) ||
        film.keywords.some(k => k.toLowerCase().includes(query))
      );
    });

    renderResults(filtered);
    updateStats(filtered.length);
  }

  function updateStats(count) {
    if (statsElement) {
      statsElement.textContent = `Showing ${count} titles from demo subset`;
    }
  }

  // ============================================================================
  // 4. RENDER LOGIC
  // ============================================================================
  function renderResults(data) {
    resultsContainer.innerHTML = "";

    if (data.length === 0) {
      resultsContainer.innerHTML = `
        <div class="no-results">
          <p>No films found matching your query in this demo subset.</p>
          <small>Try "Hitchcock", "1959", or "Drama"</small>
        </div>
      `;
      return;
    }

    data.forEach(film => {
      // Dynamic centrality bar width
      const centralityPercent = Math.round(film.centrality * 100);

      // Determine badge color based on cluster
      let badgeClass = "badge-default";
      if (film.cluster.includes("Horror")) badgeClass = "badge-red";
      if (film.cluster.includes("Classic")) badgeClass = "badge-gold";
      if (
        film.cluster.includes("Foreign") ||
        film.cluster.includes("Wave") ||
        film.cluster.includes("Samurai")
      )
        badgeClass = "badge-blue";

      const card = document.createElement("div");
      card.className = "film-result-card fade-in";
      card.innerHTML = `
        <div class="film-header">
          <h5>${film.title} <span class="film-year">(${film.year})</span></h5>
          <span class="film-cluster ${badgeClass}">${film.cluster}</span>
        </div>
        
        <div class="film-meta">
          <span>🎬 ${film.director}</span>
          <span>🏷️ ${film.genre.join(", ")}</span>
        </div>

        <div class="film-metrics">
          <div class="metric-row">
            <span class="metric-name">Graph Centrality</span>
            <div class="metric-bar-container">
              <div class="metric-bar-fill" style="width: ${centralityPercent}%"></div>
            </div>
            <span class="metric-value">${film.centrality}</span>
          </div>
          <div class="keywords-list">
            ${film.keywords
              .map(k => `<span class="keyword-tag">#${k}</span>`)
              .join("")}
          </div>
        </div>
      `;

      resultsContainer.appendChild(card);
    });
  }

  // ============================================================================
  // 5. RUN
  // ============================================================================
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
