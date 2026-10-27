/**
 * Fake News Detector — Frontend Logic
 */

(function () {
  "use strict";

  // ── DOM refs ─────────────────────────────────────────────────
  const form          = document.getElementById("detectorForm");
  const textarea      = document.getElementById("newsText");
  const analyzeBtn    = document.getElementById("analyzeBtn");
  const clearBtn      = document.getElementById("clearBtn");
  const charCounter   = document.getElementById("char-counter");
  const resultPanel   = document.getElementById("resultPanel");
  const resultIcon    = document.getElementById("resultIcon");
  const resultVerdict = document.getElementById("resultVerdict");
  const resultSubtitle= document.getElementById("resultSubtitle");
  const confidenceFill= document.getElementById("confidenceFill");
  const confidenceText= document.getElementById("confidenceText");
  const confidenceBar = document.getElementById("confidenceBar");
  const confidenceRating = document.getElementById("confidenceRating");
  const realProbEl    = document.getElementById("realProb");
  const fakeProbEl    = document.getElementById("fakeProb");
  const errorPanel    = document.getElementById("errorPanel");
  const errorText     = document.getElementById("errorText");
  const sampleCards   = document.querySelectorAll(".sample-card");

  const MAX_CHARS = 50000;

  // ── Character counter ─────────────────────────────────────────
  textarea.addEventListener("input", () => {
    const len = textarea.value.length;
    charCounter.textContent = `${len.toLocaleString()} / ${MAX_CHARS.toLocaleString()}`;
    charCounter.style.color = len > MAX_CHARS * 0.9 ? "#f59e0b" : "";
  });

  // ── Clear button ──────────────────────────────────────────────
  clearBtn.addEventListener("click", () => {
    textarea.value = "";
    charCounter.textContent = `0 / ${MAX_CHARS.toLocaleString()}`;
    hideResult();
    hideError();
    textarea.focus();
  });

  // ── Sample cards ──────────────────────────────────────────────
  sampleCards.forEach((card) => {
    card.addEventListener("click", () => {
      const text = card.getAttribute("data-text");
      textarea.value = text;
      charCounter.textContent = `${text.length.toLocaleString()} / ${MAX_CHARS.toLocaleString()}`;
      hideResult();
      hideError();
      // Scroll to form
      form.scrollIntoView({ behavior: "smooth", block: "center" });
      // Auto-submit after brief delay
      setTimeout(() => submitForm(), 400);
    });
  });

  // ── Form submit ───────────────────────────────────────────────
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    submitForm();
  });

  async function submitForm() {
    const text = textarea.value.trim();

    hideResult();
    hideError();

    // Client-side validation
    if (!text) {
      showError("Please enter some news text to analyze.");
      return;
    }
    if (text.length < 10) {
      showError("Text is too short. Please enter at least 10 characters.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();

      if (!response.ok) {
        showError(data.error || "An error occurred. Please try again.");
        return;
      }

      showResult(data);
    } catch (err) {
      showError("Network error. Please check your connection and try again.");
      console.error("Fetch error:", err);
    } finally {
      setLoading(false);
    }
  }

  // ── Show / hide helpers ───────────────────────────────────────
  function setLoading(loading) {
    const btnText   = analyzeBtn.querySelector(".btn-text");
    const btnIcon   = analyzeBtn.querySelector(".btn-icon");
    const btnLoader = analyzeBtn.querySelector(".btn-loader");

    analyzeBtn.disabled = loading;
    if (loading) {
      btnText.style.display   = "none";
      btnIcon.style.display   = "none";
      btnLoader.style.display = "inline-flex";
      btnLoader.style.alignItems = "center";
      btnLoader.style.gap     = "0.5rem";
    } else {
      btnText.style.display   = "";
      btnIcon.style.display   = "";
      btnLoader.style.display = "none";
    }
  }

  function hideResult() {
    resultPanel.style.display = "none";
    resultPanel.className = "result-panel";
  }

  function hideError() {
    errorPanel.style.display = "none";
  }

  function showError(msg) {
    errorText.textContent = msg;
    errorPanel.style.display = "flex";
    errorPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function showResult(data) {
    const isFake = data.prediction === "FAKE";
    const confidencePct = Math.round(data.confidence * 100);

    // Panel styling
    resultPanel.className = `result-panel ${isFake ? "result-fake" : "result-real"}`;
    resultPanel.style.display = "block";

    // Icon
    resultIcon.textContent = isFake ? "❌" : "✅";

    // Verdict
    resultVerdict.textContent = isFake ? "FAKE NEWS DETECTED" : "REAL NEWS";

    // Subtitle
    resultSubtitle.textContent = isFake
      ? "This article shows patterns commonly found in misinformation."
      : "This article appears to be credible based on our analysis.";

    // Confidence bar (animate)
    confidenceText.textContent = `${confidencePct}%`;
    confidenceBar.setAttribute("aria-valuenow", confidencePct);
    // Reset then animate
    confidenceFill.style.width = "0";
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        confidenceFill.style.width = `${confidencePct}%`;
      });
    });

    // Rating label
    confidenceRating.textContent = `Confidence level: ${data.confidence_label}`;

    // Probabilities
    realProbEl.textContent = `${Math.round(data.real_probability * 100)}%`;
    fakeProbEl.textContent = `${Math.round(data.fake_probability * 100)}%`;

    // Scroll into view
    resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ── Navbar smooth scroll ──────────────────────────────────────
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", (e) => {
      const href = link.getAttribute("href");
      if (href.startsWith("#")) {
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) target.scrollIntoView({ behavior: "smooth" });
      }
    });
  });

  // ── Initial health check (optional — confirms API is up) ──────
  fetch("/health")
    .then((r) => r.json())
    .then((d) => {
      if (!d.model_loaded) {
        console.warn("Model not loaded on server. Predictions may fail.");
      }
    })
    .catch(() => {
      // Silently ignore — API may not be ready during cold start
    });
})();
