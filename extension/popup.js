/**
 * popup.js — Social Media Mood Analyzer
 * ─────────────────────────────────────────────────────────────────────────────
 * Handles all UI interactions in the extension popup:
 *  • Tab switching
 *  • Page / custom text analysis
 *  • Badge-all-comments trigger
 *  • Result display with animated score bars
 *  • API health check
 */

const API_BASE = "http://localhost:5000";

// ── Colour palette matching content.js ────────────────────────────────────────
const COLOURS = {
  positive: { bar: "#28a745", bg: "#d4edda", border: "#28a745", text: "#155724" },
  negative: { bar: "#dc3545", bg: "#f8d7da", border: "#dc3545", text: "#721c24" },
  neutral:  { bar: "#6c757d", bg: "#e2e3e5", border: "#6c757d", text: "#383d41" },
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const resultCard       = $("result-card");
const resultEmoji      = $("result-emoji");
const resultSentiment  = $("result-sentiment");
const resultConfidence = $("result-confidence");
const scoreBars        = $("score-bars");
const loading          = $("loading");
const errorBox         = $("error-box");
const apiStatus        = $("api-status");
const extractedBox     = $("extracted-text-box");
const extractedText    = $("extracted-text");
const customInput      = $("custom-input");

// ── Helpers ──────────────────────────────────────────────────────────────────
function showLoading()  { loading.classList.remove("hidden"); resultCard.classList.add("hidden"); errorBox.classList.add("hidden"); }
function hideLoading()  { loading.classList.add("hidden"); }
function showError(msg) { errorBox.textContent = "⚠️ " + msg; errorBox.classList.remove("hidden"); resultCard.classList.add("hidden"); }

function showResult(data) {
  hideLoading();
  errorBox.classList.add("hidden");

  const c = COLOURS[data.sentiment] || COLOURS.neutral;

  resultEmoji.textContent     = data.emoji || "🤔";
  resultSentiment.textContent = data.sentiment.toUpperCase();
  resultSentiment.style.color = c.bar;
  resultConfidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  // Style the card border
  resultCard.style.borderLeft = `5px solid ${c.bar}`;
  resultCard.style.background = c.bg;

  // Score bars
  scoreBars.innerHTML = "";
  if (data.scores) {
    for (const [label, score] of Object.entries(data.scores)) {
      const col = COLOURS[label] || COLOURS.neutral;
      const pct = (score * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "score-row";
      row.innerHTML = `
        <span class="score-label">${label}</span>
        <div class="score-bar-track">
          <div class="score-bar-fill" style="width:0%;background:${col.bar}" data-pct="${pct}"></div>
        </div>
        <span class="score-pct">${pct}%</span>
      `;
      scoreBars.appendChild(row);
    }
    // Animate bars after a tiny delay
    setTimeout(() => {
      document.querySelectorAll(".score-bar-fill").forEach(bar => {
        bar.style.width = bar.dataset.pct + "%";
      });
    }, 50);
  }

  resultCard.classList.remove("hidden");
}

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    tab.classList.add("active");
    $("tab-" + tab.dataset.tab).classList.add("active");
    resultCard.classList.add("hidden");
    errorBox.classList.add("hidden");
  });
});

// ── Analyze current page ──────────────────────────────────────────────────────
$("btn-analyze-page").addEventListener("click", async () => {
  showLoading();
  extractedBox.classList.add("hidden");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.tabs.sendMessage(tab.id, { action: "ANALYZE_PAGE" }, response => {
      hideLoading();
      if (chrome.runtime.lastError) {
        showError("Cannot access this page. Try a regular website.");
        return;
      }
      if (response?.error) { showError(response.error); return; }
      if (response?.text) {
        extractedText.textContent = response.text.substring(0, 200) + (response.text.length > 200 ? "…" : "");
        extractedBox.classList.remove("hidden");
      }
      if (response?.result) showResult(response.result);
    });
  } catch (e) { hideLoading(); showError(e.message); }
});

// ── Badge all comments ────────────────────────────────────────────────────────
$("btn-badge-comments").addEventListener("click", async () => {
  showLoading();
  resultCard.classList.add("hidden");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.tabs.sendMessage(tab.id, { action: "ANALYZE_ALL_COMMENTS" }, response => {
      hideLoading();
      if (chrome.runtime.lastError) { showError("Cannot access this page."); return; }
      if (response?.error) { showError(response.error); return; }
      // Show a brief success note
      errorBox.style.background = "#d4edda";
      errorBox.style.color = "#155724";
      errorBox.style.borderColor = "#28a745";
      errorBox.textContent = "✅ Comments have been badged on the page!";
      errorBox.classList.remove("hidden");
      setTimeout(() => { errorBox.classList.add("hidden"); errorBox.style = ""; }, 3000);
    });
  } catch (e) { hideLoading(); showError(e.message); }
});

// ── Analyze custom text ───────────────────────────────────────────────────────
$("btn-analyze-custom").addEventListener("click", async () => {
  const text = customInput.value.trim();
  if (!text) { showError("Please enter some text."); return; }
  showLoading();
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.tabs.sendMessage(tab.id, { action: "ANALYZE_CUSTOM_TEXT", text }, response => {
      hideLoading();
      if (chrome.runtime.lastError) {
        // Fallback: call API directly from popup
        fetch(`${API_BASE}/predict`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ text }),
        })
          .then(r => r.json())
          .then(data => showResult(data))
          .catch(err => showError("API unreachable: " + err.message));
        return;
      }
      if (response?.error) { showError(response.error); return; }
      if (response?.result) showResult(response.result);
    });
  } catch (e) { hideLoading(); showError(e.message); }
});

// ── Allow Enter key in textarea (Shift+Enter = newline) ───────────────────────
customInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("btn-analyze-custom").click();
  }
});

// ── API health check ──────────────────────────────────────────────────────────
async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    const ok  = res.ok;
    apiStatus.textContent = ok ? "⬤" : "⬤";
    apiStatus.style.color = ok ? "#28a745" : "#dc3545";
    apiStatus.title       = ok ? "API Online" : "API Offline";
  } catch (_) {
    apiStatus.textContent = "⬤";
    apiStatus.style.color = "#dc3545";
    apiStatus.title       = "API Offline — start Flask server";
  }
}

checkApiHealth();
setInterval(checkApiHealth, 5000);
