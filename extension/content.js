/**
 * content.js — Social Media Mood Analyzer
 * ─────────────────────────────────────────────────────────────────────────────
 * Injected into every webpage.
 *
 * Responsibilities:
 *  1. Extract visible text from the active page (generic + social-specific).
 *  2. Listen for messages from popup.js requesting analysis.
 *  3. Call Flask API and return results.
 *  4. Optionally inject inline mood badges next to individual comments.
 */

const API_URL = "http://localhost:5000/predict";

// ── Sentiment colour map ──────────────────────────────────────────────────────
const COLOURS = {
  positive: { bg: "#d4edda", border: "#28a745", text: "#155724" },
  negative: { bg: "#f8d7da", border: "#dc3545", text: "#721c24" },
  neutral:  { bg: "#e2e3e5", border: "#6c757d", text: "#383d41" },
};

// ── Platform-specific text selectors ─────────────────────────────────────────
const SELECTORS = {
  "twitter.com":   ["[data-testid='tweetText']", "article"],
  "x.com":         ["[data-testid='tweetText']", "article"],
  "reddit.com":    ["[data-testid='comment']", ".Comment", "p"],
  "web.whatsapp.com": [".message-in .selectable-text", ".message-out .selectable-text"],
  "instagram.com": ["._a9zs", "._a9zr", "ul li span"],
};

/**
 * Extract text from the page.
 * Tries platform-specific selectors first, falls back to body text.
 * Returns the first ~500 chars of combined text.
 */
function extractPageText() {
  const host = window.location.hostname;
  let selectors = null;

  for (const [domain, sel] of Object.entries(SELECTORS)) {
    if (host.includes(domain)) {
      selectors = sel;
      break;
    }
  }

  let collected = [];

  if (selectors) {
    for (const sel of selectors) {
      document.querySelectorAll(sel).forEach(el => {
        const t = el.innerText?.trim();
        if (t && t.length > 5) collected.push(t);
      });
      if (collected.length > 0) break;
    }
  }

  // Generic fallback — grab <p> text
  if (collected.length === 0) {
    document.querySelectorAll("p, span, div").forEach(el => {
      if (el.children.length === 0) {           // leaf nodes only
        const t = el.innerText?.trim();
        if (t && t.length > 20) collected.push(t);
      }
    });
  }

  return collected.slice(0, 10).join(" ").substring(0, 500);
}

/**
 * Call the Flask API with a text string.
 * Returns the parsed JSON response.
 */
async function analyzeText(text) {
  const res = await fetch(API_URL, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

/**
 * Inject a small coloured badge after a given element.
 */
function injectBadge(element, result) {
  const existing = element.parentNode?.querySelector(".mood-badge");
  if (existing) existing.remove();

  const c = COLOURS[result.sentiment] || COLOURS.neutral;
  const badge = document.createElement("span");
  badge.className = "mood-badge";
  badge.style.cssText = `
    display: inline-block;
    margin-left: 8px;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    background: ${c.bg};
    border: 1px solid ${c.border};
    color: ${c.text};
    vertical-align: middle;
    cursor: default;
  `;
  badge.title = `Confidence: ${(result.confidence * 100).toFixed(0)}%`;
  badge.textContent = `${result.emoji} ${result.sentiment} (${(result.confidence * 100).toFixed(0)}%)`;
  element.parentNode?.insertBefore(badge, element.nextSibling);
}

/**
 * Scan all visible comment elements and badge them.
 */
async function analyzeAllComments() {
  const host = window.location.hostname;
  let selectors = SELECTORS["reddit.com"]; // sensible default

  for (const [domain, sel] of Object.entries(SELECTORS)) {
    if (host.includes(domain)) { selectors = sel; break; }
  }

  for (const sel of selectors) {
    const elements = document.querySelectorAll(sel);
    for (const el of elements) {
      const text = el.innerText?.trim();
      if (!text || text.length < 5) continue;
      try {
        const result = await analyzeText(text.substring(0, 300));
        injectBadge(el, result);
      } catch (_) { /* silently skip if API is down */ }
    }
  }
}

// ── Message listener (from popup.js) ─────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {

  if (message.action === "GET_PAGE_TEXT") {
    const text = extractPageText();
    sendResponse({ text });
    return true;
  }

  if (message.action === "ANALYZE_PAGE") {
    const text = extractPageText();
    if (!text) {
      sendResponse({ error: "No readable text found on this page." });
      return true;
    }
    analyzeText(text)
      .then(result => sendResponse({ result, text }))
      .catch(err  => sendResponse({ error: err.message }));
    return true; // keep channel open for async
  }

  if (message.action === "ANALYZE_ALL_COMMENTS") {
    analyzeAllComments()
      .then(() => sendResponse({ ok: true }))
      .catch(err => sendResponse({ error: err.message }));
    return true;
  }

  if (message.action === "ANALYZE_CUSTOM_TEXT") {
    const text = message.text;
    analyzeText(text)
      .then(result => sendResponse({ result }))
      .catch(err  => sendResponse({ error: err.message }));
    return true;
  }
});
