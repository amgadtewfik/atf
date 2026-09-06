// Pure render/decision logic for the chat stream. No DOM.
// Unit-tested by tests/test_renderer_logic.js (node).

function computeVisibility(state) {
  const { thinkBuf, ansBuf, answerStarted } = state;
  const hasThink = !!thinkBuf && thinkBuf.trim().length > 0;
  const hasAnswer = !!ansBuf && ansBuf.trim().length > 0;
  // If thinking happened, the answer only shows after think-close.
  // With no thinking at all, the answer shows as soon as it streams.
  const showAnswer = hasAnswer && (!hasThink || !!answerStarted);
  return {
    showThink: hasThink,
    showAnswer,
    thinkText: hasThink ? thinkBuf.trim() : "",
    answerText: ansBuf || "",
  };
}

function cursorTarget(state) {
  const v = computeVisibility(state);
  return v.showAnswer ? "answer" : "think";
}

// Rough token estimate (chars/4). Used ONLY for pre-flight meter display;
// the bridge reports exact prompt-token counts via "usage" events.
function estimateTokens(text) {
  return Math.max(1, Math.ceil((text || "").length / 4));
}

// Conversation title from the first user message.
function sessionTitle(text) {
  const t = (text || "").trim().replace(/\s+/g, " ");
  if (!t) return "New chat";
  return t.length > 48 ? t.slice(0, 48) + "…" : t;
}

// Which history turns survive a context budget? Mirrors the bridge's
// newest-first truncation so the UI can show honest drop counts.
function planContext(history, systemTokens, messageTokens, budgetTokens) {
  let used = (systemTokens || 0) + (messageTokens || 0);
  let kept = 0, dropped = 0;
  for (let i = history.length - 1; i >= 0; i--) {
    const t = estimateTokens(history[i].user) + estimateTokens(history[i].assistant || "");
    if (used + t > budgetTokens) { dropped++; continue; }
    used += t; kept++;
  }
  return { kept, dropped, used };
}

if (typeof module !== "undefined") {
  module.exports = { computeVisibility, cursorTarget, estimateTokens, sessionTitle, planContext };
}
