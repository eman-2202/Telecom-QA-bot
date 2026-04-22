# ================= CSS =================

CSS = """
/* =========================================================
   🌙 GLOBAL BACKGROUND + FONT
   ========================================================= */

/* Dark background for entire app */
body {
    background: #020617;
}

/* Force modern clean font across whole UI */
body, .gradio-container {
    font-family: "Inter", sans-serif !important;
}


/* =========================================================
   🧠 HEADER (TOP SECTION)
   Logo + Title + Subtitle
   ========================================================= */

/* Layout: horizontal alignment */
.header {
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom:20px;
}

/* 📡 Logo icon */
.logo { 
    font-size:32px; 
}

/* 🏷 Main title */
.title {
    font-size:26px;
    font-weight:bold;
    letter-spacing:2px;
    color:#e2e8f0 !important;
}

/* 📝 Subtitle */
.subtitle {
    color:#94a3b8;
}


/* =========================================================
   🎯 HIGHLIGHT COLORS (Used inside text)
   ========================================================= */

/* Used for PDF mentions */
.highlight-pdf { color: #ef4444; }

/* Used for Excel mentions */
.highlight-excel { color: #22c55e; }

/* Used for Runbook mentions */
.highlight-runbook { color: #3b82f6; }


/* =========================================================
   💡 EXAMPLES SECTION (Fix truncation issue)
   ========================================================= */

/* Allow example buttons to show FULL text (no truncation) */
.gradio-container .examples button {
    white-space: normal !important;   /* allow text wrapping */
    overflow: visible !important;     /* show full content */
    text-overflow: unset !important;

    display: block !important;        /* remove clamp */
    max-height: none !important;

    line-height: 1.5;
    padding: 10px;
}

/* Ensure full width layout */
.gradio-container .examples {
    width: 100%;
}


/* =========================================================
   🚫 REMOVE LOADING ICONS (Cleaner UI)
   ========================================================= */

/* Hide Gradio loading indicators */
[data-testid="status-indicator"],
[data-testid="loading"],
[data-testid="/*progress"] {
    display: none !important;
}

/* Hide floating overlay elements (bottom-right icons) */
div[style*="position: absolute"],
div[style*="position: fixed"] {
    display: none !important;
}


/* =========================================================
   💬 CHATBOT TEXT IMPROVEMENT
   ========================================================= */

/* Improve readability of chatbot messages */
.gr-chatbot {
    font-size: 15px;
    line-height: 1.6;
}


/* =========================================================
   ✨ OPTIONAL SMALL UI ENHANCEMENTS (SAFE)
   ========================================================= */

/* Smooth transitions for buttons/cards */
button {
    transition: all 0.2s ease;
}

/* Slight hover effect (modern feel) */
button:hover {
    opacity: 0.9;
}

/* 🚫 Hide the "Examples" label completely */
.gradio-container .examples .label {
    display: none !important;
}

"""
# ================= HTML =================

HEADER_HTML = """
<div class="header">
    <div class="logo">📡</div>
    <div>
        <div class="title">TELECOM Q&A BOT</div>
        <div class="subtitle">
            Multi-Source RAG over · 3GPP pdfs · KPI Tables · NOC Runbooks
        </div>
    </div>
</div>
"""

WELCOME_HTML = """
<div class="ai-bubble">
System online. I can answer questions from 
<span class="highlight-pdf">3GPP Technical Specification pdfs</span>, your 
<span class="highlight-excel">KPI threshold tables</span>, 
and <span class="highlight-runbook">NOC runbooks</span>.
</div>
"""

EMPTY_SOURCES_HTML = "<p style='color:#aaa;'>Ask a question to see sources</p>"