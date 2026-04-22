# ====================== app.py ======================
import gradio as gr

from src.query_data import query_rag                          # RAG pipeline: question → answer + sources
from config.settings import FILTER_OPTIONS, EXAMPLES         # dropdown options & example questions
from styles.ui import CSS, HEADER_HTML, WELCOME_HTML, EMPTY_SOURCES_HTML  # styling assets


# ══════════════════════════════════════════════════════════════════════════════
# CHAT PIPELINE
# Two functions chained on every Send / Enter event:
#   Step 1 → add_user_message : shows user bubble instantly (queue=False)
#   Step 2 → generate_answer  : runs RAG, updates chat + sources panel
# ══════════════════════════════════════════════════════════════════════════════

def add_user_message(message, history):
    """Append user message to chat history and pass it to the next step."""
    history = list(history or [])
    history.append({"role": "user", "content": message})
    return history, message  # message is passed through so .then() can read it


def generate_answer(history, message, filter_choice):
    """Run the full RAG pipeline and append the assistant answer."""
    # FILTER_OPTIONS maps dropdown label → doc_type string  e.g. "PDF Only" → "pdf"
    answer, sources = query_rag(message, FILTER_OPTIONS[filter_choice])

    history.append({"role": "assistant", "content": answer})
    return history, format_sources_cards(sources)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE CARD RENDERER
# Converts the sources list from query_rag() into colour-coded HTML cards.
# ══════════════════════════════════════════════════════════════════════════════

def format_sources_cards(sources):
    """Render retrieved sources as styled HTML cards for the right panel."""
    if not sources:
        return "<p style='color:#aaa;'>No sources yet</p>"

    # (border colour, display label, icon) — add new file types here
    type_map = {
        "pdf":     ("#ef4444", "PDF",     "📕"),
        "xls":     ("#22c55e", "EXCEL",   "📊"),
        "runbook": ("#3b82f6", "RUNBOOK", "📘"),
        "txt":     ("#3b82f6", "RUNBOOK", "📘"),
    }

    html = ""
    for i, s in enumerate(sources, 1):
        filename   = s["source"].replace("\\", "/").split("/")[-1]
        fname_lower = filename.lower()

        color, label, icon = next(
            (v for k, v in type_map.items() if k in fname_lower),
            ("#64748b", "DOC", "📄")   # fallback for unknown types
        )

        location = s.get("location", "N/A")  # "Page: N" | "Row: N" | "Proc: N"

        html += f"""
        <div style="
            border:1px solid {color};
            border-left:6px solid {color};
            border-radius:12px;
            padding:12px;
            margin-bottom:12px;
            background:#020617;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <b style="color:#e2e8f0;">{icon} {filename}</b>
                <span style="
                    background:{color}; color:white;
                    padding:2px 8px; border-radius:8px; font-size:12px;
                ">#{i}</span>
            </div>
            <div style="margin-top:6px; font-size:13px;">
                <span style="color:{color}; font-weight:bold;">{label}</span> |
                {location} |
                Score: {round(s['score'], 3)}
            </div>
        </div>
        """

    return html


# ══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# Header → Row → Left column (chat) + Right column (sources + examples)
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks() as demo:

    # Header banner
    gr.Markdown("""
    <div class="header">
        <div class="logo">📡</div>
        <div>
            <div class="title">TELECOM Q&A BOT</div>
            <div class="subtitle">
                Multi-Source RAG over · 3GPP pdfs · KPI Tables · NOC Runbooks
            </div>
        </div>
    </div>
    """)

    with gr.Row():

        # ── Left column: chatbot + input ──────────────────────────────────────
        with gr.Column(scale=7):

            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": """
<div class="ai-bubble">
System online. I can answer questions from 
<span class="highlight-pdf">3GPP Technical Specification pdfs</span>, your 
<span class="highlight-excel">KPI threshold tables</span>, 
and <span class="highlight-runbook">NOC runbooks</span>.
</div>
"""}],
                height=450,
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Ask a telecom question",
                    placeholder="What is the definition of PRB Utilization and its warning threshold?",
                    lines=2,
                    scale=6
                )

                # Choices & default come from config/settings.py → FILTER_OPTIONS
                filter_dropdown = gr.Dropdown(
                    choices=list(FILTER_OPTIONS.keys()),
                    value="All Sources",
                    label="Document Type",
                    scale=2
                )

                with gr.Column(scale=1):
                    send_btn  = gr.Button("➤ Send")
                    clear_btn = gr.Button("❌ Clear")

        # ── Right column: sources panel + examples ────────────────────────────
        with gr.Column(scale=3):

            gr.Markdown("### 📚 Retrieved Sources")
            sources_box = gr.HTML(
                value="<p style='color:#aaa;'>Ask a question to see sources</p>"
            )

            gr.Markdown("### 💡 Example Questions")
            # Examples list comes from config/settings.py → EXAMPLES
            gr.Examples(examples=EXAMPLES, inputs=msg, label="click to fill")


# ══════════════════════════════════════════════════════════════════════════════
# EVENT WIRING
# add_user_message runs instantly (queue=False) then generate_answer follows.
# Both Send button and Enter key trigger the same two-step chain.
# ══════════════════════════════════════════════════════════════════════════════

    # Send button
    send_btn.click(
        add_user_message,
        inputs=[msg, chatbot], outputs=[chatbot, msg], queue=False
    ).then(
        generate_answer,
        inputs=[chatbot, msg, filter_dropdown], outputs=[chatbot, sources_box]
    )

    # Enter key — same chain as Send
    msg.submit(
        add_user_message,
        inputs=[msg, chatbot], outputs=[chatbot, msg], queue=False
    ).then(
        generate_answer,
        inputs=[chatbot, msg, filter_dropdown], outputs=[chatbot, sources_box]
    )

    # Clear — resets chat, textbox, and sources panel
    clear_btn.click(
        lambda: ([], "", "<p style='color:#aaa;'>Sources cleared</p>"),
        inputs=None, outputs=[chatbot, msg, sources_box], queue=False
    )


# ══════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)