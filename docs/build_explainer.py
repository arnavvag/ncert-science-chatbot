"""Builds docs/EXPLAINER.pdf (one page, with the request-flow flowchart).

    pip install reportlab
    python docs/build_explainer.py --name "Your Name" --app "https://your-app.streamlit.app" \
        --repo "https://github.com/you/ncert-science-chatbot" \
        --notes "Raised CACHE_SIM_THRESHOLD from 0.75 to 0.80 after testing; ..."
--notes (optional) adds a final bullet to 'What did not work' - put your own real observations there.
"""
import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, Spacer, Table, TableStyle

INDIGO, ORANGE, GREEN, GREY = colors.HexColor("#3b3592"), colors.HexColor("#f5a623"), colors.HexColor("#2e8b57"), colors.HexColor("#555555")
W, H = A4
M = 34


def box(c, x, y, w, h, lines, fill=colors.HexColor("#eeedf8"), stroke=INDIGO, bold_first=True):
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(0.9)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    c.setFillColor(colors.black)
    n = len(lines); lead = 8
    ty = y + h / 2 + (n - 1) * lead / 2 - 2.4
    for i, ln in enumerate(lines):
        c.setFont("Helvetica-Bold" if (i == 0 and bold_first) else "Helvetica", 6.6)
        c.drawCentredString(x + w / 2, ty - i * lead, ln)


def arrow(c, x1, y1, x2, y2, label=None, color=INDIGO, dash=False):
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(0.9)
    c.setDash(2, 2) if dash else c.setDash()
    c.line(x1, y1, x2, y2)
    c.setDash()
    import math
    a = math.atan2(y2 - y1, x2 - x1); s = 4.2
    p = c.beginPath(); p.moveTo(x2, y2)
    p.lineTo(x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4))
    p.lineTo(x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4)); p.close()
    c.drawPath(p, fill=1, stroke=0)
    if label:
        c.setFont("Helvetica-Bold", 6.3); c.setFillColor(color)
        c.drawString((x1 + x2) / 2 + 2, (y1 + y2) / 2 + 2, label)


def flowchart(c, top):
    h1, h2 = 34, 34
    y1 = top - 14 - h1
    y2 = y1 - 40 - h2
    # row 1 (left to right)
    xs = [M, M + 90, M + 212, M + 345, M + 462]
    ws = [70, 100, 112, 100, 62]
    box(c, xs[0], y1, ws[0], h1, ["Student", "message"], fill=colors.HexColor("#fff3dc"), stroke=ORANGE)
    box(c, xs[1], y1, ws[1], h1, ["1. Classify turn", "standalone / follow-up", "meta / unclear"])
    box(c, xs[2], y1, ws[2], h1, ["2. Cache lookup", "local embedding + FAISS", "same context key only"])
    box(c, xs[3], y1, ws[3], h1, ["3. Safety guards", "numbers, units, terms,", "word order, sim >= 0.75"])
    box(c, xs[4], y1, ws[4], h1, ["All guards", "pass?"], fill=colors.HexColor("#fff3dc"), stroke=ORANGE)
    for i in range(4):
        arrow(c, xs[i] + ws[i], y1 + h1 / 2, xs[i + 1], y1 + h1 / 2)
    # row 2
    x2 = [M, M + 100, M + 212, M + 345]
    w2 = [86, 100, 112, 100]
    box(c, x2[0], y2, w2[0], h2, ["Reply", "+ chapter citations", "+ cache_hit, latency"], fill=colors.HexColor("#fff3dc"), stroke=ORANGE)
    box(c, x2[1], y2, w2[1], h2, ["6. Store in cache", "only if grounded and", "cacheable (see below)"])
    box(c, x2[2], y2, w2[2], h2, ["5. Gemini via LangChain", "answers from chunks only", "names chapters used"])
    box(c, x2[3], y2, w2[3], h2, ["4. RAG on NCERT index", "top-5 chunks (FAISS);", "low score => decline"])
    box(c, xs[4] - 8, y2, ws[4] + 8, h2, ["HIT: cached reply", "NO LLM call", "tens of ms"], fill=colors.HexColor("#e3f4ea"), stroke=GREEN)
    arrow(c, xs[4] + ws[4] / 2 + 14, y1, xs[4] + ws[4] / 2 + 14, y2 + h2, "yes", color=GREEN)
    arrow(c, xs[4] + 6, y1, x2[3] + w2[3] - 2, y2 + h2, "no", color=INDIGO)
    arrow(c, x2[3], y2 + h2 / 2, x2[2] + w2[2], y2 + h2 / 2)
    arrow(c, x2[2], y2 + h2 / 2, x2[1] + w2[1], y2 + h2 / 2)
    arrow(c, x2[1], y2 + h2 / 2, x2[0] + w2[0], y2 + h2 / 2)
    # hit path back to reply (dashed, under the row)
    yb = y2 - 8
    c.setStrokeColor(GREEN); c.setLineWidth(0.9); c.setDash(2, 2)
    c.line(xs[4] - 8 + (ws[4] + 8) / 2, y2, xs[4] - 8 + (ws[4] + 8) / 2, yb)
    c.line(xs[4] - 8 + (ws[4] + 8) / 2, yb, x2[0] + w2[0] / 2, yb)
    c.setDash()
    arrow(c, x2[0] + w2[0] / 2, yb, x2[0] + w2[0] / 2, y2, color=GREEN)
    c.setFont("Helvetica-Oblique", 6.4); c.setFillColor(GREY)
    c.drawString(M, yb - 10, "Meta turns (\"explain it more simply\"), unclear turns and turns after an uncached turn skip steps 2-3 and are never stored.")
    return yb - 14


def build(out, name, app, repo, notes):
    c = canvas.Canvas(str(out), pagesize=A4)
    c.setFillColor(INDIGO); c.rect(0, H - 52, W, 52, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 14)
    c.drawString(M, H - 28, "NCERT Class 10 Science Chatbot with Smart Caching - Approach")
    c.setFont("Helvetica", 7.6)
    c.drawString(M, H - 42, f"AI Intern Round 1 (Prepzy.ai)  |  {name}  |  App: {app}  |  Code: {repo}")
    bottom = flowchart(c, H - 58)

    body = ParagraphStyle("b", fontName="Helvetica", fontSize=8.6, leading=11, alignment=TA_LEFT, textColor=colors.black)
    h = ParagraphStyle("h", parent=body, fontName="Helvetica-Bold", fontSize=10, leading=12.5, textColor=INDIGO, spaceBefore=5, spaceAfter=1)
    bl = ParagraphStyle("bl", parent=body, leftIndent=8, bulletIndent=0, spaceAfter=1.5)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.2, leading=10.2)

    def bullets(items):
        return [Paragraph(t, bl, bulletText="•") for t in items]

    story = [
        Paragraph("How the chatbot works", h),
        Paragraph(
            "The NCERT chapter PDFs are cleaned, split into ~900-character chunks tagged with their chapter, embedded with "
            "<b>BAAI/bge-small-en-v1.5</b> (local ONNX, no API cost) and stored in a <b>FAISS</b> index through LangChain. For each "
            "question the top-5 chunks are retrieved; if even the best chunk scores below a threshold the bot politely declines "
            "<i>without calling the LLM</i>. Otherwise a LangChain chain (prompt | ChatOpenAI pointed at Gemini's OpenAI-compatible "
            "endpoint | parser) answers <b>only</b> from those chunks plus recent chat history, may return an out-of-scope sentinel, and ends "
            "with a SOURCES line that is checked against the real chapter list to produce <b>citations</b>. <b>FastAPI</b> exposes "
            "<font face='Courier'>/session</font> and <font face='Courier'>/chat</font>; <b>Streamlit</b> shows chapter, cache-hit flag and latency for every reply "
            "and has a New conversation button.", body),
    ]
    story = story[:2] + [Paragraph("How the cache decides - a hit needs <i>every</i> check to pass", h)] + bullets([
        "<b>Turn type first.</b> <i>standalone</i> (self-contained), <i>contextual</i> (\"what about its laws?\": pronouns, \"and for...\", no topic), "
        "<i>meta</i> (\"explain it more simply\", \"give an example\") or <i>unclear</i> (nothing to refer to).",
        "<b>Candidate:</b> embedding similarity >= 0.75 against cached questions in the <i>same context</i> only.",
        "<b>Guards (deterministic):</b> numbers and units identical and in the same order (R = 20 vs 30 cm; u and f swapped); set of meaningful terms identical after "
        "dropping filler (what / does / mean / define), stemming and unit synonyms, so concave vs convex, NaOH vs NaCl, H2O vs H2O2 all fail; term order identical when "
        "direction words appear (ice <b>to</b> water vs water <b>to</b> ice).",
        "<b>Follow-ups</b> are keyed to the cache entries of the previous two turns, so a follow-up is reused only after the same earlier conversation. "
        "A hit = local embedding + FAISS + guards, then return: <b>no LLM call</b>, tens of ms (target < 500 ms).",
    ])
    cache_col = [Paragraph("<b>What we cache</b>", cell)] + [Paragraph(t, cell, bulletText="•") for t in [
        "Standalone textbook answers whose best chunk scored >= 0.55 (well grounded).",
        "Follow-ups, but only under the exact preceding-turn context.",
        "Numericals: each distinct set of numbers is its own entry.",
    ]]
    never_col = [Paragraph("<b>What we never cache</b>", cell)] + [Paragraph(t, cell, bulletText="•") for t in [
        "Refusals / out-of-scope replies, clarification prompts, LLM errors.",
        "\"Explain simpler / example / summarise\": they depend on this student's last answer and a repeat needs new text.",
        "Any turn that follows an uncached turn (context is unknowable); answers from an older index version (re-ingest invalidates).",
    ]]
    t = Table([[cache_col, never_col]], colWidths=[(W - 2 * M) / 2] * 2)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.6, INDIGO),
                           ("INNERGRID", (0, 0), (-1, -1), 0.4, INDIGO), ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#e3f4ea")),
                           ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#fdeaea")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story += [Paragraph("What is cached and what is not", h), t, Paragraph("What did not work / trade-offs", h)]
    items = [
        "<b>Similarity threshold alone.</b> \"Image by a concave mirror\" and \"...convex mirror\", or R = 20 vs R = 30, differ by one token, so they embed almost identically; "
        "a threshold strict enough to block them also blocks genuine paraphrases. Hence the deterministic guards on top of embeddings.",
        "<b>Resolving follow-ups with an LLM before lookup</b> would turn every follow-up hit into an LLM call, breaking the \"hits never call an LLM\" rule. "
        "Follow-ups are keyed by conversation context instead.",
        "<b>Precision over recall.</b> A paraphrase that adds or swaps a content word (\"refraction of light\") is a miss and gets its own entry: one extra LLM call, never a wrong answer.",
        "<b>Limits:</b> cache is per-instance (SQLite + in-memory FAISS) and resets if the free Streamlit host is rebuilt; English only; thresholds depend on the embedding model.",
    ]
    if notes:
        items.append(f"<b>Observed while testing:</b> {notes}")
    story += bullets(items)

    f = Frame(M, 28, W - 2 * M, bottom - 28, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0)
    f.addFromList(story, c)
    if story:
        raise SystemExit(f"Content does not fit on one page ({len(story)} flowables left) - shorten text or --notes.")
    c.setFont("Helvetica", 6.5); c.setFillColor(GREY)
    c.drawString(M, 16, "Stack: Python, LangChain, FAISS, FastAPI, Streamlit, Gemini (OpenAI-compatible API), FastEmbed.")
    c.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="<Your Name>")
    ap.add_argument("--app", default="<Streamlit app URL>")
    ap.add_argument("--repo", default="<GitHub repo URL>")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default=str(Path(__file__).with_name("EXPLAINER.pdf")))
    a = ap.parse_args()
    build(a.out, a.name, a.app, a.repo, a.notes)
    print("wrote", a.out)
