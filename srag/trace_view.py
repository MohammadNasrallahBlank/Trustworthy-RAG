"""Render a RAGState's decision trace as HTML — the heart of the live demo.

The whole point of this architecture is that you can *see why* it did what it
did. `render_trace_html` turns the structured `state.trace` into a readable
timeline: planning, retrieval, the verifier's diagnosis, each targeted
correction, and the final answer or abstention — colour-coded by diagnosis and
action. Pure function, no dependencies, so it is unit-tested without a browser.
"""

from __future__ import annotations

import html

from .state import RAGState

_DIAGNOSIS_COLOR = {
    "pass": "#16a34a", "retrieval_fault": "#dc2626",
    "generation_fault": "#d97706", "planning_fault": "#ca8a04",
    "conflict": "#7c3aed",
}
_STATUS_COLOR = {"answered": "#16a34a", "hedged": "#d97706", "abstained": "#6b7280"}
_ACTION_COLOR = {
    "fill-and-retrieve": "#ca8a04", "reformulate": "#dc2626", "hyde": "#dc2626",
    "web": "#dc2626", "regenerate": "#d97706", "resolve-conflict": "#7c3aed",
}


def _chip(text, color):
    return (f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
            f'background:{color};color:white;font-size:12px;font-weight:600">'
            f'{html.escape(str(text))}</span>')


def _card(inner, border="#e5e7eb"):
    return (f'<div style="border:1px solid {border};border-radius:8px;padding:10px 12px;'
            f'margin:6px 0;background:#fff">{inner}</div>')


def render_trace_html(state: RAGState) -> str:
    parts: list[str] = ['<div style="font-family:ui-sans-serif,system-ui,sans-serif;'
                        'color:#111827;max-width:860px">']

    # Header: question + final status.
    status = state.answer_status
    parts.append(_card(
        f'<div style="font-size:13px;color:#6b7280">QUESTION</div>'
        f'<div style="font-size:16px;font-weight:600;margin:2px 0 8px">'
        f'{html.escape(state.question)}</div>'
        f'{_chip(status.upper(), _STATUS_COLOR.get(status, "#6b7280"))} &nbsp;'
        f'{_chip("type: " + state.question_type, "#374151")} &nbsp;'
        f'{_chip("corrections: " + str(state.correction_count), "#374151")} &nbsp;'
        f'{_chip("confidence: " + (f"{state.confidence:.2f}" if state.confidence is not None else "-"), "#374151")}',
        border="#111827"))

    # Final answer / abstention message.
    answer_html = html.escape(state.final_answer) if state.final_answer else "<i>(abstained)</i>"
    cites = " ".join(_chip(c, "#2563eb") for c in state.citations)
    parts.append(_card(
        f'<div style="font-size:13px;color:#6b7280">FINAL</div>'
        f'<div style="font-size:18px;font-weight:700;margin:2px 0">{answer_html}</div>'
        + (f'<div style="margin-top:4px">{cites}</div>' if cites else ""),
        border=_STATUS_COLOR.get(status, "#6b7280")))

    # Timeline of decisions.
    parts.append('<div style="font-size:13px;color:#6b7280;margin:12px 0 2px">DECISION TRACE</div>')
    for e in state.trace:
        ev = e.get("event")
        if ev == "plan":
            subs = "".join(f"<li>{html.escape(s)}</li>" for s in e.get("sub_queries", []))
            inner = (f'<b>plan</b> &nbsp;{_chip(e.get("question_type", ""), "#374151")}'
                     f'<ul style="margin:6px 0 0 18px;padding:0">{subs}</ul>')
            parts.append(_card(inner))
        elif ev == "retrieve":
            inner = (f'<b>retrieve</b> &nbsp;hop {html.escape(str(e.get("hop")))} — '
                     f'{e.get("new_chunks", 0)} new of {e.get("candidates", 0)} candidates')
            parts.append(_card(inner))
        elif ev == "rerank":
            inner = (f'<b>rerank</b> &nbsp;kept {e.get("kept", 0)} · '
                     f'top={e.get("top", 0):.3f} margin={e.get("margin", 0):.3f}')
            parts.append(_card(inner))
        elif ev == "generate":
            inner = (f'<b>generate</b> &nbsp;answer=<code>{html.escape(str(e.get("answer", "")))}</code> · '
                     f'{e.get("n_claims", 0)} claims · answerable={e.get("answerable")}'
                     + (' · <i>parametric</i>' if e.get("parametric") else ''))
            parts.append(_card(inner))
        elif ev == "verify":
            diag = e.get("diagnosis", "")
            color = _DIAGNOSIS_COLOR.get(diag, "#374151")
            extra = ""
            if e.get("failing_hops"):
                extra += f' · failing: {", ".join(e["failing_hops"])}'
            if e.get("unsupported_claims"):
                extra += f' · {len(e["unsupported_claims"])} unsupported'
            inner = (f'<b>verify</b> &nbsp;{_chip(diag, color)} '
                     f'confidence={e.get("confidence", 0):.2f}{html.escape(extra)}')
            parts.append(_card(inner, border=color))
        elif ev == "correct":
            action = e.get("action", "")
            color = _ACTION_COLOR.get(action, "#374151")
            inner = (f'<b>correct</b> &nbsp;{_chip(action, color)} '
                     f'(diagnosis: {html.escape(str(e.get("diagnosis")))}) · '
                     f'+{e.get("new_chunks", 0)} new chunks · pass {e.get("pass_no")}')
            parts.append(_card(inner, border=color))
        elif ev == "control":
            inner = f'<b>control</b> &nbsp;{html.escape(str(e.get("reason")))}'
            parts.append(_card(inner))
        elif ev == "finalize":
            inner = f'<b>finalize</b> &nbsp;{_chip(e.get("status", status), _STATUS_COLOR.get(e.get("status", status), "#6b7280"))}'
            parts.append(_card(inner))
        elif ev == "gate":
            inner = f'<b>gate</b> &nbsp;retrieve={e.get("retrieve")} ({html.escape(str(e.get("reason")))})'
            parts.append(_card(inner))
        elif ev == "error":
            parts.append(_card(f'<b>error</b> {html.escape(str(e.get("error")))}', border="#dc2626"))

    # Evidence.
    if state.evidence:
        parts.append('<div style="font-size:13px;color:#6b7280;margin:12px 0 2px">EVIDENCE (reranked)</div>')
        for c in state.evidence[:8]:
            snip = html.escape((c.text[:200] + "…") if len(c.text) > 200 else c.text)
            parts.append(_card(
                f'{_chip(c.id, "#2563eb")} <span style="color:#6b7280">{html.escape(c.source)}</span> '
                f'· rerank={c.rerank_score:.3f}<div style="margin-top:4px;font-size:14px">{snip}</div>'))

    parts.append("</div>")
    return "".join(parts)


def render_trace_markdown(state: RAGState) -> str:
    """A GitHub-friendly Markdown rendering of the decision trace."""
    lines = [f"**Q: {state.question}**", "",
             f"- **type:** {state.question_type} · "
             f"**status:** `{state.answer_status}` · "
             f"**corrections:** {state.correction_count} · "
             f"**confidence:** "
             f"{state.confidence:.2f}" if state.confidence is not None else "-"]
    lines.append("")
    lines.append("| step | detail |")
    lines.append("|---|---|")
    for e in state.trace:
        ev = e.get("event")
        if ev == "plan":
            d = f"type={e.get('question_type')}; hops: " + " / ".join(e.get("sub_queries", []))
        elif ev == "retrieve":
            d = f"hop {e.get('hop')}: {e.get('new_chunks', 0)} new / {e.get('candidates', 0)} candidates"
        elif ev == "rerank":
            d = f"kept {e.get('kept', 0)}, top={e.get('top', 0):.3f}, margin={e.get('margin', 0):.3f}"
        elif ev == "generate":
            d = f"answer=`{e.get('answer', '')}`, {e.get('n_claims', 0)} claims, answerable={e.get('answerable')}"
        elif ev == "verify":
            d = f"**diagnosis=`{e.get('diagnosis')}`**, confidence={e.get('confidence', 0):.2f}"
            if e.get("failing_hops"):
                d += f", failing={e['failing_hops']}"
        elif ev == "correct":
            d = f"**action=`{e.get('action')}`** (for {e.get('diagnosis')}), +{e.get('new_chunks', 0)} chunks"
        elif ev == "control":
            d = f"stop: {e.get('reason')}"
        elif ev == "finalize":
            d = f"status={e.get('status', state.answer_status)}"
        elif ev == "gate":
            d = f"retrieve={e.get('retrieve')} ({e.get('reason')})"
        elif ev == "error":
            d = f"error: {e.get('error')}"
        else:
            d = str({k: v for k, v in e.items() if k != "event"})
        lines.append(f"| `{ev}` | {d} |")
    final = state.final_answer or "_(abstained)_"
    cites = ", ".join(f"`{c}`" for c in state.citations)
    lines.append("")
    lines.append(f"**Final:** {final}" + (f"  — citations: {cites}" if cites else ""))
    return "\n".join(lines)
