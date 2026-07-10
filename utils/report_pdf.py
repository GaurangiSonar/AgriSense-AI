"""PDF report generation for AgriSense AI."""

from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils.display import (
    relevant_citations,
    confidence_bucket,
    confidence_bar,
    format_crop_label,
    format_disease_label,
    format_percent,
    timestamp_label,
)
from utils.economics import build_economic_analysis_items


def generate_pdf_report(state) -> bytes:
    """Generate a polished PDF report from the final workflow state."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="AgriSense AI Report",
        author="AgriSense AI",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AgriSenseTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#f5f2ea"),
        alignment=TA_LEFT,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "AgriSenseSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#c5a15a"),
        spaceBefore=7,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "AgriSenseBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#f0eadc"),
    )
    small_style = ParagraphStyle(
        "AgriSenseSmall",
        parent=body_style,
        fontSize=8.5,
        textColor=colors.HexColor("#b9c3b6"),
    )

    story = []
    story.append(Paragraph("AgriSense AI", title_style))
    story.append(Paragraph("Professional Crop Disease Intelligence Report", body_style))
    story.append(Paragraph(f"Generated: {timestamp_label(state.timestamp) or 'Unknown'}", small_style))
    story.append(Spacer(1, 6))

    story.extend(
        _section_table(
            "Diagnosis",
            [
                ("Crop", format_crop_label(state.crop)),
                ("Disease", format_disease_label(state.disease)),
                ("Confidence", f"{format_percent(state.confidence)} ({confidence_bucket(state.confidence)})"),
                ("Visible Symptoms", ", ".join(state.symptoms) if state.symptoms else "Not specified"),
            ],
            body_style,
            section_style,
        )
    )
    story.extend(
        _section_table(
            "Treatment",
            [
                ("Primary Treatment", state.treatment or "Unavailable"),
                ("Dosage", state.dosage or "Unavailable"),
                ("Frequency", state.frequency or "Unavailable"),
                ("Expected Recovery", state.expected_recovery or "Unavailable"),
                ("Prevention", state.prevention or "Unavailable"),
            ],
            body_style,
            section_style,
        )
    )
    economic_rows = [
        (label, value)
        for label, value, _icon in build_economic_analysis_items(
            treatment_cost=state.treatment_cost,
            estimated_yield_recovered=state.estimated_yield_recovered,
            market_value_recovered=state.market_value_recovered,
            net_profit=state.net_profit,
            roi=state.roi,
            economics_estimated=state.economics_estimated,
        )
    ]
    story.extend(
        _section_table(
            "Economic Analysis",
            economic_rows,
            body_style,
            section_style,
        )
    )
    if state.economics_estimated:
        story.append(
            Paragraph(
                "Market data was unavailable when this report was generated. "
                "Market value, net profit, and ROI are benchmark estimates.",
                body_style,
            )
        )
        story.append(Spacer(1, 4))
    story.append(Paragraph("Evidence", section_style))
    filtered_citations = relevant_citations(state.citations, state.disease, state.crop)
    if filtered_citations:
        evidence_rows = [
            [
                Paragraph("<b>Document</b>", body_style),
                Paragraph("<b>Source</b>", body_style),
                Paragraph("<b>Page</b>", body_style),
                Paragraph("<b>Retrieval Score</b>", body_style),
                Paragraph("<b>Excerpt</b>", body_style),
            ]
        ]
        for citation in filtered_citations:
            evidence_rows.append(
                [
                    Paragraph(str(citation.get("title", citation.get("source", "Document"))), body_style),
                    Paragraph(str(citation.get("source", "Unknown")), body_style),
                    Paragraph(str(citation.get("page", "Unknown")), body_style),
                    Paragraph(str(citation.get("score", "n/a")), body_style),
                    Paragraph(str(citation.get("snippet", "")), body_style),
                ]
            )
        story.append(_styled_table(evidence_rows))
    else:
        story.append(
            Paragraph(
                "No evidence directly matched the diagnosed disease. Related documents were omitted to keep the report accurate.",
                body_style,
            )
        )
    story.append(Spacer(1, 4))

    story.extend(
        _bullet_section(
            "Recommendations",
            [
                f"Disease confidence: {format_percent(state.confidence)} ({confidence_bucket(state.confidence)})",
                f"Treatment choice: {state.treatment or 'Unavailable'}",
                f"Economic reasoning: {state.market_rationale or 'Economic recommendation generated without market analysis.'}",
                f"Memory influence: {state.memory_insight or 'No prior farmer history was available.'}",
            ],
            body_style,
            section_style,
        )
    )

    story.append(Paragraph("Generated by AgriSense AI", small_style))
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


def _section_table(title: str, rows: list[tuple[str, str]], body_style, section_style):
    table_data = [[Paragraph(f"<b>{title}</b>", section_style), ""]]
    for label, value in rows:
        table_data.append([Paragraph(label, body_style), Paragraph(value, body_style)])
    return [Table(table_data, colWidths=[55 * mm, 115 * mm], style=_table_style()) , Spacer(1, 4)]


def _bullet_section(title: str, items: list[str], body_style, section_style):
    flow = [Paragraph(title, section_style)]
    for item in items:
        flow.append(Paragraph(f"• {item}", body_style))
        flow.append(Spacer(1, 2))
    return flow


def _styled_table(data):
    table = Table(data, colWidths=[30 * mm, 30 * mm, 18 * mm, 26 * mm, 80 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17352c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#223126")),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#f0eadc")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#4e623f")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _table_style():
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5f6d2c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#223126")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#f0eadc")),
            ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#4e623f")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#0f1b16"))
    canvas.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#6d8a3c"))
    canvas.rect(0, A4[1] - 18 * mm, A4[0], 18 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#f5f2ea"))
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(doc.leftMargin, A4[1] - 11 * mm, "AgriSense AI")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#b9c3b6"))
    canvas.drawString(doc.leftMargin, 8 * mm, "Generated by AgriSense AI")
    canvas.restoreState()
