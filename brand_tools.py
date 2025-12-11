import os
import json
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# --------------------------------------------------------------------
# LOAD ENV VARIABLES (for OPENAI_API_KEY from .env)
# --------------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Make sure it is in your .env file or environment variables."
    )

# Single shared LLM for all tools
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.35,
)


def _run_llm(system_prompt: str, user_prompt: str) -> str:
    """Helper to call the chat model."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content


def _build_context_text(answers: Dict[str, Any]) -> str:
    """Turn the UI answers into a clean context block for prompts."""

    # 1) If user pasted a raw brief, use that directly.
    raw_brief = answers.get("raw_brief", "")
    if isinstance(raw_brief, str) and raw_brief.strip():
        return f"""RAW CLIENT BRIEF (user-typed):

{raw_brief.strip()}

------------------------------
Structured intake fields (if any were filled):
Client / Brand: {answers.get("client_name") or "N/A"}
Industry / niche: {answers.get("industry") or "N/A"}
Target audience: {answers.get("target_audience") or "N/A"}
Main goals: {answers.get("goals") or "N/A"}
Brand vibe: {answers.get("brand_vibe") or "N/A"}
Voice & tone: {answers.get("voice_tone") or "N/A"}
Preferred colors: {answers.get("colors") or "N/A"}
Visual keywords / mood: {answers.get("visual_keywords") or "N/A"}
Main platforms: {answers.get("platforms") or "N/A"}
"""

    # 2) Otherwise, fall back to structured fields

    raw_refs = answers.get("reference_links", "")
    if isinstance(raw_refs, str):
        refs_text = raw_refs.strip()
    else:
        refs_text = ""

    uploaded_files = answers.get("uploaded_files", [])
    uploaded_files = uploaded_files or []

    files_text = ""
    if isinstance(uploaded_files, (list, tuple)) and uploaded_files:
        files_text = "Uploaded reference files (filenames only, designer will open locally):\n"
        for name in uploaded_files:
            files_text += f"- {name}\n"

    context = f"""
Client / Brand:
- Name: {answers.get("client_name") or "N/A"}
- Industry / niche: {answers.get("industry") or "N/A"}

Audience & Goals:
- Target audience: {answers.get("target_audience") or "N/A"}
- Main business / brand goals: {answers.get("goals") or "N/A"}

Brand Vibe & Personality:
- Current / desired vibe: {answers.get("brand_vibe") or "N/A"}
- Voice & tone notes: {answers.get("voice_tone") or "N/A"}

Visual Direction:
- Preferred colors or themes: {answers.get("colors") or "N/A"}
- Visual keywords / mood: {answers.get("visual_keywords") or "N/A"}

Platforms:
- Main platforms: {answers.get("platforms") or "N/A"}

References:
- Reference links (Insta, Pinterest, sites):
{refs_text or "None provided."}

{files_text or ""}
"""
    return context.strip()


# --------------------------------------------------------------------
# 1) Brand Discovery Summary
# --------------------------------------------------------------------
def generate_brand_discovery_summary(answers: Dict[str, Any]) -> str:
    context = _build_context_text(answers)

    system_prompt = """
You are a senior brand strategist and creative director.
You turn messy client intake notes into a clear, friendly brand discovery summary.

Write in a way that:
- Feels human, not corporate.
- Is structured with headings and bullet points.
- Can be shared directly with a client as a "Brand Discovery Summary" doc.
- Highlights what you *do* know and what follow-up questions you would ask.

Do NOT invent fake data like revenue or follower counts.
    """.strip()

    user_prompt = f"""
Based on the following intake notes, write a Brand Discovery Summary.

Focus on:
1) Who this brand is.
2) Who they serve.
3) Their goals.
4) Brand personality.
5) Visual direction.
6) Top 5–7 opportunities or recommendations.

INTAKE NOTES:
{context}
""".strip()

    return _run_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------
# 2) Brand Style Guide
# --------------------------------------------------------------------
def generate_brand_style_guide(answers: Dict[str, Any]) -> str:
    context = _build_context_text(answers)

    system_prompt = """
You are a brand designer and art director.

Create a *lite* but practical brand style guide that a designer could use
inside Figma / Illustrator to start designing logos, web, and social graphics.

The guide should have clearly titled sections like:

1. Brand Essence (1–2 sentences)
2. Brand Personality & Voice
3. Logo Direction (concept ideas, not final logo)
4. Color Palette (name + HEX suggestions)
5. Typography Suggestions (display + body fonts, with Google Fonts options)
6. Imagery & Photography Style
7. Iconography & Graphic Elements
8. Do / Don't examples (brief bullets)

Keep it concise but specific. Use markdown headings and bullet lists.
If information is missing, make reasonable suggestions and note them as suggestions.
    """.strip()

    user_prompt = f"""
Using the following context, create a brand style guide document.

CONTEXT:
{context}
""".strip()

    return _run_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------
# 3) 30-Day Content Calendar (JSON)
# --------------------------------------------------------------------
def generate_content_calendar(answers: Dict[str, Any]) -> str:
    context = _build_context_text(answers)

    system_prompt = """
You are a social media strategist for a creative agency.

Create a 30-day content calendar that:
- Focuses on the platforms listed (or default to Instagram + TikTok).
- Mixes value content, behind-the-scenes, promotions, and community engagement.

IMPORTANT:
- Return the result as pure JSON ONLY.
- The JSON must be a list of 30 objects.
- Each object must have exactly these keys:
  "day" (int),
  "platform" (string),
  "post_type" (string),
  "hook" (string),
  "visual_direction" (string),
  "cta" (string).

Do NOT include any extra text before or after the JSON.
Do NOT format it as markdown.
    """.strip()

    user_prompt = f"""
Based on this brand context, generate a 30-day content calendar.

CONTEXT:
{context}
""".strip()

    return _run_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------
# 4) Logo Direction Ideas
# --------------------------------------------------------------------
def generate_logo_directions(answers: Dict[str, Any]) -> str:
    context = _build_context_text(answers)

    system_prompt = """
You are a senior logo designer and creative director.

Create CONCEPT directions for a logo, NOT final artwork.
Think like you're writing notes for yourself (or another designer) before opening Illustrator.

Your output should include:

1. Brand Essence (1–2 sentences)
2. 3–5 Logo Concept Directions
   - For each, describe: idea, symbolism, shapes, and usage hints.
3. Suggested Taglines (5–10 short lines if appropriate)
4. Notes on how the logo should flex:
   - Social media avatar
   - Website header
   - Merch / apparel
   - Favicon / app icon

Write in markdown with headings and bullet points.
Make sure the directions are specific enough that a designer could sketch from them.
    """.strip()

    user_prompt = f"""
Using the following brand intake notes, create logo concept directions.

CONTEXT:
{context}
""".strip()

    return _run_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------
# 5) AI Logo Sketch Kit (concepts + AI prompts)
# --------------------------------------------------------------------
def generate_logo_sketch_kit(answers: Dict[str, Any]) -> str:
    context = _build_context_text(answers)

    system_prompt = """
You are a logo designer who also knows how to write prompts for AI image tools.

Create a "Logo Sketch Kit" that includes:

1. Quick Brand Essence
2. 3–5 Logo Sketch Concepts
   - For each: concept name, rough layout, main shapes, where it works best.
3. AI Moodboard Image Prompts
   - 5–10 prompts written for tools like DALL·E or Midjourney.
   - Each prompt should describe style, lighting, colors, and vibe.
4. Shape / SVG Ideas
   - Suggestions for simple vector shapes that could work well (e.g., thick circle badge, angled rectangle, etc.)

Write in markdown with headings and bullet points.
    """.strip()

    user_prompt = f"""
Using this brand context, create a Logo Sketch Kit.

CONTEXT:
{context}
""".strip()

    return _run_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------
# 6) Website / Landing Page Outline
# --------------------------------------------------------------------
def generate_site_outline(answers: Dict[str, Any]) -> str:
    context = _build_context_text(answers)

    system_prompt = """
You are a UX/UI designer and conversion-focused copywriter.

Your job is to produce:
- A simple sitemap suggestion
- A homepage / landing-page outline
- Copy ideas for hero, features, social proof, and calls-to-action.

Structure your answer as:

1. Recommended Sitemap
   - List of top-level pages + short notes

2. Homepage / Landing Structure (section-by-section)
   For each section, include:
   - Section name
   - Goal / purpose
   - Wireframe-style description (what elements go here)
   - Copy idea(s) for headline + subcopy
   - CTA examples (buttons / links)

3. Extra Ideas
   - Optional sections (FAQs, comparison table, trust badges, etc.)
   - Notes for future iterations (A/B test ideas, mobile-first notes, etc.)

Write in markdown with clear headings and bullet points.
Make it something a designer could turn directly into a Figma wireframe.
    """.strip()

    user_prompt = f"""
Based on this brand and project context, create a website / landing page outline.

CONTEXT:
{context}
""".strip()

    return _run_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------
# 7) Project Summary & Simple Proposal
# --------------------------------------------------------------------
def generate_project_summary_proposal(answers: Dict[str, Any]) -> str:
    context = _build_context_text(answers)

    system_prompt = """
You are a freelance creative director writing a friendly but clear project summary and proposal.

You are NOT writing a legal contract.
Think of this as a one-page Google Doc you can send a client to confirm scope.

Structure your answer as:

1. Project Overview
   - 2–3 paragraphs summarizing the project in plain language.

2. Objectives
   - Bullet list of 3–6 key goals.

3. Proposed Scope of Work
   Group items under headings like:
   - Strategy & Discovery
   - Design & Creative
   - Development / Build (if applicable)
   - Content Support (if applicable)
   Under each, use bullets that start with verbs (e.g., "Design...", "Create...", "Set up...").

4. Deliverables
   - Bullet list of tangible outputs (files, pages, templates, etc.).

5. Timeline & Phases (high-level)
   - Phase 1, Phase 2, Phase 3 (with short descriptions + rough durations)

6. Assumptions & Notes
   - Things that are included
   - Things that are explicitly out-of-scope (if relevant)
   - What you need from the client

Write in markdown. Keep the tone warm, clear, and professional.
    """.strip()

    user_prompt = f"""
Using these notes, create a project summary and simple proposal outline.

CONTEXT:
{context}
""".strip()

    return _run_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------
# 8) Color Palette Generator  (no JSON section)
# --------------------------------------------------------------------
def generate_color_palette(answers: Dict[str, Any]) -> str:
    context = _build_context_text(answers)

    system_prompt = """
You are a brand designer and color specialist.

Create a color system for this brand.

Include:

1. Palette Overview
   - Short description of the vibe and how color supports it.

2. Core Palette
   - 3–5 primary/secondary colors in a markdown table with columns:
     Name | Role | HEX

3. Neutrals
   - 3–5 neutrals (backgrounds, surfaces, text) in a table:
     Name | Role | HEX

4. Accent / Utility Colors (if appropriate)
   - e.g., success, warning, error, info.

5. Gradient Suggestions
   - 2–4 gradient ideas, format like:
     Name, HEX start, HEX end, angle or direction.

Keep everything consistent and ready to be copied into design tools.
Do NOT include any JSON, code blocks, or a section called "Palette JSON".
    """.strip()

    user_prompt = f"""
Based on this brand context, create a color palette system.

CONTEXT:
{context}
""".strip()

    return _run_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------
# 9) Brand Voice Generator
# --------------------------------------------------------------------
def generate_brand_voice(answers: Dict[str, Any]) -> str:
    context = _build_context_text(answers)

    system_prompt = """
You are a copy director creating a brand voice guide.

Create:

1. Brand Voice Overview (1–2 paragraphs)
2. Voice Pillars
   - 3–5 key traits with descriptions.
3. Do / Don't Guidelines
   - Bullets for how to sound vs what to avoid.
4. Channel Examples
   - Email example (short).
   - Instagram caption example.
   - Website hero + subheadline example.
   - Optional: short "About" intro paragraph.

Write in markdown. Keep it practical and easy for a junior writer to follow.
    """.strip()

    user_prompt = f"""
Based on this brand intake, create a brand voice guide.

CONTEXT:
{context}
""".strip()

    return _run_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------
# 10) Proposal → Invoice Outline
# --------------------------------------------------------------------
def generate_invoice_outline(answers: Dict[str, Any]) -> str:
    context = _build_context_text(answers)

    system_prompt = """
You are a freelance designer turning a scope into an invoice-style outline.

Create:

1. Invoice Header Example
   - What info should appear (client, your studio, dates, etc.).

2. Line Items Table (markdown)
   - Columns: Item, Description, Qty, Rate, Subtotal.
   - Use 3–7 sample line items based on the work implied by the context.

3. Totals Summary
   - Subtotal, tax (placeholder), total.

4. Payment Terms & Notes
   - e.g., deposit %, due dates, late fees, what is included/excluded.

This is NOT a legal or tax document, just a structured outline a designer can paste into an invoicing tool.
    """.strip()

    user_prompt = f"""
Using the project context below, create an invoice-style outline.

CONTEXT:
{context}
""".strip()

    return _run_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------
# 11) Domain Name + Tagline Ideas
# --------------------------------------------------------------------
def generate_domain_and_taglines(answers: Dict[str, Any]) -> str:
    context = _build_context_text(answers)

    system_prompt = """
You are a naming and tagline specialist.

Create:

1. Naming Direction Notes
   - 1–2 paragraphs about what the name should communicate.

2. Domain Ideas
   - List 10–20 domain ideas (with .com priority, but you can suggest .studio, .co, etc.).
   - Note if any look particularly strong.

3. Tagline Ideas
   - 10–20 short taglines that could appear under the logo or hero section.

Keep the list scannable with bullets. Assume the client will check availability themselves.
    """.strip()

    user_prompt = f"""
Based on this brand context, suggest domain names and taglines.

CONTEXT:
{context}
""".strip()

    return _run_llm(system_prompt, user_prompt)


# --------------------------------------------------------------------
# 12) Auto-Fill from Brief: parse brief into fields
# --------------------------------------------------------------------
def parse_brief_to_fields(raw_brief: str) -> Dict[str, str]:
    """
    Use the LLM to extract structured fields from a free-form brief.
    Returns a dict with keys matching the UI fields.
    """
    system_prompt = """
You are a helpful assistant that turns a messy project brief into structured fields.

Return ONLY valid JSON with this exact structure:

{
  "client_name": "",
  "industry": "",
  "target_audience": "",
  "goals": "",
  "brand_vibe": "",
  "voice_tone": "",
  "colors": "",
  "visual_keywords": "",
  "platforms": "",
  "reference_links": ""
}

Use empty strings ("") if something is missing or unclear.
Do NOT add any extra keys or commentary.
    """.strip()

    user_prompt = f"""
Parse this brief into structured fields.

BRIEF:
{raw_brief}
""".strip()

    try:
        text = _run_llm(system_prompt, user_prompt)
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback: return empty fields
    return {
        "client_name": "",
        "industry": "",
        "target_audience": "",
        "goals": "",
        "brand_vibe": "",
        "voice_tone": "",
        "colors": "",
        "visual_keywords": "",
        "platforms": "",
        "reference_links": "",
    }
