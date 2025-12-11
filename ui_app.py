import base64
import io
import json
import re
import zipfile
from typing import List

import pandas as pd
import streamlit as st
from fpdf import FPDF
from openai import OpenAI

from brand_tools import (
    generate_brand_discovery_summary,
    generate_brand_style_guide,
    generate_content_calendar,
    generate_logo_directions,
    generate_logo_sketch_kit,
    generate_site_outline,
    generate_project_summary_proposal,
    generate_color_palette,
    generate_brand_voice,
    generate_invoice_outline,
    generate_domain_and_taglines,
    parse_brief_to_fields,
)

client = OpenAI()

# --------------------------------------------------------------------
# HELPER: SIMPLE PDF MAKER (with Unicode sanitizing)
# --------------------------------------------------------------------
def make_pdf(title: str, body: str) -> bytes:
    """
    Turn plain text into a basic PDF using FPDF.
    Sanitizes Unicode so latin-1 PDF doesn't explode.
    Returns raw PDF bytes suitable for st.download_button.
    """

    def sanitize(text: str) -> str:
        # Replace “pretty” punctuation with plain ASCII equivalents
        replacements = {
            "–": "-",
            "—": "-",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "…": "...",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)

        # Best-effort: drop any remaining characters not in latin-1
        return text.encode("latin-1", "ignore").decode("latin-1")

    safe_title = sanitize(title)
    safe_body = sanitize(body)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.multi_cell(0, 10, safe_title)
    pdf.ln(4)

    # Body
    pdf.set_font("Arial", "", 11)
    for line in safe_body.split("\n"):
        pdf.multi_cell(0, 6, line if line.strip() else " ")

    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        pdf_bytes = raw.encode("latin-1", "ignore")
    else:
        pdf_bytes = raw

    return pdf_bytes


# --------------------------------------------------------------------
# HELPER: Generate logo moodboard images (AI Logo Sketch Kit)
# --------------------------------------------------------------------
def generate_logo_moodboard_images(answers: dict, n: int = 3) -> List[bytes]:
    """
    Use OpenAI images API to generate logo moodboard images.
    Returns a list of raw image bytes (PNG).
    """
    client_name = answers.get("client_name") or "the brand"
    industry = answers.get("industry") or ""
    vibe = answers.get("brand_vibe") or ""
    colors = answers.get("colors") or ""
    visuals = answers.get("visual_keywords") or ""

    prompt = (
        f"Logo moodboard for {client_name} {('in ' + industry) if industry else ''}. "
        f"Brand vibe: {vibe}. Colors: {colors}. Visual keywords: {visuals}. "
        "Show 2D flat logo explorations, clean vector style, centered composition."
    )

    resp = client.images.generate(
    model="gpt-image-1",
    prompt=prompt,
    n=n,
    size="1024x1024",
)

    images: List[bytes] = []
    for item in resp.data:
        img_bytes = base64.b64decode(item.b64_json)
        images.append(img_bytes)

    return images


# --------------------------------------------------------------------
# HELPER: Render color swatches from markdown tables
# --------------------------------------------------------------------
def render_color_swatches(markdown_text: str) -> None:
    """
    Look for markdown tables with a HEX column and render visual swatches.
    """
    pattern = r"\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|\s*#?([0-9A-Fa-f]{6})\s*\|"
    matches = re.findall(pattern, markdown_text)
    if not matches:
        return

    st.markdown("#### 🎨 Quick Color Swatches")
    for name, role, hex_code in matches:
        hex_code = hex_code.upper()
        st.markdown(
            f"""
<div style="display:flex;align-items:center;margin-bottom:4px;">
  <div style="width:32px;height:18px;background-color:#{hex_code};border-radius:4px;border:1px solid #222;margin-right:8px;"></div>
  <span style="font-size:0.9rem;">{name.strip()} – {role.strip()} – #{hex_code}</span>
</div>
""",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------
# PAGE CONFIG & BASIC STYLING
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Tru Designs Creative Assistant",
    page_icon="👩‍🎨",
    layout="wide",
)

PINK = "#ff3e8e"

st.markdown(
    f"""
    <style>
    .tru-title-bar {{
        border-bottom: 2px solid {PINK};
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }}
    .tru-footer {{
        border-top: 1px solid #333;
        margin-top: 2.5rem;
        padding-top: 0.75rem;
        font-size: 0.85rem;
        text-align: center;
        color: #bbbbbb;
    }}
    .tru-highlight {{
        color: {PINK};
        font-weight: 600;
    }}

    /* Make EVERY button Tru Designs pink (includes Generate) */
    button {{
        background-color: {PINK} !important;
        color: #ffffff !important;
        border: 1px solid {PINK} !important;
        border-radius: 999px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
    }}

    /* Hover effect for all buttons */
    button:hover {{
        background-color: #ff5aa4 !important;
        border-color: #ff5aa4 !important;
        color: #ffffff !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# SIDEBAR – MODE PICKER
# --------------------------------------------------------------------
st.sidebar.title("👩‍🎨 Tru Designs Creative Assistant")
mode = st.sidebar.radio(
    "What do you want to generate?",
    (
        "Brand Discovery Summary",
        "Brand Style Guide",
        "Logo Direction Ideas",
        "AI Logo Sketch Kit",
        "Website / Landing Page Outline",
        "Project Summary & Proposal",
        "Color Palette Generator",
        "Brand Voice Guide",
        "Proposal → Invoice Outline",
        "Domain & Tagline Ideas",
        "30-Day Content Calendar",
    ),
)

st.sidebar.markdown(
    """
This is your **Tru Designs** mini-studio:

- Collect brand info  
- Turn it into strategy  
- Save time on every new client 🔁
"""
)

# --------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------
st.markdown(
    f"""
<div class="tru-title-bar">
  <h1>👩‍🎨 Tru Designs Creative Assistant</h1>
  <p style="color:#cccccc;">
   I'm a mini creative agent built to help you run brand discovery, style guides,
    logo concepts, site outlines, and content plans — created by
    <span class="tru-highlight">Trish Bellardine | Tru Designs</span>.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

if mode == "Brand Discovery Summary":
    st.subheader("📝 Brand Discovery Session")
    st.caption(
        "First pick what you’d like me to generate in the left menu, then paste a brief "
        "or fill out the intake form below and hit Generate. Voilà — you’ve got an organized "
        "brand discovery document to start building with."
    )
elif mode == "Brand Style Guide":
    st.subheader("🎨 Brand Style Guide Generator")
    st.caption("Get a lite style guide you can refine in Figma / Illustrator.")
elif mode == "Logo Direction Ideas":
    st.subheader("🔖 Logo Direction Ideas")
    st.caption("Strategic logo concept directions and taglines.")
elif mode == "AI Logo Sketch Kit":
    st.subheader("✏️ AI Logo Sketch Kit")
    st.caption("Logo sketch concepts + AI moodboard prompts + auto-generated images.")
elif mode == "Website / Landing Page Outline":
    st.subheader("🕸️ Website / Landing Page Outline")
    st.caption("Generate sitemap ideas and a section-by-section homepage plan.")
elif mode == "Project Summary & Proposal":
    st.subheader("📑 Project Summary & Simple Proposal")
    st.caption("Turn notes into a friendly project overview and scope outline.")
elif mode == "Color Palette Generator":
    st.subheader("🎯 Color Palette Generator")
    st.caption("Generate HEX palettes and gradients.")
elif mode == "Brand Voice Guide":
    st.subheader("🗣️ Brand Voice Guide")
    st.caption("Define voice pillars, do/don'ts, and channel examples.")
elif mode == "Proposal → Invoice Outline":
    st.subheader("💸 Proposal → Invoice Outline")
    st.caption("Turn scope into a simple invoice-style breakdown.")
elif mode == "Domain & Tagline Ideas":
    st.subheader("🌐 Domain & Tagline Ideas")
    st.caption("Name and tagline options when clients come in blank.")
else:
    st.subheader("📅 30-Day Content Calendar")
    st.caption("Generate a month of content ideas tailored to the brand vibe.")

# --------------------------------------------------------------------
# INTAKE: BRIEF + AUTO-FILL (OUTSIDE FORM)
# --------------------------------------------------------------------
st.markdown("### 📝 Intake")

raw_brief = st.text_area(
    "Optional free-form brief (paste client notes here)",
    key="raw_brief_field",
    placeholder="Example: I’m launching a new art merch brand called SpotaSwag...",
    height=200,
)
st.markdown(
    "_You can still fill fields below. The brief above will lead the AI if it’s filled._"
)

if st.button("✨ Auto-fill fields from brief"):
    if not raw_brief.strip():
        st.warning("Please paste a brief first so I can auto-fill.")
    else:
        data = parse_brief_to_fields(raw_brief)
        # Set widget state BEFORE they are instantiated
        st.session_state["client_name_field"] = data.get("client_name", "")
        st.session_state["industry_field"] = data.get("industry", "")
        st.session_state["target_audience_field"] = data.get("target_audience", "")
        st.session_state["goals_field"] = data.get("goals", "")
        st.session_state["brand_vibe_field"] = data.get("brand_vibe", "")
        st.session_state["voice_tone_field"] = data.get("voice_tone", "")
        st.session_state["colors_field"] = data.get("colors", "")
        st.session_state["visual_keywords_field"] = data.get("visual_keywords", "")
        st.session_state["platforms_field"] = data.get("platforms", "")
        st.session_state["reference_links_field"] = data.get("reference_links", "")
        st.success("Fields auto-filled from brief. You can tweak them below, then hit Generate.")

# --------------------------------------------------------------------
# SHARED FORM FOR STRUCTURED FIELDS
# --------------------------------------------------------------------
with st.form("brand_form"):
    col1, col2 = st.columns(2)

    with col1:
        client_name = st.text_input(
            "Client / Brand Name",
            key="client_name_field",
            placeholder="e.g., GlowNest Cosmetics",
        )
        industry = st.text_input(
            "Industry / Niche",
            key="industry_field",
            placeholder="e.g., Skincare, SaaS, Local café",
        )
        target_audience = st.text_area(
            "Target Audience",
            key="target_audience_field",
            placeholder="Who are they trying to reach? Demographics, psychographics, pain points.",
            height=80,
        )
        goals = st.text_area(
            "Main Goals",
            key="goals_field",
            placeholder="e.g., Grow Instagram, launch new product line, clarify brand, drive bookings.",
            height=80,
        )

    with col2:
        brand_vibe = st.text_area(
            "Brand Vibe / Personality",
            key="brand_vibe_field",
            placeholder="e.g., Minimal, bold, playful, luxe, street, warm, etc.",
            height=80,
        )
        voice_tone = st.text_area(
            "Voice & Tone",
            key="voice_tone_field",
            placeholder="e.g., Friendly and honest, a little sarcastic, educational but simple.",
            height=80,
        )
        colors = st.text_input(
            "Color Preferences (optional)",
            key="colors_field",
            placeholder="e.g., Hot pink, black, off-white, muted teal. Or HEX codes.",
        )
        visual_keywords = st.text_input(
            "Visual Keywords / Mood (optional)",
            key="visual_keywords_field",
            placeholder="e.g., retro-future, neon, graffiti, beachy, soft glow, etc.",
        )

    platforms = st.text_input(
        "Main Platforms (for content calendar)",
        key="platforms_field",
        placeholder="e.g., Instagram, TikTok, YouTube, Email newsletter",
    )

    reference_links = st.text_area(
        "Reference Links (Insta / Pinterest / sites)",
        key="reference_links_field",
        placeholder="Paste any inspiration links here, one per line.",
        height=90,
    )

    uploaded_files = st.file_uploader(
        "Upload reference images / logos / PDFs (optional – filenames only go to the AI)",
        type=["png", "jpg", "jpeg", "webp", "svg", "pdf"],
        accept_multiple_files=True,
    )

    button_label = {
        "Brand Discovery Summary": "Generate Brand Discovery Summary",
        "Brand Style Guide": "Generate Brand Style Guide",
        "Logo Direction Ideas": "Generate Logo Directions",
        "AI Logo Sketch Kit": "Generate Logo Sketch Kit",
        "Website / Landing Page Outline": "Generate Site Outline",
        "Project Summary & Proposal": "Generate Project Summary & Proposal",
        "Color Palette Generator": "Generate Color Palette",
        "Brand Voice Guide": "Generate Brand Voice Guide",
        "Proposal → Invoice Outline": "Generate Invoice Outline",
        "Domain & Tagline Ideas": "Generate Domains & Taglines",
        "30-Day Content Calendar": "Generate 30-Day Content Calendar",
    }[mode]

    submitted = st.form_submit_button(button_label)

# --------------------------------------------------------------------
# HANDLE SUBMIT
# --------------------------------------------------------------------
if submitted:
    if not client_name.strip():
        st.warning("Please enter at least a client / brand name.")
        st.stop()

    answers = {
        "raw_brief": raw_brief,
        "client_name": client_name,
        "industry": industry,
        "target_audience": target_audience,
        "goals": goals,
        "brand_vibe": brand_vibe,
        "voice_tone": voice_tone,
        "colors": colors,
        "visual_keywords": visual_keywords,
        "platforms": platforms,
        "reference_links": reference_links,
        "uploaded_files": [f.name for f in uploaded_files] if uploaded_files else [],
    }

    with st.spinner("Cooking up something creative for you… ✨"):
        if mode == "Brand Discovery Summary":
            result = generate_brand_discovery_summary(answers)
        elif mode == "Brand Style Guide":
            result = generate_brand_style_guide(answers)
        elif mode == "Logo Direction Ideas":
            result = generate_logo_directions(answers)
        elif mode == "AI Logo Sketch Kit":
            result = generate_logo_sketch_kit(answers)
        elif mode == "Website / Landing Page Outline":
            result = generate_site_outline(answers)
        elif mode == "Project Summary & Proposal":
            result = generate_project_summary_proposal(answers)
        elif mode == "Color Palette Generator":
            result = generate_color_palette(answers)
        elif mode == "Brand Voice Guide":
            result = generate_brand_voice(answers)
        elif mode == "Proposal → Invoice Outline":
            result = generate_invoice_outline(answers)
        elif mode == "Domain & Tagline Ideas":
            result = generate_domain_and_taglines(answers)
        else:
            result = generate_content_calendar(answers)

    st.markdown("---")

    pdf_body = ""

    if mode == "30-Day Content Calendar":
        st.subheader("📅 Generated 30-Day Content Calendar")
        calendar_text = result

        try:
            data = json.loads(calendar_text)
            df = pd.DataFrame(data)
            df = df[
                ["day", "platform", "post_type", "hook", "visual_direction", "cta"]
            ]
            df.columns = [
                "Day",
                "Platform",
                "Post Type",
                "Hook / Caption Idea",
                "Visual Direction",
                "CTA",
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)

            pdf_body = df.to_string(index=False)
        except Exception:
            st.warning("Couldn’t parse structured calendar, showing raw text instead:")
            st.markdown(calendar_text)
            pdf_body = calendar_text
    else:
        st.subheader("📄 Generated Output")

        # For color palette: hide any JSON section & show swatches
        display_text = result
        if mode == "Color Palette Generator":
            if "Palette JSON" in display_text:
                display_text = display_text.split("Palette JSON")[0].rstrip()
            if "```json" in display_text:
                display_text = display_text.split("```json")[0].rstrip()

        st.markdown(display_text)
        pdf_body = display_text

        if mode == "Color Palette Generator":
            render_color_swatches(display_text)

        # Extra: moodboard images for AI Logo Sketch Kit
        if mode == "AI Logo Sketch Kit":
            st.markdown("### 🖼️ Logo Moodboard Images")
            try:
                imgs = generate_logo_moodboard_images(answers, n=3)
                img_cols = st.columns(len(imgs))
                for col, img_bytes in zip(img_cols, imgs):
                    col.image(img_bytes)
            except Exception as e:
                st.warning(f"Image generation failed: {e}")
                st.info(
                    "You can still copy the AI prompts above into DALL·E, Midjourney, etc."
                )

    # ----------------------------------------------------------------
    # PDF DOWNLOAD + PROJECT ZIP
    # ----------------------------------------------------------------
    if pdf_body.strip():
        pdf_title = f"{client_name} - {mode}"
        pdf_bytes = make_pdf(pdf_title, pdf_body)

        safe_mode_slug = (
            mode.replace(" ", "_").replace("→", "to").replace("&", "and").lower()
        )
        filename = f"{client_name or 'brand'}_{safe_mode_slug}.pdf"

        st.download_button(
            label="⬇️ Download as PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
        )

        # Save into session for project ZIP
        if "project_files" not in st.session_state:
            st.session_state["project_files"] = {}
        st.session_state["project_files"][filename] = pdf_bytes

    # Project folder export
    if st.session_state.get("project_files"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            for fname, data_bytes in st.session_state["project_files"].items():
                z.writestr(fname, data_bytes)
        buffer.seek(0)

        st.download_button(
            label="📁 Download Project Folder (.zip)",
            data=buffer.getvalue(),
            file_name=f"{client_name or 'client'}_project.zip",
            mime="application/zip",
        )

# --------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------
st.markdown(
    """
<div class="tru-footer">
  Developed by <strong>Trish Bellardine</strong> | Tru Designs • 2025<br>
  <em>Tru Designs Creative Assistant – internal studio tool & portfolio piece.</em>
</div>
""",
    unsafe_allow_html=True,
)
