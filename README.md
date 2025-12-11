🎨 Tru Designs Creative Assistant

A mini creative agent for brand discovery, style guides, logo concepts, site outlines, and strategy documents — built by Trish Bellardine | Tru Designs.

🚀 Overview

The Tru Designs Creative Assistant is an AI-powered studio tool designed to streamline the entire creative intake and strategy process for designers, agencies, and entrepreneurs.

Instead of digging through chaotic client messages, PDFs, and random notes, this assistant turns info into clear, structured deliverables — instantly.

Built using Streamlit, OpenAI, and custom brand logic, it functions as a creative co-pilot inside your workflow or portfolio.

✨ Features
1. Brand Discovery Summary

Transforms raw notes into a clean, organized brand overview.

2. Brand Style Guide Generator

Generates palettes, typography notes, style direction, and visual identity guidance.

3. Logo Direction Ideas

Provides conceptual logo paths with moods, shapes, symbolism, and creative prompts.

4. AI Logo Sketch Kit

10 logo concept prompts

Suggested shapes & styles

Auto-generated moodboard images

PDF export

5. Website / Landing Page Outline

Creates structured sitemaps + homepage wireframe sections.

6. Project Summary & Proposal

Turns notes into a friendly proposal and scope outline.

7. Color Palette Generator

Generates colors, gradients, and visual swatches (no JSON clutter).

8. Brand Voice Guide

Defines tone pillars, dos/don’ts, and channel examples.

9. Domain & Tagline Ideas

Name + tagline concepts based on brand vibe.

10. 30-Day Content Calendar

Platform-specific content ideas with hooks, visuals, and CTAs.

🛠️ Tech Stack

Python 3.10+

Streamlit (full UI)

OpenAI API (GPT for text + image generation)

FPDF (PDF output)

Regex-based parsing for smart auto-fill intake

📦 Installation

Clone the repo:

git clone https://github.com/trudesigns/tru-designs-creative-assistant.git
cd tru-designs-creative-assistant


Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Create a .env file with your OpenAI API key:

OPENAI_API_KEY=your-key-here


Run the app:

streamlit run ui_app.py

📱 UX Notes

Sidebar is collapsed by default for mobile-first usability.

Buttons use Tru Designs hot-pink branding.

Image generator returns 1024×1024 assets.

🎯 Purpose

This tool serves as:

A client-facing productivity tool

An internal workflow accelerator

A portfolio showcase of Trish Bellardine’s creative + AI skills

A foundation for future SaaS concepts (Spotaswag, Tru Tools, etc.)

🖋️ Creator

Trish Bellardine
Multidisciplinary Creative Director | UX Designer | AI Entrepreneur
📍 Los Angeles, CA
🌐 https://tridesigns.com
 (placeholder if needed)

📄 License

This project is for portfolio + internal use only.
Not licensed for commercial resale.
