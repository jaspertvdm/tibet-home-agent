"""AInternet Home Agent — BYOK Mode 3 relay.

Run on your laptop, paired to your `.aint` sub-identity (e.g.
`home.vandemeent.aint`). The phone's K/IT app sends chat prompts via
I-Poll to your home agent; the daemon dispatches to whatever upstream
AI you have configured locally (Gemini API key on the laptop, OpenAI,
Anthropic, or — eventually — a desktop Claude / ChatGPT app via MCP).

Why this matters: many users already pay for Claude Desktop / ChatGPT
Plus / Gemini Pro. They don't want to buy a *second* API key for the
phone. Mode 3 lets them reuse what they have, with no API key on the
phone, and prompts never leaving their hardware.
"""

__version__ = "0.4.0"
