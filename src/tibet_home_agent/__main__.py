"""CLI entrypoint: `python -m tibet_home_agent` or `tibet-home-agent`."""
from __future__ import annotations

import sys
from .daemon import run


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help", "help"}:
        print(__doc__ or "")
        print("Usage: tibet-home-agent")
        print()
        print("Environment variables:")
        print("  HOME_AGENT_AINT      — your home-agent sub-domain (e.g. home.vandemeent)")
        print("  HOME_AGENT_TOKEN     — session token issued at claim time")
        print("  BRAIN_URL            — brain API URL (default https://brein.jaspervandemeent.nl)")
        print("  HOME_AGENT_PROVIDER  — upstream: 'echo' | 'gemini' | 'anthropic' | 'openai' | 'claude_cli'")
        print("  GEMINI_API_KEY       — when provider=gemini")
        print("  ANTHROPIC_API_KEY    — when provider=anthropic")
        print("  OPENAI_API_KEY       — when provider=openai")
        print("  HOME_AGENT_MODEL     — when provider=claude_cli (default claude-sonnet-4-6)")
        print("  HOME_AGENT_CLAUDE_CLI — path to `claude` binary (default 'claude' on PATH)")
        print("  HOME_AGENT_CLAUDE_MODE — 'simple' (default, stdin pipe, ~3-5s) or 'upip' (work-dir, ~12-17s, rich payloads)")
        print("  HOME_AGENT_TIMEOUT   — claude_cli subprocess timeout seconds (default 25)")
        print("  HOME_AGENT_MAX_TURNS — claude_cli max turns in upip mode (default 3)")
        print("  POLL_INTERVAL        — inbox poll interval seconds (default 2)")
        return 0
    try:
        run()
    except KeyboardInterrupt:
        print("\n[home-agent] stopped", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"[home-agent] FATAL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
