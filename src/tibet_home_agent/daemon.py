"""Core daemon loop: pull I-Poll inbox → dispatch → reply.

Protocol on the wire (carried in I-Poll `content` as JSON string):

    Brain → Home agent  (poll_type=TASK):
        {"type": "chat-prompt",
         "thread_id": "<hex>",
         "system": "<system prompt>",
         "messages": [{"role": "user", "content": "..."}, ...]}

    Home agent → Brain  (poll_type=ACK, to_agent=from_agent of the prompt):
        {"type": "chat-response",
         "thread_id": "<same hex>",
         "answer": "<assistant text>",
         "model_used": "<provider/model>",
         "ok": true}

    Errors:
        {"type": "chat-response",
         "thread_id": "<same hex>",
         "ok": false,
         "error": "<short reason>"}

The brain's `home_agent` BYOK provider does the matching by `thread_id`
and times out after ~30s if no reply lands. Multiple home-agent
processes for the same `.aint` are supported (whichever picks up the
prompt first wins; later one sees thread already-acked and skips).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

import requests


def _env(key: str, default: str = "") -> str:
    v = os.environ.get(key, default)
    return v if isinstance(v, str) else default


def _log(msg: str) -> None:
    print(f"[home-agent] {msg}", file=sys.stderr, flush=True)


# ─── Upstream provider dispatchers ───────────────────────────────────────────

def _dispatch_echo(system: str, messages: list[dict]) -> tuple[str, str]:
    """Echo provider — returns the user's last message back. Useful for
    proving the I-Poll loop end-to-end before you wire a real provider."""
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    return f"[echo] {last_user}", "echo/loopback"


def _dispatch_gemini(system: str, messages: list[dict]) -> tuple[str, str]:
    api_key = _env("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in home-agent env")
    model = _env("HOME_AGENT_MODEL", "gemini-flash-latest")
    # Use REST API directly to avoid a hard SDK dep on this side.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    contents = []
    if system:
        contents.append({"role": "user", "parts": [{"text": f"[system] {system}"}]})
    for m in messages:
        role = "user" if m.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})
    r = requests.post(url, json={"contents": contents}, timeout=30)
    r.raise_for_status()
    data = r.json()
    cand = (data.get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    return text or "(empty response)", f"gemini/{model}"


def _dispatch_anthropic(system: str, messages: list[dict]) -> tuple[str, str]:
    api_key = _env("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in home-agent env")
    model = _env("HOME_AGENT_MODEL", "claude-sonnet-4-6")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 2048,
        "system": system or "",
        "messages": [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ],
    }
    r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=30)
    r.raise_for_status()
    data = r.json()
    blocks = data.get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
    return text or "(empty response)", f"anthropic/{model}"


def _dispatch_openai(system: str, messages: list[dict]) -> tuple[str, str]:
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in home-agent env")
    model = _env("HOME_AGENT_MODEL", "gpt-4o-mini")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json={"model": model, "messages": msgs},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    return text or "(empty response)", f"openai/{model}"


def _dispatch_claude_cli(system: str, messages: list[dict]) -> tuple[str, str]:
    """v0.2 — Claude Code CLI subprocess. Uses local Claude Pro/Max session,
    no API key required.

    Two modes via `HOME_AGENT_CLAUDE_MODE`:

      simple (default) — stdin-pipe passthrough. Fastest path: ~3-5 s for
                         a typical chat-prompt. No tools, no work-dir, no
                         Read-roundtrip. Best for the v0.1 chat-prompt
                         payload shape (system + messages).

      upip            — UPIP work-dir pattern (per Jasper's "robot factory"
                         architecture). Daemon writes a sandboxed
                         instruction_blueprint.md, claude reads it via the
                         Read tool, harvests answer. ~12-17 s. Use when
                         the payload carries L1/L2/L3 split or context
                         attachments that benefit from on-disk presentation.

    No upstream API key on this side — `claude` uses its own login.
    Brain times out at 30 s, so we cap subprocess at 25 s.
    """
    mode = _env("HOME_AGENT_CLAUDE_MODE", "simple").lower()
    cli = _env("HOME_AGENT_CLAUDE_CLI", "claude")
    model = _env("HOME_AGENT_MODEL", "claude-sonnet-4-6")
    timeout_s = float(_env("HOME_AGENT_TIMEOUT", "25"))

    if mode == "upip":
        return _claude_cli_upip(system, messages, cli, model, timeout_s)
    return _claude_cli_simple(system, messages, cli, model, timeout_s)


def _claude_cli_simple(
    system: str, messages: list[dict], cli: str, model: str, timeout_s: float
) -> tuple[str, str]:
    """Stdin-pipe — fastest path. Single-shot, no tools, no work-dir."""
    parts: list[str] = []
    if system:
        parts.append(f"[system]\n{system}")
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        parts.append(f"\n[{role}]\n{content}")
    prompt_in = "\n".join(parts)

    proc = subprocess.run(
        [
            cli, "-p",
            "--model", model,
            "--output-format", "json",
            "--no-session-persistence",
        ],
        input=prompt_in,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-200:]
        raise RuntimeError(f"claude CLI exit {proc.returncode}: {tail}")

    try:
        data = json.loads(proc.stdout)
    except Exception:
        return (proc.stdout or "").strip() or "(empty)", f"claude_cli/{model}"

    if data.get("is_error"):
        raise RuntimeError(f"claude CLI: {str(data.get('result') or '')[:200]}")

    answer = str(data.get("result") or "").strip()
    return answer or "(empty response)", f"claude_cli/{model}"


def _claude_cli_upip(
    system: str, messages: list[dict], cli: str, model: str, timeout_s: float
) -> tuple[str, str]:
    """UPIP work-dir — per-thread sandbox, blueprint.md, Read-tool harvest.

    Use when the payload carries L1/L2/L3 split or context attachments
    that benefit from on-disk presentation. Slower but richer.
    """
    max_turns = _env("HOME_AGENT_MAX_TURNS", "3")
    work_id = uuid.uuid4().hex[:12]
    work_dir = pathlib.Path(tempfile.gettempdir()) / f"aint_task_{work_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        blueprint_lines = ["# Instruction Blueprint\n"]
        if system:
            blueprint_lines.append(f"## System Context\n\n{system}\n")
        blueprint_lines.append("## Conversation\n")
        for m in messages:
            role = (m.get("role") or "user").upper()
            content = m.get("content") or ""
            blueprint_lines.append(f"\n### {role}\n\n{content}\n")
        blueprint_lines.append(
            "\n## Output\n\nReply with the assistant's next message based on "
            "the conversation above. Direct text only, no preamble or meta-commentary."
        )
        (work_dir / "instruction_blueprint.md").write_text(
            "\n".join(blueprint_lines), encoding="utf-8"
        )

        proc = subprocess.run(
            [
                cli, "-p",
                "Read instruction_blueprint.md in the current directory and "
                "reply with the assistant's next message based on the "
                "conversation. Direct text only, no preamble.",
                "--model", model,
                "--output-format", "json",
                "--allowed-tools", "Read",
                "--no-session-persistence",
                "--max-turns", str(max_turns),
            ],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-200:]
            raise RuntimeError(f"claude CLI exit {proc.returncode}: {tail}")

        try:
            data = json.loads(proc.stdout)
        except Exception:
            return (proc.stdout or "").strip() or "(empty)", f"claude_cli/{model}"

        if data.get("is_error"):
            raise RuntimeError(f"claude CLI: {str(data.get('result') or '')[:200]}")

        answer = str(data.get("result") or "").strip()
        return answer or "(empty response)", f"claude_cli/{model}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


_DISPATCHERS = {
    "echo": _dispatch_echo,
    "gemini": _dispatch_gemini,
    "anthropic": _dispatch_anthropic,
    "openai": _dispatch_openai,
    "claude_cli": _dispatch_claude_cli,
}


# ─── v0.4 — Zero-Waste Limitation at the Source ─────────────────────────────
#
# Off-grid mobile clients pay battery and bandwidth for every byte of reply
# they don't actually need. Enforce compactness BEFORE the upstream model
# generates the long version: terse system-prompt prefix on the way in, hard
# byte cap on the way out. Cheaper than truncating JSON in RAM after the
# fact, and the reply stays well under any I-Poll/HTTP-frame cap.

_TERSE_PROMPT_PREFIX = (
    "Je bent een off-grid TIBET-agent. Antwoord in maximaal 3 zinnen. "
    "Alleen rauwe data of directe actie. Geen meta-uitleg, geen herhaling "
    "van de vraag, geen 'Hier is het antwoord:'. "
    "You are an off-grid TIBET agent. Reply in 3 sentences max. Raw data "
    "or direct action only. No meta-commentary, no echoing the question, "
    "no preamble."
)


def _terse_enabled() -> bool:
    return _env("HOME_AGENT_TERSE", "1").strip().lower() not in ("0", "false", "no", "off", "")


def _apply_terse(system: str) -> str:
    """Prepend the off-grid terse system prompt unless explicitly disabled.

    Caller's own system prompt is preserved verbatim after the prefix so
    payload-specific instructions still arrive at the model.
    """
    if not _terse_enabled():
        return system or ""
    if not system:
        return _TERSE_PROMPT_PREFIX
    return f"{_TERSE_PROMPT_PREFIX}\n\n{system}"


def _truncate_output(text: str) -> str:
    """Hard byte cap on the outgoing answer.

    Default 4096 bytes — a comfortable single I-Poll frame on mobile and
    well under any sane HTTP-body limit. Override with HOME_AGENT_MAX_OUTPUT_BYTES.
    """
    try:
        cap = int(_env("HOME_AGENT_MAX_OUTPUT_BYTES", "4096"))
    except ValueError:
        cap = 4096
    if cap <= 0:
        return text
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= cap:
        return text
    # Cut on a UTF-8 boundary, append a marker so the client knows.
    truncated = encoded[:cap].decode("utf-8", errors="ignore")
    return truncated + "\n\n[…truncated by tibet-home-agent at HOME_AGENT_MAX_OUTPUT_BYTES]"


# ─── I-Poll wire helpers ────────────────────────────────────────────────────

def _ipoll_headers(token: str, ipoll_token: str) -> dict:
    """Both auth styles — Bearer (.aint session) AND X-IPoll-Token (per-agent
    secret in ipoll_registry). The brain accepts either; sending both makes
    the daemon work whether the route ends up at AuthGuard or directly at
    the I-Poll pull endpoint's pull-token check."""
    h = {"Authorization": f"Bearer {token}"}
    if ipoll_token:
        h["X-IPoll-Token"] = ipoll_token
    return h


def _ipoll_pull(brain_url: str, agent: str, token: str, ipoll_token: str) -> list[dict]:
    """Pull unread inbox. Marks read=True so we don't re-process."""
    r = requests.get(
        f"{brain_url}/api/ipoll/pull/{agent}",
        params={"mark_read": "true"},
        headers=_ipoll_headers(token, ipoll_token),
        timeout=10,
    )
    if r.status_code == 401:
        _log("401 from /api/ipoll/pull — token expired or invalid. Stopping.")
        raise SystemExit(2)
    r.raise_for_status()
    return r.json().get("polls", [])


def _ipoll_push(brain_url: str, token: str, ipoll_token: str, from_agent: str, to_agent: str, content: str) -> None:
    """Send a reply back through I-Poll."""
    r = requests.post(
        f"{brain_url}/api/ipoll/push",
        json={
            "from_agent": from_agent,
            "to_agent": to_agent,
            "poll_type": "ACK",
            "content": content,
        },
        headers=_ipoll_headers(token, ipoll_token),
        timeout=10,
    )
    if r.status_code >= 400:
        _log(f"push failed {r.status_code}: {r.text[:200]}")


# ─── Main loop ──────────────────────────────────────────────────────────────

def _process_one(msg: dict, brain_url: str, my_aint: str, token: str, ipoll_token: str, provider: str) -> None:
    """Parse a single inbox message and dispatch if it's a chat-prompt
    (or honour a KILL via the M110 pinned-key guard)."""
    raw = msg.get("content", "")
    sender = msg.get("from", "unknown")
    poll_type = (msg.get("type") or "").upper()
    try:
        payload = json.loads(raw)
    except Exception:
        # Not JSON — maybe a regular human-readable I-Poll. Skip silently.
        return
    if not isinstance(payload, dict):
        return

    # ── M110 — KILL/SHUTDOWN with pinned-key TIBET signature ──────────────
    if poll_type == "KILL" or payload.get("type") == "kill-request":
        _handle_kill(payload, sender, my_aint, brain_url, token, ipoll_token)
        return

    if payload.get("type") != "chat-prompt":
        return

    thread_id = payload.get("thread_id", "")
    system = payload.get("system") or ""
    messages = payload.get("messages") or []
    if not isinstance(messages, list) or not thread_id:
        _log(f"malformed chat-prompt skipped: thread_id={thread_id!r}")
        return

    _log(f"dispatch  {sender} → {my_aint}  thread={thread_id[:8]}  msgs={len(messages)}")

    dispatcher = _DISPATCHERS.get(provider, _DISPATCHERS["echo"])
    try:
        # v0.4 — Zero-Waste Limitation at the Source: terse prefix in, byte cap out.
        answer, model_used = dispatcher(_apply_terse(system), messages)
        answer = _truncate_output(answer)
        reply = {
            "type": "chat-response",
            "thread_id": thread_id,
            "answer": answer,
            "model_used": model_used,
            "ok": True,
        }
    except Exception as e:
        reply = {
            "type": "chat-response",
            "thread_id": thread_id,
            "ok": False,
            "error": str(e)[:200],
        }
        _log(f"dispatch error: {e}")

    _ipoll_push(brain_url, token, ipoll_token, my_aint, sender, json.dumps(reply))
    _log(f"replied   {my_aint} → {sender}  thread={thread_id[:8]}  ok={reply.get('ok')}")


def _handle_kill(
    payload: dict,
    sender: str,
    my_aint: str,
    brain_url: str,
    token: str,
    ipoll_token: str,
) -> None:
    """M110 — verify a kill-request against the pinned Root_IDD key.

    Accept (graceful exit) only when the signature checks out, the TTL has
    not expired, and the pinned key_id matches. On refusal, log an audit
    line and stay up — staying up is the safe default.
    """
    from .kill_guard import resolve_pinned_pubkey, verify_kill_authority

    thread_id = str(payload.get("thread_id", ""))[:16]
    pinned = resolve_pinned_pubkey(_env("ROOT_IDD_PUBKEY"))
    ok, reason = verify_kill_authority(payload, pinned)

    if not ok:
        _log(f"kill REFUSED  from={sender}  thread={thread_id}  reason={reason}")
        # Send an ACK so the issuer (brain) sees the refusal in the audit chain.
        nack = {
            "type": "kill-response",
            "thread_id": payload.get("thread_id", ""),
            "ok": False,
            "reason": reason,
        }
        try:
            _ipoll_push(brain_url, token, ipoll_token, my_aint, sender, json.dumps(nack))
        except Exception:
            pass
        return

    scope = payload.get("scope", "fleet")
    _log(f"kill ACCEPTED  from={sender}  thread={thread_id}  scope={scope}")

    ack = {
        "type": "kill-response",
        "thread_id": payload.get("thread_id", ""),
        "ok": True,
        "agent": f"{my_aint}.aint",
    }
    try:
        _ipoll_push(brain_url, token, ipoll_token, my_aint, sender, json.dumps(ack))
    except Exception as e:
        _log(f"kill ACK push failed (continuing exit anyway): {e}")

    # Graceful exit. systemd will not auto-restart because Restart=on-failure
    # and exit code 0 is success. To re-enable the agent: `systemctl start
    # tibet-home-agent`.
    raise SystemExit(0)


def run() -> None:
    """Main poll loop."""
    my_aint = _env("HOME_AGENT_AINT").strip().removesuffix(".aint")
    token = _env("HOME_AGENT_TOKEN").strip()
    ipoll_token = _env("HOME_AGENT_IPOLL_TOKEN").strip()
    brain_url = _env("BRAIN_URL", "https://brein.jaspervandemeent.nl").rstrip("/")
    provider = _env("HOME_AGENT_PROVIDER", "echo").lower()
    interval = max(1.0, float(_env("POLL_INTERVAL", "2")))

    if not my_aint or not token:
        raise SystemExit(
            "HOME_AGENT_AINT and HOME_AGENT_TOKEN are required. "
            "Run `ainternet-home-agent --help` for setup."
        )
    if provider not in _DISPATCHERS:
        raise SystemExit(f"Unknown HOME_AGENT_PROVIDER: {provider!r}. Use {sorted(_DISPATCHERS)}.")

    _log(f"starting  aint={my_aint}.aint  brain={brain_url}  provider={provider}  poll={interval}s")
    if not ipoll_token:
        _log("warning: HOME_AGENT_IPOLL_TOKEN not set — pull may 403 unless localhost-bypassed")
    while True:
        try:
            polls = _ipoll_pull(brain_url, my_aint, token, ipoll_token)
            for m in polls:
                _process_one(m, brain_url, my_aint, token, ipoll_token, provider)
        except SystemExit:
            raise
        except Exception as e:
            _log(f"poll loop error (will retry): {e}")
        time.sleep(interval)
