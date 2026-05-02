# Changelog

All notable changes to the `tibet-home-agent` package are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-05-02

### Added — Zero-Waste Limitation at the Source

The off-grid mobile target pays battery and bandwidth for every byte of
reply it does not actually need. v0.4 enforces compactness BEFORE the
upstream model generates the long version, instead of truncating JSON in
RAM after the fact.

- **Terse system-prompt prefix** is now prepended to every dispatched
  prompt (NL + EN, "off-grid TIBET-agent. Reply in 3 sentences max. Raw
  data or direct action only. No meta-commentary."). Default ON.
  Disable with `HOME_AGENT_TERSE=0` per-deployment.
- **Hard byte cap on the outgoing answer** via
  `HOME_AGENT_MAX_OUTPUT_BYTES` (default `4096`). A reply over the cap is
  cut on a UTF-8 boundary and tagged with a single trailing
  `[…truncated by tibet-home-agent at HOME_AGENT_MAX_OUTPUT_BYTES]`
  marker so the client sees that it was clipped server-side, not by the
  network.
- The terse prefix and the byte cap apply to all five dispatchers
  (`echo`, `gemini`, `anthropic`, `openai`, `claude_cli`), so the
  contract stays uniform regardless of which BYOK backend the user has
  configured.

`__version__` in `tibet_home_agent/__init__.py` was stuck on `0.1.0`
since the initial release; now synced to `0.4.0`.

### Why
Predictable memory footprint, predictable I/O, no after-the-fact JSON
truncation in RAM. Default-on terse mode keeps mobile replies short and
well within a single I-Poll/HTTP frame. This is the agent-side mirror of
the brain-side cap that landed in `byok_providers._call_byok_home_agent`
at the same time.

## [0.3.0] — 2026-04-30

### Added — M110 / SWARM-005 remediation
- `kill_guard.py` — Ed25519 verification of incoming KILL/SHUTDOWN
  payloads. The daemon refuses any kill-request not signed by the
  pinned `Root_IDD` key (`PINNED_KEY_ID = "root_idd.v1"`). 30 s clock
  skew tolerance, 600 s TTL cap.
- `_handle_kill()` wired into `_process_one`, gracefully exits 0 on a
  verified kill so systemd's `Restart=on-failure` does not auto-revive
  the daemon. Refused kills emit a `kill-response` NACK with reason and
  stay up.
- 9-case test suite in `tests/test_kill_guard.py` (valid-accept,
  expired, future-dated, wrong key_id, non-Ed25519 algorithm, tampered
  payload, signed-by-other-key, no pinned pubkey, ttl exceeds cap).

This addresses Red Specter's RS-2026-001 SWARM-005 finding — kill-switch
survival in agent runtime. The fix is per-agent autonomous defence; the
network perimeter (SNAFT) cannot tell whether a KILL signal is rogue,
only the agent itself can.

## [0.2.1] — 2026-04-30

### Added
- `simple` (stdin-pipe) mode for `_dispatch_claude_cli`, gated by
  `HOME_AGENT_CLAUDE_MODE`. Default. ~3-5 s round-trip; 3-5x faster
  than the UPIP work-dir mode for typical chat-prompt payloads.

### Changed
- `upip` mode (the original v0.2 work-dir / blueprint.md /
  Read-tool-harvest path) is now the explicit opt-in, kept available for
  payloads that benefit from on-disk presentation of L1/L2/L3 context.

## [0.2.0] — 2026-04-30

### Added
- Initial `claude_cli` provider — subprocess-based. Uses a local
  `claude` CLI session, no upstream API key on this side.
- UPIP work-dir pattern with `instruction_blueprint.md` and harvested
  answer file. Multi-turn capable via `HOME_AGENT_MAX_TURNS`.

### Notes
- Daemon was renamed from `ainternet-home-agent` → `tibet-home-agent`
  during the same window. Systemd unit + EnvironmentFile paths updated;
  see service file shipped in `packaging/`.

## [0.1.0] — 2026-04-30

Initial release. Mode 3 BYOK relay with `echo`, `gemini`, `anthropic`,
and `openai` dispatchers. I-Poll long-poll + push protocol, systemd unit
suitable for laptop-side deployment.
