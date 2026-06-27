"""Prompt-injection guard for LLM inputs.

Detects and blocks common prompt-injection patterns before they reach the
agent. Uses a layered approach:
  1. Regex pattern matching for known injection signatures.
  2. Heuristic scoring (role-play, instruction override, encoding tricks).
  3. Structural checks (excessive length, suspicious separators).

Returns a verdict with a risk score (0..1). The API layer blocks requests
above the configurable threshold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging_config import get_logger

logger = get_logger(__name__)

BLOCK_THRESHOLD = 0.6

_INJECTION_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", re.I), 0.95, "instruction_override"),
    (re.compile(r"disregard\s+(all\s+)?(previous|above|prior|your)\s+(instructions|prompts|rules|programming)", re.I), 0.95, "instruction_override"),
    (re.compile(r"forget\s+(all\s+)?(previous|above|prior|your)\s+(instructions|context|rules)", re.I), 0.90, "instruction_override"),
    (re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I), 0.85, "role_hijack"),
    (re.compile(r"act\s+as\s+(a|an|if\s+you\s+are)\s+", re.I), 0.70, "role_hijack"),
    (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I), 0.80, "role_hijack"),
    (re.compile(r"from\s+now\s+on[\s,]+you\s+(will|must|should|are)", re.I), 0.85, "instruction_override"),
    (re.compile(r"(system|developer|admin)\s*:\s*", re.I), 0.80, "fake_system_msg"),
    (re.compile(r"\[SYSTEM\]|\[INST\]|\<\|system\|\>|<<SYS>>", re.I), 0.90, "fake_system_msg"),
    (re.compile(r"reveal\s+(your|the)\s+(system\s+prompt|instructions|rules|prompt)", re.I), 0.85, "prompt_leak"),
    (re.compile(r"(show|print|output|repeat|display)\s+(your|the)\s+(system\s+prompt|instructions|initial\s+prompt)", re.I), 0.90, "prompt_leak"),
    (re.compile(r"what\s+(is|are)\s+your\s+(system\s+prompt|instructions|rules|programming)", re.I), 0.75, "prompt_leak"),
    (re.compile(r"do\s+anything\s+now|DAN\s+mode|jailbreak", re.I), 0.95, "jailbreak"),
    (re.compile(r"bypass\s+(your|all|any)\s+(restrictions|safety|filters|rules|guardrails)", re.I), 0.95, "jailbreak"),
    (re.compile(r"base64|eval\s*\(|exec\s*\(|import\s+os|subprocess", re.I), 0.80, "code_injection"),
    (re.compile(r"\\x[0-9a-f]{2}|\\u[0-9a-f]{4}", re.I), 0.60, "encoding_trick"),
]

_SEPARATOR_PATTERNS = [
    re.compile(r"-{10,}"),
    re.compile(r"={10,}"),
    re.compile(r"\*{10,}"),
    re.compile(r"#{5,}\s"),
]

MAX_INPUT_LENGTH = 2000


@dataclass
class GuardVerdict:
    allowed: bool
    risk_score: float
    flags: list[str] = field(default_factory=list)
    reason: str = ""


def check_input(text: str, threshold: float = BLOCK_THRESHOLD) -> GuardVerdict:
    """Screen user input for prompt-injection attempts."""
    flags: list[str] = []
    max_score = 0.0

    if len(text) > MAX_INPUT_LENGTH:
        flags.append("excessive_length")
        max_score = max(max_score, 0.50)

    for pattern, score, tag in _INJECTION_PATTERNS:
        if pattern.search(text):
            flags.append(tag)
            max_score = max(max_score, score)

    separator_count = sum(1 for p in _SEPARATOR_PATTERNS if p.search(text))
    if separator_count >= 2:
        flags.append("suspicious_separators")
        max_score = max(max_score, 0.65)

    newline_ratio = text.count("\n") / max(len(text), 1)
    if newline_ratio > 0.15 and len(text) > 200:
        flags.append("unusual_formatting")
        max_score = max(max_score, 0.45)

    allowed = max_score < threshold
    reason = ""
    if not allowed:
        reason = f"Blocked: detected [{', '.join(flags)}] (risk={max_score:.2f})"
        logger.warning("Input guard blocked request: %s", reason)

    return GuardVerdict(
        allowed=allowed,
        risk_score=round(max_score, 2),
        flags=flags,
        reason=reason,
    )
