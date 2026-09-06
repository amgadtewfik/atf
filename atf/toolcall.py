"""Tool-calling helpers for the OpenAI-compatible server.

Pure functions/classes only -- no model/engine imports -- so they can be
unit-tested without loading MLX or FastAPI.

Implements the Qwen3 Hermes tool-call convention:
- prompt side: tools are injected into the system message as a
  "# Tools" block; assistant history renders <tool_call> blocks;
  tool results render as <tool_response> inside a user turn.
- output side: generated text is scanned for
  <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
  blocks which are converted to OpenAI-style tool_calls objects.
"""
from __future__ import annotations

import json
import logging
import re
import uuid

log = logging.getLogger("atf.toolcall")

TAG_OPEN = "<tool_call>"
TAG_CLOSE = "</tool_call>"

TOOLS_BLOCK_TEMPLATE = (
    "\n\n# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:"
    "\n<tools>\n{tools_json}\n</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
    "</tool_call>"
)


def build_tools_block(tools: list[dict] | None, tool_choice=None) -> str:
    """Canonical Qwen3/Hermes '# Tools' system block for a tools list."""
    if not tools:
        return ""
    choice = tool_choice
    must_call = choice == "required" or (
        isinstance(choice, dict) and choice.get("type") == "function")
    directive = (
        "When a supplied function is appropriate, you MUST call it and must "
        "not answer with a prose or code representation of the call."
        if must_call else
        "When a supplied function is appropriate, emit a tool call instead "
        "of describing the command in prose."
    )
    return TOOLS_BLOCK_TEMPLATE.format(
        tools_json=json.dumps(tools, ensure_ascii=False)) + "\n" + directive


def append_tools_to_system(system_content: str, tools: list[dict] | None,
                           tool_choice=None) -> str:
    """Append the tools block to an existing system message."""
    if not tools:
        return system_content or ""
    return (system_content or "") + build_tools_block(tools, tool_choice)


def _content_str(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict))
    return content if isinstance(content, str) else ""


def render_assistant_tool_calls(msg: dict) -> str:
    """Render msg['tool_calls'] as Hermes <tool_call> blocks (no wrapper)."""
    blocks = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:      # keep raw string if not JSON
                pass
        payload = {"name": name, "arguments": args}
        blocks.append(f"{TAG_OPEN}\n"
                      f"{json.dumps(payload, ensure_ascii=False)}\n"
                      f"{TAG_CLOSE}")
    return "".join(blocks)


def render_chatml_messages(messages: list[dict], tools: list[dict] | None = None,
                           tool_choice=None) -> str:
    """Full ChatML rendering including tool conventions (Qwen3 Hermes style).

    - tools go into the FIRST system message (a system message is injected
      when none exists)
    - assistant messages with tool_calls render <tool_call> blocks after
      their content
    - role=='tool' renders as <tool_response> inside a user turn
    """
    parts: list[str] = []
    sys_done = False
    msgs = list(messages)

    def _system(content: str):
        nonlocal sys_done
        body = append_tools_to_system(content, None if sys_done else tools,
                           tool_choice)
        sys_done = True
        parts.append(f"<|im_start|>system\n{body}<|im_end|>\n")

    if tools and not any(m.get("role") == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": ""})

    for msg in msgs:
        role = msg.get("role", "user")
        content = _content_str(msg.get("content", ""))
        if role == "system":
            _system(content)
        elif role == "assistant":
            body = content + render_assistant_tool_calls(msg)
            parts.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
        elif role == "tool":
            parts.append(f"<|im_start|>user\n<tool_response>\n"
                         f"{content}\n</tool_response><|im_end|>\n")
        else:
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# --------------------------------------------------------------------------
# Output-side parsing
# --------------------------------------------------------------------------

def _partial_suffix_len(buf: str, tag: str) -> int:
    """Length of the longest suffix of buf that is a proper prefix of tag."""
    for k in range(min(len(buf), len(tag) - 1), 0, -1):
        if buf.endswith(tag[:k]):
            return k
    return 0


def _parse_inner(inner: str):
    """Parse one <tool_call> body -> {name, arguments} or None (logged)."""
    text = inner.strip()
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict) or "name" not in obj:
            raise ValueError("tool_call JSON must be an object with 'name'")
        args = obj.get("arguments", {})
        if isinstance(args, str) and args.strip():
            args = json.loads(args)
        if not isinstance(args, dict):
            raise ValueError("'arguments' must be a JSON object")
        return {"name": str(obj["name"]), "arguments": args}
    except Exception as exc:
        log.warning("dropping invalid tool call block: %s (%s)",
                    exc, text[:200])
        return None


def format_openai_tool_calls(calls: list[dict]) -> list[dict]:
    """[{name, arguments(dict)}] -> OpenAI chat.completion tool_calls array."""
    out = []
    for c in calls:
        out.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": json.dumps(c["arguments"], ensure_ascii=False),
            },
        })
    return out


class ToolCallStreamParser:
    """Incremental splitter for streamed model output.

    feed(delta) returns ONLY text safe to send as 'content' (holds back
    partial '<tool_call>' prefixes split across chunk boundaries and swallows
    complete blocks). finish() flushes state and reports parsed calls.

    `disabled=True` switches to passthrough mode: every byte of model
    output is returned as content verbatim, no <tool_call> blocks are
    parsed, and calls/invalid stay empty. This is the right mode when
    the original request did NOT pass an OpenAI `tools` list -- the
    model can still hallucinate a tool call, but honoring it as real
    would make the client execute the phantom tool, feed the result
    back, and loop forever. Passing it through as text shows the user
    what the model produced and keeps the turn shape stable.
    """

    def __init__(self, disabled: bool = False):
        self.disabled = disabled
        self.buf = ""
        self.in_block = False
        self.calls: list[dict] = []       # parsed {name, arguments}
        self.invalid: list[str] = []      # raw inner text of bad blocks
        self.finished = False
        self.phantom_warned = False       # one warning per stream

    # -- internals ---------------------------------------------------------
    def _record(self, inner: str):
        if self.disabled:
            # No real tools were advertised; a parseable block is a
            # hallucination. Don't surface it as a tool call, but log
            # once so the user knows the model is reaching for tools
            # the client never asked for.
            if not self.phantom_warned:
                self.phantom_warned = True
                log.warning("model emitted a <tool_call> with no tools "
                            "advertised in the request; passing through as "
                            "text to prevent the client from looping")
            return
        parsed = _parse_inner(inner)
        if parsed is None:
            self.invalid.append(inner.strip())
        else:
            self.calls.append(parsed)

    # -- public API ----------------------------------------------------------
    def feed(self, delta: str) -> str:
        """Consume a delta; return safe-to-emit content (never markup)."""
        if self.finished or not delta:
            return ""
        if self.disabled:
            # Passthrough: every byte is content, even literal <tool_call>
            # markup from a hallucinating model.
            return delta
        self.buf += delta
        out: list[str] = []
        while True:
            if not self.in_block:
                idx = self.buf.find(TAG_OPEN)
                if idx != -1:
                    out.append(self.buf[:idx])
                    self.buf = self.buf[idx + len(TAG_OPEN):]
                    self.in_block = True
                    continue
                hold = _partial_suffix_len(self.buf, TAG_OPEN)
                emit_len = len(self.buf) - hold
                if emit_len > 0:
                    out.append(self.buf[:emit_len])
                    self.buf = self.buf[emit_len:]
                break
            end = self.buf.find(TAG_CLOSE)
            if end != -1:
                self._record(self.buf[:end])
                self.buf = self.buf[end + len(TAG_CLOSE):]
                self.in_block = False
                continue
            break                        # wait for more data inside a block
        return "".join(out)

    def finish(self) -> tuple[str, bool]:
        """Flush at end of stream.

        Returns (trailing_content, has_tool_calls). Anything still buffered
        after the last </tool_call> is trimmed; an unterminated block is
        dropped (kept OUT of content) and logged.
        """
        if self.finished:
            return "", bool(self.calls)
        self.finished = True
        if self.disabled:
            self.buf = ""
            return "", False
        if self.in_block:
            log.warning("stream ended inside an unterminated <tool_call>; "
                        "dropping %d chars", len(self.buf))
            self.invalid.append(self.buf)
            self.buf = ""
            return "", bool(self.calls)
        if self.calls:
            # Per spec trim anything after the last </tool_call>.
            trailing = ""
        else:
            # No tool call appeared: flush any held-back partial-tag text.
            trailing = self.buf
        self.buf = ""
        return trailing, bool(self.calls)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.calls)


def parse_output(text: str, disabled: bool = False) -> tuple[str | None, list[dict], list[str]]:
    """Non-streaming parse of full generated text.

    Returns (remaining_content_or_None, calls, invalid_raw_blocks).
    Content is the text BEFORE the first <tool_call> only: all blocks are
    stripped and anything after the last </tool_call> is trimmed.
    Unparsable blocks never reach content.

    `disabled=True` is for requests that did NOT pass a `tools` list.
    The model may still hallucinate <tool_call> blocks, but honoring
    them as real would loop the client (it would execute the phantom
    tool, feed the result back, and the model would hallucinate
    again). In disabled mode we return the full text unchanged as
    content, calls stays empty, and the markup reaches the user as
    visible text.
    """
    if disabled:
        # Passthrough: every byte of model output is content, even a
        # hallucinated <tool_call>...</tool_call> block.  No calls.
        content = text.strip() or None
        return content, [], []
    calls: list[dict] = []
    invalid: list[str] = []
    pos = 0
    first_open: int | None = None
    while True:
        s = text.find(TAG_OPEN, pos)
        if s == -1:
            break
        if first_open is None:
            first_open = s
        e = text.find(TAG_CLOSE, s + len(TAG_OPEN))
        if e == -1:
            log.warning("unterminated <tool_call> at end of output; "
                        "dropped %d chars", len(text) - s)
            invalid.append(text[s + len(TAG_OPEN):])
            break
        parsed = _parse_inner(text[s + len(TAG_OPEN):e])
        if parsed is None:
            invalid.append(text[s + len(TAG_OPEN):e].strip())
        else:
            calls.append(parsed)
        pos = e + len(TAG_CLOSE)
    if invalid:
        log.warning("dropped %d unparsable tool call block(s)", len(invalid))
    content = (text if first_open is None else text[:first_open]).strip() or None
    return content, calls, invalid
