"""Fast BPE tokenizer via gigatoken (Rust-backed)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Tokenizer:
    _impl: object
    vocab_size: int
    eos_token_id: int

    @classmethod
    def from_hf(cls, model_name: str) -> Tokenizer:
        import json
        import os

        import gigatoken as gt
        from gigatoken._load.hub import download_hub_file

        impl = gt.Tokenizer(model_name)
        vocab_size = impl.vocab_size

        # gamma/v1 patch: if model_name is a local directory, read the
        # tokenizer files from disk instead of hitting the HF Hub. The
        # original implementation always calls download_hub_file() which
        # interprets the path as a HF repo_id and 404s on local paths.
        # See qwen38_flash_next_analysis.md §3.2 + the model-process
        # discipline (no Hub fetches during stats runs).
        if os.path.isdir(model_name):
            def _local_read(rel: str) -> str:
                with open(os.path.join(model_name, rel), "r", encoding="utf-8") as f:
                    return f.read()
            _read = _local_read
        else:
            def _read(rel: str) -> str:
                return download_hub_file(model_name, rel)

        # gigatoken.Tokenizer loads tokenizer.json only; it never exposes
        # eos_token_id (not on Tokenizer, and the raw Rust backend doesn't
        # have it either -- see notes in
        # /Users/amgad/Desktop/Ai/Claude/atf-qwen3-tokenizer-bug/notes.md).
        # eos_token lives in tokenizer_config.json, so fetch it separately
        # (same cache/auth gigatoken itself uses) and resolve the id via
        # added_tokens_decoder.
        config = json.loads(_read("tokenizer_config.json"))
        eos_token = config.get("eos_token")
        # eos_token is usually a bare string, but some tokenizer_config.json
        # variants store it as {"content": "...", ...} instead.
        if isinstance(eos_token, dict):
            eos_token = eos_token.get("content")

        eos = None

        # 1) tokenizer_config.json -> added_tokens_decoder content-match
        #    (not guaranteed present/populated on every repo -- this is why
        #    the previous single-source fix broke again on a different repo)
        if eos_token is not None:
            for tid, tok in (config.get("added_tokens_decoder") or {}).items():
                if tok.get("content") == eos_token:
                    eos = int(tid)
                    break

        # 2) tokenizer.json -> top-level added_tokens list. This is the file
        #    gigatoken itself already fetches/parses, and is the canonical
        #    source added_tokens_decoder is normally just a dump of -- more
        #    reliably present across repos than (1).
        if eos is None and eos_token is not None:
            try:
                tok_json = json.loads(_read("tokenizer.json"))
            except Exception:
                tok_json = None
            if tok_json is not None:
                for entry in tok_json.get("added_tokens") or []:
                    if entry.get("content") == eos_token:
                        eos = int(entry["id"])
                        break

        # 3) tokenizer_config.json -> direct eos_token_id int field. Some
        #    repos store this plainly instead of (or alongside) the string.
        if eos is None:
            raw = config.get("eos_token_id")
            if isinstance(raw, int):
                eos = raw
            elif isinstance(raw, list) and raw and isinstance(raw[0], int):
                eos = raw[0]  # some repos give a list of eos ids; use the first

        if eos is None:
            raise ValueError(
                f"could not resolve eos_token_id for {model_name!r}: "
                f"eos_token={eos_token!r} not found in added_tokens_decoder, "
                f"tokenizer.json added_tokens, or a direct eos_token_id field"
            )

        return cls(_impl=impl, vocab_size=vocab_size, eos_token_id=eos)

    def encode(self, text: str) -> list[int]:
        return [int(x) for x in self._impl.encode(text)]

    def decode(self, token_ids: list[int]) -> str:
        raw = self._impl.decode(token_ids)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    def decode_token(self, token_id: int) -> str:
        return self.decode([token_id])
