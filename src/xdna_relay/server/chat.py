"""Chat messages -> prompt text, with the checkpoint's own Jinja chat template.

From ``chat_template.jinja`` or ``tokenizer_config.json`` (``chat_template``) in a Hugging Face
directory, rendered in Jinja's sandbox as transformers does (``add_generation_prompt``,
``raise_exception``, ``strftime_now``). Extra template variables (Qwen3's
``enable_thinking``) come from the request's ``chat_template_kwargs``.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path


class ChatTemplate:
    def __init__(self, source: str, bos_token: str = "", defaults: dict | None = None):
        from jinja2 import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message):
            raise TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.globals.update(raise_exception=raise_exception,
                           strftime_now=lambda f: datetime.datetime.now().strftime(f))
        self.template = env.from_string(source)
        self.bos_token = bos_token
        self.defaults = dict(defaults or {})

    @classmethod
    def from_hf_dir(cls, hf_dir: str, defaults: dict | None = None) -> "ChatTemplate":
        d = Path(hf_dir)
        cfg = json.loads((d / "tokenizer_config.json").read_text()) if (d / "tokenizer_config.json").exists() else {}
        if (d / "chat_template.jinja").exists():
            source = (d / "chat_template.jinja").read_text()
        elif isinstance(cfg.get("chat_template"), str):
            source = cfg["chat_template"]
        else:
            raise ValueError(f"{hf_dir}: no chat template (chat_template.jinja / tokenizer_config.json)")
        bos = cfg.get("bos_token") or ""
        if isinstance(bos, dict):
            bos = bos.get("content", "")
        return cls(source, bos, defaults)

    def render(self, messages: list[dict], **kwargs) -> str:
        """Prompt text for ``messages``, the assistant's turn opened. The BOS token, where the
        template writes one, is part of the text (tokenize without adding specials)."""
        return self.template.render(messages=messages, add_generation_prompt=True,
                                    bos_token=self.bos_token, **{**self.defaults, **kwargs})
