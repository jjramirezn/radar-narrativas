"""Claude calls with cost tracking. Every call is optional: without credentials or network,
callers get None back and fall back to a non-AI path, so the pipeline always finishes."""
import json
from concurrent.futures import ThreadPoolExecutor

from .context import Context

# USD per million tokens (input, output), first-party API; Haiku's rate is for prompts up to 100K tokens.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


class LLM:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.model = ctx.settings.llm_model
        self.client = None
        self.disabled_reason = None
        if not ctx.settings.use_llm:
            self.disabled_reason = "desactivada con --sin-ia"
            return
        try:
            import anthropic

            self.client = anthropic.Anthropic(max_retries=4, timeout=120.0)
            # Credentials are only resolved on the first request: check once, cheaply, up front.
            self.client.with_options(timeout=15.0, max_retries=1).models.retrieve(self.model)
        except Exception as exc:  # noqa: BLE001 - no credentials or no network: run without AI
            self.client = None
            self.disabled_reason = f"{type(exc).__name__}: {str(exc)[:100]}"
        if self.disabled_reason:
            print(f"[radar]   IA no disponible ({self.disabled_reason}); se sigue sin IA", flush=True)

    @property
    def available(self) -> bool:
        return self.client is not None

    def _track(self, usage):
        price_in, price_out = PRICES.get(self.model, PRICES["claude-opus-5-5"])
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        costs = self.ctx.costs
        costs.llm_calls += 1
        costs.input_tokens += usage.input_tokens + cache_write + cache_read
        costs.output_tokens += usage.output_tokens
        costs.usd += (usage.input_tokens * price_in + cache_write * price_in * 1.25 + cache_read * price_in * 0.1
                      + usage.output_tokens * price_out) / 1e6

    def json(self, system: str, prompt: str, schema: dict, max_tokens: int = 8000) -> dict | None:
        """One call whose answer must match `schema`. Returns the parsed object, or None on any failure."""
        if not self.available:
            return None
        import anthropic

        kwargs = {
            "model": self.model, "max_tokens": max_tokens, "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}, "effort": "low"},
        }
        if self.model.startswith("claude-haiku"):
            kwargs["thinking"] = {"type": "disabled"}  # labeling needs no reasoning trace
        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            print("[radar]   IA: credenciales inválidas; se sigue sin IA", flush=True)
            self.client = None
            return None
        except Exception as exc:  # noqa: BLE001 - any failure falls back to the non-AI path
            print(f"[radar]   IA: error {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            return None
        self._track(response.usage)
        if response.stop_reason in ("refusal", "max_tokens"):
            return None
        text = next((b.text for b in response.content if b.type == "text"), None)
        try:
            return json.loads(text) if text else None
        except json.JSONDecodeError:
            return None

    def json_many(self, jobs: list[tuple[str, str, dict]], workers: int = 8) -> list[dict | None]:
        """Run several independent calls in parallel, results in input order."""
        if not self.available or not jobs:
            return [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda job: self.json(*job), jobs))
