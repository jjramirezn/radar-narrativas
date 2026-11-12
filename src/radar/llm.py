"""Claude calls with cost tracking and a replay record.

Every answer is appended to registro_ia/<dataset>.jsonl, keyed by a hash of the exact request.
A later run that makes the same request reuses the recorded answer instead of calling the API,
so re-running the pipeline on the same data reproduces the same result, with or without a key.
`--ia-nueva` ignores the record and asks again.

Every call is optional: without credentials, network or a recorded answer, callers get None
back and fall back to a non-AI path, so the pipeline always finishes.
"""
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .context import Context

# USD per million tokens (input, output), first-party API; Haiku's rate is for prompts up to 100K tokens.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
RECORD_DIR = Path(__file__).resolve().parents[2] / "registro_ia"


class LLM:
    def __init__(self, ctx: Context, refresh: bool = False):
        self.ctx = ctx
        self.model = ctx.settings.llm_model
        self.client = None
        self.disabled_reason = None
        self.recorded = {}
        self.record_path = RECORD_DIR / f"{ctx.datos.resolve().name}.jsonl"
        self._lock = threading.Lock()
        if not ctx.settings.use_llm:
            self.disabled_reason = "desactivada con --sin-ia"
            return
        if refresh:
            self.record_path.unlink(missing_ok=True)
        elif self.record_path.exists():
            for line in self.record_path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self.recorded[entry["clave"]] = entry
        try:
            import anthropic

            self.client = anthropic.Anthropic(max_retries=4, timeout=120.0)
            # Credentials are only resolved on the first request: check once, cheaply, up front.
            self.client.with_options(timeout=15.0, max_retries=1).models.retrieve(self.model)
        except Exception as exc:  # noqa: BLE001 - no credentials or no network: run without AI
            self.client = None
            self.disabled_reason = f"{type(exc).__name__}: {str(exc)[:100]}"
        if self.disabled_reason:
            replay = f"; se reutilizan {len(self.recorded)} respuestas registradas" if self.recorded else "; se sigue sin IA"
            print(f"[radar]   IA no disponible ({self.disabled_reason}){replay}", flush=True)

    @property
    def available(self) -> bool:
        """True if some request can be answered: live, or from the record."""
        return self.client is not None or bool(self.recorded)

    def _usd(self, input_tokens: int, output_tokens: int) -> float:
        price_in, price_out = PRICES.get(self.model, PRICES["claude-opus-5-5"])
        return (input_tokens * price_in + output_tokens * price_out) / 1e6

    def json(self, system: str, prompt: str, schema: dict, max_tokens: int = 8000) -> dict | None:
        """One call whose answer must match `schema`. Returns the parsed object, or None on any failure."""
        kwargs = {
            "model": self.model, "max_tokens": max_tokens, "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}, "effort": "low"},
        }
        if self.model.startswith("claude-haiku"):
            kwargs["thinking"] = {"type": "disabled"}  # labeling needs no reasoning trace
        key = hashlib.sha256(json.dumps(kwargs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        costs = self.ctx.costs
        if key in self.recorded:
            entry = self.recorded[key]
            with self._lock:
                costs.llm_replayed += 1
                costs.usd_replayed += entry["usd"]
            return entry["respuesta"]
        if self.client is None:
            return None
        import anthropic

        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            print("[radar]   IA: credenciales inválidas; se sigue sin IA", flush=True)
            self.client = None
            return None
        except Exception as exc:  # noqa: BLE001 - any failure falls back to the non-AI path
            print(f"[radar]   IA: error {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            return None
        u = response.usage
        tokens_in = u.input_tokens + (getattr(u, "cache_creation_input_tokens", 0) or 0) \
            + (getattr(u, "cache_read_input_tokens", 0) or 0)
        usd = self._usd(tokens_in, u.output_tokens)
        with self._lock:
            costs.llm_calls += 1
            costs.input_tokens += tokens_in
            costs.output_tokens += u.output_tokens
            costs.usd += usd
        if response.stop_reason in ("refusal", "max_tokens"):
            return None
        text = next((b.text for b in response.content if b.type == "text"), None)
        try:
            answer = json.loads(text) if text else None
        except json.JSONDecodeError:
            return None
        if answer is not None:
            entry = {"clave": key, "modelo": self.model, "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "tokens_entrada": tokens_in, "tokens_salida": u.output_tokens, "usd": round(usd, 6),
                     "pedido": prompt, "respuesta": answer}
            with self._lock:
                self.recorded[key] = entry
                self.record_path.parent.mkdir(parents=True, exist_ok=True)
                with self.record_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return answer

    def json_many(self, jobs: list[tuple[str, str, dict]], workers: int = 8) -> list[dict | None]:
        """Run several independent calls in parallel, results in input order."""
        if not self.available or not jobs:
            return [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda job: self.json(*job), jobs))
