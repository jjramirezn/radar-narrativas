"""Run context shared by every pipeline step: paths, the DuckDB connection, settings, costs."""
import json
import os
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import duckdb


@dataclass
class Settings:
    embedding_model: str = "intfloat/multilingual-e5-small"
    llm_model: str = "claude-haiku-5-5"
    use_llm: bool = True
    seed: int = 0


@dataclass
class Costs:
    """Tracks wall-clock time per stage and AI usage, and writes costo.json."""
    started: float = field(default_factory=time.time)
    stages: dict = field(default_factory=dict)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    llm_replayed: int = 0      # answers reused from registro_ia/ instead of calling the API
    usd_replayed: float = 0.0  # what those answers cost when they were first requested

    @contextmanager
    def stage(self, name: str):
        t0 = time.time()
        print(f"[radar] {name}...", flush=True)
        try:
            yield
        finally:
            self.stages[name] = round(time.time() - t0, 1)
            print(f"[radar] {name} listo en {self.stages[name]}s", flush=True)

    def to_json(self, comments: int, settings: Settings) -> dict:
        seconds = round(time.time() - self.started, 1)
        return {
            "comentarios_procesados": comments,
            "segundos": seconds,
            "tokens_ia": self.input_tokens + self.output_tokens,
            "tokens_ia_entrada": self.input_tokens,
            "tokens_ia_salida": self.output_tokens,
            "llamadas_ia": self.llm_calls,
            "usd_ia": round(self.usd, 4),
            "usd_por_mil_comentarios": round(self.usd / comments * 1000, 6) if comments else 0.0,
            "llamadas_ia_reutilizadas": self.llm_replayed,
            "usd_ia_original_de_reutilizadas": round(self.usd_replayed, 4),
            "modelo_ia": settings.llm_model if settings.use_llm else None,
            "modelo_embeddings": settings.embedding_model,
            "hardware": hardware(),
            "almacenamiento": "DuckDB en proceso sobre archivos Parquet locales",
            "etapas_segundos": self.stages,
        }


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    except (ValueError, OSError, AttributeError):
        ram_gb = None
    ram = f", {ram_gb} GB RAM" if ram_gb else ""
    return f"{cpu}, {os.cpu_count()} hilos{ram}, sin GPU"


@dataclass
class Context:
    datos: Path
    entrega: Path
    trabajo: Path
    settings: Settings
    con: duckdb.DuckDBPyConnection = field(default_factory=duckdb.connect)
    costs: Costs = field(default_factory=Costs)
    report: dict = field(default_factory=dict)
    # Filled by the narratives step and reused downstream.
    comments: object = None
    embeddings: object = None

    def __post_init__(self):
        self.trabajo.mkdir(parents=True, exist_ok=True)
        self.entrega.mkdir(parents=True, exist_ok=True)
        # Naive timestamps in the input are Argentine local time.
        self.con.execute("SET TimeZone = 'America/Argentina/Buenos_Aires'")

    def parquet(self, table: str) -> Path:
        return self.trabajo / f"{table}.parquet"

    def sql(self, query: str, params=None) -> list:
        """Run a statement and consume its result (DuckDB 2 blocks the connection on unread results)."""
        return self.con.execute(query, params or []).fetchall()

    def put_table(self, name: str, frame):
        """Store a pandas DataFrame as a DuckDB table."""
        self.con.register("_frame", frame)
        self.sql(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _frame")
        self.con.unregister("_frame")

    def save_report(self):
        (self.trabajo / "reporte.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2, default=str),
                                                   encoding="utf-8")
