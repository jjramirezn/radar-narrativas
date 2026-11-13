"""One command, end to end: radar --datos <carpeta> --entrega <carpeta>."""
import argparse
import os
from pathlib import Path

from . import coordination, ingest, naming, narratives, output, report, stance
from .context import Context, Settings
from .llm import LLM


def load_dotenv():
    """Read KEY=VALUE lines from a .env file (git-ignored) into the environment, never overriding it."""
    for path in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.removeprefix("export ").split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip("'\""))
            return


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(prog="radar", description="Narrativas y coordinación a escala (desafío QualyData).")
    ap.add_argument("--datos", type=Path, required=True, help="Carpeta con comentarios.csv.gz, cuentas.csv.gz, ...")
    ap.add_argument("--entrega", type=Path, default=Path("entrega"), help="Dónde escribir los 4 archivos")
    ap.add_argument("--trabajo", type=Path, default=None, help="Carpeta de trabajo (Parquet, cachés)")
    ap.add_argument("--sin-ia", action="store_true", help="No llamar a ningún modelo de lenguaje")
    ap.add_argument("--ia-nueva", action="store_true",
                    help="Ignorar las respuestas registradas en registro_ia/ y volver a pedirlas")
    ap.add_argument("--modelo-ia", default=Settings.llm_model)
    ap.add_argument("--modelo-embeddings", default=Settings.embedding_model)
    args = ap.parse_args()

    settings = Settings(embedding_model=args.modelo_embeddings, llm_model=args.modelo_ia, use_llm=not args.sin_ia)
    trabajo = args.trabajo or Path("trabajo") / args.datos.resolve().name
    ctx = Context(datos=args.datos, entrega=args.entrega, trabajo=trabajo, settings=settings)

    with ctx.costs.stage("ingesta"):
        ingest.run(ctx)
    llm = LLM(ctx, refresh=args.ia_nueva)
    with ctx.costs.stage("narrativas"):
        narratives.run(ctx)
    with ctx.costs.stage("nombres"):
        claims = naming.run(ctx, llm)
    with ctx.costs.stage("postura"):
        df = stance.run(ctx, llm, claims)
    ctx.put_table("etiquetas", df[["comentario_id", "narrativa", "postura"]])
    with ctx.costs.stage("coordinacion"):
        coordination.run(ctx)
    with ctx.costs.stage("polaridad"):
        if stance.fix_polarity(ctx, llm):
            ctx.put_table("etiquetas", ctx.comments[["comentario_id", "narrativa", "postura"]])
    with ctx.costs.stage("salida"):
        cost = output.write(ctx)
    with ctx.costs.stage("tablero"):
        report.write_dashboard(ctx, cost)
    ctx.save_report()
    print(f"[radar] listo: {cost['comentarios_procesados']} comentarios en {cost['segundos']}s -> {ctx.entrega}")


if __name__ == "__main__":
    main()
