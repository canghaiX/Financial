"""Parse local PDFs with MinerU/magic-pdf.

Default GPU policy uses CUDA device 1 so CUDA device 0 can serve the local vLLM.
The script records per-file status in
data/processed/mineru_parse_report.json.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_agentic_rag.config import load_yaml, resolve_project_path
from financial_agentic_rag.utils.io import write_json


def build_command(command: str, pdf_path: Path, output_dir: Path, method: str) -> list[str]:
    if command == "mineru":
        return [
            sys.executable,
            "-m",
            "mineru.cli.client",
            "-p",
            str(pdf_path),
            "-o",
            str(output_dir),
            "-m",
            method,
        ]
    if command == "magic-pdf":
        return [
            command,
            "-p",
            str(pdf_path),
            "-o",
            str(output_dir),
            "-m",
            method,
        ]
    return [command, str(pdf_path), str(output_dir)]


def ensure_mineru_backend_ready(command: str, backend: str | None) -> None:
    """Fail early if the current Python env cannot run MinerU's selected backend."""

    if command != "mineru" or not backend:
        return
    try:
        from mineru.cli.common import ensure_backend_dependencies
    except ImportError as exc:
        raise RuntimeError(
            "MinerU is not importable in the current Python environment. "
            f"Current Python: {sys.executable}"
        ) from exc
    try:
        ensure_backend_dependencies(backend)
    except Exception as exc:
        raise RuntimeError(
            f"MinerU backend '{backend}' is not ready in current Python: {sys.executable}. "
            'For local parsing, run: pip install -U "mineru[pipeline]>=3.2.3"'
        ) from exc
    if backend in {"pipeline", "hybrid-auto-engine"}:
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV cannot be imported by MinerU. If the error mentions 'libGL.so.1', "
                "install it with: conda install -n financial-rag -c conda-forge libgl libglib -y"
            ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval_config.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    mineru_config = config.get("mineru", {})
    command = mineru_config.get("command", "magic-pdf")
    if command != "mineru" and shutil.which(command) is None:
        raise RuntimeError(
            f"MinerU command '{command}' was not found. Install MinerU/magic-pdf first."
        )

    pdf_dir = resolve_project_path(mineru_config.get("source_pdf_dir", "pdf"))
    output_dir = resolve_project_path(mineru_config.get("output_dir", "data/processed/mineru"))
    report_path = resolve_project_path(
        config.get("output_paths", {}).get("parse_report", "data/processed/mineru_parse_report.json")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    overwrite = args.overwrite or bool(mineru_config.get("overwrite", False))
    method = mineru_config.get("method", "auto")
    backend = mineru_config.get("backend")
    lang = mineru_config.get("lang")
    ensure_mineru_backend_ready(command, backend)

    env = os.environ.copy()
    configured_cuda = os.getenv(
        "MINERU_CUDA_VISIBLE_DEVICES",
        str(mineru_config.get("cuda_visible_devices", "1")),
    )
    env.setdefault("CUDA_VISIBLE_DEVICES", configured_cuda)
    env.setdefault("MINERU_MODEL_SOURCE", str(mineru_config.get("model_source", "local")))

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    report: list[dict[str, str | int]] = []
    for pdf_path in pdfs:
        expected_dir = output_dir / pdf_path.stem
        if expected_dir.exists() and not overwrite:
            report.append({"file": str(pdf_path), "status": "skipped_exists", "returncode": 0})
            continue
        cmd = build_command(command, pdf_path, output_dir, method)
        if command == "mineru":
            if backend:
                cmd.extend(["-b", str(backend)])
            if lang:
                cmd.extend(["-l", str(lang)])
        result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
        report.append(
            {
                "file": str(pdf_path),
                "status": "ok" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1200:],
                "stderr_tail": result.stderr[-1200:],
            }
        )
        if result.returncode != 0:
            print(f"[failed] {pdf_path.name}: {result.stderr[-400:]}", file=sys.stderr)
        else:
            print(f"[ok] {pdf_path.name}")

    write_json(report_path, report)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
