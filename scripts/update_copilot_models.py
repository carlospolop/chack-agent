"""Fetch the list of models supported by GitHub Copilot CLI.

The Copilot CLI embeds an authoritative model catalog in its
``help config`` output under the ``model`` config key.  This script
parses that list so it can be stored in a static JSON file and kept
up-to-date via a GitHub Actions workflow.

Usage::

    python scripts/update_copilot_models.py [--output chack_agent/config/copilot_models.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_OUTPUT_PATH = Path("chack_agent/config/copilot_models.json")


def _find_copilot_cli() -> str:
    configured = os.environ.get("COPILOT_CLI_PATH", "").strip()
    if configured:
        found = shutil.which(configured)
        if found:
            return found
    found = shutil.which("copilot")
    if found:
        return found
    raise RuntimeError(
        "Could not find 'copilot' CLI on PATH. "
        "Install it (e.g. brew install copilot-cli) or set COPILOT_CLI_PATH."
    )


def fetch_copilot_models(copilot_path: str | None = None) -> list[str]:
    """Extract models from ``copilot help config`` output."""
    cli = copilot_path or _find_copilot_cli()
    result = subprocess.run(
        [cli, "help", "config"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"copilot help config exited with code {result.returncode}: "
            f"{(result.stderr or result.stdout)[:500]}"
        )

    output = result.stdout
    models: list[str] = []
    in_model_section = False
    for line in output.splitlines():
        stripped = line.strip()
        # Detect the start of the model section
        if stripped.startswith('"model"') or stripped.startswith("`model`"):
            in_model_section = True
            continue
        if in_model_section:
            # Each model line looks like:   - "claude-sonnet-4.6"
            match = re.match(r'^-\s+"([^"]+)"', stripped)
            if match:
                models.append(match.group(1))
            elif stripped.startswith("-"):
                # Possibly a bare model name
                bare = stripped.lstrip("- ").strip().strip('"').strip("'")
                if bare and "/" not in bare and len(bare) < 80:
                    models.append(bare)
            elif models:
                # No more model lines → end of section
                break
    return models


def write_models_json(models: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"models": models}
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(models)} copilot models to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch GitHub Copilot CLI model list and write JSON."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "--copilot-path",
        default="",
        help="Path to copilot CLI binary (default: auto-detect)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copilot_path = args.copilot_path.strip() or None
    models = fetch_copilot_models(copilot_path)
    if not models:
        raise RuntimeError("No models found in copilot help output")
    write_models_json(models, Path(args.output))


if __name__ == "__main__":
    main()
