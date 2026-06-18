"""Replace compact-scene placeholders in technical_report.md with mirrored molecular structure."""
from __future__ import annotations

import re
from pathlib import Path

REPORT = Path(
    r"D:\Users\Escritorio\U\Ot2026-2doSemM\tesis-magister-latex\technical_report.md"
)

VERSION_SECTIONS = [
    ("3.1.1", "fst_gpu", "grid1–grid4"),
    ("3.1.2", "scnd_gpu", "grid1–grid3"),
    ("3.1.3", "std", "grid1–grid4"),
    ("3.1.4", "hybrid", "grid1–grid4"),
]

INTRO_OLD = (
    "Molecular (sparse PDB) scenes are fully documented below. Compact (synthetic grid) scenes were\n"
    "tested with the same protocol; per-metric write-ups for compact scenes are reserved as placeholders\n"
    "until results are added."
)

INTRO_NEW = (
    "Molecular (sparse PDB) scenes are fully documented below. Compact (synthetic grid) scenes use the\n"
    "same protocol and metrics; figures are linked under `img/chapter3/results/compact/` with per-metric\n"
    "analysis left as placeholders (`Análisis (compact)`) for future write-up."
)

COMPACT_PLACEHOLDER = re.compile(
    r"\n##### Compact \(synthetic\) scenes\n\n"
    r"> \*\*Placeholder\.\*\*[^\n]*\n",
    re.MULTILINE,
)


def _adapt_caption(line: str, grids_label: str) -> str:
    line = line.replace("distance from the molecule", "distance from the scene")
    line = re.sub(
        r"across different scenes and camera r[a-z\.]*",
        f"across compact grid scenes {grids_label}",
        line,
        flags=re.IGNORECASE,
    )
    line = line.replace(
        "Each line corresponds to a different scene.",
        "Each line corresponds to a different grid (grid1, grid2, …).",
    )
    line = line.replace(
        "Each line corresponds to a different scene",
        "Each line corresponds to a different grid (grid1, grid2, …)",
    )
    return line


def build_compact_section(mol_block: str, version_folder: str, grids_label: str) -> str:
    lines_out = ["##### Compact (synthetic) scenes", ""]
    mol_prefix = f"img/chapter3/results/{version_folder}/"
    compact_prefix = f"img/chapter3/results/compact/{version_folder}/"

    for raw in mol_block.splitlines():
        s = raw.strip()
        if s.startswith("##### "):
            lines_out.extend(["", raw, ""])
            continue
        if s.startswith("![") and mol_prefix in raw:
            lines_out.extend(["", raw.replace(mol_prefix, compact_prefix), ""])
            continue
        if s.startswith("*Caption:"):
            lines_out.extend(
                ["", _adapt_caption(raw, grids_label), "", "> **Análisis (compact).**", ""]
            )
            continue

    return "\n".join(lines_out).rstrip() + "\n"


def patch_report() -> None:
    text = REPORT.read_text(encoding="utf-8")
    text = text.replace(INTRO_OLD, INTRO_NEW)

    for sec_id, version_folder, grids_label in VERSION_SECTIONS:
        sec_re = re.compile(
            rf"(#### {re.escape(sec_id)}[^\n]*\n)(.*?)(?=\n#### 3\.1\.|\n## |\Z)",
            re.DOTALL,
        )
        m = sec_re.search(text)
        if not m:
            raise SystemExit(f"Section {sec_id} not found")
        header, body = m.group(1), m.group(2)

        mol_m = re.search(
            r"##### Molecular \(sparse\) scenes\n(.*?)(?=\n##### Compact \(synthetic\) scenes)",
            body,
            flags=re.DOTALL,
        )
        compact_m = COMPACT_PLACEHOLDER.search(body)
        if not mol_m or not compact_m:
            raise SystemExit(f"No molecular/compact block in section {sec_id}")
        compact = build_compact_section(mol_m.group(1), version_folder, grids_label)
        new_body = body[: compact_m.start()] + "\n" + compact + body[compact_m.end() :]
        text = text[: m.start()] + header + new_body + text[m.end() :]

    REPORT.write_text(text, encoding="utf-8")
    print(f"Updated {REPORT} ({len(VERSION_SECTIONS)} compact sections)")


if __name__ == "__main__":
    patch_report()
