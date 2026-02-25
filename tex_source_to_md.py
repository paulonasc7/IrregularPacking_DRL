#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def extract_body(tex: str) -> str:
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", tex, flags=re.S)
    return m.group(1) if m else tex


def strip_block_envs(tex: str) -> str:
    envs = [
        "figure",
        "figure*",
        "table",
        "table*",
        "wrapfigure",
        "equation*",
        "align*",
    ]
    for env in envs:
        tex = re.sub(rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}", "", tex, flags=re.S)
    return tex


def drop_noise_commands(tex: str) -> str:
    # Avoid broad multiline regex stripping to prevent accidental content loss.
    out_lines = []
    for line in tex.splitlines():
        s = line.strip()
        if s.startswith("\\maketitle"):
            continue
        if s.startswith("\\IEEEpeerreviewmaketitle"):
            continue
        if s.startswith("\\markboth"):
            continue
        if s.startswith("\\bibliographystyle"):
            continue
        if s.startswith("\\bibliography"):
            continue
        if s.startswith("\\IEEEoverridecommandlockouts"):
            continue
        out_lines.append(line)
    out = "\n".join(out_lines)
    out = re.sub(r"\\label\{[^{}]*\}", "", out)
    out = re.sub(r"\\vspace\{[^{}]*\}", "", out)
    out = re.sub(r"\\hspace\{[^{}]*\}", "", out)
    out = re.sub(r"\\centering", "", out)
    return out


def convert_structure(tex: str) -> str:
    # Keep core sections as Markdown headings.
    tex = re.sub(r"\\title\{(.*?)\}", r"# \1", tex, flags=re.S)
    tex = re.sub(r"\\author\{(.*?)\}", r"**Authors:** \1", tex, flags=re.S)
    tex = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", r"## Abstract\n\1", tex, flags=re.S)
    tex = re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", r"## Keywords\n\1", tex, flags=re.S)

    tex = re.sub(r"\\section\*?\{(.*?)\}", r"\n## \1\n", tex)
    tex = re.sub(r"\\subsection\*?\{(.*?)\}", r"\n### \1\n", tex)
    tex = re.sub(r"\\subsubsection\*?\{(.*?)\}", r"\n#### \1\n", tex)

    # Keep equations as display math blocks.
    tex = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", r"\n$$\n\1\n$$\n", tex, flags=re.S)
    tex = re.sub(r"\\begin\{align\}(.*?)\\end\{align\}", r"\n$$\n\1\n$$\n", tex, flags=re.S)
    tex = re.sub(r"\\\[(.*?)\\\]", r"\n$$\n\1\n$$\n", tex, flags=re.S)

    # Convert simple lists.
    tex = re.sub(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", lambda m: _convert_enumerate(m.group(1)), tex, flags=re.S)
    tex = re.sub(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", lambda m: _convert_itemize(m.group(1)), tex, flags=re.S)

    return tex


def _convert_enumerate(body: str) -> str:
    items = [s.strip() for s in re.split(r"\\item", body) if s.strip()]
    return "\n" + "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1)) + "\n"


def _convert_itemize(body: str) -> str:
    items = [s.strip() for s in re.split(r"\\item", body) if s.strip()]
    return "\n" + "\n".join(f"- {item}" for item in items) + "\n"


def clean_inline_latex(tex: str) -> str:
    # References/citations -> bracket tags.
    tex = re.sub(r"~?\\cite\{(.*?)\}", lambda m: "[" + m.group(1).replace(",", "; ") + "]", tex)
    tex = re.sub(r"\\ref\{(.*?)\}", r"\1", tex)
    tex = re.sub(r"\\eqref\{(.*?)\}", r"(\1)", tex)

    # Common formatting commands.
    tex = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", tex, flags=re.S)
    tex = re.sub(r"\\textit\{(.*?)\}", r"*\1*", tex, flags=re.S)
    tex = re.sub(r"\\emph\{(.*?)\}", r"*\1*", tex, flags=re.S)
    tex = re.sub(r"\\underline\{(.*?)\}", r"\1", tex, flags=re.S)

    # Handle escaped characters.
    tex = tex.replace(r"\%", "%").replace(r"\_", "_").replace(r"\&", "&")

    # Remove leftover command names but keep their content in braces.
    tex = re.sub(r"\\[a-zA-Z]+\*?", "", tex)

    # Cleanup braces used only for grouping.
    tex = tex.replace("{", "").replace("}", "")
    tex = tex.replace("\\", "")

    # Normalize whitespace.
    tex = re.sub(r"[ \t]+", " ", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    tex = re.sub(r"^\s+|\s+$", "", tex, flags=re.S)
    return tex + "\n"


def tex_to_md(input_path: Path, output_path: Path) -> None:
    tex = input_path.read_text(encoding="utf-8", errors="ignore")
    body = extract_body(tex)
    body = strip_block_envs(body)
    body = drop_noise_commands(body)
    body = convert_structure(body)
    md = clean_inline_latex(body)
    output_path.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LaTeX source paper into rough Markdown.")
    parser.add_argument("input_tex", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    tex_to_md(args.input_tex, args.output)


if __name__ == "__main__":
    main()
