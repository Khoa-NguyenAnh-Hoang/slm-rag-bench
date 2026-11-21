# slm-rag-bench — paper scaffold (Springer LNCS)

This directory holds the LaTeX source for the thesis paper:

**Cost, Latency, and Faithfulness: A Fixed-Retriever Benchmark for Small-Model RAG**

## Structure

```
paper/
├── main.tex                  % Entry point (Overleaf: upload this directory)
├── llncs.cls                 % Springer LNCS class (copy of "latex example paper")
├── splncs04.bst              % BibTeX style
├── README.md                 % This file
│
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_benchmark_design.tex
│   ├── 04_methodology.tex
│   ├── 05_results.tex        % Placeholder tables/figures; -- = await GPU run
│   ├── 06_threats.tex
│   ├── 07_reproducibility.tex
│   ├── 08_conclusion.tex
│   └── references.bib
│
├── tables/                   % tab_positioning, tab_scorecard, tab_paired, ...
└── figures/                  % Placeholders; export PNGs from results/report.html
```

## Compile (Overleaf recommended)

1. Upload the whole `paper/` directory to Overleaf.
2. Add `llncs.cls` (already present, copied from `latex example paper`).
3. Compiler: pdfLaTeX, then BibTeX, then pdfLaTeX x2:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Status

All result tables use `--` placeholders; every figure is a TODO.
Replace the placeholders with numbers from `results/scorecard_summary.md`
and `results/report.html` after the GPU matrix lands, then fill
Section~\ref{sec:results} prose.
