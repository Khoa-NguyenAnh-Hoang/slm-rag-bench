# Cost, Latency, and Faithfulness: A Fixed-Retriever Benchmark for Small-Model RAG

## Revised Full Paper (After Applying the Section-by-Section Fixes)

---

## Abstract

Small open-weight language models are attractive for retrieval-augmented generation (RAG) because they are deployable under tighter cost and hardware budgets, but prior work shows that RAG systems still struggle with faithfulness, negative rejection, and noisy evidence handling, especially in small-to-mid model settings [mala2026faithfulness, chen2023rgb, zeng2025rare]. Existing evaluations often emphasize answer quality while underreporting latency, TTFT, and monetary cost, which makes deployability comparisons incomplete [wu2024survey, brown2025systematic]. This paper presents a fixed-retriever benchmark for small-model RAG on a HotpotQA distractor subset, varying only generator choice and a corrective retrieval stage while logging retrieval quality, adjusted accuracy, faithfulness, abstention, latency, TTFT, and two cost axes from reproducible per-query artifacts [mala2026faithfulness, es2023ragas, saadfalcon2023ares, elkiran2026empirical]. Retrieval is held constant across all cells (BM25 sparse candidates followed by a MiniLM cross-encoder reranker), and only the generator and the presence of a corrective retrieval stage vary across a 3 × 2 grid of three small open models and two pipelines. Every cell is scored on retrieval quality, adjusted accuracy, faithfulness, abstention, end-to-end latency, time to first token, and two cost axes: an API-equivalent cost and a GPU-amortised cost. We report results for n=300 queries per cell on a HotpotQA distractor subset. [TODO: one-sentence headline result once the matrix has run.] We release the harness, the per-query records, and the scorecards so that every reported number is reproducible from committed artifacts.

**Keywords:** Retrieval-augmented generation · Small language models · Evaluation benchmark · Faithfulness · Latency · Inference cost

---

## 1. Introduction

Retrieval-augmented generation (RAG) grounds a language model's answer in retrieved evidence and is now a standard pattern for question answering over private or fast-changing corpora [lewis2020rag]. In deployment, the useful question is rarely "which model is most accurate?" but "which model gives the best answer per unit of money and time?" Small open-weight models in the 1–8B parameter range are attractive on that second question: they can be served on a single consumer or workstation GPU, their weights and prompts stay private, and their per-query cost is predictable. Prior work shows that RAG systems still struggle with faithfulness, negative rejection, and noisy evidence handling, especially in small-to-mid model settings [mala2026faithfulness, chen2023rgb, zeng2025rare]. The trade-off is not simply that small models are worse than proprietary ones; it is that RAG evaluation is multi-component, and errors can originate from retrieval, grounding, or generation [murugaraj2025ragvue]. Model choice should therefore be framed as a quality–latency–cost trade-off rather than an accuracy-only comparison [wu2024survey, sharma2025survey].

Existing evaluations capture pieces of this picture but not the whole of it. RAGAS provides reference-free faithfulness and relevance metrics [es2023ragas]. ARES adds lightweight trained judges for domain-shift evaluation [saadfalcon2023ares]. eRAG evaluates retrieval quality in a way that correlates better with downstream performance than raw relevance labels [salemi2024erag]. RAGVUE decomposes evaluation into claim-level and diagnostic components [murugaraj2025ragvue]. RARE stresses perturbation robustness and time-sensitive corpora [zeng2025rare]. RGB stresses noise robustness and negative rejection [chen2023rgb]. Mala et al. give a systematic study of faithfulness hallucinations across 2–8B models and several retrievers, and introduce adjusted accuracy and rejection rate [mala2026faithfulness]. EvaRAG and Rasheed et al. add cost and latency to the RAG scorecard and use statistical tests to rank pipeline components [elkiran2026empirical, rasheed2025empirical]. Many evaluations still underreport efficiency and cost relative to answer quality [wu2024survey, saadfalcon2023ares]. What this paper does is combine these elements in one controlled benchmark design: (i) retrieval is held fixed so that two cells differ only by the model or the corrective stage, (ii) faithfulness, abstention, latency, and cost are all reported for the same queries, and (iii) every number is traceable to a committed per-query record. When retrieval is allowed to vary between model runs, a quality difference can be a retrieval difference in disguise, which is exactly the confound that makes quality-only rankings misleading [rasheed2025empirical].

This paper makes four contributions:

1. **A fixed-retriever benchmark design.** We hold the retriever constant (BM25 sparse candidates, then a MiniLM cross-encoder reranker) and vary only the generator and the corrective stage, giving a 3 × 2 grid of three small open models and two pipelines (Section 3). The design isolates generator and corrective-stage effects under a constant retrieval policy; it does not claim to identify the globally best RAG architecture [elkiran2026empirical].

2. **A multi-axis scorecard.** Each cell is scored on retrieval quality, adjusted accuracy, hallucination and rejection rate, RAGAS-style faithfulness, end-to-end latency, time to first token, and two cost axes (Section 4).

3. **A measurement contract.** Rather than trusting notebook conventions, the harness enforces seven rules about what counts as a valid measurement, and fails a run that violates them (Section 4).

4. **An open, reproducible artifact.** The harness, the configuration, the per-query records, and the scorecards are released together, so a reported number can be traced back to the run that produced it (Section 7).

We organise the study around three research questions:

- **RQ1.** Among small open models evaluated under a fixed retriever, which gives the best faithfulness for a given latency and cost budget?
- **RQ2.** Does adding a corrective retrieval stage improve the cost–latency–faithfulness frontier, or only shift it?
- **RQ3.** Does the model ranking change when latency and cost are included in the comparison instead of quality alone?

The rest of the paper is organised as follows. Section 2 reviews related work and positions this benchmark. Section 3 describes the benchmark design. Section 4 defines the metrics, the judge protocol, and the statistical tests. Section 5 reports results and discussion. Section 6 discusses threats to validity, Section 7 covers reproducibility, and Section 8 concludes.

---

## 2. Related Work

**RAG evaluation frameworks.** RAGAS introduced reference-free metrics for the retriever and the generator separately, including faithfulness, answer relevance, and context relevance, and validated them against human judgements [es2023ragas]. It measures text quality in isolation and does not account for latency or monetary cost. ARES adds lightweight trained judges and evaluates under domain shift, improving on pure prompting-based judges but still centering on answer quality [saadfalcon2023ares]. eRAG evaluates retrieval quality in a way that correlates better with downstream RAG performance than raw relevance labels, reinforcing that retrieval and generation should be evaluated jointly but separately [salemi2024erag]. RAGVUE decomposes evaluation into claim-level and diagnostic components, making it possible to attribute failures to retrieval, grounding, or reasoning [murugaraj2025ragvue]. RARE stresses perturbation robustness and time-sensitive corpora, showing that RAG systems can degrade under realistic distribution shifts [zeng2025rare]. RGB reframes evaluation as a stress test of four capabilities: noise robustness, negative rejection, information integration, and counterfactual robustness, and reports rejection rate alongside accuracy [chen2023rgb]. These frameworks score a pipeline; most still treat cost and latency as secondary rather than first-class axes.

**Corrective and self-correcting retrieval.** CRAG adds a lightweight retrieval evaluator that scores each retrieved document and triggers one of three actions—_Correct_, _Incorrect_, or _Ambiguous_—then refines the context with a decompose-then-recompose step [yan2024crag]. It improves robustness when retrieval degrades, but published overhead numbers exclude retrieval and preprocessing, and the method assumes an external evaluator checkpoint plus an optional web-search fallback [yan2024crag]. The version evaluated here uses a local proxy evaluator and no live search (Section 3); it is therefore a derivative corrective design, not a full CRAG reproduction.

**Cost, latency, and component analysis.** EvaRAG builds a large controlled configuration space over indexing, similarity, retrieval, and reranking, and logs semantic quality together with mean and P95 latency, token usage, and monetary cost per query [elkiran2026empirical]. Rasheed et al. apply factorial ANOVA and effect-size analysis to the same family of RAG components and show that retriever and similarity choices dominate quality variance while index choice mainly trades latency [rasheed2025empirical]. These works establish the scorecard this paper adopts, but they study large hosted models and treat retrieval as an experimental axis. Fixing retrieval and moving to small open models changes the question from "which pipeline component matters?" to "which deployable model should I pick?".

**Faithfulness in small models.** Mala et al. benchmark faithfulness hallucinations for 2–8B open models across sparse, dense, hybrid, and reranked retrieval, and show that stronger retrieval can raise raw accuracy while also raising hallucination rate; they introduce adjusted accuracy, which scores precision over attempted answers only [mala2026faithfulness]. Their study motivates our main faithfulness metric and our separation of rejection rate from accuracy. Importantly, faithfulness here means support by retrieved evidence, not factual truth; the two can diverge [murugaraj2025ragvue].

Table 1 summarises how existing work differs from this benchmark.

---

## 3. Benchmark Design

**Retrieval is fixed, and that is the point.** The benchmark answers "which small model should I pick under a fixed retrieval policy?", so retrieval must not vary between cells—otherwise two cells can differ by more than their model, which is exactly the confound that makes quality-only rankings misleading [rasheed2025empirical]. Component studies show that retriever choice dominates much of the variance in downstream quality, while rerankers can still shift correctness [elkiran2026empirical]. Every cell therefore uses the same fixed retriever:

1. BM25 sparse candidate pool via `bm25s` (depth 30),
2. MiniLM-L6 cross-encoder reranker over the 30 candidates, returning the top 10.

There is no dense retrieval arm. On the full 483,696-paragraph HotpotQA distractor corpus, the sparse pool already holds at least one gold paragraph 98.00% of the time, and the reranked top-10 reaches 97.67%, so even a perfect dense fusion could add at most 0.33 percentage points of recall@10—far too little to justify varying the retriever between cells. The reranker runs in every cell and is not an experimental axis. This design isolates **generator and corrective-stage effects** under a constant retrieval policy; it is not a claim about the globally best RAG architecture [elkiran2026empirical].

**Matrix.** Three small open generator models × two pipelines (gives 6 cells, Table 2): `Qwen2.5-1.5B-Instruct`, `Qwen3-4B-Instruct-2507`, and `LLaMA-3.1-8B-Instruct` (Llama requires an accepted licence). The pipelines are:

- `baseline`: retrieved context is passed directly to the generator.
- `crag`: each retrieved context is scored by a local corrective evaluator (a `flan-t5-base` proxy for CRAG's trained 0.77B T5-large evaluator) and one of `Correct`/`Incorrect`/`Ambiguous` is triggered. On `Correct` and `Ambiguous`, context is refined with decompose-then-recompose strip filtering; on `Incorrect`, context is emptied and the model must abstain or answer from parametric memory—there is no live web-search fallback, so an `Incorrect` action is treated as a measured abstention rather than rescued accuracy. This is a derivative corrective design, not a full CRAG reproduction [yan2024crag].

**Judge.** Faithfulness and relevance are scored by a separate judge model (`Mistral-7B-Instruct-v0.3`, served as an AWQ checkpoint) on its own vLLM endpoint, never concurrently with the generator. The server identities of both judge and generator are recorded in every scorecard; a gate enforces _judge ≠ generator_ so no cell can be self-scored by the model that wrote its answer (Section 4).

**Data.** HotpotQA distractor subset, sampled once at n=300 queries with seed 42 from the train split that also builds the corpus—90,447 rows yielding 483,696 paragraphs after title–text deduplication. Eval and corpus come from the same split by design, so every query's gold paragraphs are present in the index (this makes the bench in-domain rather than out-of-domain; see Threat T3, Section 6). HotpotQA is a useful multi-hop benchmark but is Wikipedia-centric and less reflective of proprietary or dynamic production corpora [wu2024survey, mala2026faithfulness]; Section 6 discusses this limitation.

---

## 4. Methodology

**Metrics.** Each cell reports the following numbers, measured end-to-end rather than assembled from stage timings (see Rule R1 below). Retrieval, answer quality, and grounding are evaluated separately [murugaraj2025ragvue, mortaheb2025ragcheck].

- **Retrieval quality**—MAP@k, NDCG@k, Recall@k, MRR, and Coverage, recorded at the actual retrieval depth (queries at different depths are not comparable) [rasheed2025empirical].
- **Adjusted accuracy**—precision over attempted answers only:

  AdjAcc = Correct / (Correct + Hallucinated) × 100%

  This follows Mala et al. and rewards abstention when context is insufficient [mala2026faithfulness].

- **Hallucination rate** and **rejection rate**—the share of answers judged unsupported or refused (_Refusal_), respectively. Abstention and rejection are reported separately because robustness work shows RAG systems often fail on negative rejection and risk control [chen2023rgb, chen2024counterfactual].
- **Faithfulness**—RAGAS-style generator groundedness: an LLM decomposes the answer into statements and judges each against the retrieved context; faithfulness is the share of supported statements [es2023ragas]. Faithfulness here means support by retrieved evidence, not factual truth; the two can diverge [murugaraj2025ragvue]. Statements and verdicts are produced by the fixed judge model (Section 4).
- **Latency**—mean and P95 end-to-end latency per query (seconds), plus mean TTFT (time to first token), plus tokens per second. P95 tail matters because a few slow queries dominate user-visible latency in a served system.
- **Cost**—two numbers per cell:
  - API-equivalent cost from prompt/completion tokens and a published provider price table (provider and verification date recorded);
  - GPU-amortised cost from measured wall-clock and an hourly rate.

  Neither number is allowed to default to 0.0; an unavailable measurement is recorded as "None" rather than zero [elkiran2026empirical].

**Measurement contract (enforced).** The harness runs seven rules and fails a cell that breaks any of them:

1. Latency is measured with one outer timer per query, never summed from stage timings.
2. An unavailable measurement is `None`, never `0.0`.
3. A missing reference answer is `Unknown`, never a fabricated label.
4. Unknown keys, missing usage blocks, or missing config raise; there is no silent fallback.
5. Scores carry their retrieval depth; costs carry pricing version and verification metadata.
6. The cell manifest is written before any model is loaded.
7. Every value is read from `configs/experiment.yaml` at exactly one place.

A source-tree scanner greps the code for fabricated-zero patterns (docstrings blanked first, so prose about the rules does not trip it), and four of the rules run as hard gates inside each cell.

**Judge protocol.** RAGAS-style faithfulness and relevance are produced by the judge model, not by the generator (Section 3). Both serve identities—generator and judge—are written into every scorecard, and a hard gate enforces _judge ≠ generator_. The judge runs streaming _after_ the generator server has been stopped, so GPU memory can be reused. LLM judges are useful but prompt-sensitive and unstable [murugaraj2025ragvue]; we therefore use one judge consistently across all cells and report this as a limitation (Section 6).

**Statistical analysis plan.** The comparison of interest is crag vs. baseline within each model (paired by query). We plan:

- McNemar's exact test on per-query correctness (discordant counts: crag correct / baseline wrong vs. reverse).
- Wilcoxon signed-rank test on per-query latency differences.
- Benjamini–Hochberg false-discovery-rate adjustment across all reported model × pipeline comparisons.

All three tests are implemented in `analysis/scorecard.py` and run against committed scorecards; they are scaffolded here but their results are filled in after the matrix runs (Section 5).

---

## 5. Results & Discussion

[TODO: Replace every "--" value in this section once the matrix has run. All numbers here come from committed `results/*/scorecard.json`.]

### 5.1 Scorecard

Table 3 gives the cross-cell scorecard. Each row is one cell of the 3 × 2 matrix, evaluated on n=300 queries. We present results as a **frontier comparison** rather than a single scalar winner, because recent work shows that no single RAG configuration dominates all quality and efficiency axes [sharma2025survey, brown2025systematic].

[TODO: Reading guide once numbers exist: highlight which cell minimises cost per faithfulness point, which minimises P95 latency, and whether crag beats baseline within each model. Do not lead with a single "best" cell.]

### 5.2 Paired crag vs. baseline tests

Because each model pair shares the same queries, we test the crag effect within models. Table 4 reports McNemar's exact test for correctness and Wilcoxon signed-rank for latency, with Benjamini–Hochberg false-discovery-rate adjustment across all comparisons.

[TODO: Interpretation template: significance means the corrective stage reliably shifts a given axis; non-significance means the added evaluator latency is not justified by the measured gain. Interpret gains cautiously: corrective pipelines can improve citation faithfulness even when fluency gains are modest [michaels2026dsgt].]

### 5.3 Pareto fronts

Figures 1 and 2 plot faithfulness against P95 latency and against total cost respectively. Cells on the front are candidates for "best deployable model" under different budget constraints.

**Figure 1:** Faithfulness vs. end-to-end P95 latency.

**Figure 2:** Faithfulness vs. cost per query.

### 5.4 Human-label agreement

[TODO: Table: agreement percentage and Cohen's kappa from `src.monitor.labels score` on B2's 50 hand labels (*Factual / Hallucination / Refusal*).]

### 5.5 CRAG proxy validity

[TODO: Table: proxy/action agreement against vendored CRAG reference labels for popqa, bio, pubqa, arc_challenge, from `tools/crag_validity.py`. The smoke result already shows the proxy is non-discriminative on negatives (0% agreement on Incorrect). Do not interpret the Incorrect path as measured accuracy.]

### 5.6 Deviation log

Any departure from the plan is recorded with its consequence:

[TODO: Current deviations: (1) self-judge in Phase A, closed by giving the judge its own server; (2) flan-t5-base proxy non-discriminative on CRAG Incorrect negatives; interpret with caution.]

### 5.7 Discussion

HotpotQA-based findings may not transfer to multilingual, domain-specific, or time-sensitive corpora [elkiran2026empirical, zeng2025rare, li2025biomedical]. Robustness under noise, contradiction, and multi-hop integration remains a broader open problem beyond this benchmark [chen2023rgb, zeng2025rare, li2025biomedical]. The contribution here is a reproducible comparison harness, not a definitive answer to all production RAG selection questions [wu2024survey, sharma2025survey]. Where possible, we distinguish retrieval-supported failures from generation failures, following disentangling and diagnostic evaluation work [mala2026faithfulness, murugaraj2025ragvue].

---

## 6. Threats to Validity

**T1 Construct validity (what we measure).** Faithfulness is computed by an LLM judge rather than by human annotators on every answer, so it inherits judge sensitivity—prompt phrasing, temperature, and the underlying model can all move the score [es2023ragas, murugaraj2025ragvue]. The 50-sample human-label agreement study (Section 5) calibrates this risk. Faithfulness can also diverge from factual correctness: an answer may be supported by retrieved evidence yet factually wrong, or factually right yet unsupported by the retrieved context [murugaraj2025ragvue]. API-equivalent cost is also a proxy: published provider prices vary by up to an order of magnitude for the same model, so the GPU-amortised number is the primary cost result.

**T2 Internal validity (fairness of the comparison).** Four risks were addressed before the runs: (i) self-judge—the judge is now a separate server, enforced by a hard gate; (ii) CRAG proxy correctness—the flan-t5-base evaluator is not discriminative on Incorrect passages, so its action distribution is reported with an explicit caveat rather than treated as ground truth; (iii) data leakage—eval and corpus are drawn from the same train split by design (in-domain by construction), so retrieval metrics are not confounded by missing gold paragraphs; (iv) LLM-as-judge instability—judge prompt and decoding settings are fixed across cells, and the limitation is reported rather than hidden [murugaraj2025ragvue].

**T3 External validity (what generalises).** Results are single-corpus, English-only, single-split (HotpotQA distractor train), no dense retriever arm, no live web search, and batch-1 serving under vLLM defaults. HotpotQA is Wikipedia-centric, which limits transfer to multilingual, domain-specific, and time-sensitive production corpora [wu2024survey, elkiran2026empirical, zeng2025rare, li2025biomedical]. Findings do not generalise out-of-domain, to other languages, or to high-concurrency serving. A small model may also answer from parametric memory rather than retrieved context.

**T4 Statistical validity (significance claims).** n=300 per cell gives roughly ±2.9 percentage points standard error at a 50% rate. Paired tests plus Benjamini–Hochberg adjustment limit false discoveries across the 3 × 2 comparisons; effect sizes should accompany any claim of significance rather than p-values alone.

**T5 System validity (pipeline-specific).** The fixed chunking strategy, retrieval depth (30 candidates → top 10), and embedding assumption are design choices, not neutral defaults; different depths or chunk sizes can shift both retrieval and downstream correctness [elkiran2026empirical]. The local CRAG proxy also omits the published web-search rescue path and replaces the trained evaluator with a smaller derivative model, so the crag cells are a derivative corrective design rather than a full CRAG reproduction [yan2024crag].

---

## 7. Reproducibility

**Artifacts.** Every finished cell writes, under `results/<cell_id>/`:

- `manifest.json`—configuration, corpus size, pricing version, git revision, hardware;
- `queries.jsonl`—one runtime record per query (question, answer, contexts, retrieved ids, gold ids, latency, TTFT, tokens, cost, label, extra fields);
- `scorecard.json`—aggregate summary, retrieval metrics, adjusted accuracy, RAGAS faithfulness, gate outcomes, judge and generator serve identifiers.

`queries.jsonl` and `scorecard.json` are committed artifacts; re-running a cell deletes its own `queries.jsonl` first, so partial output is never mixed with a fresh run. Cross-cell summary and statistics are generated by `analysis/scorecard.py` into `results/scorecard_summary.md`, and a static report is rendered by `analysis/report.py` into `results/report.html`. Traceable per-query artifacts respond to broader calls for provenance-aware, reproducible RAG evaluation [brown2025systematic, murugaraj2025ragvue, saadfalcon2023ares, sharma2025survey].

**Checks before the GPU ladder.** `tools.selfcheck` runs nine CPU checks covering retrieval, metrics logging, the pipeline, the cell gates, the matrix, data loading, the retriever, and the CRAG evaluator; every gate condition is verified to fire when broken. No cell had run before this paper was written, so Tables 3–8 contain "--" placeholders and await the GPU matrix.

**Availability.** The harness, configs, vendor data, and this paper's source are at https://github.com/Khoa-NguyenAnh-Hoang/slm-rag-bench (placeholder DOI to be assigned at release). Costs assume `configs/pricing.yaml`; two renter-specific values (`serve.gpu_memory_utilization` and `gpu_usd_per_hour`) must be set before a rented-box run and are recorded in every manifest.

---

## 8. Conclusion

[TODO: One-paragraph conclusion once the matrix has run: which cell dominates the faithfulness–cost–latency frontier, whether crag is worth its evaluator latency, and the model recommendation under a fixed budget.]

This paper offers a **controlled, reproducible benchmark** for comparing small-model RAG under joint quality, faithfulness, latency, and cost reporting. It does not claim to establish a new standard or to settle all production RAG selection questions; broader questions of domain transfer, dynamic corpora, and retrieval-policy adaptation remain open [brown2025systematic, wu2024survey, sharma2025survey, li2025biomedical].

**Limitations.** The grid compares whole pipelines rather than isolating CRAG components; the corrective stage bundles evaluation, strip editing, and evaluator latency. The retriever is fixed, so no claim is made about dense vs. sparse retrieval. Costs depend on provider pricing at a single verification date, and the data is English-only, single-split HotpotQA distractor. The corrective pipeline is a derivative design, not a full CRAG reproduction.

**Future work.** Add a dense retrieval arm to quantify its recall ceiling; try alternative judges to bound judge sensitivity; extend to multilingual small-model subsets; evaluate batched serving rather than batch-1; sweep AWQ vs. fp16 serving settings; and test on time-sensitive or domain-specific corpora to probe external validity [zeng2025rare, li2025biomedical, almohaimeed2026arabic].

[TODO: Add Acknowledgements, Funding, and Ethics statements if required by the venue.]

---

## Revised Bibliography (Additions)

The following entries should be added to `sections/references.bib` to support the revised claims:

```bibtex
@article{saadfalcon2023ares,
  author    = {Saad-Falcon, Jon and Khattab, Omar and Potts, Christopher and Zaharia, Matei},
  title     = {ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems},
  journal   = {arXiv preprint arXiv:2311.09476},
  year      = {2023}
}

@inproceedings{salemi2024erag,
  author    = {Salemi, Alireza and Zamani, Hamed},
  title     = {Evaluating Retrieval Quality in Retrieval-Augmented Generation},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year      = {2024},
  doi       = {10.1145/3626772.3657957}
}

@article{murugaraj2025ragvue,
  author    = {Murugaraj, K. and Lamsiyah, S. and Theobald, M.},
  title     = {RAGVUE: A Diagnostic View for Explainable and Automated Evaluation of Retrieval-Augmented Generation},
  pages     = {512--526},
  year      = {2025},
  doi       = {10.48550/arxiv.2601.04196}
}

@article{zeng2025rare,
  author    = {Zeng, Y. and Cao, T. and Wang, D.-Q. and Zhao, X.-R. and Qiu, Z. and Ziyadi, M. and Wu, T.-S. and Li, L.},
  title     = {RARE: Retrieval-Aware Robustness Evaluation for Retrieval-Augmented Generation Systems},
  journal   = {arXiv preprint arXiv:2506.00789},
  year      = {2025}
}

@article{brown2025systematic,
  author    = {Brown, A. and Roman, M. and Devereux, B.},
  title     = {A Systematic Literature Review of Retrieval-Augmented Generation: Techniques, Metrics, and Challenges},
  journal   = {arXiv preprint arXiv:2508.06401},
  year      = {2025}
}

@article{wu2024survey,
  author    = {Wu, S.-Y. and Xiong, Y. and Cui, Y.-F. and Wu, H.-L. and Chen, C. and Yuan, Y. and Huang, L.-M. and Liu, X. and Kuo, T.-W. and Guan, N. and Xue, C.},
  title     = {Retrieval-augmented generation for natural language processing: a survey},
  journal   = {Artificial Intelligence Review},
  volume    = {59},
  year      = {2024},
  doi       = {10.1007/s10462-026-11605-7}
}

@article{sharma2025survey,
  author    = {Sharma, C.},
  title     = {Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers},
  journal   = {arXiv preprint arXiv:2506.00054},
  year      = {2025}
}

@article{li2025biomedical,
  author    = {Li, M. and Zhan, Z.-F. and Yang, H. and Xiao, Y. and Zhou, H.-X. and Huang, J.-T. and Zhang, R.},
  title     = {Benchmarking retrieval-augmented large language models in biomedical NLP: Application, robustness, and self-awareness},
  journal   = {Science Advances},
  volume    = {11},
  year      = {2025},
  doi       = {10.1126/sciadv.adr1443}
}

@inproceedings{chen2024counterfactual,
  author    = {Chen, L. and Zhang, R.-Q. and Guo, J. and Fan, Y.-X. and Cheng, X.-Q.},
  title     = {Controlling Risk of Retrieval-augmented Generation: A Counterfactual Prompting Framework},
  pages     = {2380--2393},
  year      = {2024},
  doi       = {10.48550/arxiv.2409.16146}
}

@article{almohaimeed2026arabic,
  author    = {Almohaimeed, S. and Alabduljabbar, A. and Jari, M. and Alkhowaiter, M. and Almohaimeed, S. and Rahhal, M. A. M.},
  title     = {Benchmarking retrieval augmented generation LLMs for Arabic noise robustness},
  journal   = {Frontiers in Big Data},
  volume    = {9},
  year      = {2026},
  doi       = {10.3389/fdata.2026.1884673}
}

@article{michaels2026dsgt,
  author    = {Michaels, B. and Johnson, B.},
  title     = {DS@GT ARC at LongEval: Citation Integrity and Factual Grounding in Scientific QA},
  journal   = {arXiv preprint arXiv:2607.14400},
  year      = {2026}
}

@article{mortaheb2025ragcheck,
  author    = {Mortaheb, M. and Khojastepour, M. and Chakradhar, S. T. and Ulukus, S.},
  title     = {RAG-Check: Evaluating Multimodal Retrieval Augmented Generation Performance},
  journal   = {arXiv preprint arXiv:2501.03995},
  year      = {2025}
}
```

Additionally, update the existing `elkiran2025evarag` entry to reflect the 2026 publication details:

```bibtex
@article{elkiran2026empirical,
  author    = {Elkiran, Harun and Rasheed, Jawad},
  title     = {An Empirical Evaluation of Retrieval, Reranking, and Similarity for a Q\&A-Based Retrieval Augmented Generation System},
  journal   = {IEEE Access},
  volume    = {14},
  pages     = {26053--26066},
  year      = {2026},
  doi       = {10.1109/access.2026.3664852}
}
```

(Replace all `\cite{elkiran2025evarag}` in the text with `\cite{elkiran2026empirical}`.)

---

## Summary of Applied Fixes

| Section                  | Key Changes Applied                                                                                                                                                                                                                                                                             |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Abstract**             | Replaced "more prone than proprietary" with cited struggle framing; replaced "rarely" with "often underreport"; added HotpotQA scope; added ARES and Elkiran & Rasheed citations; softened novelty claim.                                                                                       |
| **Introduction**         | Removed blanket proprietary comparison; added multi-component/diagnostic rationale; added deployability trade-off framing; reframed fixed retrieval as confound control; replaced "what remains uncommon" with "what this paper does is combine"; added scoped-design caveat in contribution 1. |
| **Related Work**         | Added ARES, eRAG, RAGVUE, RARE; sharpened CRAG caveat (overhead excludes retrieval/preprocessing, assumes external evaluator); distinguished faithfulness from factual correctness; added Mala motivation sentence.                                                                             |
| **Benchmark Design**     | Added explicit "controlled benchmark, not universal recommendation" framing; justified BM25+MiniLM via component-variance literature; added HotpotQA Wikipedia-centric limitation; described crag as derivative design.                                                                         |
| **Methodology**          | Defined faithfulness as evidence-support, not truth; justified abstention/rejection metrics with citations; acknowledged judge instability; separated retrieval/answer/grounding evaluation.                                                                                                    |
| **Results**              | Reframed as frontier comparison, not scalar winner; added cautious interpretation of corrective gains; added retrieval-vs-generation failure distinction.                                                                                                                                       |
| **Discussion (new 5.7)** | Tempered generalization; noted HotpotQA transfer limits; framed contribution as harness, not definitive answer.                                                                                                                                                                                 |
| **Threats**              | Expanded T1 (faithfulness/correctness divergence), T2 (judge instability), T3 (Wikipedia-centric, multilingual/domain limits), T5 (new system validity: chunking, depth, CRAG proxy).                                                                                                           |
| **Reproducibility**      | Tied artifact traceability to provenance-aware evaluation literature.                                                                                                                                                                                                                           |
| **Conclusion**           | Removed "new standard" implication; scoped contribution to controlled benchmark; added external-validity future work.                                                                                                                                                                           |
