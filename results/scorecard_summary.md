| cell | n | adj acc % | faithfulness | lat p95 ms | ttft p95 ms | api $ | gpu $ | MAP@3 | recall@10 | crag dist |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-llama3.1-8b | 30 | 68.8 | 0.3888888888888889 | 5332 | 2055.1 | 0.0008 | 0.0709 | 0.875 | 0.783 | {} |
| baseline-qwen2.5-1.5b | 30 | 53.6 | 0.5773809523809523 | 4329 | 702.9 | 0.0016 | 0.0668 | 0.875 | 0.783 | {} |
| baseline-qwen3-4b | 30 | 77.3 | 0.5752380952380952 | 8171 | 1745.8 | 0.0004 | 0.0787 | 0.875 | 0.783 | {} |
| crag-llama3.1-8b | 30 | 66.7 | 0.38055555555555554 | 5087 | 562.0 | 0.0003 | 0.0523 | 0.875 | 0.783 | {'Ambiguous': 13, 'Correct': 15, 'Incorrect': 2} |
| crag-qwen2.5-1.5b | 30 | 53.8 | 0.48166666666666663 | 5427 | 173.0 | 0.0007 | 0.0515 | 0.875 | 0.783 | {'Ambiguous': 13, 'Correct': 15, 'Incorrect': 2} |
| crag-qwen3-4b | 30 | 81.2 | 0.5088888888888888 | 8856 | 404.8 | 0.0002 | 0.0559 | 0.875 | 0.783 | {'Ambiguous': 13, 'Correct': 15, 'Incorrect': 2} |

Pareto:
- faithfulness_vs_latency: ['baseline-qwen2.5-1.5b']
- faithfulness_vs_gpu_cost: ['baseline-qwen2.5-1.5b', 'crag-qwen2.5-1.5b', 'crag-qwen3-4b']

Stats:
- crag-llama3.1-8b vs baseline-llama3.1-8b round-trip latency: p=0.824, q_fdr=0.988
- crag-llama3.1-8b vs baseline-llama3.1-8b correctness (McNemar): p=0.453, q_fdr=0.906
- crag-qwen2.5-1.5b vs baseline-qwen2.5-1.5b round-trip latency: p=4.42e-06, q_fdr=2.65e-05
- crag-qwen2.5-1.5b vs baseline-qwen2.5-1.5b correctness (McNemar): p=1, q_fdr=1
- crag-qwen3-4b vs baseline-qwen3-4b round-trip latency: p=0.641, q_fdr=0.961
- crag-qwen3-4b vs baseline-qwen3-4b correctness (McNemar): p=0.219, q_fdr=0.656
