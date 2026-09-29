"""CRAG-style retrieval evaluator + knowledge-strip refinement.

Per-document confidence in [-1,1], three actions (Correct/Incorrect/Ambiguous) chosen by
upper/lower thresholds, then decompose-then-recompose strips of 3 sentences keeping the top 5.

The scorer is `google/flan-t5-base` (248M), a prompt-based yes/no decision standing in for a
trained confidence regressor: it is small enough to stay resident on a T4 alongside vLLM and the
retrieval models. It is a proxy, not the real thing — a 248M prompt-based scorer scores highly
on-topic passages almost always, so the Correct/Incorrect split is much less discriminative than
the design assumes. Read the action distribution before trusting it.

No live web search on Incorrect (the $0 constraint): the caller receives empty strips and must
fall back to an abstention/parametric prompt.
"""
from __future__ import annotations

import re

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

PROMPT_TMPL = (
    "Decide whether the passage helps answer the question. Answer only yes or no.\n"
    "Question: {q}\nPassage: {d}\nAnswer:"
)


class CRAGEvaluator:
    def __init__(self, model_name: str = "google/flan-t5-base",
                 upper_threshold: float = 0.5, lower_threshold: float = -0.5,
                 strip_top_n: int = 5, device: str = "cuda") -> None:
        assert lower_threshold < upper_threshold
        self.tok = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.upper, self.lower, self.strip_top_n = upper_threshold, lower_threshold, strip_top_n
        self._yes = self.tok("yes", add_special_tokens=False).input_ids[0]
        self._no = self.tok("no", add_special_tokens=False).input_ids[0]

    @torch.no_grad()
    def _score(self, query: str, passage: str) -> float:
        """Confidence in [-1, 1] = p(yes) - p(no) on the first generated token.

        `decoder_input_ids` is explicit because transformers 5 no longer infers it from
        `input_ids` in the encoder-decoder path — omitting it raises "You must specify exactly
        one of input_ids or inputs_embeds" (T5ForConditionalGeneration.forward, v5). Passing a
        single start token makes this exactly one decode step, so logits[0, -1] is the
        distribution over the first output token, which is the quantity the metric is defined on.
        """
        enc = self.tok(PROMPT_TMPL.format(q=query, d=passage[:1000]),
                       return_tensors="pt", truncation=True, max_length=512)
        start = torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device)
        logits = self.model(input_ids=enc.input_ids.to(self.device),
                            attention_mask=enc.attention_mask.to(self.device),
                            decoder_input_ids=start).logits[0, -1]
        probs = torch.softmax(logits[[self._yes, self._no]], dim=-1)
        return float(probs[0] - probs[1])

    @staticmethod
    def decompose(passage: str, strip_len: int = 3) -> list[str]:
        """Split into strips of `strip_len` sentences."""
        sents = [s for s in re.split(r"(?<=[.!?])\s+", passage) if s.strip()]
        return [" ".join(sents[i:i + strip_len]) for i in range(0, len(sents), strip_len)] or [passage]

    def evaluate(self, query: str, docs: list[str]) -> dict:
        """Return {"action", "strips", "doc_scores"} — strips recomposed in source order."""
        if not docs:
            return {"action": "Incorrect", "strips": [], "doc_scores": []}
        doc_scores = [self._score(query, d) for d in docs]
        if max(doc_scores) >= self.upper:
            action = "Correct"
        elif min(doc_scores) < self.lower and max(doc_scores) < self.lower:
            action = "Incorrect"
        else:
            action = "Ambiguous"
        # Refinement runs for Correct|Ambiguous. Incorrect => no internal knowledge.
        strips: list[tuple[str, float]] = []
        if action in ("Correct", "Ambiguous"):
            for d in docs:
                for s in self.decompose(d):
                    strips.append((s, self._score(query, s)))
        ranked = sorted(enumerate(strips), key=lambda p: p[1][1], reverse=True)[: self.strip_top_n]
        keep = [strips[i] for i, _ in sorted(ranked)]      # top-N strips, back in source order
        return {
            "action": action,
            "strips": [s for s, _ in keep],
            "strip_scores": [round(v, 3) for _, v in keep],
            "doc_scores": [round(v, 3) for v in doc_scores],
        }


if __name__ == "__main__":
    ev = CRAGEvaluator(device="cpu")
    q = "Who painted the Mona Lisa?"
    # Plain strings, NOT one-element lists. Wrapping them in brackets and then passing
    # `[good]` again makes `docs` [[str]] and decompose() gets a list — this smoke test raised
    # TypeError on its first honest run, i.e. it had never executed.
    good = ("The Mona Lisa was painted by Leonardo da Vinci in the early 1500s. "
            "It hangs in the Louvre. Millions visit it annually.")
    junk = ("Bananas are yellow and rich in potassium. Monkeys like fruit. "
            "The zoo opens at nine.")
    r1, r2 = ev.evaluate(q, [good]), ev.evaluate(q, [junk])
    assert r1["action"] == "Correct" and r1["strips"], r1
    assert r2["action"] == "Incorrect" and r2["strips"] == [], r2
    # The strip path must actually fire, not merely return a non-empty list: 3 sentences of
    # `good` decompose into exactly 1 strip of 3, and the doc score is the whole-doc confidence.
    assert r1["strips"] == [good], r1["strips"]
    assert all(-1.0 <= s <= 1.0 for s in r1["doc_scores"] + r1["strip_scores"]), r1
    print("crag_module smoke OK:", r1["action"], r1["doc_scores"], "|", r2["action"])
