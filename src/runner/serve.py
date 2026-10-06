"""vLLM serving shim: one AWQ model copy on GPU behind an OpenAI-compatible endpoint.

The runner does NOT load model weights in-process. This module starts vLLM as a managed
subprocess, talks to it over HTTP, and stops it reliably — so peak VRAM is the max over roles
rather than the sum, and the retrieval models (MPNet, MiniLM, flan-t5) can stay CO-RESIDENT with
the server through the generation pass.

Why a subprocess rather than an inline call: the runner must reliably kill the server even when a
pass fails. A managed Popen with a health-check loop does that. `--enforce-eager` is deliberate:
no CUDA-graph capture on a 16 GB T4 halves the risk of first-token OOM, at the cost of some
throughput — this benchmark is latency-measured, not throughput-measured.

ROLE-PARAMETERISED: the same class serves `generator` and `judge`, so a different model can sit
on the judging endpoint without a second code path.

STREAMING IS REQUIRED FOR TTFT: a non-streaming completion has no first-chunk timestamp, so TTFT
cannot be measured from it at all. `stream=True` plus `stream_options={"include_usage": True}`
gives both the first-token time AND the server-side usage block. Token counts come from that
block, never from a local tokenizer — the runner holds no weights, and server-side counting is
what vLLM bills. If the usage block is missing, `serve_generate` raises rather than returning
zeros: a silent 0 would flow straight into the cost axis and the tokens/s figure.
"""
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class VLLMServer:
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    gpu_memory_utilization: float = 0.5
    max_model_len: int = 4096
    quantization: str | None = "awq"
    startup_timeout_s: int = 900
    extra_args: list[str] = field(default_factory=list)
    log_path: str = "outputs/vllm_server.log"
    _proc: subprocess.Popen | None = field(default=None, repr=False, init=False)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _wait_until(self, path: str) -> None:
        deadline = time.time() + self.startup_timeout_s
        err: Exception | None = None
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"vllm exited early (rc={self._proc.returncode}) — "
                                   f"see {self.log_path}")
            try:
                with urllib.request.urlopen(self.url + path, timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception as e:      # server not up yet
                err = e
            time.sleep(5)
        raise TimeoutError(f"vllm not ready after {self.startup_timeout_s}s: {err!r} "
                           f"— see {self.log_path}")

    def launch(self) -> "VLLMServer":
        if shutil.which("vllm") is None and not _module_present("vllm"):
            raise RuntimeError("`vllm` not found — install the serve extra "
                               "(uv sync --extra serve)")
        s = socket.socket()
        try:
            s.bind((self.host, self.port))
        except OSError as e:
            raise RuntimeError(f"{self.host}:{self.port} already in use — "
                               f"stop the old server first: {e}")
        finally:
            s.close()

        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", self.model, "--host", self.host, "--port", str(self.port)]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        cmd += ["--gpu-memory-utilization", str(self.gpu_memory_utilization),
                "--max-model-len", str(self.max_model_len),
                "--enforce-eager", *self.extra_args]

        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        log = open(self.log_path, "a", encoding="utf-8")
        # start_new_session: own process group, so killpg kills the engine's children too.
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        self._wait_until("/models")
        self._wait_until("/health")
        return self

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            os.killpg(self._proc.pid, signal.SIGTERM)
            try:
                self._proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self._proc.pid, signal.SIGKILL)
        self._proc = None


def _module_present(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


@contextmanager
def serving(server: VLLMServer):
    """`with serving(server_from_config(cfg, role)) as ep:` — launches, yields, and ALWAYS stops,
    so a failed pass or a failing gate assert still releases the GPU for the next role."""
    server.launch()
    try:
        yield server
    finally:
        server.stop()


def server_from_config(cfg: dict, role: str = "generator") -> VLLMServer:
    """Build a VLLMServer for a role.

    With `judge.serve_id` null, both roles resolve to the generator's weights — the self-judge
    limitation, carried in the manifest as a deviation. Setting `judge.serve_id`
    makes the two roles diverge; the two servers then need different ports, which is why the
    port is per-role rather than global. Note the judge inherits the generator's `quantization`
    unless `judge_quantization` is set — correct today only because the judge is served AWQ.
    """
    assert role in ("generator", "judge"), role
    gen = cfg["models"][cfg["_model_key"]]
    judge_id = cfg["judge"].get("serve_id")
    # A role's model is either its own serve_id or, by default, the generator's.
    model = gen["serve_id"] if role == "generator" else (judge_id or gen["serve_id"])
    ep = cfg["serve"]
    return VLLMServer(
        model=model,
        host=ep.get("host", "127.0.0.1"),
        port=int(ep.get("port", 8000) if role == "generator" else ep.get("judge_port", 8001)),
        gpu_memory_utilization=float(ep.get("gpu_memory_utilization", 0.5)),
        max_model_len=int(ep.get("max_model_len", 4096)),
        # The judge needs no --quantization override of its own; it inherits the generator's
        # until experiment.yaml declares otherwise.
        quantization=ep.get(f"{role}_quantization", gen.get("quantization")),
        extra_args=list(ep.get("extra_args", [])),
    )


def serve_client(server: VLLMServer):
    from openai import OpenAI

    return OpenAI(base_url=server.url, api_key="local", timeout=600.0)


def serve_generate(client, model: str, prompt: str, max_new_tokens: int,
                   temperature: float = 0.0) -> tuple[str, int, int, float]:
    """One streaming chat completion -> (text, prompt_tokens, completion_tokens, ttft_ms)."""
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_new_tokens, stream=True,
        extra_body={"stream_options": {"include_usage": True}},
    )
    chunks: list[str] = []
    ttft_ms: float | None = None
    usage = None
    t0 = time.perf_counter()
    for ev in stream:
        if getattr(ev, "usage", None):
            usage = ev.usage
        choices = getattr(ev, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0].delta, "content", None)
        if delta:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            chunks.append(delta)
    if usage is None:
        raise RuntimeError(
            "vLLM returned no usage block. Token counts (and therefore the whole cost axis) "
            "depend on it. Check the server log for a rejected stream_options, or pin an "
            "openai client version that sends it."
        )
    return ("".join(chunks).strip(), usage.prompt_tokens, usage.completion_tokens,
            ttft_ms if ttft_ms is not None else -1.0)


def ragas_compat() -> None:
    """Stub the module ragas cannot be imported without. ragas 0.4.3 runs
    `from langchain_community.chat_models.vertexai import ChatVertexAI` at import time
    (ragas/llms/base.py:12), and that file no longer ships in langchain-community >= 0.3
    (0.4.2 has only embeddings/llms/utilities vertexai) — NO installable package satisfies it.
    ChatVertexAI is only ever used in a MULTIPLE_COMPLETION_SUPPORTED isinstance() check on
    the legacy Langchain path (base.py:48-54), which the instructor judge here never takes; a
    real class (not None) keeps that check safe if it ever does.
    It lives HERE, not in the caller, because the judge runs inside a `python -m` SUBPROCESS —
    a sys.modules stub in an interactive kernel does not travel with it.
    """
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = mod


def build_judge(server: VLLMServer, mode: str = "JSON"):
    """Instructor-style judge the ragas (PyPI 0.4.3) collections API accepts
    (metrics/collections/base.py:113 _validate_llm requires InstructorBaseRagasLLM;
    llm_factory without a client raises, llms/base.py:579).
    Mode.JSON first (vLLM guided-JSON honors response_format); Mode.MD_JSON (plain-text JSON
    parsing) is the sanctioned fallback — passed as judge.mode in the config.
    """
    ragas_compat()          # must precede the ragas import — see ragas_compat()
    import instructor
    from openai import OpenAI
    from ragas.llms import llm_factory

    return llm_factory(server.model, provider="openai",
                       client=instructor.from_openai(
                           OpenAI(base_url=server.url, api_key="local"),
                           mode=getattr(instructor.Mode, mode.upper(), instructor.Mode.JSON)))


def build_embeddings(device: str = "cuda"):
    """Local MPNet embeddings, now wrapped in ragas' HuggingFaceEmbeddings
    (ragas/embeddings/huggingface_provider.py), which _validate_embeddings
    (metrics/collections/base.py:123) accepts.

    This is the model's ONLY use, which is why pinning the id matters - a bare
    "all-mpnet-base-v2" could resolve to a different revision and put AnswerRelevancy in a
    different embedding space than the one the number was reported under."""
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2", device=device)
