import os
import re
import json
import time
import argparse
import hashlib
import shutil
import io
import logging
import threading
import warnings
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Reuse loaded embedding models and silence noisy HF/transformers load reports.
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_RUNTIME_DEVICE_LOGGED = False


def _embedding_device() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _log_runtime_embedding_device_once() -> None:
    global _RUNTIME_DEVICE_LOGGED
    if _RUNTIME_DEVICE_LOGGED:
        return

    device = _embedding_device()
    msg = f"[Runtime] Embedding device={device}"
    if device == "cuda":
        try:
            import torch  # type: ignore

            msg += f" gpu={torch.cuda.get_device_name(0)}"
        except Exception:
            pass
    print(msg)
    _RUNTIME_DEVICE_LOGGED = True


def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        device = _embedding_device()

        hf_logger = logging.getLogger("huggingface_hub")
        st_logger = logging.getLogger("sentence_transformers")
        prev_hf_level = hf_logger.level
        prev_st_level = st_logger.level

        tf_logging = None
        prev_tf_verbosity = None
        try:
            from transformers.utils import logging as tf_logging  # type: ignore
            prev_tf_verbosity = tf_logging.get_verbosity()
            tf_logging.set_verbosity_error()
        except Exception:
            tf_logging = None

        hf_logger.setLevel(logging.ERROR)
        st_logger.setLevel(logging.ERROR)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    model = SentenceTransformer(model_name, device=device)
        finally:
            hf_logger.setLevel(prev_hf_level)
            st_logger.setLevel(prev_st_level)
            if tf_logging is not None and prev_tf_verbosity is not None:
                tf_logging.set_verbosity(prev_tf_verbosity)

        _MODEL_CACHE[model_name] = model
        return model

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class DocumentChunk:
    chunk_id: int
    source_path: str
    text: str

@dataclass
class RAGResult:
    answer: str
    used_sources: List[str]
    missing_terms: List[str]
    top_scores: List[float]
    had_low_confidence: bool
    judge_rejected: bool = False  # whether the entailment judge rejected the answer
    _selected_chunks: Optional[List[DocumentChunk]] = None
    _context: str = ""

@dataclass
class AgentSpec:
    name: str
    description: str
    data_dir: str

@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    text: str
    ts: float

# =============================================================================
# Chunking (paragraph packing)
# =============================================================================

def split_into_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", text.strip())
    paras = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"[ \t]+", " ", p)
        paras.append(p)
    return paras

def sentence_split(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]

def chunk_text_semantic(text: str, max_chars: int = 900, overlap_units: int = 1) -> List[str]:
    text = text.strip()
    if not text:
        return []

    units: List[str] = []
    paras = split_into_paragraphs(text)

    for p in paras:
        if len(p) <= max_chars:
            units.append(p)
        else:
            sents = sentence_split(p)
            if not sents:
                units.append(p[:max_chars])
            else:
                units.extend(sents)

    if overlap_units <= 0:
        # Pack without overlap
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0

        def flush():
            nonlocal cur, cur_len
            if cur:
                chunk = "\n\n".join(cur).strip()
                if chunk:
                    chunks.append(chunk)
            cur = []
            cur_len = 0

        for u in units:
            u = u.strip()
            if not u:
                continue
            add_len = len(u) + (2 if cur else 0)
            if cur and cur_len + add_len > max_chars:
                flush()
            cur.append(u)
            cur_len += add_len
        flush()
        return chunks

    # Overlapped repack in unit-space
    overlapped_chunks: List[str] = []
    i = 0
    n = len(units)

    while i < n:
        cur: List[str] = []
        cur_len = 0
        start_i = i

        while i < n:
            u = units[i].strip()
            if not u:
                i += 1
                continue
            add_len = len(u) + (2 if cur else 0)
            if cur and cur_len + add_len > max_chars:
                break
            cur.append(u)
            cur_len += add_len
            i += 1

        chunk = "\n\n".join(cur).strip()
        if chunk:
            overlapped_chunks.append(chunk)

        if i >= n:
            break

        i = max(i - overlap_units, start_i + 1)

    return overlapped_chunks

# =============================================================================
# File loading
# =============================================================================

def load_text_files(data_dir: str) -> List[Tuple[str, str]]:
    files = []
    for root, _, filenames in os.walk(data_dir):
        for fname in filenames:
            if not fname.lower().endswith(".txt"):
                continue
            full_path = os.path.join(root, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(full_path, "r", encoding="latin-1") as f:
                    text = f.read()
            files.append((full_path, text))
    return files

# =============================================================================
# Fingerprint + cache paths (docs embedding cache)
# =============================================================================

def compute_dir_fingerprint(data_dir: str) -> str:
    h = hashlib.sha256()
    data_dir = os.path.abspath(data_dir)

    records = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.lower().endswith(".txt"):
                continue
            p = os.path.join(root, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            rel = os.path.relpath(p, data_dir)
            records.append((rel, st.st_mtime_ns, st.st_size))

    records.sort()
    for rel, mtime_ns, size in records:
        h.update(rel.encode("utf-8"))
        h.update(str(mtime_ns).encode("utf-8"))
        h.update(str(size).encode("utf-8"))

    return h.hexdigest()

def cache_dir_for_agent(data_dir: str, model_name: str) -> str:
    base = os.path.join(os.path.abspath(data_dir), ".rag_cache")
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)
    return os.path.join(base, safe_model)

# =============================================================================
# Build DocumentChunks (uses semantic chunking)
# =============================================================================

def build_document_chunks(data_dir: str, max_chars: int = 900, overlap_units: int = 1) -> List[DocumentChunk]:
    files = load_text_files(data_dir)
    chunks: List[DocumentChunk] = []
    chunk_id = 0

    for path, text in files:
        text_chunks = chunk_text_semantic(text, max_chars=max_chars, overlap_units=overlap_units)
        for ch in text_chunks:
            chunks.append(DocumentChunk(chunk_id=chunk_id, source_path=path, text=ch))
            chunk_id += 1

    if not chunks:
        raise ValueError(f"No text chunks created from directory: {data_dir}")

    return chunks

# =============================================================================
# Embedding Index for DOCUMENTS (persisted)
# =============================================================================

class EmbeddingIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.chunks: List[DocumentChunk] = []
        self.cache_dir = cache_dir

    def load_model(self):
        if self.model is None:
            self.model = _load_sentence_transformer(self.model_name)

    def _paths(self) -> Dict[str, str]:
        if not self.cache_dir:
            raise RuntimeError("cache_dir not set on EmbeddingIndex.")
        os.makedirs(self.cache_dir, exist_ok=True)
        return {
            "meta": os.path.join(self.cache_dir, "meta.json"),
            "chunks": os.path.join(self.cache_dir, "chunks.jsonl"),
            "emb": os.path.join(self.cache_dir, "embeddings.npy"),
        }

    def try_load(self, expected_fingerprint: str) -> bool:
        p = self._paths()
        if not (os.path.exists(p["meta"]) and os.path.exists(p["chunks"]) and os.path.exists(p["emb"])):
            return False

        try:
            with open(p["meta"], "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("fingerprint") != expected_fingerprint:
                return False
            if meta.get("model_name") != self.model_name:
                return False

            loaded_chunks: List[DocumentChunk] = []
            with open(p["chunks"], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    loaded_chunks.append(
                        DocumentChunk(
                            chunk_id=int(obj["chunk_id"]),
                            source_path=obj["source_path"],
                            text=obj["text"],
                        )
                    )

            emb = np.load(p["emb"])
            if emb.dtype != np.float32:
                emb = emb.astype(np.float32, copy=False)

            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
            emb = emb / norms

            self.chunks = loaded_chunks
            self.chunk_embeddings = emb
            return True
        except Exception:
            return False

    def save(self, fingerprint: str):
        if self.chunk_embeddings is None or not self.chunks:
            raise RuntimeError("Nothing to save: build index first.")

        p = self._paths()
        meta = {
            "fingerprint": fingerprint,
            "model_name": self.model_name,
            "num_chunks": len(self.chunks),
            "emb_shape": list(self.chunk_embeddings.shape),
        }
        with open(p["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with open(p["chunks"], "w", encoding="utf-8") as f:
            for ch in self.chunks:
                f.write(json.dumps({"chunk_id": ch.chunk_id, "source_path": ch.source_path, "text": ch.text}) + "\n")

        emb = self.chunk_embeddings.astype(np.float32, copy=False)
        np.save(p["emb"], emb)

    def fit(self, chunks: List[DocumentChunk]):
        self.load_model()
        self.chunks = chunks
        texts = [ch.text for ch in chunks]
        print(f"[Embeddings] Encoding {len(texts)} chunks with {self.model_name} ...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32, copy=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        self.chunk_embeddings = embeddings / norms
        print("[Embeddings] Done.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if self.chunk_embeddings is None or not self.chunks:
            raise RuntimeError("Index not built. Call fit() or try_load() first.")

        self.load_model()
        q_emb = self.model.encode([query], convert_to_numpy=True)[0].astype(np.float32, copy=False)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)

        scores = np.dot(self.chunk_embeddings, q_emb)
        top_k = min(top_k, len(self.chunks))
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[idx], float(scores[idx])) for idx in idxs]

# =============================================================================
# Term extraction + missing terms
# =============================================================================

BASIC_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "and", "or", "for", "to", "with",
    "some", "please", "give", "me", "about",
    "how", "do", "i", "is", "are", "you", "can", "my"
}

def normalize_token(token: str) -> str:
    return "".join(c.lower() for c in token if c.isalnum())

def extract_content_terms(text: str) -> List[str]:
    terms = []
    for raw in text.split():
        tok = normalize_token(raw)
        if not tok:
            continue
        if tok in BASIC_STOPWORDS:
            continue
        terms.append(tok)
    return terms

def find_missing_query_terms(query: str, chunks: List[DocumentChunk]) -> List[str]:
    query_terms = extract_content_terms(query)
    combined_text = " ".join(ch.text.lower() for ch in chunks)

    missing = []
    for term in query_terms:
        if term not in combined_text:
            missing.append(term)

    seen = set()
    unique_missing = []
    for t in missing:
        if t not in seen:
            seen.add(t)
            unique_missing.append(t)
    return unique_missing

def build_context_from_chunks(
    chunks_with_scores: List[Tuple[DocumentChunk, float]],
    min_score: float = 0.15,
    max_tokens_approx: int = 1500,
) -> Tuple[str, List[DocumentChunk]]:
    selected_chunks: List[DocumentChunk] = []
    context_pieces = []
    total_chars = 0

    for chunk, score in chunks_with_scores:
        if score < min_score:
            continue
        text = chunk.text.strip()
        if not text:
            continue

        if total_chars + len(text) + 50 > max_tokens_approx * 4:
            break

        context_pieces.append(
            f"[SOURCE: {chunk.source_path} | CHUNK_ID: {chunk.chunk_id} | SCORE: {score:.3f}]\n{text}"
        )
        selected_chunks.append(chunk)
        total_chars += len(text) + 50

    context = "\n\n---\n\n".join(context_pieces)
    return context, selected_chunks

# =============================================================================
# Ollama LLM call utilities (router / summarizer / rewriter / answer)
# =============================================================================

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract the first JSON object from a string.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def call_ollama_generate(
    prompt: str,
    model: str = "llama3:8b",
    base_url: str = "http://localhost:11434",
    timeout_connect: float = 3.0,
    timeout_read: float = 120.0,
    stream: bool = False,
) -> str:
    payload = {"model": model, "prompt": prompt, "stream": stream}
    url = f"{base_url}/api/generate"
    resp = requests.post(url, json=payload, timeout=(timeout_connect, timeout_read), stream=stream)
    resp.raise_for_status()

    if not stream:
        data = resp.json()
        return (data.get("response") or "").strip()

    out_parts = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        token = obj.get("response")
        if token:
            out_parts.append(token)
        if obj.get("done") is True:
            break
    return "".join(out_parts).strip()

def _is_abstention(text: str) -> bool:
    t = (text or "").lower()
    return (
        "final_answer:" in t and "i don't know" in t
    ) or ("i don't know based on the provided documents" in t)

def entailment_judge(
    context: str,
    user_query: str,
    model_output: str,
    model: str,
    base_url: str,
    timeout_read: float = 120.0,
) -> Tuple[bool, str]:
    """
    Returns (supported, reason). Strictly checks whether FINAL_ANSWER is supported by CONTEXT.
    IMPORTANT: Absence of evidence is NOT evidence of absence.
    IMPORTANT: Abstentions are allowed.
    """
    prompt = f"""
You are a strict verifier for a Retrieval-Augmented Generation system.

USER_QUESTION:
{user_query}

CONTEXT (the ONLY allowed source of facts):
{context}

MODEL_OUTPUT:
{model_output}

Rules:
- The FINAL_ANSWER must be supported by explicit statements in the CONTEXT.
- Absence of evidence is NOT evidence of absence.
  Example: If the context does not mention X, you cannot conclude "X is not true".
- Negative claims (e.g., "does not", "no", "never", "cannot") require explicit negation/support in the CONTEXT.
- IMPORTANT: An abstention is allowed.
  If FINAL_ANSWER says "I don't know based on the provided documents", then supported=true
  UNLESS the CONTEXT explicitly contains the answer to the USER_QUESTION.

Return ONLY valid JSON:
{{
  "supported": true/false,
  "reason": "short explanation"
}}
""".strip()

    raw = call_ollama_generate(
        prompt=prompt,
        model=model,
        base_url=base_url,
        timeout_read=timeout_read,
        stream=False,
    )
    obj = _extract_json_object(raw) or {}
    supported = bool(obj.get("supported", False))
    reason = str(obj.get("reason", "")).strip()
    return supported, reason

# =============================================================================
# Three-tier conversation memory
# =============================================================================

class RecentWindowMemory:
    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns
        self.turns: List[ChatTurn] = []

    def add(self, role: str, text: str):
        self.turns.append(ChatTurn(role=role, text=text, ts=time.time()))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def clear(self):
        self.turns = []

    def as_text(self, max_lines: int = 12) -> str:
        lines = []
        for t in self.turns[-max_lines:]:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.text}")
        return "\n".join(lines).strip()

class SummaryStateMemory:
    """
    Tier [2]: A running summary updated via LLM after each turn.
    """
    def __init__(self, initial_summary: str = ""):
        self.summary = initial_summary.strip()

    def clear(self):
        self.summary = ""

    def update_with_llm(
        self,
        user_text: str,
        assistant_text: str,
        model: str,
        base_url: str,
        timeout_read: float = 120.0,
    ):
        prompt = f"""
You maintain a compact, factual, running summary of a conversation.

Current summary:
{self.summary if self.summary else "[empty]"}

New turn to incorporate:
User: {user_text}
Assistant: {assistant_text}

Update the summary so it remains:
- compact (aim for <= 12 bullet points),
- factual (no guesses),
- includes stable user preferences, important entities, decisions, tasks, and open threads,
- excludes fluff or one-off jokes.

Return ONLY the updated summary as plain text (no JSON).
""".strip()

        new_summary = call_ollama_generate(
            prompt=prompt,
            model=model,
            base_url=base_url,
            timeout_read=timeout_read,
            stream=False,
        )
        self.summary = (new_summary or "").strip()

class LongTermConversationStore:
    """
    Tier [3]: Hybrid retrieval (vector + TF-IDF) over ALL conversation turns.
    """
    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        alpha: float = 0.6,
        persist_path: str = ".conv_memory",
    ):
        self.embed_model_name = embed_model_name
        self.alpha = float(alpha)
        self.persist_path = persist_path

        self.model: SentenceTransformer = _load_sentence_transformer(self.embed_model_name)

        self.docs: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None

        self._load_if_exists()

    def _paths(self) -> Dict[str, str]:
        os.makedirs(self.persist_path, exist_ok=True)
        return {
            "docs": os.path.join(self.persist_path, "turns.jsonl"),
            "emb": os.path.join(self.persist_path, "embeddings.npy"),
        }

    def _load_if_exists(self):
        p = self._paths()
        if os.path.exists(p["docs"]):
            loaded_docs = []
            loaded_meta = []
            with open(p["docs"], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    loaded_docs.append(obj["text"])
                    loaded_meta.append({"role": obj.get("role"), "ts": obj.get("ts")})
            self.docs = loaded_docs
            self.meta = loaded_meta

        if os.path.exists(p["emb"]):
            emb = np.load(p["emb"])
            if emb.dtype != np.float32:
                emb = emb.astype(np.float32, copy=False)
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
            emb = emb / norms
            self.embeddings = emb

        if self.docs:
            self._rebuild_tfidf()

    def _persist(self):
        p = self._paths()
        with open(p["docs"], "w", encoding="utf-8") as f:
            for text, m in zip(self.docs, self.meta):
                f.write(json.dumps({"text": text, "role": m.get("role"), "ts": m.get("ts")}) + "\n")
        if self.embeddings is not None:
            np.save(p["emb"], self.embeddings.astype(np.float32, copy=False))

    def _rebuild_tfidf(self):
        self._tfidf = TfidfVectorizer(
            lowercase=True,
            max_features=50000,
            ngram_range=(1, 2),
        )
        self._tfidf_matrix = self._tfidf.fit_transform(self.docs)

    def add_turn(self, role: str, text: str):
        doc = f"{role.capitalize()}: {text}".strip()
        self.docs.append(doc)
        self.meta.append({"role": role, "ts": time.time()})

        new_emb = self.model.encode([doc], convert_to_numpy=True, show_progress_bar=False).astype(np.float32, copy=False)
        new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-10)

        if self.embeddings is None:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])

        self._rebuild_tfidf()
        self._persist()

    def search(self, query: str, top_k: int = 8) -> List[Tuple[str, float]]:
        if not self.docs or self.embeddings is None or self._tfidf is None or self._tfidf_matrix is None:
            return []

        q_emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32, copy=False)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        vec_scores = np.dot(self.embeddings, q_emb)

        q_t = self._tfidf.transform([query])
        doc_dot = (self._tfidf_matrix @ q_t.T).toarray().reshape(-1)
        doc_norms = np.sqrt(self._tfidf_matrix.multiply(self._tfidf_matrix).sum(axis=1)).A1 + 1e-10
        q_norm = np.sqrt(q_t.multiply(q_t).sum()) + 1e-10
        tfidf_scores = doc_dot / (doc_norms * q_norm)

        hybrid = self.alpha * vec_scores + (1.0 - self.alpha) * tfidf_scores
        idxs = np.argsort(hybrid)[::-1][: min(top_k, len(self.docs))]
        return [(self.docs[i], float(hybrid[i])) for i in idxs]

# =============================================================================
# LLM Router + LLM Query Rewrite
# =============================================================================

@dataclass
class MemoryDecision:
    tier: int  # 1,2,3
    need_retrieval: bool
    retrieval_query: str
    rationale: str

class MemoryRouter:
    def __init__(self, ollama_model: str, base_url: str, timeout_read: float = 120.0):
        self.ollama_model = ollama_model
        self.base_url = base_url
        self.timeout_read = timeout_read

    def _looks_recent_reference(self, user_query: str) -> bool:
        q = (user_query or "").lower()
        if not q:
            return False
        referential_patterns = [
            r"\b(this|that|these|those|it|they|them|he|she|his|her|its|their)\b",
            r"\b(above|earlier|previous|before|last time|as mentioned|you said|we said|again)\b",
            r"\b(follow up|follow-up|continue|same as|what about that|that one)\b",
        ]
        return any(re.search(p, q) for p in referential_patterns)

    def _looks_self_contained_factual(self, user_query: str) -> bool:
        q = (user_query or "").strip().lower()
        if not q:
            return False

        starts_like_question = bool(re.match(r"^(do|does|did|is|are|can|could|what|which|who|where|when|why|how)\b", q))
        has_question_mark = "?" in q
        content_tokens = re.findall(r"[a-z]{3,}", q)
        return (starts_like_question or has_question_mark) and len(content_tokens) >= 2

    def decide(self, user_query: str, recent_window: str, summary_state: str) -> MemoryDecision:
        prompt = f"""
You are a routing module for a 3-tier memory system.

Goal: choose the CHEAPEST memory tier that is sufficient to interpret the user query.
Tiers:
1) RECENT: last N conversation turns only
2) SUMMARY: running summary of older context (plus recent if needed)
3) RETRIEVAL: search long-term conversation history (vector + TF-IDF)

Return ONLY valid JSON with this schema:
{{
  "tier": 1|2|3,
  "need_retrieval": true|false,
  "retrieval_query": "a standalone query to retrieve needed past conversation items (empty if not needed)",
  "rationale": "short reason"
}}

Decision policy:
- Use Tier 1 ONLY when the user clearly refers to very recent turns (pronouns/deixis like "that", "it", "as you said").
- Use Tier 2 for self-contained factual/domain questions that do not depend on recent turns.
- Use Tier 3 only if exact older conversation details are needed and likely not covered by recent+summary.

Do not claim "recent conversation likely contains the answer" for standalone factual questions.
Example: "do dogs eat ramen" is standalone factual and should not be routed to Tier 1 by default.

User query:
{user_query}

Recent window:
{recent_window if recent_window else "[empty]"}

Summary state:
{summary_state if summary_state else "[empty]"}
""".strip()

        raw = call_ollama_generate(
            prompt=prompt,
            model=self.ollama_model,
            base_url=self.base_url,
            timeout_read=self.timeout_read,
            stream=False,
        )
        obj = _extract_json_object(raw) or {}
        tier = int(obj.get("tier", 3)) if str(obj.get("tier", "")).isdigit() else 3
        tier = tier if tier in (1, 2, 3) else 3
        need_retrieval = bool(obj.get("need_retrieval", tier == 3))
        retrieval_query = str(obj.get("retrieval_query", "")).strip()
        rationale = str(obj.get("rationale", "")).strip()
        if tier != 3:
            need_retrieval = False
            retrieval_query = ""

        # Guardrail: avoid over-selecting tier 1 for standalone factual questions.
        if tier == 1 and self._looks_self_contained_factual(user_query) and not self._looks_recent_reference(user_query):
            tier = 2
            need_retrieval = False
            retrieval_query = ""
            rationale = "Self-contained factual query with explicit entities; avoid recent-window bias."

        if tier == 3 and not retrieval_query:
            retrieval_query = user_query
        return MemoryDecision(tier=tier, need_retrieval=need_retrieval, retrieval_query=retrieval_query, rationale=rationale)

    def rewrite_for_retrieval_and_routing(
        self,
        user_query: str,
        recent_window: str,
        summary_state: str,
        retrieved_memory_snippets: str,
    ) -> str:
        prompt = f"""
Rewrite the user's query into a standalone, explicit query suitable for retrieval and agent routing.

Hard rules:
- Return EXACTLY ONE LINE.
- Do NOT add quotes.
- Do NOT add explanations or prefaces.
- KEEP ALL IMPORTANT KEYWORDS from the user query verbatim (especially named foods, products, animals).
- Do NOT generalize (e.g., do not replace "ramen" with "food" or "diet").
- If no pronouns/references exist, return the original query unchanged.

Use provided conversation memory ONLY to resolve references; do not invent facts.

User query:
{user_query}

Recent window:
{recent_window if recent_window else "[empty]"}

Summary state:
{summary_state if summary_state else "[empty]"}

Retrieved long-term memory snippets (may be empty):
{retrieved_memory_snippets if retrieved_memory_snippets else "[empty]"}
""".strip()

        rewritten = call_ollama_generate(
            prompt=prompt,
            model=self.ollama_model,
            base_url=self.base_url,
            timeout_read=self.timeout_read,
            stream=False,
        )
        return (rewritten or "").strip() or user_query

    def sanitize_rewrite_output(self, raw: str, fallback: str) -> str:
        """
        Prevent topic-drift: if the rewrite doesn't overlap with the original query, keep original.
        Also strips common junk, quotes, and multi-line outputs.
        """
        if not raw:
            return fallback

        t = raw.strip()
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

        bad_prefixes = (
            "here is", "rewritten query", "i kept", "since there", "as per", "hard rules"
        )
        filtered = []
        for ln in lines:
            low = ln.lower()
            if any(low.startswith(bp) for bp in bad_prefixes):
                continue
            if low in {"query:", "rewritten query:", "rewritten:", "output:"}:
                continue
            filtered.append(ln)

        if not filtered:
            filtered = lines

        # Strip surrounding quotes if any
        filtered = [ln.strip().strip('"').strip("'").strip() for ln in filtered if ln.strip()]

        fb_terms = set(extract_content_terms(fallback))

        def overlap_count(ln: str) -> int:
            return len(fb_terms & set(extract_content_terms(ln)))

        # pick best overlap first; tie-breaker: shorter
        best = max(filtered, key=lambda ln: (overlap_count(ln), -len(ln)))
        if overlap_count(best) == 0:
            return fallback  # <- CRITICAL: no overlap => don't drift to unrelated topic
        return best.strip() or fallback

# =============================================================================
# RAG (per-agent)
# =============================================================================

class SimpleRAG:
    def __init__(self, data_dir: str, embed_model_name: str = "all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        cache_dir = cache_dir_for_agent(data_dir, embed_model_name)
        self.index = EmbeddingIndex(model_name=embed_model_name, cache_dir=cache_dir)
        self._built = False

    def build_index(self):
        print(f"[RAG] Preparing index for: {self.data_dir}")

        fp = compute_dir_fingerprint(self.data_dir)
        if self.index.try_load(expected_fingerprint=fp):
            print(f"[RAG] Loaded cached index from: {self.index.cache_dir}")
            print(f"[RAG] Cached chunks: {len(self.index.chunks)}\n")
            self._built = True
            return

        print(f"[RAG] Cache miss (or data changed). Building index...")
        chunks = build_document_chunks(self.data_dir, max_chars=900, overlap_units=1)
        print(f"[RAG] Number of chunks: {len(chunks)}")
        print("[RAG] Building embedding index...")
        self.index.fit(chunks)
        self.index.save(fp)
        print(f"[RAG] Saved cached index to: {self.index.cache_dir}")
        self._built = True
        print("[RAG] Index built.\n")

    def answer(
        self,
        user_query: str,
        retrieval_query: str,
        top_k: int = 5,
        min_score: float = 0.15,
        ollama_model: str = "llama3:8b",
        conversation_history_block: str = "",
        timeout_read: float = 120.0,
        base_url: str = "http://localhost:11434",
    ) -> RAGResult:
        if not self._built:
            raise RuntimeError("Index not built. Call build_index() first.")

        retrieved = self.index.search(retrieval_query, top_k=top_k)
        top_scores = [float(s) for _, s in retrieved] if retrieved else []

        if not retrieved or retrieved[0][1] < min_score:
            return RAGResult(
                answer="I couldn’t find any document that seems relevant.\nTry rephrasing your query.",
                used_sources=[],
                missing_terms=extract_content_terms(user_query),
                top_scores=top_scores,
                had_low_confidence=True,
                judge_rejected=False,
                _selected_chunks=[],
                _context="",
            )

        context, selected_chunks = build_context_from_chunks(retrieved, min_score=min_score)
        if not selected_chunks:
            return RAGResult(
                answer="Documents retrieved, but none were confidently relevant.\nTry rephrasing your query.",
                used_sources=[],
                missing_terms=extract_content_terms(user_query),
                top_scores=top_scores,
                had_low_confidence=True,
                judge_rejected=False,
                _selected_chunks=[],
                _context="",
            )

        missing_terms = find_missing_query_terms(user_query, selected_chunks)

        prompt = f"""
You are a helpful assistant answering the user's question using ONLY the provided Context.

Conversation memory (ONLY for co-reference; NOT factual evidence):
{conversation_history_block if conversation_history_block else "[none]"}

User question:
{user_query}

Important query terms that were NOT found in any retrieved chunk:
{", ".join(missing_terms) if missing_terms else "none"}

Context (multiple chunks, each with [SOURCE: path | CHUNK_ID | SCORE]):
{context}

STRICT RULES:
- Use ONLY the Context for factual claims. Do NOT use outside knowledge.
- If the Context does not explicitly support the answer, you MUST say:
  FINAL_ANSWER: I don't know based on the provided documents.
- EVIDENCE bullets must be exact verbatim quotes copied from the Context.
- Do NOT write meta-statements like "none of the chunks mention X" as evidence.

OUTPUT FORMAT (exactly):
FINAL_ANSWER: <one short paragraph>

EVIDENCE:
- "<verbatim quote 1 from Context>"
- "<verbatim quote 2 from Context>"
""".strip()

        answer_body = call_ollama_generate(
            prompt=prompt,
            model=ollama_model,
            base_url=base_url,
            timeout_read=timeout_read,
            stream=False,
        )

        unique_paths = []
        for chunk, score in retrieved:
            if score < min_score:
                continue
            if chunk.source_path not in unique_paths:
                unique_paths.append(chunk.source_path)

        if unique_paths:
            srcs = "\n".join(f"- {p}" for p in unique_paths)
            answer_body = (answer_body or "").strip() + f"\n\nSOURCES:\n{srcs}"

        return RAGResult(
            answer=(answer_body or "").strip(),
            used_sources=unique_paths,
            missing_terms=missing_terms,
            top_scores=top_scores,
            had_low_confidence=False,
            judge_rejected=False,
            _selected_chunks=selected_chunks,
            _context=context,
        )

class RAGAgent:
    def __init__(self, spec: AgentSpec, embed_model_name: str):
        self.spec = spec
        self.rag = SimpleRAG(data_dir=spec.data_dir, embed_model_name=embed_model_name)

    def build(self):
        self.rag.build_index()

    def run(
        self,
        user_query: str,
        retrieval_query: str,
        top_k: int,
        min_score: float,
        ollama_model: str,
        conversation_history_block: str,
        timeout_read: float,
        base_url: str,
    ) -> RAGResult:
        return self.rag.answer(
            user_query=user_query,
            retrieval_query=retrieval_query,
            top_k=top_k,
            min_score=min_score,
            ollama_model=ollama_model,
            conversation_history_block=conversation_history_block,
            timeout_read=timeout_read,
            base_url=base_url,
        )

    def peek_retrieval_score(self, query: str) -> float:
        """
        Cheap routing primitive: "how well does this agent match the query?"
        """
        try:
            hits = self.rag.index.search(query, top_k=1)
            return float(hits[0][1]) if hits else -1.0
        except Exception:
            return -1.0

# =============================================================================
# Planner (not used, retrieval-based routing used instead)
# =============================================================================

class Planner:
    def __init__(self, agent_specs: List[AgentSpec], model_name: str = "all-MiniLM-L6-v2"):
        self.model = _load_sentence_transformer(model_name)
        self.agent_specs: List[AgentSpec] = []
        self.agent_desc_emb: Optional[np.ndarray] = None
        self.refresh(agent_specs)

    def refresh(self, agent_specs: List[AgentSpec]):
        self.agent_specs = list(agent_specs)
        if not self.agent_specs:
            self.agent_desc_emb = None
            return

        texts = [f"{a.name}: {a.description}" for a in self.agent_specs]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        self.agent_desc_emb = emb

    def rank_agents(self, query: str, history: str = "") -> List[Tuple[AgentSpec, float]]:
        if not self.agent_specs or self.agent_desc_emb is None:
            return []

        q = query if not history else f"{query}\n\nHistory:\n{history}"
        q_emb = self.model.encode([q], convert_to_numpy=True)[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)

        scores = np.dot(self.agent_desc_emb, q_emb)
        order = np.argsort(scores)[::-1]
        return [(self.agent_specs[i], float(scores[i])) for i in order]

# =============================================================================
# Evaluator + Orchestrator
# =============================================================================

class AnswerEvaluator:
    def __init__(
        self,
        max_missing_terms: int = 3,
        require_sources: bool = True,
        min_top_score: float = 0.15,
        min_top_score_for_abstention: float = 0.30,
    ):
        self.max_missing_terms = max_missing_terms
        self.require_sources = require_sources
        self.min_top_score = min_top_score
        self.min_top_score_for_abstention = min_top_score_for_abstention

    def is_good(self, query: str, result: RAGResult, selected_chunks: List[DocumentChunk]) -> Tuple[bool, str]:
        if result.judge_rejected:
            return False, "Verifier rejected answer (not supported by retrieved context)"

        if result.had_low_confidence:
            return False, "Low retrieval confidence flag"

        if result.top_scores and result.top_scores[0] < self.min_top_score:
            return False, f"Top retrieval score too low ({result.top_scores[0]:.3f})"

        lowered = (result.answer or "").lower()
        is_abstention = ("final_answer:" in lowered and "i don't know" in lowered) or ("i don't know based on the provided documents" in lowered)

        if self.require_sources and not result.used_sources and not is_abstention:
            return False, "No sources used"

        if len(result.missing_terms) > self.max_missing_terms:
            return False, f"Too many missing terms ({len(result.missing_terms)}): {result.missing_terms}"

        # Abstentions are allowed only when retrieval is still topically relevant.
        if is_abstention:
            if result.top_scores and result.top_scores[0] < self.min_top_score_for_abstention:
                return False, f"Abstention from weakly relevant retrieval ({result.top_scores[0]:.3f})"
            query_terms = extract_content_terms(query)
            if query_terms and len(result.missing_terms) >= len(query_terms):
                return False, "Abstention from context with zero query-term coverage"
            return True, "OK (abstained)"

        # For non-abstaining answers, require explicit coverage of key query terms in retrieved context.
        if result.missing_terms:
            return False, f"Missing key query terms in evidence: {result.missing_terms}"

        return True, "OK"

class Orchestrator:
    def __init__(
        self,
        agents: List[RAGAgent],
        planner: Planner,
        evaluator: AnswerEvaluator,
        memory_recent: RecentWindowMemory,
        memory_summary: SummaryStateMemory,
        memory_longterm: LongTermConversationStore,
        router: MemoryRouter,
        max_attempts: int = 3,
        debug: bool = True,
        base_url: str = "http://localhost:11434",
        timeout_read: float = 120.0,
    ):
        self.agents_by_name: Dict[str, RAGAgent] = {a.spec.name: a for a in agents}
        self.planner = planner  # kept (not used for routing now)
        self.evaluator = evaluator
        self.max_attempts = max_attempts
        self.debug = debug

        self.memory_recent = memory_recent
        self.memory_summary = memory_summary
        self.memory_longterm = memory_longterm
        self.router = router

        self.base_url = base_url
        self.timeout_read = timeout_read

    def register_agent(self, agent: RAGAgent):
        self.agents_by_name[agent.spec.name] = agent
        self.planner.refresh([a.spec for a in self.agents_by_name.values()])

    def print_agent_description(self, agent_name: str) -> None:
        agent = self.agents_by_name.get(agent_name)
        if agent is None:
            print(f"[Describe] No agent named '{agent_name}'.")
            if self.agents_by_name:
                print("[Describe] Available agents:", ", ".join(sorted(self.agents_by_name.keys())))
            return

        desc = (agent.spec.description or "").strip()
        print(f"[Describe] Agent: {agent.spec.name}")
        print(f"[Describe] Description: {desc if desc else '(empty)'}")



    def rank_agents_by_retrieval(self, query: str, top_k_peek: int = 1) -> List[Tuple[AgentSpec, float]]:
        """
        Stable agent routing: pick the agent whose corpus retrieves best for the query.
        """
        scored: List[Tuple[AgentSpec, float]] = []
        for agent in self.agents_by_name.values():
            sc = agent.peek_retrieval_score(query)
            scored.append((agent.spec, float(sc)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def answer(
        self,
        user_query: str,
        top_k: int,
        min_score: float,
        ollama_model: str,
    ) -> Tuple[RAGResult, Dict[str, Any]]:
        tried = set()
        last_result: Optional[RAGResult] = None

        recent_text = self.memory_recent.as_text()
        summary_text = self.memory_summary.summary

        # 1) Tier decision via LLM router
        decision = self.router.decide(
            user_query=user_query,
            recent_window=recent_text,
            summary_state=summary_text,
        )

        # 2) Tier 3 retrieval over conversation history
        retrieved_snips = []
        if decision.tier == 3:
            retrieved_snips = self.memory_longterm.search(decision.retrieval_query, top_k=8)

        retrieved_block = ""
        if retrieved_snips:
            lines = []
            for i, (txt, sc) in enumerate(retrieved_snips, start=1):
                lines.append(f"[MEMORY {i} | score={sc:.3f}] {txt}")
            retrieved_block = "\n".join(lines)

        # 3) Build co-reference-only memory block
        if decision.tier == 1:
            convo_block = f"RECENT WINDOW:\n{recent_text}"
        elif decision.tier == 2:
            convo_block = f"RECENT WINDOW:\n{recent_text}\n\nSUMMARY STATE:\n{summary_text}"
        else:
            convo_block = (
                f"RECENT WINDOW:\n{recent_text}\n\nSUMMARY STATE:\n{summary_text}\n\nRETRIEVED LONG-TERM MEMORY:\n{retrieved_block}"
            )

        # 4) Rewrite query (single-line) + sanitize against topic drift
        raw_rewrite = self.router.rewrite_for_retrieval_and_routing(
            user_query=user_query,
            recent_window=recent_text if decision.tier >= 1 else "",
            summary_state=summary_text if decision.tier >= 2 else "",
            retrieved_memory_snippets=retrieved_block if decision.tier == 3 else "",
        )
        retrieval_query = self.router.sanitize_rewrite_output(raw_rewrite, user_query)

        # 5) Multi-attempt: try best-retrieving agents first
        history_notes: List[str] = []

        for attempt in range(self.max_attempts):
            ranked = self.rank_agents_by_retrieval(retrieval_query, top_k_peek=1)
            if not ranked:
                return (
                    RAGResult(
                        answer="No agents available to answer the question.",
                        used_sources=[],
                        missing_terms=extract_content_terms(user_query),
                        top_scores=[],
                        had_low_confidence=True,
                    ),
                    {"tier": decision.tier, "router_rationale": decision.rationale, "retrieved_memory": retrieved_snips, "rewritten_query": retrieval_query},
                )

            chosen_spec = None
            chosen_score = None
            for spec, score in ranked:
                if spec.name not in tried:
                    chosen_spec = spec
                    chosen_score = score
                    break
            if chosen_spec is None:
                chosen_spec, chosen_score = ranked[0]

            tried.add(chosen_spec.name)
            agent = self.agents_by_name[chosen_spec.name]

            if self.debug:
                print(f"[Router] tier={decision.tier} rationale={decision.rationale}")
                print(f"[Rewrite] {retrieval_query}")
                print(f"[AgentRoute] Attempt {attempt+1}/{self.max_attempts} | Chosen agent: {chosen_spec.name} (peek_top_score={chosen_score:.3f})")

            result = agent.run(
                user_query=user_query,
                retrieval_query=retrieval_query,
                top_k=top_k,
                min_score=min_score,
                ollama_model=ollama_model,
                conversation_history_block=convo_block,
                timeout_read=self.timeout_read,
                base_url=self.base_url,
            )
            last_result = result

            # 6) Entailment verification gate (abstentions allowed)
            if (not result.had_low_confidence) and result._context:
                # If the model abstained properly, skip judging.
                if _is_abstention(result.answer):
                    supported, why = True, "Abstention allowed"
                else:
                    supported, why = entailment_judge(
                        context=result._context,
                        user_query=user_query,
                        model_output=result.answer,
                        model=ollama_model,
                        base_url=self.base_url,
                        timeout_read=self.timeout_read,
                    )

                if self.debug:
                    print(f"[Judge] supported={supported} reason={why}")

                if not supported:
                    result.judge_rejected = True
                    # Do NOT mark had_low_confidence; retrieval might be strong, but answer is ungrounded.

            ok, reason = self.evaluator.is_good(user_query, result, selected_chunks=result._selected_chunks or [])

            if self.debug:
                ts = result.top_scores[0] if result.top_scores else 0.0
                print(f"[Eval] ok={ok} reason={reason} top_score={ts:.3f} missing_terms={len(result.missing_terms)} sources={len(result.used_sources)}")

            if ok:
                # Avoid misleading citations when the model abstains.
                if _is_abstention(result.answer):
                    result.answer = re.sub(r"\n\nSOURCES:\n(?:- .*(?:\n|$))+", "", result.answer.strip(), flags=re.IGNORECASE).strip()
                    result.used_sources = []
                    return result, {
                        "tier": decision.tier,
                        "router_rationale": decision.rationale,
                        "retrieved_memory": retrieved_snips,
                        "rewritten_query": retrieval_query,
                    }

                result.answer = f"[Agent: {chosen_spec.name}]\n" + result.answer
                return result, {
                    "tier": decision.tier,
                    "router_rationale": decision.rationale,
                    "retrieved_memory": retrieved_snips,
                    "rewritten_query": retrieval_query,
                }

            history_notes.append(
                f"Attempt {attempt+1} used agent '{chosen_spec.name}' but was inadequate ({reason}). "
                f"Missing terms: {result.missing_terms}. Sources: {result.used_sources}."
            )

        if last_result is None:
            last_result = RAGResult(
                answer="No attempt was made (unexpected).",
                used_sources=[],
                missing_terms=[],
                top_scores=[],
                had_low_confidence=True,
            )

        # Best-effort wrapping
        if history_notes:
            last_result.answer = "FINAL_ANSWER: I don't know based on the provided documents."
            last_result.used_sources = []
            last_result.judge_rejected = False
            last_result.had_low_confidence = True
            return last_result, {
                "tier": decision.tier,
                "router_rationale": decision.rationale,
                "retrieved_memory": retrieved_snips,
                "rewritten_query": retrieval_query,
                "attempt_failures": history_notes,
            }

        if not last_result.had_low_confidence and not last_result.judge_rejected:
            last_result.answer = "[Orchestrator] Best-effort answer (confidence checks did not fully pass):\n\n" + last_result.answer

        return last_result, {
            "tier": decision.tier,
            "router_rationale": decision.rationale,
            "retrieved_memory": retrieved_snips,
            "rewritten_query": retrieval_query,
        }

# =============================================================================
# Auto agent creation (folder => agent)
# =============================================================================

def looks_like_path(text: str) -> bool:
    t = text.strip().strip('"').strip("'")
    return os.path.exists(t)

def parse_add_command(user_text: str) -> Optional[str]:
    m = re.match(r"^\s*add\s+(.+?)\s*$", user_text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip().strip('"').strip("'")

def infer_agent_name_from_path(path: str) -> str:
    p = os.path.abspath(path)
    if os.path.isdir(p):
        return os.path.basename(p) or "Agent"
    parent = os.path.basename(os.path.dirname(p))
    return parent or "Agent"

def default_agent_description(name: str) -> str:
    nice = name.replace("_", " ").replace("-", " ").strip()
    return f"Documents in folder '{nice}'. Answers questions grounded in those files."

def ensure_agent_from_path(
    path: str,
    embed_model_name: str,
    orchestrator: Orchestrator,
    debug: bool = True,
) -> Optional[str]:
    p = os.path.abspath(path)
    if not os.path.exists(p):
        return None

    if os.path.isfile(p):
        if not p.lower().endswith(".txt"):
            return None
        p = os.path.dirname(p)

    if not os.path.isdir(p):
        return None

    has_txt = False
    for root, _, files in os.walk(p):
        if any(f.lower().endswith(".txt") for f in files):
            has_txt = True
            break
    if not has_txt:
        return None

    agent_name = infer_agent_name_from_path(p)

    if agent_name in orchestrator.agents_by_name:
        if debug:
            print(f"[AutoAgent] Agent '{agent_name}' already exists. Using existing agent.")
        return agent_name

    spec = AgentSpec(name=agent_name, description=default_agent_description(agent_name), data_dir=p)
    agent = RAGAgent(spec, embed_model_name=embed_model_name)

    print(f"[AutoAgent] Creating new agent '{agent_name}' from: {p}")
    agent.build()
    orchestrator.register_agent(agent)
    print(f"[AutoAgent] Registered agent '{agent_name}'. Total agents: {len(orchestrator.agents_by_name)}")
    return agent_name

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Agentic RAG with three-tier memory (recent -> summary -> retrieval).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing agent subfolders of .txt files.")
    parser.add_argument("--embed_model_name", type=str, default="all-MiniLM-L6-v2", help="Sentence-transformers model name for embeddings.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--min_score", type=float, default=0.15, help="Minimum cosine similarity score for chunk inclusion.")
    parser.add_argument("--ollama_model", type=str, default="llama3:8b", help="Which Ollama model to use.")
    parser.add_argument("--max_attempts", type=int, default=3, help="Max orchestrator attempts (try different agents).")
    parser.add_argument("--debug", action="store_true", help="Print router/planner/evaluator debug logs.")
    parser.add_argument("--ollama_base_url", type=str, default="http://localhost:11434", help="Ollama base URL.")
    parser.add_argument("--ollama_timeout_read", type=float, default=120.0, help="Ollama read timeout in seconds.")
    parser.add_argument("--memory_recent_turns", type=int, default=12, help="Recent window size (turns).")
    parser.add_argument("--memory_alpha", type=float, default=0.6, help="Hybrid memory alpha: vector weight vs TF-IDF.")
    parser.add_argument("--memory_store_dir", type=str, default=".conv_memory", help="Persist dir for long-term conv memory.")
    return parser.parse_args()

def discover_agents_from_root(data_dir: str) -> List[AgentSpec]:
    specs = []

    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)

        if not os.path.isdir(full_path):
            continue

        # Check if folder contains at least one .txt file (recursive)
        has_txt = False
        for root, _, files in os.walk(full_path):
            if any(f.lower().endswith(".txt") for f in files):
                has_txt = True
                break

        if not has_txt:
            continue

        specs.append(
            AgentSpec(
                name=entry,
                description=default_agent_description(entry),
                data_dir=full_path,
            )
        )

    return specs



# =============================================================================
# Reusable system builder for evaluation / scripting
# =============================================================================

@dataclass
class SystemConfig:
    data_dir: str
    embed_model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    min_score: float = 0.15
    ollama_model: str = "llama3:8b"
    max_attempts: int = 3
    debug: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_read: float = 120.0
    memory_recent_turns: int = 12
    memory_alpha: float = 0.6
    memory_store_dir: str = ".conv_memory"

def config_from_args(args) -> SystemConfig:
    return SystemConfig(
        data_dir=args.data_dir,
        embed_model_name=args.embed_model_name,
        top_k=args.top_k,
        min_score=args.min_score,
        ollama_model=args.ollama_model,
        max_attempts=args.max_attempts,
        debug=args.debug,
        ollama_base_url=args.ollama_base_url,
        ollama_timeout_read=args.ollama_timeout_read,
        memory_recent_turns=args.memory_recent_turns,
        memory_alpha=args.memory_alpha,
        memory_store_dir=args.memory_store_dir,
    )

def build_system(config: SystemConfig) -> Tuple[Orchestrator, SystemConfig]:
    agent_specs = discover_agents_from_root(config.data_dir)

    if not agent_specs:
        raise ValueError(f"No valid agent folders found in {config.data_dir}")

    agents: List[RAGAgent] = []
    for spec in agent_specs:
        if not os.path.isdir(spec.data_dir):
            raise ValueError(
                f"Missing agent folder: {spec.data_dir}\n"
                f"Expected subfolders under --data_dir."
            )
        agent = RAGAgent(spec, embed_model_name=config.embed_model_name)
        agents.append(agent)

    print("[System] Building indices for all agents...")
    for a in agents:
        print(f"\n[System] Building agent '{a.spec.name}' from: {a.spec.data_dir}")
        a.build()

    memory_recent = RecentWindowMemory(max_turns=config.memory_recent_turns)
    memory_summary = SummaryStateMemory(initial_summary="")
    memory_longterm = LongTermConversationStore(
        embed_model_name=config.embed_model_name,
        alpha=config.memory_alpha,
        persist_path=config.memory_store_dir,
    )
    router = MemoryRouter(
        ollama_model=config.ollama_model,
        base_url=config.ollama_base_url,
        timeout_read=config.ollama_timeout_read,
    )

    planner = Planner(agent_specs=agent_specs, model_name=config.embed_model_name)
    evaluator = AnswerEvaluator(
        max_missing_terms=3,
        require_sources=True,
        min_top_score=config.min_score,
    )

    orch = Orchestrator(
        agents=agents,
        planner=planner,
        evaluator=evaluator,
        memory_recent=memory_recent,
        memory_summary=memory_summary,
        memory_longterm=memory_longterm,
        router=router,
        max_attempts=config.max_attempts,
        debug=config.debug,
        base_url=config.ollama_base_url,
        timeout_read=config.ollama_timeout_read,
    )
    return orch, config

def build_system_from_args(args) -> Tuple[Orchestrator, SystemConfig]:
    return build_system(config_from_args(args))
def main():
    args = parse_args()
    _log_runtime_embedding_device_once()

    agent_specs = discover_agents_from_root(args.data_dir)

    if not agent_specs:
        raise ValueError(f"No valid agent folders found in {args.data_dir}")

    agents: List[RAGAgent] = []
    for spec in agent_specs:
        if not os.path.isdir(spec.data_dir):
            raise ValueError(
                f"Missing agent folder: {spec.data_dir}\n"
                f"Expected subfolders under --data_dir: recipes/, tech/, animals/"
            )
        agent = RAGAgent(spec, embed_model_name=args.embed_model_name)
        agents.append(agent)

    print("[System] Building indices for all agents...")
    for a in agents:
        print(f"\n[System] Building agent '{a.spec.name}' from: {a.spec.data_dir}")
        a.build()

    memory_recent = RecentWindowMemory(max_turns=args.memory_recent_turns)
    memory_summary = SummaryStateMemory(initial_summary="")
    memory_longterm = LongTermConversationStore(
        embed_model_name=args.embed_model_name,
        alpha=args.memory_alpha,
        persist_path=args.memory_store_dir,
    )
    router = MemoryRouter(
        ollama_model=args.ollama_model,
        base_url=args.ollama_base_url,
        timeout_read=args.ollama_timeout_read,
    )

    planner = Planner(agent_specs=agent_specs, model_name=args.embed_model_name)  # kept
    evaluator = AnswerEvaluator(max_missing_terms=3, require_sources=True, min_top_score=args.min_score)

    orch = Orchestrator(
        agents=agents,
        planner=planner,
        evaluator=evaluator,
        memory_recent=memory_recent,
        memory_summary=memory_summary,
        memory_longterm=memory_longterm,
        router=router,
        max_attempts=args.max_attempts,
        debug=args.debug,
        base_url=args.ollama_base_url,
        timeout_read=args.ollama_timeout_read,
    )

    print("\nAgentic RAG System ready (Three-tier memory)")
    print(f"LLM Model: {args.ollama_model}")
    print("Agents: " + ", ".join(sorted(orch.agents_by_name.keys())))
    print("Commands:")
    print("  - add <path-to-folder-or-txt>   (auto create agent + index)")
    print("  - describe <agent-name>          (show agent description)")
    print("  - clear                         (clear recent + summary)")
    print("  - reset                         (clear recent+summary and delete long-term store on disk)")
    print("  - exit / quit")
    print("Ask a question. You can also paste a file/folder path.\n")

    while True:
        try:
            raw = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        cmd = raw.lower().strip()
        if cmd in {"exit", "quit"}:
            print("Goodbye.")
            break

        if cmd == "clear":
            memory_recent.clear()
            memory_summary.clear()
            print("\nSystem:\nCleared recent window + summary.\n" + "=" * 80 + "\n")
            continue

        if cmd == "reset":
            memory_recent.clear()
            memory_summary.clear()
            # Delete long-term memory store on disk
            shutil.rmtree(args.memory_store_dir, ignore_errors=True)
            # Recreate store object empty (so further adds work)
            memory_longterm = LongTermConversationStore(
                embed_model_name=args.embed_model_name,
                alpha=args.memory_alpha,
                persist_path=args.memory_store_dir,
            )
            orch.memory_longterm = memory_longterm
            print("\nSystem:\nReset memory (recent+summary cleared, long-term store deleted).\n" + "=" * 80 + "\n")
            continue

        if cmd.startswith("describe "):
            agent_name = raw.split(maxsplit=1)[1].strip()
            orch.print_agent_description(agent_name)
            continue

        # store user in tier-1 window + tier-3 store
        memory_recent.add("user", raw)
        memory_longterm.add_turn("user", raw)

        # explicit add command
        add_path = parse_add_command(raw)
        if add_path:
            created = ensure_agent_from_path(add_path, args.embed_model_name, orch, debug=args.debug)
            msg = (
                f"Added/available agent: {created}. Current agents: {', '.join(sorted(orch.agents_by_name.keys()))}"
                if created else
                f"Could not create agent from path: {add_path} (must be a folder with .txt or a .txt file)."
            )
            print("\nSystem:\n")
            print(msg)
            print("\n" + "=" * 80 + "\n")

            memory_recent.add("assistant", msg)
            memory_summary.update_with_llm(
                user_text=raw,
                assistant_text=msg,
                model=args.ollama_model,
                base_url=args.ollama_base_url,
                timeout_read=args.ollama_timeout_read,
            )
            continue

        # implicit path detection
        if looks_like_path(raw.strip('"').strip("'")):
            created = ensure_agent_from_path(raw.strip('"').strip("'"), args.embed_model_name, orch, debug=args.debug)
            if created:
                msg = f"Detected path and added/available agent: {created}. Now ask your question about those files."
                print("\nSystem:\n")
                print(msg)
                print("\n" + "=" * 80 + "\n")

                memory_recent.add("assistant", msg)
                memory_summary.update_with_llm(
                    user_text=raw,
                    assistant_text=msg,
                    model=args.ollama_model,
                    base_url=args.ollama_base_url,
                    timeout_read=args.ollama_timeout_read,
                )
                continue

        # normal QA
        result, _dbg = orch.answer(
            user_query=raw,
            top_k=args.top_k,
            min_score=args.min_score,
            ollama_model=args.ollama_model,
        )

        print("\nRAG:\n")
        print(result.answer)
        print("\n" + "=" * 80 + "\n")

        assistant_text_for_memory = (result.answer or "")[:1200]
        memory_recent.add("assistant", assistant_text_for_memory)

        memory_summary.update_with_llm(
            user_text=raw,
            assistant_text=assistant_text_for_memory,
            model=args.ollama_model,
            base_url=args.ollama_base_url,
            timeout_read=args.ollama_timeout_read,
        )

if __name__ == "__main__":
    main()
