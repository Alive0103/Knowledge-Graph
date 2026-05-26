from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np

from vendor_utils import ensure_vendor_path


logger = logging.getLogger(__name__)

_HAS_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_HAS_LATIN_RE = re.compile(r"[A-Za-z]")


def contains_cjk(text: str) -> bool:
    return bool(_HAS_CJK_RE.search(text))


def normalize_alias(alias: str) -> str:
    return re.sub(r"\s+", " ", alias.strip())


def alias_key(alias: str) -> str:
    return alias.casefold() if not contains_cjk(alias) else alias


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = alias_key(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def normalize_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return None
    return vector / norm


@dataclass(frozen=True)
class AliasEntry:
    alias: str
    length: int
    is_latin: bool
    pattern: re.Pattern[str] | None = None

    def iter_matches(self, text: str) -> Iterator[tuple[int, int, str]]:
        if self.pattern is not None:
            for match in self.pattern.finditer(text):
                yield match.start(), match.end(), match.group(0)
            return

        start = 0
        while True:
            index = text.find(self.alias, start)
            if index == -1:
                break
            end = index + len(self.alias)
            yield index, end, text[index:end]
            start = index + 1


@dataclass(frozen=True)
class MentionSpan:
    start: int
    end: int
    text: str


class DictionaryMentionExtractor:
    """
    Alias-dictionary mention extractor.

    This is the practical recovery path when the original NER model is missing.
    """

    def __init__(self, aliases: Sequence[str]) -> None:
        cleaned = []
        for alias in aliases:
            candidate = normalize_alias(alias)
            if not candidate:
                continue
            if len(candidate) > 80:
                continue
            if candidate.startswith("http://") or candidate.startswith("https://"):
                continue
            cleaned.append(candidate)

        cleaned = unique_preserve_order(cleaned)
        entries: list[AliasEntry] = []
        for alias in cleaned:
            is_latin = bool(_HAS_LATIN_RE.search(alias)) and not contains_cjk(alias)
            pattern = None
            if is_latin:
                escaped = re.escape(alias)
                pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
            entries.append(AliasEntry(alias=alias, length=len(alias), is_latin=is_latin, pattern=pattern))

        self.entries = sorted(entries, key=lambda item: (-item.length, item.alias.casefold()))

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "DictionaryMentionExtractor":
        aliases: list[str] = []
        for record in records:
            aliases.extend(collect_aliases(record))
        logger.info("Built dictionary extractor with %s unique aliases", len(unique_preserve_order(aliases)))
        return cls(aliases)

    def extract_spans(self, text: str, min_length: int) -> list[MentionSpan]:
        if not text:
            return []

        occupied = [False] * len(text)
        accepted: list[MentionSpan] = []
        seen: set[str] = set()

        for entry in self.entries:
            if entry.length < min_length:
                continue
            for start, end, matched_text in entry.iter_matches(text):
                key = alias_key(matched_text)
                if key in seen:
                    continue
                if any(occupied[start:end]):
                    continue
                for index in range(start, end):
                    occupied[index] = True
                accepted.append(MentionSpan(start=start, end=end, text=matched_text))
                seen.add(key)
                break

        accepted.sort(key=lambda item: item.start)
        return accepted

    def extract(self, text: str, min_length: int) -> list[str]:
        return [span.text for span in self.extract_spans(text, min_length)]


class TransformerMentionExtractor:
    """Optional NER-backed extractor for a closer reproduction of the original pipeline."""

    def __init__(self, model_name_or_path: str, label_mapping_path: str | None = None, max_length: int = 512) -> None:
        ensure_vendor_path()
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers/torch are required for transformer extraction") from exc

        self.torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.id_to_label = self._load_id_to_label(label_mapping_path)

    def _load_id_to_label(self, label_mapping_path: str | None) -> dict[int, str]:
        if label_mapping_path:
            mapping_file = Path(label_mapping_path)
        else:
            mapping_file = Path(self.model.name_or_path) / "label_mapping.json"

        if mapping_file.exists():
            with open(mapping_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            raw_id_to_label = payload.get("id_to_label", {})
            if raw_id_to_label:
                return {int(key): value for key, value in raw_id_to_label.items()}

        config_mapping = getattr(self.model.config, "id2label", None) or {}
        if config_mapping:
            return {int(key): value for key, value in config_mapping.items()}

        raise RuntimeError("No usable label mapping found for transformer extractor")

    def extract(self, text: str, min_length: int) -> list[str]:
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with self.torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = self.torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        entities: list[str] = []
        current_tokens: list[str] = []

        for token, label_id in zip(tokens, predictions):
            if token in {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}:
                continue
            label = self.id_to_label.get(label_id, "O")
            token = token.replace("##", "")

            if label.startswith("B-"):
                if current_tokens:
                    entity = self._decode_tokens(current_tokens)
                    if entity and len(entity) >= min_length:
                        entities.append(entity)
                current_tokens = [token]
            elif label.startswith("I-"):
                if current_tokens:
                    current_tokens.append(token)
            else:
                if current_tokens:
                    entity = self._decode_tokens(current_tokens)
                    if entity and len(entity) >= min_length:
                        entities.append(entity)
                current_tokens = []

        if current_tokens:
            entity = self._decode_tokens(current_tokens)
            if entity and len(entity) >= min_length:
                entities.append(entity)

        return unique_preserve_order(entity.strip() for entity in entities if entity.strip())

    def _decode_tokens(self, tokens: Sequence[str]) -> str:
        token_ids = self.tokenizer.convert_tokens_to_ids(list(tokens))
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


class HybridMentionExtractor:
    """
    Prefer transformer predictions, but fall back to dictionary matches when
    the weakly trained CPU smoke model extracts nothing.
    """

    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback

    def extract(self, text: str, min_length: int) -> list[str]:
        primary_result = self.primary.extract(text, min_length)
        if primary_result:
            return primary_result
        return self.fallback.extract(text, min_length)


class NullVectorizer:
    def vectorize_terms(self, terms: Sequence[str]) -> None:
        return None


class HashingVectorizer:
    """Deterministic lightweight fallback used for smoke tests and recovery runs."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def vectorize_terms(self, terms: Sequence[str]) -> Optional[list[float]]:
        term_vectors = [self._vectorize_single(term) for term in terms if str(term).strip()]
        if not term_vectors:
            return None
        merged = np.mean(np.array(term_vectors), axis=0)
        normalized = normalize_vector(merged)
        return normalized.tolist() if normalized is not None else None

    def _vectorize_single(self, term: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in self._term_features(term):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[-1] % 2 == 0 else -1.0
            vector[index] += sign
        normalized = normalize_vector(vector)
        return normalized if normalized is not None else vector

    @staticmethod
    def _term_features(term: str) -> list[str]:
        text = term.casefold().strip()
        if not text:
            return []
        compact = text.replace(" ", "")
        features = {text, compact}
        for size in (2, 3):
            if len(compact) >= size:
                for index in range(len(compact) - size + 1):
                    features.add(compact[index:index + size])
        return sorted(features)


class TransformerVectorizer:
    def __init__(self, model_name_or_path: str, dim: int = 1024, batch_size: int = 32) -> None:
        ensure_vendor_path()
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers/torch are required for transformer vectorization") from exc

        self.torch = torch
        self.dim = dim
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def vectorize_terms(self, terms: Sequence[str]) -> Optional[list[float]]:
        texts = [str(term).strip() for term in terms if str(term).strip()]
        if not texts:
            return None

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            for vector in batch_vectors:
                resized = self._resize(vector)
                normalized = normalize_vector(resized)
                if normalized is not None:
                    vectors.append(normalized)

        if not vectors:
            return None

        merged = np.mean(np.array(vectors), axis=0)
        normalized = normalize_vector(merged)
        return normalized.tolist() if normalized is not None else None

    def _resize(self, vector: np.ndarray) -> np.ndarray:
        if len(vector) == self.dim:
            return vector.astype(np.float32)
        if len(vector) < self.dim:
            return np.pad(vector.astype(np.float32), (0, self.dim - len(vector)), constant_values=0)
        return vector[:self.dim].astype(np.float32)


def collect_aliases(record: dict) -> list[str]:
    aliases: list[str] = []
    label = record.get("label")
    if isinstance(label, str):
        aliases.append(label)
    for key in ("zh_aliases", "en_aliases", "aliases_zh", "aliases_en"):
        value = record.get(key, [])
        if isinstance(value, list):
            aliases.extend(str(item) for item in value if str(item).strip())
    return aliases
