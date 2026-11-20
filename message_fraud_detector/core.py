"""Core immutable types used by the detector."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class TextFeatures:
    """Normalized features derived from a message."""

    message_id: str
    text: str
    normalized_text: str
    char_length: int
    word_count: int
    unique_word_count: int
    digit_count: int
    uppercase_count: int
    punctuation_count: int
    url_count: int
    email_count: int
    phone_count: int
    urgency_terms: int
    credential_terms: int
    payment_terms: int
    threat_terms: int
    emoji_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuleHit:
    """A single explainable MutationGuard-lite rule result."""

    rule_id: str
    category: str
    severity: str
    weight: float
    evidence: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalHit:
    """A nearest known example returned by TF-IDF retrieval."""

    label: str
    score: float
    text: str
    source: str
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreResult:
    """Complete score returned for one message."""

    message_id: str
    text: str
    label: str
    score: float
    confidence: float
    risk_level: str
    rule_hits: List[RuleHit] = field(default_factory=list)
    retrieval_hits: List[RetrievalHit] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["rule_hits"] = [hit.to_dict() for hit in self.rule_hits]
        value["retrieval_hits"] = [hit.to_dict() for hit in self.retrieval_hits]
        return value


@dataclass(frozen=True)
class Example:
    """A labeled or unlabeled example stored in the retrieval index."""

    text: str
    label: str
    source: str = "input"
    message_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectorConfig:
    """Runtime detector settings."""

    retrieval_weight: float = 0.55
    rule_weight: float = 0.45
    fraud_threshold: float = 0.48
    suspicious_threshold: float = 0.28
    top_k: int = 5
    retrieval_min_score: float = 0.08
    rule_cap: float = 1.0
    lower_retrieve: bool = True
    strip_accents: bool = True
    max_features: int = 5000
    ngram_min: int = 1
    ngram_max: int = 2
    sublinear_tf: bool = True
    model_version: str = "1.0"

    def validate(self) -> None:
        if not 0.0 <= self.retrieval_weight <= 1.0:
            raise ValueError("retrieval_weight must be between 0 and 1")
        if not 0.0 <= self.rule_weight <= 1.0:
            raise ValueError("rule_weight must be between 0 and 1")
        if abs(self.retrieval_weight + self.rule_weight - 1.0) > 0.0001:
            raise ValueError("retrieval_weight and rule_weight must sum to 1")
        if not 0.0 <= self.fraud_threshold <= 1.0:
            raise ValueError("fraud_threshold must be between 0 and 1")
        if not 0.0 <= self.suspicious_threshold <= 1.0:
            raise ValueError("suspicious_threshold must be between 0 and 1")
        if self.suspicious_threshold >= self.fraud_threshold:
            raise ValueError("suspicious_threshold must be below fraud_threshold")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.max_features < 1:
            raise ValueError("max_features must be at least 1")
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ValueError("invalid ngram configuration")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def require_text(value: Any, field_name: str = "text") -> str:
    """Validate and return a string field."""
    if not isinstance(value, str):
        raise ValueError("{} must be a string".format(field_name))
    if not value.strip():
        raise ValueError("{} must not be empty".format(field_name))
    return value.strip()

"""Text normalization and lightweight feature extraction."""

import re
import unicodedata
from typing import Iterable, List, Sequence



TOKEN_PATTERN = re.compile(r"[^\w]+", re.UNICODE)
URL_PATTERN = re.compile(r"(https?://|www\.)[^\s]+", re.I)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}")
DIGIT_PATTERN = re.compile(r"\d")
UPPER_PATTERN = re.compile(r"[A-Z]")
PUNCT_PATTERN = re.compile(r"[^\w\s]")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]",
    re.UNICODE,
)

URGENCY_TERMS = {
    "urgent", "immediately", "now", "today", "quick", "quickly", "expire",
    "expired", "deadline", "limited", "last", "hurry", "act", "warning",
}
CREDENTIAL_TERMS = {
    "password", "passcode", "otp", "pin", "login", "username", "credential",
    "verify", "verification", "account", "security", "ssn", "identity",
}
PAYMENT_TERMS = {
    "gift", "card", "crypto", "bitcoin", "ethereum", "cash", "wire",
    "transfer", "payment", "pay", "fee", "refund", "reward", "prize",
    "bank", "zelle", "paypal", "deposit", "tax",
}
THREAT_TERMS = {
    "suspend", "suspended", "locked", "blocked", "arrest", "legal",
    "police", "terminate", "closure", "deleted", "penalty", "fine",
    "lawsuit", "shutdown",
}


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize(text: str, strip_accents_enabled: bool = True) -> str:
    if strip_accents_enabled:
        text = strip_accents(text)
    text = text.lower()
    text = text.replace("http://", " http:// ").replace("https://", " https:// ")
    text = text.replace("www.", " www. ")
    text = re.sub(r"\s+", " ", text)
    text = TOKEN_PATTERN.sub(" ", text)
    return " ".join(text.split())


def tokenize(text: str, strip_accents_enabled: bool = True) -> List[str]:
    normalized = normalize(text, strip_accents_enabled)
    tokens = normalized.split()
    expanded: List[str] = []
    for token in tokens:
        expanded.append(token)
    return expanded


def character_ngrams(text: str, size: int = 3) -> Iterable[str]:
    compact = normalize(text)
    if size < 1:
        raise ValueError("ngram size must be positive")
    if len(compact) < size:
        yield compact
        return
    for index in range(len(compact) - size + 1):
        yield compact[index:index + size]


def count_terms(words: Sequence[str], vocabulary: set[str]) -> int:
    return sum(1 for word in words if word in vocabulary)


def extract_features(message_id: str, text: str, strip_accents_enabled: bool = True) -> TextFeatures:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    normalized = normalize(text, strip_accents_enabled)
    words = normalized.split()
    digit_count = len(DIGIT_PATTERN.findall(text))
    uppercase_count = len(UPPER_PATTERN.findall(text))
    punctuation_count = len(PUNCT_PATTERN.findall(text))
    urls = len(URL_PATTERN.findall(text))
    emails = len(EMAIL_PATTERN.findall(text))
    phones = len(PHONE_PATTERN.findall(text))
    emoji_count = len(EMOJI_PATTERN.findall(text))
    return TextFeatures(
        message_id=message_id,
        text=text,
        normalized_text=normalized,
        char_length=len(text),
        word_count=len(words),
        unique_word_count=len(set(words)),
        digit_count=digit_count,
        uppercase_count=uppercase_count,
        punctuation_count=punctuation_count,
        url_count=urls,
        email_count=emails,
        phone_count=phones,
        urgency_terms=count_terms(words, URGENCY_TERMS),
        credential_terms=count_terms(words, CREDENTIAL_TERMS),
        payment_terms=count_terms(words, PAYMENT_TERMS),
        threat_terms=count_terms(words, THREAT_TERMS),
        emoji_count=emoji_count,
    )

"""Pure-Python TF-IDF vectorizer and cosine similarity index."""

import math
from collections import Counter
from typing import Iterable, List, Sequence, Tuple



class TfIdfVectorizer:
    """Small TF-IDF implementation with word unigrams and bigrams."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_min: int = 1,
        ngram_max: int = 2,
        sublinear_tf: bool = True,
        strip_accents: bool = True,
    ) -> None:
        self.max_features = max_features
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.sublinear_tf = sublinear_tf
        self.strip_accents = strip_accents
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.document_count = 0
        self.fitted = False

    def ngrams(self, text: str) -> List[str]:
        words = normalize(text, self.strip_accents).split()
        terms: List[str] = []
        for size in range(self.ngram_min, self.ngram_max + 1):
            for start in range(0, max(0, len(words) - size + 1)):
                terms.append(" ".join(words[start:start + size]))
        return terms

    def fit(self, documents: Sequence[str]) -> "TfIdfVectorizer":
        if not documents:
            raise ValueError("cannot fit TF-IDF on an empty corpus")
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(self.ngrams(document)))

        ranked = sorted(
            document_frequency,
            key=lambda term: (-document_frequency[term], term),
        )
        selected = ranked[:self.max_features]
        self.vocabulary = {term: index for index, term in enumerate(selected)}
        self.document_count = len(documents)
        self.idf = {}
        for term in selected:
            frequency = document_frequency[term]
            self.idf[term] = math.log((1.0 + self.document_count) / (1.0 + frequency)) + 1.0
        self.fitted = True
        return self

    def fit_transform(self, documents: Sequence[str]) -> List[dict[int, float]]:
        self.fit(documents)
        return [self.transform(document) for document in documents]

    def transform(self, text: str) -> dict[int, float]:
        if not self.fitted:
            raise RuntimeError("vectorizer must be fitted before transform")
        counts = Counter(self.ngrams(text))
        values: dict[int, float] = {}
        norm = 0.0
        for term, count in counts.items():
            index = self.vocabulary.get(term)
            if index is None:
                continue
            weight = (1.0 + math.log(count)) if self.sublinear_tf and count > 0 else float(count)
            weight *= self.idf[term]
            values[index] = weight
            norm += weight * weight
        if norm:
            norm = math.sqrt(norm)
            values = {index: weight / norm for index, weight in values.items()}
        return values

    def cosine_sparse(self, left: dict[int, float], right: dict[int, float]) -> float:
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        score = 0.0
        for index, weight in left.items():
            score += weight * right.get(index, 0.0)
        return max(0.0, min(1.0, score))

    def pairwise_cosine(
        self,
        left_documents: Sequence[str],
        right_documents: Sequence[str],
    ) -> List[List[float]]:
        left_vectors = [self.transform(document) for document in left_documents]
        right_vectors = [self.transform(document) for document in right_documents]
        return [
            [self.cosine_sparse(left, right) for right in right_vectors]
            for left in left_vectors
        ]

    def to_dict(self) -> dict:
        return {
            "max_features": self.max_features,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "sublinear_tf": self.sublinear_tf,
            "strip_accents": self.strip_accents,
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "document_count": self.document_count,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TfIdfVectorizer":
        if not isinstance(value, dict):
            raise ValueError("TF-IDF state must be a JSON object")
        vectorizer = cls(
            max_features=int(value.get("max_features", 5000)),
            ngram_min=int(value.get("ngram_min", 1)),
            ngram_max=int(value.get("ngram_max", 2)),
            sublinear_tf=bool(value.get("sublinear_tf", True)),
            strip_accents=bool(value.get("strip_accents", True)),
        )
        vocabulary = value.get("vocabulary")
        idf = value.get("idf")
        if not isinstance(vocabulary, dict) or not isinstance(idf, dict):
            raise ValueError("TF-IDF state is missing vocabulary or idf")
        vectorizer.vocabulary = {str(term): int(index) for term, index in vocabulary.items()}
        vectorizer.idf = {str(term): float(weight) for term, weight in idf.items()}
        vectorizer.document_count = int(value.get("document_count", 0))
        vectorizer.fitted = bool(value.get("fitted", False))
        if not vectorizer.fitted:
            raise ValueError("serialized TF-IDF vectorizer is not fitted")
        return vectorizer

"""TRE-style nearest-neighbor retrieval index."""

import heapq
from typing import Iterable, List, Sequence



class RetrievalIndex:
    """In-memory TF-IDF retrieval over labeled examples."""

    def __init__(self, vectorizer: TfIdfVectorizer) -> None:
        self.vectorizer = vectorizer
        self.examples: List[Example] = []
        self.vectors: List[dict[int, float]] = []

    def fit(self, examples: Sequence[Example]) -> "RetrievalIndex":
        if not examples:
            raise ValueError("retrieval index needs at least one training example")
        labels = {example.label for example in examples}
        if not labels:
            raise ValueError("retrieval examples must have labels")
        for example in examples:
            if not isinstance(example.label, str) or not example.label.strip():
                raise ValueError("retrieval example label must not be empty")
            if not isinstance(example.text, str) or not example.text.strip():
                raise ValueError("retrieval example text must not be empty")
        self.examples = list(examples)
        texts = [example.text for example in self.examples]
        self.vectors = self.vectorizer.fit_transform(texts)
        return self

    def add(self, example: Example) -> None:
        if not isinstance(example.text, str) or not example.text.strip():
            raise ValueError("example text must not be empty")
        self.examples.append(example)
        self.vectors.append(self.vectorizer.transform(example.text))

    def search(self, text: str, top_k: int = 5) -> List[RetrievalHit]:
        if not self.examples:
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query = self.vectorizer.transform(text)
        candidates = []
        for index, (example, vector) in enumerate(zip(self.examples, self.vectors)):
            score = self.vectorizer.cosine_sparse(query, vector)
            candidates.append((-score, index, example))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [
            RetrievalHit(
                label=example.label,
                score=max(0.0, min(1.0, -negative_score)),
                text=example.text,
                source=example.source,
                index=index,
            )
            for negative_score, index, example in candidates[:top_k]
        ]

    def label_scores(self, hits: Sequence[RetrievalHit], top_k: int = 5) -> dict[str, float]:
        totals: dict[str, float] = {}
        weights: dict[str, float] = {}
        for rank, hit in enumerate(hits[:top_k]):
            rank_weight = 1.0 / (rank + 1.0)
            totals[hit.label] = totals.get(hit.label, 0.0) + hit.score * rank_weight
            weights[hit.label] = weights.get(hit.label, 0.0) + rank_weight
        return {
            label: totals[label] / weights[label]
            for label in totals
        }

    def nearest_label(self, text: str, top_k: int = 5) -> tuple[str, float, List[RetrievalHit]]:
        hits = self.search(text, top_k)
        if not hits:
            return "unknown", 0.0, []
        scores = self.label_scores(hits, top_k)
        label = max(sorted(scores), key=lambda key: (scores[key], key == "fraud"))
        weighted = scores[label]
        return label, weighted, hits

    def size(self) -> int:
        return len(self.examples)

    def to_dict(self) -> dict:
        return {
            "vectorizer": self.vectorizer.to_dict(),
            "examples": [example.to_dict() for example in self.examples],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "RetrievalIndex":
        if not isinstance(value, dict):
            raise ValueError("retrieval index state must be a JSON object")
        vectorizer = TfIdfVectorizer.from_dict(value["vectorizer"])
        raw_examples = value.get("examples")
        if not isinstance(raw_examples, list):
            raise ValueError("retrieval examples must be a list")
        examples = []
        for index, raw in enumerate(raw_examples):
            if not isinstance(raw, dict):
                raise ValueError("example {} must be an object".format(index))
            examples.append(Example(
                text=raw["text"],
                label=raw["label"],
                source=raw.get("source", "input"),
                message_id=raw.get("message_id", ""),
                metadata=raw.get("metadata", {}),
            ))
        return cls(vectorizer).fit(examples)

"""MutationGuard-lite behavioral rule engine."""

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence



@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    category: str
    severity: str
    weight: float
    description: str


RULES = (
    RuleDefinition("urgency_pressure", "behavior", "high", 0.18, "Uses urgency or artificial deadline pressure"),
    RuleDefinition("credential_request", "identity", "high", 0.22, "Requests passwords, OTP, PIN, or account verification"),
    RuleDefinition("payment_request", "payment", "high", 0.20, "Requests payment, transfer, refund, gift card, or crypto"),
    RuleDefinition("threat_coercion", "behavior", "high", 0.18, "Uses suspension, arrest, legal, or account termination threats"),
    RuleDefinition("shortener_or_external_link", "link", "medium", 0.10, "Contains external links or shortened-looking domains"),
    RuleDefinition("unusual_phone_request", "channel", "medium", 0.10, "Uses a phone number or asks to move to another channel"),
    RuleDefinition("impersonation_claim", "identity", "medium", 0.16, "Claims to be support, bank, government, or a known service"),
    RuleDefinition("prize_reward", "payment", "medium", 0.14, "Promises an unexpected reward or prize"),
    RuleDefinition("all_caps", "stylistic", "low", 0.07, "Uses strong all-uppercase emphasis"),
    RuleDefinition("excessive_punctuation", "stylistic", "low", 0.06, "Contains excessive punctuation"),
)


SHORT_LINK_PATTERN = re.compile(
    r"\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|is\.gd|buff\.ly|rebrand\.ly)/?\S*",
    re.I,
)
PHONE_CHANNEL_PATTERN = re.compile(
    r"\b(?:call|text|whatsapp|telegram|phone|sms|message me|contact me)\b",
    re.I,
)
IMPERSONATION_PATTERN = re.compile(
    r"\b(?:bank|paypal|amazon|apple|microsoft|google|irs|police|federal|"
    r"customer service|support team|security team)\b",
    re.I,
)
PRIZE_PATTERN = re.compile(
    r"\b(?:won|winner|selected|congratulations|reward|prize|lottery|claim)\b",
    re.I,
)


class MutationGuardLite:
    """Deterministic behavioral rules with evidence and severity."""

    def evaluate(self, text: str, message_id: str = "") -> List[RuleHit]:
        features = extract_features(message_id, text)
        words = features.normalized_text.split()
        hits: List[RuleHit] = []
        by_id = {rule.rule_id: rule for rule in RULES}

        if features.urgency_terms:
            rule = by_id["urgency_pressure"]
            evidence = ", ".join(sorted(set(words).intersection({
                "urgent", "immediately", "now", "today", "expire", "expired",
                "deadline", "limited", "hurry", "warning",
            })))
            hits.append(RuleHit(rule.rule_id, rule.category, rule.severity, rule.weight,
                                evidence or "urgency language", rule.description))

        if features.credential_terms:
            rule = by_id["credential_request"]
            evidence = ", ".join(sorted(set(words).intersection({
                "password", "passcode", "otp", "pin", "login", "username",
                "credential", "verify", "verification", "account", "security", "ssn",
            })))
            hits.append(RuleHit(rule.rule_id, rule.category, rule.severity, rule.weight,
                                evidence or "credential language", rule.description))

        if features.payment_terms:
            rule = by_id["payment_request"]
            evidence = ", ".join(sorted(set(words).intersection({
                "gift", "card", "crypto", "bitcoin", "cash", "wire", "transfer",
                "payment", "pay", "fee", "refund", "reward", "prize", "bank",
                "zelle", "paypal", "deposit", "tax",
            })))
            hits.append(RuleHit(rule.rule_id, rule.category, rule.severity, rule.weight,
                                evidence or "payment language", rule.description))

        if features.threat_terms:
            rule = by_id["threat_coercion"]
            evidence = ", ".join(sorted(set(words).intersection({
                "suspend", "suspended", "locked", "blocked", "arrest", "legal",
                "police", "terminate", "closure", "deleted", "penalty", "fine",
            })))
            hits.append(RuleHit(rule.rule_id, rule.category, rule.severity, rule.weight,
                                evidence or "coercive language", rule.description))

        if features.url_count or SHORT_LINK_PATTERN.search(text):
            rule = by_id["shortener_or_external_link"]
            evidence = "shortened link" if SHORT_LINK_PATTERN.search(text) else "external URL"
            hits.append(RuleHit(rule.rule_id, rule.category, rule.severity, rule.weight,
                                evidence, rule.description))

        if features.phone_count or PHONE_CHANNEL_PATTERN.search(text):
            rule = by_id["unusual_phone_request"]
            evidence = "phone number or alternate channel requested" if (
                features.phone_count or PHONE_CHANNEL_PATTERN.search(text)
            ) else "channel switch"
            hits.append(RuleHit(rule.rule_id, rule.category, rule.severity, rule.weight,
                                evidence, rule.description))

        if IMPersonation := IMPersonationPattern(text):
            rule = by_id["impersonation_claim"]
            hits.append(RuleHit(rule.rule_id, rule.category, rule.severity, rule.weight,
                                IMPersonation, rule.description))

        if match := PRIZE_PATTERN.search(text):
            rule = by_id["prize_reward"]
            hits.append(RuleHit(rule.rule_id, rule.category, rule.severity, rule.weight,
                                match.group(0).lower(), rule.description))

        letters = sum(char.isalpha() for char in text)
        capitals = sum(char.isupper() for char in text)
        if letters >= 8 and capitals / letters >= 0.7:
            rule = by_id["all_caps"]
            hits.append(RuleHit(rule.rule_id, rule.category, rule.severity, rule.weight,
                                "{:.0%} uppercase".format(capitals / letters), rule.description))

        punctuation = sum(not char.isalnum() and not char.isspace() for char in text)
        if len(text) >= 12 and punctuation / max(len(text), 1) >= 0.12:
            rule = by_id["excessive_punctuation"]
            hits.append(RuleHit(rule.rule_id, rule.category, rule.severity, rule.weight,
                                "{} punctuation characters".format(punctuation), rule.description))
        return hits

    def weighted_score(self, hits: Sequence[RuleHit], cap: float = 1.0) -> float:
        return min(cap, sum(hit.weight for hit in hits))


def IMPersonationPattern(text: str) -> str:
    match = IMPERSONATION_PATTERN.search(text)
    return match.group(0).lower() if match else ""

"""JSON-backed configuration and environment overrides."""

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping



DEFAULT_CONFIG_NAME = "detector.json"


def default_config() -> DetectorConfig:
    return DetectorConfig()


def load_config(path: str | os.PathLike[str] | None = None) -> DetectorConfig:
    """Load a DetectorConfig from JSON, with optional environment overrides."""
    config = default_config()
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError("configuration file not found: {}".format(config_path))
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a JSON object")
        unknown = set(raw) - set(config.to_dict())
        if unknown:
            raise ValueError("unknown configuration keys: {}".format(", ".join(sorted(unknown))))
        config = DetectorConfig(**raw)

    overrides: Dict[str, Any] = {}
    for key in config.to_dict():
        env_name = "MESSAGE_FRAUD_" + key.upper()
        if env_name not in os.environ:
            continue
        env_value = os.environ[env_name]
        current = getattr(config, key)
        try:
            if isinstance(current, bool):
                overrides[key] = env_value.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int):
                overrides[key] = int(env_value)
            elif isinstance(current, float):
                overrides[key] = float(env_value)
            elif isinstance(current, str):
                overrides[key] = env_value
        except ValueError as exc:
            raise ValueError("invalid environment override {}".format(env_name)) from exc
    if overrides:
        config = replace(config, **overrides)
    config.validate()
    return config


def save_config(config: DetectorConfig, path: str | os.PathLike[str]) -> None:
    """Persist a validated configuration as formatted JSON."""
    config.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)


def merge_overrides(config: DetectorConfig, overrides: Mapping[str, Any]) -> DetectorConfig:
    allowed = set(config.to_dict())
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError("unknown configuration keys: {}".format(", ".join(sorted(unknown))))
    merged = replace(config, **dict(overrides))
    merged.validate()
    return merged

"""Top-level TRE-style retrieval detector with behavioral rules."""

from typing import Iterable, List, Sequence



class MessageFraudDetector:
    """Combines nearest-neighbor retrieval and deterministic behavior rules."""

    def __init__(
        self,
        index: RetrievalIndex,
        config: DetectorConfig | None = None,
    ) -> None:
        if config is None:
            config = load_config()
        config.validate()
        self.index = index
        self.config = config
        self.guard = MutationGuardLite()

    @classmethod
    def train(
        cls,
        examples: Sequence[Example],
        config: DetectorConfig | None = None,
    ) -> "MessageFraudDetector":
        if config is None:
            config = load_config()
        config.validate()
        if not examples:
            raise ValueError("training requires at least one example")
        normalized_labels = {example.label.lower().strip() for example in examples}
        allowed = {"fraud", "legitimate", "suspicious", "unknown"}
        unknown = normalized_labels - allowed
        if unknown:
            raise ValueError("unsupported labels: {}".format(", ".join(sorted(unknown))))
        normalized = [
            Example(
                text=example.text.strip(),
                label=example.label.lower().strip(),
                source=example.source,
                message_id=example.message_id,
                metadata=dict(example.metadata),
            )
            for example in examples
            if example.text.strip()
        ]
        if not normalized:
            raise ValueError("training corpus contains no non-empty examples")
        vectorizer = TfIdfVectorizer(
            max_features=config.max_features,
            ngram_min=config.ngram_min,
            ngram_max=config.ngram_max,
            sublinear_tf=config.sublinear_tf,
            strip_accents=config.strip_accents,
        )
        index = RetrievalIndex(vectorizer).fit(normalized)
        return cls(index, config)

    def score(self, text: str, message_id: str = "") -> ScoreResult:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")
        if len(text) > 100000:
            raise ValueError("text exceeds maximum length of 100000 characters")

        nearest_label, retrieval_score, hits = self.index.nearest_label(
            text,
            self.config.top_k,
        )
        if retrieval_score < self.config.retrieval_min_score:
            retrieval_score = 0.0
        if nearest_label not in ("fraud", "suspicious"):
            retrieval_score = 0.0

        rule_hits = self.guard.evaluate(text, message_id)
        rule_score = self.guard.weighted_score(rule_hits, self.config.rule_cap)
        combined = (
            self.config.retrieval_weight * retrieval_score
            + self.config.rule_weight * rule_score
        )
        if nearest_label == "fraud" and retrieval_score > 0:
            combined = max(combined, retrieval_score * 0.85)
        combined = max(0.0, min(1.0, combined))

        if combined >= self.config.fraud_threshold:
            label = "fraud"
            risk_level = "high"
        elif combined >= self.config.suspicious_threshold:
            label = "suspicious"
            risk_level = "medium"
        else:
            label = "legitimate"
            risk_level = "low"

        confidence = abs(combined - 0.5) * 2.0
        confidence = max(0.0, min(1.0, confidence))
        reasons = self.build_reasons(retrieval_score, nearest_label, rule_hits)
        return ScoreResult(
            message_id=message_id,
            text=text,
            label=label,
            score=round(combined, 6),
            confidence=round(confidence, 6),
            risk_level=risk_level,
            rule_hits=rule_hits,
            retrieval_hits=hits,
            reasons=reasons,
            metadata={
                "retrieval_label": nearest_label,
                "retrieval_score": round(retrieval_score, 6),
                "rule_score": round(rule_score, 6),
                "model_version": self.config.model_version,
                "threshold": self.config.fraud_threshold,
            },
        )

    def score_many(
        self,
        texts: Iterable[tuple[str, str]],
    ) -> List[ScoreResult]:
        results = []
        for item in texts:
            if isinstance(item, str):
                message_id = ""
                text = item
            elif len(item) == 2:
                message_id, text = item
            else:
                raise ValueError("score_many items must be text or (message_id, text)")
            results.append(self.score(text, message_id))
        return results

    def build_reasons(
        self,
        retrieval_score: float,
        retrieval_label: str,
        rule_hits: Sequence,
    ) -> List[str]:
        reasons: List[str] = []
        if retrieval_score > 0:
            reasons.append(
                "Nearest retrieval label is '{}' at similarity {:.3f}".format(
                    retrieval_label,
                    retrieval_score,
                )
            )
        for hit in rule_hits:
            reasons.append(
                "{}: {} (evidence: {})".format(
                    hit.rule_id,
                    hit.description,
                    hit.evidence,
                )
            )
        if not reasons:
            reasons.append("No strong fraud retrieval or MutationGuard-lite signals")
        return reasons

    def explain(self, text: str, message_id: str = "") -> dict:
        return self.score(text, message_id).to_dict()

    def to_dict(self) -> dict:
        return {
            "format": "message-fraud-detector",
            "format_version": 1,
            "config": self.config.to_dict(),
            "index": self.index.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "MessageFraudDetector":
        if not isinstance(value, dict):
            raise ValueError("model state must be a JSON object")
        if value.get("format") != "message-fraud-detector":
            raise ValueError("unsupported model format")
        if value.get("format_version") != 1:
            raise ValueError("unsupported model format version")
        config = DetectorConfig(**value["config"])
        index = RetrievalIndex.from_dict(value["index"])
        return cls(index, config)

"""Threshold presets and threshold validation utilities."""

from typing import Dict, Iterable, Mapping



PRESETS: Dict[str, Dict[str, float]] = {
    "balanced": {
        "fraud_threshold": 0.48,
        "suspicious_threshold": 0.28,
        "retrieval_weight": 0.55,
        "rule_weight": 0.45,
    },
    "high_precision": {
        "fraud_threshold": 0.65,
        "suspicious_threshold": 0.38,
        "retrieval_weight": 0.60,
        "rule_weight": 0.40,
    },
    "high_recall": {
        "fraud_threshold": 0.38,
        "suspicious_threshold": 0.20,
        "retrieval_weight": 0.50,
        "rule_weight": 0.50,
    },
}


def list_presets() -> dict:
    return {name: dict(values) for name, values in sorted(PRESETS.items())}


def preset(name: str) -> Dict[str, float]:
    normalized = name.lower().strip()
    if normalized not in PRESETS:
        raise ValueError("unknown threshold preset: {}".format(name))
    return dict(PRESETS[normalized])


def apply_preset(config: DetectorConfig, name: str) -> DetectorConfig:
    values = preset(name)
    from dataclasses import replace

    merged = replace(config, **values)
    merged.validate()
    return merged


def validate_threshold_pair(fraud_threshold: float, suspicious_threshold: float) -> None:
    if not 0.0 <= suspicious_threshold <= 1.0:
        raise ValueError("suspicious threshold must be between 0 and 1")
    if not 0.0 <= fraud_threshold <= 1.0:
        raise ValueError("fraud threshold must be between 0 and 1")
    if suspicious_threshold >= fraud_threshold:
        raise ValueError("suspicious threshold must be below fraud threshold")

"""Public constants and output field names."""


SUPPORTED_LABELS = (
    "fraud",
    "legitimate",
    "suspicious",
)

RISK_LEVELS = (
    "low",
    "medium",
    "high",
)

SUPPORTED_INPUT_FORMATS = (
    "csv",
    "jsonl",
    "json",
    "auto",
)

RESULT_FIELDS = (
    "message_id",
    "text",
    "label",
    "score",
    "confidence",
    "risk_level",
    "rule_hits",
    "retrieval_hits",
    "reasons",
)

MODEL_FORMAT = "message-fraud-detector"
MODEL_FORMAT_VERSION = 1
