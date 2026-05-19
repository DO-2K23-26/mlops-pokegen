"""Rule-based keyword extraction and normalization for Pokemon card generation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

DEFAULT_NORMALIZATION_MAP: Dict[str, List[str]] = {
    "normal": ["basic", "common", "regular", "plain", "normaltype"],
    "fire": ["flame", "flames", "burn", "burning", "blaze", "fiery", "heat", "inferno"],
    "water": ["wave", "ocean", "sea", "river", "aqua", "splash", "tidal"],
    "grass": ["plant", "leaf", "forest", "nature", "vine", "seed", "flora"],
    "flying": ["air", "wind", "sky", "wing", "wings", "flight", "soar"],
    "fighting": ["punch", "kick", "strike", "martial", "combat", "brawl"],
    "poison": ["toxic", "venom", "acid", "poisonous", "toxin"],
    "electric": ["lightning", "thunder", "shock", "volt", "spark", "electricity"],
    "ground": ["earth", "soil", "sand", "mud", "quake", "dust"],
    "rock": ["stone", "boulder", "crystal", "rocky", "pebble"],
    "psychic": ["mind", "mental", "telepathy", "psyonic", "brain", "illusion"],
    "ice": ["freeze", "frozen", "snow", "frost", "blizzard", "icy"],
    "bug": ["insect", "ant", "beetle", "spider", "crawler"],
    "ghost": ["spirit", "phantom", "haunt", "shadow", "specter"],
    "steel": ["metal", "iron", "armor", "blade", "alloy"],
    "dragon": ["drake", "wyrm", "serpent", "legendary"],
    "dark": ["shadow", "evil", "night", "doom", "darkness"],
    "fairy": ["magic", "magical", "sparkle", "light", "charm"],
    "explosion": ["explosive", "explode", "blast"],
}

DEFAULT_ALLOWED_POS: Tuple[str, ...] = ("NOUN", "ADJ", "VERB")
DEFAULT_IGNORED_KEYWORDS: Set[str] = {"preevolution", "pokmon"}
DEFAULT_POS_WEIGHTS: Dict[str, float] = {
    "NOUN": 3.0,
    "ADJ": 2.0,
    "VERB": 1.0,
}
DEFAULT_KEEP_RATIO = 0.8
DEFAULT_MIN_KEYWORDS = 12
DEFAULT_MAX_KEYWORDS = 30


def _invert_normalization_map(normalization_map: Mapping[str, Iterable[str]]) -> Dict[str, str]:
    """Build synonym -> canonical mapping for O(1) normalization lookup."""
    inverse: Dict[str, str] = {}
    for canonical, synonyms in normalization_map.items():
        canonical_normalized = canonical.strip().lower()
        inverse[canonical_normalized] = canonical_normalized
        for synonym in synonyms:
            synonym_normalized = synonym.strip().lower()
            if synonym_normalized:
                inverse[synonym_normalized] = canonical_normalized
    return inverse


def _tokenize_keyword_phrase(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


@dataclass
class KeywordExtractor:
    """Deterministic spaCy + YAKE + rule-based normalization pipeline."""

    nlp: Any
    normalization_map: Mapping[str, Iterable[str]] = field(default_factory=lambda: DEFAULT_NORMALIZATION_MAP)
    allowed_pos: Sequence[str] = field(default_factory=lambda: DEFAULT_ALLOWED_POS)
    ignored_keywords: Set[str] = field(default_factory=lambda: set(DEFAULT_IGNORED_KEYWORDS))
    pos_weights: Mapping[str, float] = field(default_factory=lambda: DEFAULT_POS_WEIGHTS)
    keep_ratio: float = DEFAULT_KEEP_RATIO
    min_keywords: int = DEFAULT_MIN_KEYWORDS
    max_keywords: int = DEFAULT_MAX_KEYWORDS
    use_yake: bool = True

    def __post_init__(self) -> None:
        self._normalization_lookup = _invert_normalization_map(self.normalization_map)
        self._allowed_pos_set = set(self.allowed_pos)
        self._ignored_keywords = {keyword.lower().strip() for keyword in self.ignored_keywords}
        self._pos_weight_lookup = {k.upper(): float(v) for k, v in self.pos_weights.items()}

    @classmethod
    def from_default_model(
        cls,
        model_name: str = "en_core_web_sm",
        normalization_map: Optional[Mapping[str, Iterable[str]]] = None,
        allowed_pos: Sequence[str] = DEFAULT_ALLOWED_POS,
        ignored_keywords: Optional[Set[str]] = None,
        pos_weights: Mapping[str, float] = DEFAULT_POS_WEIGHTS,
        keep_ratio: float = DEFAULT_KEEP_RATIO,
        min_keywords: int = DEFAULT_MIN_KEYWORDS,
        max_keywords: int = DEFAULT_MAX_KEYWORDS,
        use_yake: bool = True,
    ) -> "KeywordExtractor":
        """Initialize extractor with a spaCy English pipeline."""
        try:
            import spacy

            nlp = spacy.load(model_name)
        except OSError as exc:
            raise OSError(
                f"spaCy model '{model_name}' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "spaCy could not be loaded in this Python environment. "
                "Try Python 3.13 or lower, then install spaCy and en_core_web_sm."
            ) from exc

        return cls(
            nlp=nlp,
            normalization_map=normalization_map or DEFAULT_NORMALIZATION_MAP,
            allowed_pos=allowed_pos,
            ignored_keywords=ignored_keywords or set(DEFAULT_IGNORED_KEYWORDS),
            pos_weights=pos_weights,
            keep_ratio=keep_ratio,
            min_keywords=min_keywords,
            max_keywords=max_keywords,
            use_yake=use_yake,
        )

    def extract(self, text: str) -> List[str]:
        """Extract, normalize and rank keywords from already-cleaned text."""
        if not text or not text.strip():
            return []

        doc = self.nlp(text)

        # Step 1: POS filtering + lowercase lemma/token extraction.
        raw_keywords: List[Tuple[str, str]] = []
        for token in doc:
            if token.is_stop or token.is_punct or token.pos_ not in self._allowed_pos_set:
                continue

            base = token.lemma_.lower().strip() if token.lemma_ and token.lemma_ != "-PRON-" else token.text.lower().strip()
            if base and base not in self._ignored_keywords:
                raw_keywords.append((base, token.pos_))

        # Step 2: Deduplicate before domain normalization.
        deduplicated: List[Tuple[str, str]] = []
        seen_raw: Set[str] = set()
        for keyword, pos in raw_keywords:
            if keyword in seen_raw:
                continue
            seen_raw.add(keyword)
            deduplicated.append((keyword, pos))

        # Step 3: Normalize and deduplicate canonical forms.
        unique_entries: List[Tuple[str, str, str, int]] = []
        seen_normalized: Set[str] = set()
        for index, (original_keyword, pos) in enumerate(deduplicated):
            normalized_keyword = self._normalize_keyword(original_keyword)
            if normalized_keyword in seen_normalized:
                continue
            seen_normalized.add(normalized_keyword)
            unique_entries.append((original_keyword, normalized_keyword, pos, index))

        if not unique_entries:
            return []

        if not self.use_yake:
            return [normalized_keyword for _, normalized_keyword, _, _ in unique_entries]

        # Step 4: YAKE scoring + conservative selection to preserve detail.
        yake_scores = self._extract_yake_scores(text)
        if not yake_scores:
            return [normalized_keyword for _, normalized_keyword, _, _ in unique_entries]

        ranked: List[Tuple[float, int, str]] = []
        for original_keyword, normalized_keyword, pos, index in unique_entries:
            score_candidates: List[float] = []
            if original_keyword in yake_scores:
                score_candidates.append(yake_scores[original_keyword])
            if normalized_keyword in yake_scores:
                score_candidates.append(yake_scores[normalized_keyword])

            # Missing score is treated as moderately relevant to avoid over-pruning.
            yake_penalty = min(score_candidates) if score_candidates else 0.45
            pos_weight = self._pos_weight_lookup.get(pos.upper(), 1.0)
            combined_score = (1.0 - yake_penalty) * pos_weight
            ranked.append((combined_score, index, normalized_keyword))

        target_count = self._compute_target_count(len(ranked))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = ranked[:target_count]
        selected.sort(key=lambda item: item[1])

        return [keyword for _, _, keyword in selected]

    def _compute_target_count(self, total_keywords: int) -> int:
        if total_keywords <= 0:
            return 0

        target = max(self.min_keywords, math.ceil(total_keywords * self.keep_ratio))
        if self.max_keywords > 0:
            target = min(target, self.max_keywords)
        return min(target, total_keywords)

    def _extract_yake_scores(self, text: str) -> Dict[str, float]:
        try:
            import yake
        except Exception:
            return {}

        text_token_count = len(text.split())
        top_n = max(20, min(80, text_token_count * 2))

        try:
            extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=top_n)
            phrase_scores = extractor.extract_keywords(text)
        except Exception:
            return {}

        token_scores: Dict[str, float] = {}
        for phrase, score in phrase_scores:
            for token in _tokenize_keyword_phrase(phrase):
                existing = token_scores.get(token)
                if existing is None or score < existing:
                    token_scores[token] = score

        if not token_scores:
            return {}

        values = list(token_scores.values())
        min_score = min(values)
        max_score = max(values)

        if math.isclose(min_score, max_score):
            return {token: 0.5 for token in token_scores}

        # Normalize so 0.0=most important and 1.0=least important.
        return {
            token: (score - min_score) / (max_score - min_score)
            for token, score in token_scores.items()
        }

    def _normalize_keyword(self, keyword: str) -> str:
        keyword_lower = keyword.lower()
        return self._normalization_lookup.get(keyword_lower, keyword_lower)


def extract_keywords(
    text: str,
    extractor: Optional[KeywordExtractor] = None,
) -> List[str]:
    """Convenience API to extract keywords with default extractor config."""
    active_extractor = extractor or KeywordExtractor.from_default_model()
    return active_extractor.extract(text)
