"""Rule-based keyword extraction and normalization for Pokemon card generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# Canonical concept -> synonym list
from typing import Dict, List

DEFAULT_NORMALIZATION_MAP: Dict[str, List[str]] = {
    "normal": ["basic", "common", "regular", "plain"],
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
}

DEFAULT_ALLOWED_POS: Tuple[str, ...] = ("NOUN", "ADJ", "VERB")


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


def _deduplicate_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    output: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


@dataclass
class KeywordExtractor:
    """Deterministic spaCy + rule-based keyword extraction pipeline."""

    nlp: Any
    normalization_map: Mapping[str, Iterable[str]] = field(default_factory=lambda: DEFAULT_NORMALIZATION_MAP)
    allowed_pos: Sequence[str] = field(default_factory=lambda: DEFAULT_ALLOWED_POS)

    def __post_init__(self) -> None:
        self._normalization_lookup = _invert_normalization_map(self.normalization_map)
        self._allowed_pos_set = set(self.allowed_pos)

    @classmethod
    def from_default_model(
        cls,
        model_name: str = "en_core_web_sm",
        normalization_map: Optional[Mapping[str, Iterable[str]]] = None,
        allowed_pos: Sequence[str] = DEFAULT_ALLOWED_POS,
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
        )

    def extract(self, text: str) -> List[str]:
        """Extract and normalize keywords from already-cleaned text."""
        if not text or not text.strip():
            return []

        doc = self.nlp(text)

        # Step 1: POS filtering + base normalization to lowercase lemmas/tokens.
        raw_keywords: List[str] = []
        for token in doc:
            if token.is_stop or token.is_punct or token.pos_ not in self._allowed_pos_set:
                continue

            # Use lemma where possible to collapse inflections.
            base = token.lemma_.lower().strip() if token.lemma_ and token.lemma_ != "-PRON-" else token.text.lower().strip()
            if base:
                raw_keywords.append(base)

        # Step 2: Deduplicate before domain normalization (as requested in README).
        deduplicated = _deduplicate_preserve_order(raw_keywords)

        # Step 3: Map variants/synonyms to canonical concepts.
        normalized = [self._normalize_keyword(keyword) for keyword in deduplicated]

        # Step 4: Deduplicate again, since multiple words can map to one concept.
        return _deduplicate_preserve_order(normalized)

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
