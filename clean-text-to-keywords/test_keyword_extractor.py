import unittest

from keyword_extractor import KeywordExtractor


class FakeToken:
    def __init__(self, text: str, pos: str, lemma: str, is_stop: bool) -> None:
        self.text = text
        self.pos_ = pos
        self.lemma_ = lemma
        self.is_stop = is_stop
        self.is_punct = not any(ch.isalnum() for ch in text)


class FakeNLP:
    def __init__(self, tag_map, stopwords) -> None:
        self.tag_map = tag_map
        self.stopwords = stopwords

    def __call__(self, text: str):
        tokens = []
        for raw in text.split():
            token_text = raw.strip()
            lowered = token_text.lower()
            tokens.append(
                FakeToken(
                    text=token_text,
                    pos=self.tag_map.get(lowered, "NOUN"),
                    lemma=lowered,
                    is_stop=lowered in self.stopwords,
                )
            )
        return tokens


class TestableKeywordExtractor(KeywordExtractor):
    def __init__(self, *args, yake_scores=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_yake_scores = yake_scores or {}

    def _extract_yake_scores(self, text: str):
        return self._test_yake_scores


class KeywordExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tag_map = {
            "fiery": "ADJ",
            "dragon": "NOUN",
            "attack": "VERB",
            "explosive": "ADJ",
            "flames": "NOUN",
            "burning": "ADJ",
            "creature": "NOUN",
            "with": "ADP",
            "blaze": "NOUN",
            "and": "CCONJ",
            "dangerous": "ADJ",
            "electric": "ADJ",
            "mouse": "NOUN",
            "using": "VERB",
            "thunder": "NOUN",
            "shock": "NOUN",
            "strong": "ADJ",
            "furret": "NOUN",
            "long": "ADJ",
            "slender": "ADJ",
            "soft": "ADJ",
            "fur": "NOUN",
            "flexible": "ADJ",
            "body": "NOUN",
            "move": "VERB",
            "gracefully": "ADJ",
            "narrow": "ADJ",
            "tunnel": "NOUN",
            "tail": "NOUN",
            "smash": "VERB",
            "opponent": "NOUN",
            "battle": "NOUN",
            "cheerful": "ADJ",
            "endurance": "NOUN",
        }

        stopwords = {
            "a",
            "very",
            "and",
            "with",
            "the",
            "it",
            "to",
            "its",
            "that",
            "through",
            "in",
        }
        cls.nlp = FakeNLP(tag_map=tag_map, stopwords=stopwords)
        cls.extractor = KeywordExtractor(nlp=cls.nlp, use_yake=False)

    def test_readme_main_example(self) -> None:
        text = "fiery dragon attack explosive flames"
        result = self.extractor.extract(text)
        self.assertEqual(result, ["fire", "dragon", "attack", "explosion"])

    def test_synonym_normalization(self) -> None:
        text = "burning creature with blaze power"
        result = self.extractor.extract(text)
        self.assertEqual(result, ["fire", "creature", "power"])

    def test_mixed_types(self) -> None:
        text = "electric mouse using thunder shock"
        result = self.extractor.extract(text)
        self.assertEqual(result, ["electric", "mouse", "using"])

    def test_noise_input(self) -> None:
        text = "a very very strong and dangerous creature"
        result = self.extractor.extract(text)
        self.assertEqual(result, ["strong", "dangerous", "creature"])

    def test_yake_keeps_detailed_information(self) -> None:
        text = (
            "furret long slender creature soft fur flexible body move gracefully narrow tunnel "
            "tail smash opponent battle cheerful endurance"
        )

        yake_scores = {
            "furret": 0.00,
            "creature": 0.05,
            "tail": 0.08,
            "battle": 0.10,
            "smash": 0.12,
            "tunnel": 0.14,
            "endurance": 0.18,
            "body": 0.20,
            "cheerful": 0.22,
            "slender": 0.26,
            "flexible": 0.28,
            "gracefully": 0.34,
            "narrow": 0.40,
            "long": 0.42,
            "soft": 0.44,
            "fur": 0.45,
            "move": 0.48,
            "opponent": 0.52,
        }
        extractor = TestableKeywordExtractor(
            nlp=self.nlp,
            use_yake=True,
            keep_ratio=0.8,
            min_keywords=10,
            max_keywords=30,
            yake_scores=yake_scores,
        )

        result = extractor.extract(text)

        self.assertGreaterEqual(len(result), 10)
        self.assertIn("furret", result)
        self.assertIn("creature", result)
        self.assertIn("tail", result)
        self.assertIn("tunnel", result)


if __name__ == "__main__":
    unittest.main()
