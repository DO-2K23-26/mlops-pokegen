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
            "power": "NOUN",
            "electric": "ADJ",
            "mouse": "NOUN",
            "using": "VERB",
            "thunder": "NOUN",
            "shock": "NOUN",
            "a": "DET",
            "very": "ADV",
            "strong": "ADJ",
            "and": "CCONJ",
            "dangerous": "ADJ",
        }

        stopwords = {"a", "very", "and", "with"}
        cls.nlp = FakeNLP(tag_map=tag_map, stopwords=stopwords)
        cls.extractor = KeywordExtractor(nlp=cls.nlp)

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


if __name__ == "__main__":
    unittest.main()
