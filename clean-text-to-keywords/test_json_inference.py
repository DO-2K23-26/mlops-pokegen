import unittest

from json_inference import fill_template_from_keywords, infer_profile_from_keywords


class JsonInferenceTests(unittest.TestCase):
    def test_profile_inference_basics(self) -> None:
        keywords = [
            "zapthorn",
            "electric",
            "wolf",
            "thunder",
            "claw",
            "speed",
            "storm",
            "agile",
            "forest",
            "recover",
            "energy",
        ]

        profile = infer_profile_from_keywords(keywords)

        self.assertEqual(profile["name"], "Zapthorn")
        self.assertEqual(profile["type"], "electric")
        self.assertIn("thunder", profile["attacks"])
        self.assertIn("claw", profile["attacks"])
        self.assertIn("recover", profile["abilities"])
        self.assertEqual(profile["habitat"], "forest")
        self.assertGreaterEqual(profile["speed"], 80)

    def test_fill_key_only_template(self) -> None:
        template = {
            "name": "",
            "type": "",
            "secondary_type": None,
            "attacks": [],
            "abilities": [],
            "habitat": "",
            "personality": [],
            "hp": None,
            "attack": None,
            "defense": None,
            "speed": None,
            "description": "",
            "keywords": [],
        }

        keywords = [
            "furret",
            "normal",
            "tail",
            "smash",
            "tunnel",
            "agile",
            "cheerful",
            "explore",
            "endurance",
        ]

        result = fill_template_from_keywords(template, keywords)

        self.assertEqual(result["name"], "Furret")
        self.assertEqual(result["type"], "normal")
        self.assertIn("smash", result["attacks"])
        self.assertIn("explore", result["abilities"])
        self.assertEqual(result["habitat"], "tunnel")
        self.assertIn("cheerful", result["personality"])
        self.assertIsInstance(result["description"], str)
        self.assertGreater(len(result["description"]), 20)

    def test_fill_tcg_style_template(self) -> None:
        template = {
            "category": "Pokemon",
            "name": "",
            "hp": "",
            "types": [""],
            "description": "",
            "stage": "",
            "attacks": [
                {"cost": [""], "name": "", "effect": ""},
                {"cost": [""], "name": "", "effect": "", "damage": 0},
            ],
            "weaknesses": [{"type": "", "value": ""}],
            "retreat": 0,
        }

        keywords = [
            "zapthorn",
            "electric",
            "thunder",
            "claw",
            "speed",
            "storm",
            "energy",
        ]

        result = fill_template_from_keywords(template, keywords)

        self.assertEqual(result["name"], "Zapthorn")
        self.assertEqual(result["types"], ["electric"])
        self.assertEqual(result["stage"], "Basic")
        self.assertTrue(result["hp"].isdigit())
        self.assertEqual(result["weaknesses"][0]["type"], "ground")
        self.assertEqual(result["weaknesses"][0]["value"], "x2")
        self.assertEqual(result["attacks"][0]["name"], "Thunder")
        self.assertEqual(result["attacks"][1]["name"], "Claw")
        self.assertEqual(result["attacks"][0]["cost"], ["Electric"])
        self.assertGreaterEqual(result["retreat"], 0)

    def test_name_fallback_to_unknown_for_generic_tokens(self) -> None:
        keywords = [
            "black",
            "fur",
            "giant",
            "electric",
            "claw",
            "speed",
            "storm",
        ]

        profile = infer_profile_from_keywords(keywords)
        self.assertEqual(profile["name"], "Unknown")

    def test_preserves_existing_values(self) -> None:
        template = {
            "name": "CustomName",
            "type": "electric",
            "attacks": [],
            "description": "Already set",
        }
        keywords = ["furret", "normal", "attack"]

        result = fill_template_from_keywords(template, keywords)

        self.assertEqual(result["name"], "CustomName")
        self.assertEqual(result["type"], "electric")
        self.assertEqual(result["description"], "Already set")
        self.assertIn("attack", result["attacks"])


if __name__ == "__main__":
    unittest.main()
