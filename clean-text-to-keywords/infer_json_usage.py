import argparse
import json
import os
import re
from typing import Sequence

from keyword_extractor import KeywordExtractor
from json_inference import fill_template_from_keywords


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract keywords and infer values into a JSON template.",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Input description text.",
    )
    parser.add_argument(
        "--template",
        default="",
        help="Path to JSON template file with keys only. If omitted, full inferred JSON is returned.",
    )
    parser.add_argument(
        "--model",
        default="en_core_web_sm",
        help="spaCy model name (default: en_core_web_sm).",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Provide keywords directly instead of raw text.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only inferred JSON (skip keyword list output).",
    )
    return parser


def _load_template(path: str):
    if not path:
        return {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"Template file not found: {path}")

    with open(path, "r", encoding="utf-8") as file_handle:
        raw = file_handle.read().strip()
        if not raw:
            return {}
        return json.loads(raw)


def _parse_keywords_fragment(raw: str):
    if not raw.strip():
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip().lower() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass

    tokens = re.findall(r"[a-zA-Z0-9_-]+", raw.lower())
    return [token for token in tokens if token]


def _extract_keywords(args):
    if args.keywords:
        return [word.strip().lower() for word in args.keywords if word.strip()]

    if args.template and not os.path.exists(args.template) and args.template.lstrip().startswith("["):
        raw = " ".join([args.template] + args.text)
        return _parse_keywords_fragment(raw)

    if not args.text:
        raise ValueError("Provide input text or use --keywords.")

    text = " ".join(args.text)
    extractor = KeywordExtractor.from_default_model(model_name=args.model)
    return extractor.extract(text)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    keywords = _extract_keywords(args)

    template_path = args.template
    if args.template and not os.path.exists(args.template) and args.template.lstrip().startswith("["):
        template_path = ""

    template = _load_template(template_path)
    inferred_json = fill_template_from_keywords(template, keywords)

    if args.json_only:
        print(json.dumps(inferred_json, indent=2))
        return

    print(json.dumps(keywords))
    print(json.dumps(inferred_json, indent=2))


if __name__ == "__main__":
    main()
