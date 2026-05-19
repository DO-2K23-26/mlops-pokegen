import argparse
import json
from typing import Sequence

from keyword_extractor import KeywordExtractor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract normalized keywords from cleaned text.",
    )
    parser.add_argument(
        "text",
        nargs="+",
        help="Input text to process. Pass as one quoted string or multiple words.",
    )
    parser.add_argument(
        "--model",
        default="en_core_web_sm",
        help="spaCy model name (default: en_core_web_sm).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    text = " ".join(args.text)
    extractor = KeywordExtractor.from_default_model(model_name=args.model)
    keywords = extractor.extract(text)
    print(json.dumps(keywords))


if __name__ == "__main__":
    main()
