# Pokemon Text-to-JSON Pipeline

This project converts free-form Pokemon description text into:

1. A normalized keyword list
2. A populated Pokemon JSON object (from a blank/key-only template)

The pipeline is deterministic and rule-based.

## Architecture

### Stage 1: Keyword Extraction

File: `keyword_extractor.py`

Input: raw text description

Core logic:

- spaCy tokenization and POS tagging
- POS filtering (`NOUN`, `ADJ`, `VERB`)
- stopword and punctuation removal
- lemma-based normalization
- domain synonym normalization (example: `flames -> fire`)
- optional YAKE relevance scoring
- conservative retention policy so detail is not over-pruned

Output: ordered list of normalized keywords

### Stage 2: JSON Inference

File: `json_inference.py`

Input: keyword list + optional JSON template

Core logic:

- infer primary/secondary type
- infer name candidate
- infer attacks, abilities, habitat, personality
- infer basic stats (`hp`, `attack`, `defense`, `speed`)
- fill nested TCG-like template fields (`types`, `attacks`, `weaknesses`, `stage`, `retreat`, etc.)
- preserve already non-empty values in the provided template

Output: inferred JSON profile

### Stage 3: Orchestration CLI

File: `infer_json_usage.py`

This is the main entrypoint for end-to-end usage.

Default behavior:

1. prints extracted keyword list
2. prints inferred JSON

## Project Structure

- `keyword_extractor.py`: keyword extraction engine
- `json_inference.py`: keyword-to-JSON inference logic
- `infer_json_usage.py`: end-to-end CLI
- `example_usage.py`: keyword extraction only CLI
- `json_template_example.json`: sample blank/key-only template
- `test_keyword_extractor.py`: extraction tests
- `test_json_inference.py`: inference tests
- `requirements.txt`: Python dependencies

## Requirements

- Python 3.13 or lower is recommended for spaCy compatibility
- pip

Dependencies in `requirements.txt`:

- `spacy>=3.7.0`
- `yake>=0.4.2`

## Setup

1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Install spaCy English model

```bash
python -m spacy download en_core_web_sm
```

## How To Run

### A) Extract keywords only

```bash
python example_usage.py "furret long slender agile creature with soft fur"
```

Output: JSON list of keywords.

### B) End-to-end: text -> keywords -> JSON

```bash
python infer_json_usage.py --template json_template_example.json "furret long slender agile creature with soft fur"
```

Output order:

1. keyword list
2. inferred JSON

### C) End-to-end but JSON only

```bash
python infer_json_usage.py --json-only --template json_template_example.json "furret long slender agile creature with soft fur"
```

### D) Start from keywords directly

```bash
python infer_json_usage.py --template json_template_example.json --keywords furret normal tail smash tunnel agile cheerful explore endurance
```

Tip: If you pass `--keywords`, text extraction is skipped.

## Template Behavior

If `--template` is omitted, inference returns a full inferred profile object.

If `--template` is provided:

- empty fields are populated from inferred values
- non-empty fields are preserved

Current sample template supports nested card-like data including:

- `types`
- `attacks` with `cost`, `name`, `effect`, `damage`
- `weaknesses` with `type`, `value`
- `stage`, `retreat`, `legal`

## Tests

Run all tests:

```bash
python -m unittest -q
```

## Troubleshooting

### 1) spaCy model not found

Error mentions `en_core_web_sm` not installed.

Fix:

```bash
python -m spacy download en_core_web_sm
```

### 2) spaCy import/runtime problems on very new Python versions

Use Python 3.13 or lower and reinstall requirements.

### 3) `--template` path errors

Ensure `--template` points to a valid file path, for example:

```bash
--template json_template_example.json
```

If your input is already a keyword list, use `--keywords` instead of putting the list in `--template`.

## Design Notes

- deterministic and explainable (no LLM calls)
- domain mappings are easy to extend in `keyword_extractor.py` and `json_inference.py`
- scoring and template fill rules are intentionally simple and stable for game-content generation
