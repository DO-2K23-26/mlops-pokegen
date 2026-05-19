"""Infer Pokemon-like JSON values from extracted keywords."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Sequence

POKEMON_TYPES = {
    "normal",
    "fire",
    "water",
    "grass",
    "electric",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
}

HABITAT_KEYWORDS = {
    "forest",
    "field",
    "cave",
    "mountain",
    "river",
    "ocean",
    "sea",
    "tunnel",
    "nest",
    "sky",
    "desert",
    "swamp",
    "volcano",
}

PERSONALITY_KEYWORDS = {
    "calm",
    "gentle",
    "agile",
    "playful",
    "cheerful",
    "energetic",
    "curious",
    "fierce",
    "brave",
    "loyal",
    "timid",
    "bold",
}

MOVE_KEYWORDS = {
    "attack",
    "smash",
    "strike",
    "kick",
    "punch",
    "shock",
    "thunder",
    "bolt",
    "blast",
    "explosion",
    "freeze",
    "bite",
    "claw",
    "tail",
    "fight",
}

ABILITY_KEYWORDS = {
    "recover",
    "endurance",
    "explore",
    "hide",
    "wander",
    "bond",
    "speed",
    "power",
    "energy",
    "flexible",
}

STAT_HINTS = {
    "hp": {"endurance", "recover", "energy", "stamina", "healthy", "vital"},
    "attack": {"attack", "smash", "strike", "punch", "kick", "claw", "fight", "power"},
    "defense": {"armor", "shield", "tough", "hard", "resist", "solid"},
    "speed": {"speed", "swift", "agile", "quick", "fast", "dash"},
}

KEY_ALIASES = {
    "name": {"name", "pokemon_name"},
    "type": {"type", "primary_type", "pokemon_type"},
    "secondary_type": {"secondary_type", "type2", "secondary"},
    "attacks": {"attacks", "moves", "skills", "offense"},
    "abilities": {"abilities", "traits", "passives", "special_abilities"},
    "habitat": {"habitat", "environment", "region"},
    "personality": {"personality", "temperament", "nature"},
    "description": {"description", "flavor_text", "summary", "lore"},
    "keywords": {"keywords", "tags"},
    "hp": {"hp", "health", "health_points"},
    "attack": {"attack", "atk"},
    "defense": {"defense", "def"},
    "speed": {"speed", "spd"},
}

GENERIC_NAME_BLACKLIST = {
    "black",
    "white",
    "yellow",
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "pink",
    "gray",
    "grey",
    "brown",
    "fur",
    "body",
    "tail",
    "claw",
    "storm",
    "cloud",
    "enemy",
    "super",
    "scary",
    "giant",
    "speed",
}

TYPE_WEAKNESS = {
    "normal": "fighting",
    "fire": "water",
    "water": "electric",
    "grass": "fire",
    "electric": "ground",
    "ice": "fire",
    "fighting": "psychic",
    "poison": "ground",
    "ground": "water",
    "flying": "electric",
    "psychic": "dark",
    "bug": "fire",
    "rock": "water",
    "ghost": "dark",
    "dragon": "fairy",
    "dark": "fighting",
    "steel": "fire",
    "fairy": "steel",
}


def _title_case(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split())


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _canonical_key(key: str) -> str:
    lowered = key.lower().strip()
    for canonical, aliases in KEY_ALIASES.items():
        if lowered in aliases:
            return canonical
    return lowered


def _pick_name(keywords: Sequence[str]) -> str:
    for keyword in keywords:
        if keyword in POKEMON_TYPES:
            continue
        if keyword in HABITAT_KEYWORDS:
            continue
        if keyword in MOVE_KEYWORDS:
            continue
        if keyword in ABILITY_KEYWORDS:
            continue
        if keyword in PERSONALITY_KEYWORDS:
            continue
        if keyword in GENERIC_NAME_BLACKLIST:
            continue
        if len(keyword) < 4:
            continue
        return _title_case(keyword)
    return "Unknown"


def _pick_types(keywords: Sequence[str]) -> List[str]:
    types: List[str] = []
    for keyword in keywords:
        if keyword in POKEMON_TYPES and keyword not in types:
            types.append(keyword)
        if len(types) >= 2:
            break
    if not types:
        types.append("normal")
    return types


def _pick_habitat(keywords: Sequence[str]) -> str:
    habitats = [word for word in keywords if word in HABITAT_KEYWORDS]
    if not habitats:
        return "unknown"
    return habitats[0]


def _pick_personality(keywords: Sequence[str]) -> List[str]:
    result: List[str] = []
    for keyword in keywords:
        if keyword in PERSONALITY_KEYWORDS and keyword not in result:
            result.append(keyword)
    return result[:3]


def _pick_attacks(keywords: Sequence[str]) -> List[str]:
    attacks: List[str] = []
    for keyword in keywords:
        if keyword in MOVE_KEYWORDS and keyword not in attacks:
            attacks.append(keyword)
    return attacks[:4]


def _pick_abilities(keywords: Sequence[str]) -> List[str]:
    abilities: List[str] = []
    for keyword in keywords:
        if keyword in ABILITY_KEYWORDS and keyword not in abilities:
            abilities.append(keyword)
    return abilities[:4]


def _score_stat(base: int, keywords: Sequence[str], hints: Iterable[str]) -> int:
    hint_set = set(hints)
    matches = sum(1 for keyword in keywords if keyword in hint_set)
    # Each match adds 10 points; keep stats in [40, 160].
    return max(40, min(160, base + (matches * 10)))


def _build_description(name: str, primary_type: str, attacks: Sequence[str], abilities: Sequence[str], habitat: str) -> str:
    attack_text = ", ".join(attacks) if attacks else "basic combat"
    ability_text = ", ".join(abilities) if abilities else "balanced adaptation"
    return (
        f"{name} is a {primary_type}-type Pokemon often found in {habitat}. "
        f"It commonly uses {attack_text} and shows abilities like {ability_text}."
    )


def _retreat_cost_from_speed(speed: int) -> int:
    if speed >= 120:
        return 0
    if speed >= 90:
        return 1
    if speed >= 70:
        return 2
    return 3


def _attack_damage_from_attack_stat(attack_stat: int, index: int) -> int:
    # Keep card damage in simple 10-step increments.
    base = 30 + max(0, attack_stat - 70) // 2
    adjusted = base + (index * 10)
    return max(10, min(160, (adjusted // 10) * 10))


def _energy_name_for_type(pokemon_type: str) -> str:
    if pokemon_type == "normal":
        return "Colorless"
    return _title_case(pokemon_type)


def _fill_tcg_like_template(output: Dict[str, Any], inferred: Mapping[str, Any]) -> None:
    if "name" in output and _is_empty_value(output.get("name")):
        output["name"] = inferred["name"]

    if "description" in output and _is_empty_value(output.get("description")):
        output["description"] = inferred["description"]

    if "hp" in output and _is_empty_value(output.get("hp")):
        hp_value = inferred["hp"]
        output["hp"] = str(hp_value) if isinstance(output.get("hp"), str) else hp_value

    if "types" in output and isinstance(output.get("types"), list):
        types_value = output["types"]
        if len(types_value) == 0 or all(_is_empty_value(item) for item in types_value):
            inferred_types = [inferred["type"]]
            if inferred.get("secondary_type"):
                inferred_types.append(inferred["secondary_type"])
            output["types"] = inferred_types

    if "stage" in output and _is_empty_value(output.get("stage")):
        output["stage"] = "Basic"

    if "retreat" in output and (output.get("retreat") in (None, 0, "")):
        output["retreat"] = _retreat_cost_from_speed(int(inferred["speed"]))

    if "weaknesses" in output and isinstance(output.get("weaknesses"), list):
        weaknesses = output["weaknesses"]
        if weaknesses:
            weakness_type = TYPE_WEAKNESS.get(inferred["type"], "fighting")
            first = weaknesses[0]
            if isinstance(first, dict):
                if _is_empty_value(first.get("type")):
                    first["type"] = weakness_type
                if _is_empty_value(first.get("value")):
                    first["value"] = "x2"

    if "attacks" in output and isinstance(output.get("attacks"), list):
        attack_entries = output["attacks"]
        inferred_attacks = inferred["attacks"]
        inferred_type = inferred["type"]
        for idx, attack_entry in enumerate(attack_entries):
            if not isinstance(attack_entry, dict):
                continue

            attack_name = inferred_attacks[idx] if idx < len(inferred_attacks) else "tackle"
            attack_title = _title_case(attack_name)
            if _is_empty_value(attack_entry.get("name")):
                attack_entry["name"] = attack_title
            if _is_empty_value(attack_entry.get("effect")):
                attack_entry["effect"] = f"Deals damage with {attack_name}."

            if "damage" in attack_entry and (attack_entry.get("damage") in (None, 0, "")):
                attack_entry["damage"] = _attack_damage_from_attack_stat(int(inferred["attack"]), idx)

            if "cost" in attack_entry and isinstance(attack_entry.get("cost"), list):
                current_cost = attack_entry["cost"]
                if len(current_cost) == 0 or all(_is_empty_value(item) for item in current_cost):
                    attack_entry["cost"] = [_energy_name_for_type(inferred_type)]


def infer_profile_from_keywords(keywords: Sequence[str]) -> Dict[str, Any]:
    cleaned = [k.strip().lower() for k in keywords if k and k.strip()]

    name = _pick_name(cleaned)
    types = _pick_types(cleaned)
    attacks = _pick_attacks(cleaned)
    abilities = _pick_abilities(cleaned)
    habitat = _pick_habitat(cleaned)
    personality = _pick_personality(cleaned)

    hp = _score_stat(70, cleaned, STAT_HINTS["hp"])
    attack = _score_stat(70, cleaned, STAT_HINTS["attack"])
    defense = _score_stat(70, cleaned, STAT_HINTS["defense"])
    speed = _score_stat(70, cleaned, STAT_HINTS["speed"])

    return {
        "name": name,
        "type": types[0],
        "secondary_type": types[1] if len(types) > 1 else None,
        "attacks": attacks,
        "abilities": abilities,
        "habitat": habitat,
        "personality": personality,
        "hp": hp,
        "attack": attack,
        "defense": defense,
        "speed": speed,
        "keywords": cleaned,
        "description": _build_description(name, types[0], attacks, abilities, habitat),
    }


def fill_template_from_keywords(template: Mapping[str, Any], keywords: Sequence[str]) -> Dict[str, Any]:
    """Fill a key-only template by inferring values from keywords.

    Existing non-empty values in template are preserved.
    """
    inferred = infer_profile_from_keywords(keywords)
    output: Dict[str, Any] = deepcopy(dict(template))

    if not output:
        return inferred

    _fill_tcg_like_template(output, inferred)

    for key, current_value in output.items():
        canonical = _canonical_key(key)
        if canonical not in inferred:
            continue
        if _is_empty_value(current_value):
            output[key] = inferred[canonical]

    return output
