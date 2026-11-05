"""Internal numerical support selected by the local experiments."""
import json
from pathlib import Path
import torch

def town_slug(town: str) -> str:
    text = town.strip().lower().replace('/', '_').replace(' ', '_')
    return ''.join((char if char.isalnum() or char == '_' else '_' for char in text))

def save_town_states(directory: Path, layers: int, columns: list[str], numeric: dict, target_center: float, target_spread: float, states: dict[str, dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    town_dir = directory / 'towns'
    town_dir.mkdir(parents=True, exist_ok=True)
    names = {}
    for town, state in states.items():
        slug = town_slug(town)
        torch.save(state, town_dir / f'{slug}.pt')
        names[slug] = town
    payload = {'columns': columns, 'layers': layers, 'numeric': {column: {key: float(value) for key, value in spec.items()} for column, spec in numeric.items()}, 'target_center': target_center, 'target_spread': target_spread, 'towns': names}
    (directory / 'config.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

def load_config(directory: Path) -> dict:
    return json.loads((directory / 'config.json').read_text(encoding='utf-8'))

def load_state(directory: Path, slug: str) -> dict:
    return torch.load(directory / 'towns' / f'{slug}.pt', map_location='cpu', weights_only=True)
