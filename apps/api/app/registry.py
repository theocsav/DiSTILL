import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .settings import DATASETS_REGISTRY_PATH, PRESETS_DIR


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def list_datasets() -> List[Dict[str, Any]]:
    if not DATASETS_REGISTRY_PATH.exists():
        return []
    data = _load_json(DATASETS_REGISTRY_PATH)
    if isinstance(data, dict):
        data = data.get("datasets", [])
    if not isinstance(data, list):
        return []
    return data


def save_datasets(datasets: List[Dict[str, Any]]) -> None:
    if DATASETS_REGISTRY_PATH.exists():
        current = _load_json(DATASETS_REGISTRY_PATH)
        if isinstance(current, dict):
            current["datasets"] = datasets
            _write_json(DATASETS_REGISTRY_PATH, current)
            return
    _write_json(DATASETS_REGISTRY_PATH, datasets)


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    for dataset in list_datasets():
        if dataset.get("id") == dataset_id:
            return dataset
    return None


def upsert_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    dataset_id = dataset.get("id")
    if not dataset_id:
        raise ValueError("Dataset id is required.")
    datasets = list_datasets()
    replaced = False
    for index, existing in enumerate(datasets):
        if existing.get("id") == dataset_id:
            datasets[index] = dataset
            replaced = True
            break
    if not replaced:
        datasets.append(dataset)
    save_datasets(datasets)
    return dataset


def update_dataset(dataset_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    datasets = list_datasets()
    for index, existing in enumerate(datasets):
        if existing.get("id") == dataset_id:
            merged = dict(existing)
            merged.update(updates)
            datasets[index] = merged
            save_datasets(datasets)
            return merged
    return None


def delete_dataset(dataset_id: str) -> bool:
    datasets = list_datasets()
    new_datasets = [item for item in datasets if item.get("id") != dataset_id]
    if len(new_datasets) == len(datasets):
        return False
    save_datasets(new_datasets)
    return True


def list_presets() -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    if not PRESETS_DIR.exists():
        return presets
    for path in sorted(PRESETS_DIR.glob("*.json")):
        try:
            preset = _load_json(path)
        except json.JSONDecodeError:
            continue
        if not isinstance(preset, dict):
            continue
        preset.setdefault("id", path.stem)
        preset["path"] = path.as_posix()
        presets.append(preset)
    return presets


def dataset_manifest_hash(dataset: Dict[str, Any]) -> str:
    payload = {
        "schema_manifest": dataset.get("schema_manifest", {}),
        "metadata_columns": dataset.get("metadata_columns", []),
        "staged_path": dataset.get("staged_path"),
        "cell_metadata_path": dataset.get("cell_metadata_path"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
