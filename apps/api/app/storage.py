from pathlib import Path
from pathlib import Path
from typing import Dict, Iterable, List


def safe_join(base: Path, target: str) -> Path:
    base = base.resolve()
    candidate = (base / target).resolve()
    if base not in candidate.parents and candidate != base:
        raise ValueError("Invalid path")
    return candidate


def is_allowed_path(path: Path, allowed_roots: Iterable[Path]) -> bool:
    path = path.resolve()
    for root in allowed_roots:
        root = root.resolve()
        if root in path.parents or root == path:
            return True
    return False


def enforce_allowed_path(path: Path, allowed_roots: Iterable[Path]) -> None:
    if not is_allowed_path(path, allowed_roots):
        raise ValueError("Path is outside allowed roots")


def list_artifacts(base: Path, subpath: str = "", max_depth: int = 4) -> List[Dict[str, str]]:
    root = safe_join(base, subpath) if subpath else base.resolve()
    items: List[Dict[str, str]] = []
    if not root.exists():
        return items

    base_resolved = base.resolve()
    root_depth = len(root.relative_to(base_resolved).parts) if root != base_resolved else 0
    if root_depth > max_depth:
        return items

    def walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(current.iterdir())
        except PermissionError:
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if depth < max_depth:
                    walk(entry, depth + 1)
                continue
            rel = entry.relative_to(base_resolved)
            if len(rel.parts) > max_depth:
                continue
            items.append(
                {
                    "path": str(rel).replace("\\", "/"),
                    "size": str(entry.stat().st_size),
                }
            )

    walk(root, root_depth)
    return sorted(items, key=lambda x: x["path"])
