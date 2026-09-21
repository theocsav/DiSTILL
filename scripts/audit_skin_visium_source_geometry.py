#!/usr/bin/env python3
"""Audit Visium source geometry from Space Ranger metadata inside ZIP archives.

This script intentionally reads only ``tissue_positions*.csv`` and
``scalefactors_json.json``.  It never extracts an archive or opens expression
matrices.  Real shared-data audits must be submitted through SLURM; local
synthetic fixtures are suitable for tests.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable


class SourceGeometryError(ValueError):
    """Malformed or unsafe source geometry input."""


NAMED_COLUMNS = (
    "barcode", "in_tissue", "array_row", "array_col",
    "pxl_row_in_fullres", "pxl_col_in_fullres",
)
LEGACY_COLUMNS = NAMED_COLUMNS
NEIGHBOR_OFFSETS = ((0, 2), (1, -1), (1, 1))


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SourceGeometryError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise SourceGeometryError(f"{name} is non-finite")
    return result


def _members_for(archive: zipfile.ZipFile, suffix: str) -> list[str]:
    return sorted(
        name for name in archive.namelist()
        if not name.startswith("__MACOSX/") and name.rstrip("/").endswith(suffix)
    )


def _read_positions(raw: bytes, archive_name: str, member: str) -> list[dict[str, Any]]:
    text = raw.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise SourceGeometryError(f"{archive_name}:{member} is empty")
    header = [cell.strip() for cell in rows[0]]
    named = all(column in header for column in NAMED_COLUMNS)
    if named:
        indices = {column: header.index(column) for column in NAMED_COLUMNS}
        data_rows = rows[1:]
    else:
        if len(header) < len(LEGACY_COLUMNS):
            raise SourceGeometryError(
                f"{archive_name}:{member} has neither the named-column nor six-column legacy format"
            )
        indices = {column: index for index, column in enumerate(LEGACY_COLUMNS)}
        data_rows = rows
    result: list[dict[str, Any]] = []
    seen_barcodes: dict[str, int] = {}
    for line_number, row in enumerate(data_rows, 2 if named else 1):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) <= max(indices.values()):
            raise SourceGeometryError(f"{archive_name}:{member} row {line_number} is truncated")
        barcode = row[indices["barcode"]].strip()
        if not barcode:
            raise SourceGeometryError(f"{archive_name}:{member} row {line_number} has an empty barcode")
        if barcode in seen_barcodes:
            raise SourceGeometryError(
                f"{archive_name}:{member} duplicate barcode {barcode!r} at row {line_number}; "
                f"first seen at row {seen_barcodes[barcode]}"
            )
        seen_barcodes[barcode] = line_number
        try:
            in_tissue = int(row[indices["in_tissue"]])
            array_row = int(row[indices["array_row"]])
            array_col = int(row[indices["array_col"]])
        except (TypeError, ValueError) as exc:
            raise SourceGeometryError(f"{archive_name}:{member} row {line_number} has invalid lattice values") from exc
        if in_tissue not in (0, 1):
            raise SourceGeometryError(f"{archive_name}:{member} row {line_number} has invalid in_tissue={in_tissue}")
        result.append({
            "barcode": barcode, "in_tissue": in_tissue,
            "array_row": array_row, "array_col": array_col,
            "pxl_row_in_fullres": _finite(row[indices["pxl_row_in_fullres"]], "pixel row"),
            "pxl_col_in_fullres": _finite(row[indices["pxl_col_in_fullres"]], "pixel column"),
        })
    if not result:
        raise SourceGeometryError(f"{archive_name}:{member} has no positions")
    return result


def canonical_neighbor_pairs(coords: Iterable[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return deterministic undirected Visium hex-neighbor pairs."""
    coordinate_list = list(coords)
    points = set(coordinate_list)
    if len(points) != len(coordinate_list):
        # This branch is mainly defensive for callers passing a reusable list;
        # audit_archive performs its own duplicate check before calling here.
        raise SourceGeometryError("duplicate lattice coordinates")
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for row, col in sorted(points):
        for dr, dc in NEIGHBOR_OFFSETS:
            other = (row + dr, col + dc)
            if other in points:
                pairs.append(((row, col), other))
    return pairs


def _topology(coords: Iterable[tuple[int, int]]) -> dict[str, int]:
    coordinate_list = list(coords)
    pairs = canonical_neighbor_pairs(coordinate_list)
    degree = {coord: 0 for coord in set(coordinate_list)}
    for left, right in pairs:
        degree[left] += 1
        degree[right] += 1
    adjacency = {coord: set() for coord in degree}
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = 0
    unseen = set(degree)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return {
        "edges": len(pairs),
        "zero_degree": sum(value == 0 for value in degree.values()),
        "connected_components": components,
    }


def _edges(rows: list[dict[str, Any]], subset: Iterable[dict[str, Any]] | None = None) -> list[float]:
    selected = list(subset if subset is not None else rows)
    by_coord = {(row["array_row"], row["array_col"]): row for row in selected}
    if len(by_coord) != len(selected):
        raise SourceGeometryError("duplicate lattice coordinates")
    distances: list[float] = []
    for left_coord, right_coord in canonical_neighbor_pairs(by_coord):
        left, right = by_coord[left_coord], by_coord[right_coord]
        distances.append(math.hypot(
            left["pxl_row_in_fullres"] - right["pxl_row_in_fullres"],
            left["pxl_col_in_fullres"] - right["pxl_col_in_fullres"],
        ))
    return distances


def _quartiles(values: list[float]) -> tuple[float, float, float, float]:
    ordered = sorted(values)
    if not ordered:
        raise SourceGeometryError("no canonical lattice neighbor pairs found")
    def quantile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return quantile(0.25), quantile(0.5), quantile(0.75), ordered[0]


def audit_archive(path: str | Path) -> dict[str, Any]:
    """Audit one ZIP without extracting it or reading expression data."""
    archive_path = Path(path)
    if not archive_path.is_file() or archive_path.suffix.lower() != ".zip":
        raise SourceGeometryError(f"not a ZIP archive: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        position_members = _members_for(archive, "spatial/tissue_positions.csv")
        if not position_members:
            position_members = _members_for(archive, "spatial/tissue_positions_list.csv")
        scale_members = _members_for(archive, "spatial/scalefactors_json.json")
        if len(position_members) != 1 or len(scale_members) != 1:
            raise SourceGeometryError(
                f"{archive_path.name}: expected one positions and one scalefactors member; "
                f"found {position_members!r}, {scale_members!r}"
            )
        positions_member, scale_member = position_members[0], scale_members[0]
        rows = _read_positions(archive.read(positions_member), archive_path.name, positions_member)
        try:
            scales = json.loads(archive.read(scale_member).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceGeometryError(f"{archive_path.name}:{scale_member} is not valid JSON") from exc
        if not isinstance(scales, dict):
            raise SourceGeometryError(f"{archive_path.name}:{scale_member} must contain an object")
        diameter = _finite(scales.get("spot_diameter_fullres"), "spot_diameter_fullres")
        if diameter <= 0:
            raise SourceGeometryError("spot_diameter_fullres must be positive")
        lattice = [(row["array_row"], row["array_col"]) for row in rows]
        if len(set(lattice)) != len(lattice):
            raise SourceGeometryError(f"{archive_path.name}: duplicate lattice coordinates")
        pixels = [(row["pxl_row_in_fullres"], row["pxl_col_in_fullres"]) for row in rows]
        if len(set(pixels)) != len(pixels):
            raise SourceGeometryError(f"{archive_path.name}: duplicate pixel coordinates")
        tissue_rows = [row for row in rows if row["in_tissue"] == 1]
        all_distances = _edges(rows)
        tissue_distances = _edges(rows, tissue_rows) if tissue_rows else []
        all_topology = _topology(lattice)
        tissue_topology = _topology([(row["array_row"], row["array_col"]) for row in tissue_rows])
        if not all_distances:
            raise SourceGeometryError(f"{archive_path.name}: no canonical lattice neighbor pairs")
        if any((not math.isfinite(distance)) or distance <= 0 for distance in all_distances):
            raise SourceGeometryError(f"{archive_path.name}: nonpositive or nonfinite lattice pixel pitch")
        q1, median, q3, minimum = _quartiles(all_distances)
        maximum = max(all_distances)
        current_scale = 55.0 / diameter
        nominal_scale = 100.0 / median
        hires = scales.get("tissue_hires_scalef")
        lowres = scales.get("tissue_lowres_scalef")
        fiducial = scales.get("fiducial_diameter_fullres")
        result: dict[str, Any] = {
            "sample": archive_path.stem,
            "source_zip": str(archive_path.resolve()),
            "positions_member": positions_member,
            "scalefactors_member": scale_member,
            "positions_total": len(rows), "positions_in_tissue": len(tissue_rows),
            "duplicate_lattice_coords": 0, "duplicate_pixel_coords": 0,
            "canonical_lattice_edges_all": all_topology["edges"],
            "canonical_lattice_edges_in_tissue": tissue_topology["edges"],
            "lattice_zero_degree_all": all_topology["zero_degree"],
            "lattice_zero_degree_in_tissue": tissue_topology["zero_degree"],
            "lattice_connected_components_all": all_topology["connected_components"],
            "lattice_connected_components_in_tissue": tissue_topology["connected_components"],
            "pixel_pitch_median": median, "pixel_pitch_min": minimum,
            "pixel_pitch_max": maximum, "pixel_pitch_iqr": q3 - q1,
            "spot_diameter_fullres": diameter,
            "current_55um_scale_um_per_pixel": current_scale,
            "current_55um_resulting_median_um": median * current_scale,
            "nominal_100um_array_pitch_scale_um_per_pixel": nominal_scale,
            "implied_spot_diameter_under_nominal_pitch_um": diameter * nominal_scale,
            "scale_ratio_nominal_to_current": nominal_scale / current_scale,
            "scale_discrepancy_fraction": nominal_scale / current_scale - 1.0,
            "tissue_hires_scalef": _finite(hires, "tissue_hires_scalef") if hires is not None else None,
            "tissue_lowres_scalef": _finite(lowres, "tissue_lowres_scalef") if lowres is not None else None,
            "fiducial_diameter_fullres": _finite(fiducial, "fiducial_diameter_fullres") if fiducial is not None else None,
        }
        return result


def discover_archives(source_dir: str | Path) -> list[Path]:
    root = Path(source_dir)
    if not root.is_dir():
        raise SourceGeometryError(f"source directory does not exist: {root}")
    archives = [p for p in root.glob("*.zip") if not p.name.startswith("Stereo_seq_")]
    if not archives:
        raise SourceGeometryError(f"no eligible ZIP archives found in {root}")
    return sorted(archives, key=lambda path: path.name)


def audit_source(source_dir: str | Path) -> list[dict[str, Any]]:
    return [audit_archive(path) for path in discover_archives(source_dir)]


def write_audit(rows: list[dict[str, Any]], output_dir: str | Path, *, proposed_manifest: bool = False) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise SourceGeometryError(f"refusing existing output directory: {output}")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".partial", dir=parent))
    try:
        fields = list(rows[0]) if rows else []
        with (staging / "skin_visium_source_geometry_audit.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        payload = {
            "scope": "metadata-only Visium source ZIP geometry audit",
            "edge_definition": "canonical undirected array-coordinate offsets (0,2), (1,-1), (1,1); no nearest-neighbor inference",
            "source_files": [row["source_zip"] for row in rows],
            "samples": rows,
            "caveat": "Canonical edge, zero-degree, and component counts describe observed positions only and do not detect missing array sites. The nominal 100 um array-pitch scale is a sensitivity calibration candidate, not an independent microscope calibration; no correction is selected or applied.",
        }
        (staging / "skin_visium_source_geometry_audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if proposed_manifest:
            with (staging / "skin_visium_nominal_sensitivity_candidate_scales.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "microns_per_pixel", "scale_source"])
                writer.writeheader()
                for row in rows:
                    writer.writerow({"sample_id": row["sample"], "microns_per_pixel": row["nominal_100um_array_pitch_scale_um_per_pixel"], "scale_source": "nominal_100um_array_pitch"})
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proposed-manifest", action="store_true", help="emit an explicitly named sensitivity candidate scales file; never an operational input manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        write_audit(audit_source(args.source_dir), args.output_dir, proposed_manifest=args.proposed_manifest)
    except (OSError, zipfile.BadZipFile, SourceGeometryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
