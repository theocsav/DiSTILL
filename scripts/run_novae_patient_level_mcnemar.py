#!/usr/bin/env python3
"""Compute a predeclared exact patient-level McNemar report.

Only the completed immutable patient-level pooling output is read.  This is a
reporting operation: it does not retrain, re-pool, or inspect any FOV data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

# Permit direct invocation from any working directory on a compute node.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd
from scipy.stats import binomtest

INPUT_DEFAULT = Path(
    "/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/"
    "runs/novae_patient_level_pooling_20260924T161344Z_3258476/"
    "historical164_patient_level_pooling"
)
CONFIGURATIONS = ("full", "composition_only", "composition_enrichment", "composition_niche")
ARMS = ("nmf", "novae")
METHODS = ("primary_mean", "sensitivity_median", "sensitivity_majority_vote")
LABELS = ("healthy", "systemic_sclerosis")
PRIMARY_CONFIGURATION = "full"
PRIMARY_METHOD = "primary_mean"
EXPECTED_ROWS = 336
EXPECTED_PATIENTS = 14
REQUIRED_COLUMNS = {
    "configuration",
    "arm",
    "method",
    "patient",
    "outer_fold",
    "fov_count",
    "score",
    "vote_fraction",
    "tie",
    "true_label",
    "predicted_label",
    "correct",
}


class ContractError(ValueError):
    """Raised when immutable pooling output violates the frozen contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing immutable pooling manifest: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid pooling manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"pooling manifest must be an object: {path}")
    return value


def _inventory(manifest: dict[str, Any]) -> dict[str, Any]:
    for key in ("output_sha256", "output_hashes", "outputs"):
        value = manifest.get(key)
        if isinstance(value, dict) and value:
            return value
    raise ContractError("pooling manifest has no output hash inventory")


def _expected_hash(entry: Any) -> str:
    value = entry.get("sha256") if isinstance(entry, dict) else entry
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"invalid SHA256 entry in pooling manifest: {entry!r}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContractError(f"invalid SHA256 entry in pooling manifest: {entry!r}") from exc
    return value


def validate_pooling_protocol(manifest: dict[str, Any]) -> None:
    """Require the exact frozen protocol published by patient-level pooling."""
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ContractError("pooling manifest is missing its protocol object")
    expected = {
        "cohort": "historical-164",
        "unit": "FOV predictions pooled to patient",
        "threshold": 0.5,
        "positive_class": "systemic_sclerosis",
        "methods": list(METHODS),
        "configurations": list(CONFIGURATIONS),
        "arms": list(ARMS),
        "training": "none; completed immutable out-of-fold predictions only",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ContractError(f"pooling manifest protocol differs from frozen protocol: {key}")


def validate_manifest_inventory(manifest_path: Path, root: Path) -> dict[str, str]:
    """Validate every declared output hash, including the required patient CSV."""
    manifest = _json(manifest_path)
    inventory = _inventory(manifest)
    excludes = {str(item) for item in manifest.get("output_sha256_excludes", []) if isinstance(item, str)}
    result: dict[str, str] = {}
    root_resolved = root.resolve(strict=False)
    for relative, entry in inventory.items():
        if not isinstance(relative, str):
            raise ContractError(f"invalid output inventory path: {relative!r}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ContractError(f"output inventory path escapes pooling root: {relative!r}")
        # A manifest may explicitly exclude itself from its own inventory.
        if relative in excludes or (relative_path == Path(manifest_path.name) and manifest_path.parent == root):
            continue
        path = root / relative_path
        try:
            path.resolve(strict=False).relative_to(root_resolved)
        except ValueError as exc:
            raise ContractError(f"output inventory path escapes pooling root: {relative!r}") from exc
        expected = _expected_hash(entry)
        if not path.is_file() or sha256(path) != expected:
            raise ContractError(f"output hash mismatch: {relative}")
        result[relative] = expected
    required = "patient_predictions_long.csv"
    if required not in result:
        raise ContractError("pooling manifest inventory does not declare patient_predictions_long.csv")
    return result


def _parse_bool(value: Any, column: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ContractError(f"{column} contains a non-boolean value: {value!r}")


def validate_patient_predictions(frame: pd.DataFrame, source: Path | str = "patient_predictions_long.csv") -> pd.DataFrame:
    """Validate the complete 336-row pooled patient table and recompute correctness."""
    source = Path(source)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ContractError(f"{source} missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_ROWS:
        raise ContractError(f"{source} must contain exactly {EXPECTED_ROWS} rows")
    frame = frame.copy()
    for column in ("configuration", "arm", "method", "patient", "true_label", "predicted_label"):
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().isin({"", "nan", "none", "null"}).any():
            raise ContractError(f"{source} has blank values in {column}")
    if set(frame.configuration) != set(CONFIGURATIONS):
        raise ContractError("pooled output has an unexpected configuration set")
    if set(frame.arm) != set(ARMS):
        raise ContractError("pooled output has an unexpected arm set")
    if set(frame.method) != set(METHODS):
        raise ContractError("pooled output has an unexpected method set")
    if set(frame.true_label) != set(LABELS) or set(frame.predicted_label) - set(LABELS):
        raise ContractError("pooled output labels must be healthy/systemic_sclerosis")
    try:
        for column in ("outer_fold", "fov_count"):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        for column in ("score", "vote_fraction"):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{source} has non-numeric patient metadata or scores") from exc
    if frame[["score", "vote_fraction"]].isna().any().any() or not frame[["score", "vote_fraction"]].map(lambda x: x == x and abs(x) != float("inf")).all().all():
        raise ContractError(f"{source} scores must be finite")
    if not frame.score.between(0.0, 1.0).all() or not frame.vote_fraction.between(0.0, 1.0).all():
        raise ContractError(f"{source} scores must be in [0,1]")
    if not frame.outer_fold.between(1, EXPECTED_PATIENTS).all() or (frame.fov_count <= 0).any():
        raise ContractError(f"{source} has invalid outer fold or FOV count")
    frame["tie"] = [_parse_bool(value, "tie") for value in frame["tie"]]
    frame["correct"] = [_parse_bool(value, "correct") for value in frame["correct"]]
    duplicate_keys = ["configuration", "arm", "method", "patient"]
    if frame.duplicated(duplicate_keys).any():
        raise ContractError(f"{source} has duplicate configuration/arm/method/patient rows")
    expected_cells = len(CONFIGURATIONS) * len(ARMS) * len(METHODS)
    cell_sizes = frame.groupby(["configuration", "arm", "method"], sort=False).size()
    if len(cell_sizes) != expected_cells or not cell_sizes.eq(EXPECTED_PATIENTS).all():
        raise ContractError("pooled output must have exactly one row per patient in every cell")
    if frame.patient.nunique() != EXPECTED_PATIENTS:
        raise ContractError("pooled output must contain exactly 14 patients")
    if frame.true_label.groupby(frame.patient).nunique().ne(1).any():
        raise ContractError("patient labels are not aligned within patients")
    patient_labels = frame.groupby("patient", sort=False).true_label.first()
    if patient_labels.value_counts().to_dict() != {"healthy": 4, "systemic_sclerosis": 10}:
        raise ContractError("pooled output must contain 4 healthy and 10 systemic_sclerosis patients")
    if frame.groupby("patient").outer_fold.nunique().ne(1).any() or set(frame.outer_fold) != set(range(1, EXPECTED_PATIENTS + 1)):
        raise ContractError("patient outer-fold alignment must be one patient per fold 1..14")
    if frame.groupby("patient").fov_count.nunique().ne(1).any() or int(frame.groupby("patient").fov_count.first().sum()) != 164:
        raise ContractError("patient FOV counts must align and sum to the historical 164 FOVs")
    recomputed = frame.predicted_label.eq(frame.true_label)
    if not recomputed.equals(frame.correct):
        raise ContractError("stored correct booleans do not match predicted_label == true_label")

    # The patient key, label, fold, and FOV count are metadata that must align
    # across all 24 model/method cells, independently of row order.
    metadata = ["outer_fold", "fov_count", "true_label"]
    reference = frame[(frame.configuration == CONFIGURATIONS[0]) & (frame.arm == ARMS[0]) & (frame.method == METHODS[0])].set_index("patient").sort_index()
    for key, group in frame.groupby(["configuration", "arm", "method"], sort=False):
        candidate = group.set_index("patient").sort_index()
        for column in metadata:
            if not candidate[column].equals(reference[column]):
                raise ContractError(f"patient alignment mismatch in {key} column {column}")
    return frame


def paired_correctness(frame: pd.DataFrame, configuration: str, method: str) -> pd.DataFrame:
    """Return an aligned per-patient NMF/NOVAE correctness table."""
    selected = frame[(frame.configuration == configuration) & (frame.method == method)]
    left = selected[selected.arm == "nmf"].rename(columns={"correct": "nmf_correct", "predicted_label": "nmf_predicted_label", "score": "nmf_score"})
    right = selected[selected.arm == "novae"].rename(columns={"correct": "novae_correct", "predicted_label": "novae_predicted_label", "score": "novae_score"})
    columns = ["patient", "outer_fold", "true_label", "nmf_correct", "nmf_predicted_label", "nmf_score"]
    right_columns = ["patient", "novae_correct", "novae_predicted_label", "novae_score"]
    paired = left[columns].merge(right[right_columns], on="patient", how="inner", validate="one_to_one")
    if len(paired) != EXPECTED_PATIENTS:
        raise ContractError(f"paired table for {configuration}/{method} is not 14 patients")
    return paired.sort_values(["outer_fold", "patient"], kind="stable").reset_index(drop=True)


def exact_mcnemar(paired: pd.DataFrame, configuration: str = "", method: str = "", *, primary: bool = False) -> dict[str, Any]:
    """Compute the exact two-sided paired McNemar/binomial test from booleans."""
    required = {"nmf_correct", "novae_correct"}
    if not required.issubset(paired.columns):
        raise ContractError("paired table is missing correctness booleans")
    nmf = paired.nmf_correct.astype(bool)
    novae = paired.novae_correct.astype(bool)
    both_correct = int((nmf & novae).sum())
    both_wrong = int((~nmf & ~novae).sum())
    nmf_only = int((nmf & ~novae).sum())
    novae_only = int((~nmf & novae).sum())
    n = int(len(paired))
    discordant = nmf_only + novae_only
    p_value = 1.0 if discordant == 0 else float(binomtest(nmf_only, n=discordant, p=0.5, alternative="two-sided").pvalue)
    warning = "exploratory; exact two-sided McNemar; no equivalence or significance claim"
    if not primary:
        warning += "; secondary descriptive sensitivity, multiplicity-unadjusted, not selected"
    return {
        "configuration": configuration,
        "method": method,
        "primary": bool(primary),
        "n": n,
        "n_patients": n,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "nmf_only_correct": nmf_only,
        "novae_only_correct": novae_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
        "p_value": p_value,
        "test": "scipy.stats.binomtest(successes=nmf_only_correct, n=discordant, p=0.5, alternative='two-sided')",
        "warning": warning,
    }


def _source_hashes(input_root: Path, inventory: dict[str, str]) -> dict[str, str]:
    paths = {"pooling_manifest.json": sha256(input_root / "pooling_manifest.json")}
    paths.update(inventory)
    return paths


def _tree_hashes(root: Path, exclude: Iterable[Path] = ()) -> dict[str, str]:
    excluded = {path.resolve() for path in exclude}
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file() and path.resolve() not in excluded}


def run_mcnemar(*, input_root: Path, output_root: Path) -> dict[str, Any]:
    """Validate immutable pooling output and atomically publish exact report files."""
    input_root, output_root = Path(input_root), Path(output_root)
    manifest_path = input_root / "pooling_manifest.json"
    prediction_path = input_root / "patient_predictions_long.csv"
    if output_root.exists():
        raise ContractError(f"refusing existing output root (no overwrite): {output_root}")
    resolved_output = output_root.resolve(strict=False)
    resolved_input = input_root.resolve(strict=False)
    if resolved_output == resolved_input or resolved_output in resolved_input.parents or resolved_input in resolved_output.parents:
        raise ContractError("output root overlaps immutable input root")
    manifest = _json(manifest_path)
    validate_pooling_protocol(manifest)
    inventory = validate_manifest_inventory(manifest_path, input_root)
    if not prediction_path.is_file():
        raise ContractError(f"missing immutable patient predictions: {prediction_path}")
    before = _source_hashes(input_root, inventory)
    frame = validate_patient_predictions(pd.read_csv(prediction_path), prediction_path)
    all_rows: list[dict[str, Any]] = []
    paired_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for configuration in CONFIGURATIONS:
        for method in METHODS:
            paired = paired_correctness(frame, configuration, method)
            paired_tables[configuration, method] = paired
            all_rows.append(exact_mcnemar(paired, configuration, method, primary=(configuration == PRIMARY_CONFIGURATION and method == PRIMARY_METHOD)))
    all_results = pd.DataFrame(all_rows)
    primary_paired = paired_tables[PRIMARY_CONFIGURATION, PRIMARY_METHOD].copy()
    primary_paired.insert(0, "configuration", PRIMARY_CONFIGURATION)
    primary_paired.insert(1, "method", PRIMARY_METHOD)
    primary_paired["discordance"] = primary_paired.nmf_correct.ne(primary_paired.novae_correct)
    primary_paired["direction"] = "neither"
    primary_paired.loc[primary_paired.nmf_correct & ~primary_paired.novae_correct, "direction"] = "nmf_only_correct"
    primary_paired.loc[~primary_paired.nmf_correct & primary_paired.novae_correct, "direction"] = "novae_only_correct"
    warnings = [
        "Primary comparison is predeclared full/primary_mean NMF versus NOVAE over 14 paired patients.",
        "Secondary configurations and pooling methods are descriptive sensitivity analyses only, multiplicity-unadjusted and not selected.",
        "Exact two-sided McNemar tests are reported without asymptotic chi-square, mid-p, one-sided headlines, equivalence, or significance claims.",
    ]
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=parent))
    try:
        all_results.to_csv(stage / "exact_mcnemar_all.csv", index=False)
        primary_record = all_results[(all_results.configuration == PRIMARY_CONFIGURATION) & (all_results.method == PRIMARY_METHOD)].iloc[0].to_dict()
        primary_payload = {"protocol": {"configuration": PRIMARY_CONFIGURATION, "method": PRIMARY_METHOD, "arms": list(ARMS), "unit": "patient", "test": "exact two-sided McNemar via scipy.stats.binomtest", "p_successes": "nmf_only_correct", "null": "p=0.5", "n": EXPECTED_PATIENTS}, "result": primary_record, "warnings": warnings}
        (stage / "primary_exact_mcnemar.json").write_text(json.dumps(primary_payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
        primary_paired.to_csv(stage / "paired_primary_patients.csv", index=False)
        after_before = _source_hashes(input_root, inventory)
        if before != after_before:
            raise ContractError("immutable pooling input changed before report publication")
        output_hashes = _tree_hashes(stage)
        code_hashes = {"report_script": sha256(Path(__file__)), "launcher": sha256(REPO / "scripts" / "submit_novae_patient_level_mcnemar.sh"), "protocol_document": sha256(REPO / "docs" / "NOVAE_PATIENT_LEVEL_MCNEMAR.md")}
        report_manifest = {
            "protocol": {"cohort": "historical-164", "row_unit": "patient", "source": "completed immutable patient_predictions_long.csv", "configurations": list(CONFIGURATIONS), "arms": list(ARMS), "methods": list(METHODS), "primary_configuration": PRIMARY_CONFIGURATION, "primary_method": PRIMARY_METHOD, "expected_rows": EXPECTED_ROWS, "expected_patients": EXPECTED_PATIENTS, "labels": list(LABELS), "test": "exact two-sided McNemar via scipy.stats.binomtest", "zero_discordances": "p=1.0"},
            "warnings": warnings,
            "input": {"root": str(input_root), "manifest": str(manifest_path), "manifest_sha256": before["pooling_manifest.json"], "patient_predictions_sha256": before["patient_predictions_long.csv"], "validated_output_inventory": inventory},
            "source_unchanged_before_after": before == _source_hashes(input_root, inventory),
            "code_sha256": code_hashes,
            "output_sha256": output_hashes,
            "output_sha256_excludes": ["mcnemar_manifest.json"],
        }
        (stage / "mcnemar_manifest.json").write_text(json.dumps(report_manifest, indent=2, default=_json_default) + "\n", encoding="utf-8")
        if output_root.exists():
            raise ContractError(f"refusing existing output root (no overwrite): {output_root}")
        os.replace(stage, output_root)
        return report_manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_mcnemar(input_root=args.input_root, output_root=args.output_root)
    except Exception as exc:
        print(f"exact patient-level McNemar report refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
