from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_INPUT_DIR = Path("skin_dataset")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "processed"
DEFAULT_DATASET_ID = "skin_visium_ssc"
DEFAULT_EXCLUDE_PREFIXES = ("Stereo_seq_",)
VISIUM_SPOT_DIAMETER_UM = 55.0
IMAGE_MEMBERS = (
    "spatial/tissue_hires_image.png",
    "spatial/tissue_lowres_image.png",
    "spatial/cytassist_image.tiff",
    "spatial/aligned_tissue_image.jpg",
    "spatial/detected_tissue_image.jpg",
    "spatial/aligned_fiducials.jpg",
)


def infer_disease_state(sample_id: str) -> str:
    return "healthy" if sample_id.upper().startswith("HC") else "systemic_sclerosis"


def discover_sample_ids(input_dir: Path, exclude_prefixes: tuple[str, ...]) -> list[str]:
    sample_ids = []
    for path in sorted(input_dir.glob("*.zip")):
        stem = path.stem
        if any(stem.startswith(prefix) for prefix in exclude_prefixes):
            continue
        sample_ids.append(stem)
    if not sample_ids:
        raise FileNotFoundError(f"No Visium ZIP files found in {input_dir}")
    return sample_ids


def _read_zip_json(zf: zipfile.ZipFile, member_name: str) -> dict:
    with zf.open(member_name) as handle:
        return json.load(handle)


def _read_zip_csv(zf: zipfile.ZipFile, member_name: str) -> pd.DataFrame:
    with zf.open(member_name) as handle:
        return pd.read_csv(handle)


def _extract_zip_member(zf: zipfile.ZipFile, member_name: str, output_path: Path) -> Path | None:
    try:
        data = zf.read(member_name)
    except KeyError:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return output_path


def _load_visium_counts(zf: zipfile.ZipFile, sample_id: str) -> ad.AnnData:
    member_name = f"{sample_id}/filtered_feature_bc_matrix.h5"
    payload = zf.read(member_name)
    with h5py.File(io.BytesIO(payload), "r") as handle:
        matrix_group = handle["matrix"]
        data = matrix_group["data"][()]
        indices = matrix_group["indices"][()]
        indptr = matrix_group["indptr"][()]
        shape = tuple(matrix_group["shape"][()].tolist())
        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape).transpose().tocsr()

        barcodes = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in matrix_group["barcodes"][()]]
        features = matrix_group["features"]
        feature_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in features["name"][()]
        ]
        feature_ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in features["id"][()]
        ]
        var = pd.DataFrame(
            {
                "gene_ids": feature_ids,
                "feature_name": feature_names,
            },
            index=pd.Index(feature_names, name="gene"),
        )
        obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
        adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def _build_obs_frame(
    sample_id: str,
    positions: pd.DataFrame,
    scalefactors: dict,
    barcodes: pd.Index,
    pseudo_fov_tile_um: float | None = None,
) -> pd.DataFrame:
    if "barcode" not in positions.columns:
        raise RuntimeError(f"{sample_id}: tissue_positions.csv is missing the 'barcode' column.")

    positions = positions.copy()
    positions["barcode"] = positions["barcode"].astype(str)
    positions = positions.set_index("barcode", drop=False)
    positions = positions.loc[positions.index.intersection(barcodes)].copy()
    if "in_tissue" in positions.columns:
        positions = positions[positions["in_tissue"].astype(int) == 1].copy()

    spot_diameter = float(scalefactors.get("spot_diameter_fullres", 0.0))
    width = spot_diameter
    height = spot_diameter
    radius = spot_diameter / 2.0
    area = 3.141592653589793 * (radius**2)

    positions["sample_id"] = sample_id
    positions["patient"] = sample_id
    positions["Disease_State"] = infer_disease_state(sample_id)
    positions["disease_state"] = positions["Disease_State"]
    positions["cell_ID"] = positions["barcode"].astype(str)
    positions["cell_id"] = positions["barcode"].astype(str)
    positions["unique_cell_id"] = sample_id + "_" + positions["barcode"].astype(str)
    positions["CenterX_global_px"] = positions["pxl_col_in_fullres"].astype(float)
    positions["CenterY_global_px"] = positions["pxl_row_in_fullres"].astype(float)
    positions["CenterX_local_px"] = positions["CenterX_global_px"]
    positions["CenterY_local_px"] = positions["CenterY_global_px"]
    positions["Width"] = width
    positions["Height"] = height
    positions["Area"] = area
    positions["assay_type"] = "Visium"
    positions["platform"] = "visium"
    positions["slide_ID"] = sample_id

    if pseudo_fov_tile_um is not None:
        px_per_um = width / VISIUM_SPOT_DIAMETER_UM
        tile_px = pseudo_fov_tile_um * px_per_um
        x0 = float(positions["CenterX_global_px"].min())
        y0 = float(positions["CenterY_global_px"].min())
        positions["pseudo_fov_tile_um"] = float(pseudo_fov_tile_um)
        positions["pseudo_fov_tile_px"] = float(tile_px)
        positions["tile_x"] = np.floor((positions["CenterX_global_px"] - x0) / tile_px).astype(int)
        positions["tile_y"] = np.floor((positions["CenterY_global_px"] - y0) / tile_px).astype(int)
        positions["fov"] = (
            "tile"
            + positions["tile_x"].astype(str)
            + "_"
            + positions["tile_y"].astype(str)
        )
    else:
        positions["fov"] = "1"

    positions["field_of_view"] = sample_id + "_" + positions["fov"].astype(str)

    positions = positions.set_index("unique_cell_id", drop=False)
    return positions


def load_sample(
    zip_path: Path,
    output_image_dir: Path,
    pseudo_fov_tile_um: float | None = None,
) -> tuple[ad.AnnData, pd.DataFrame, dict]:
    sample_id = zip_path.stem
    with zipfile.ZipFile(zip_path) as zf:
        adata = _load_visium_counts(zf, sample_id)
        positions = _read_zip_csv(zf, f"{sample_id}/spatial/tissue_positions.csv")
        scalefactors = _read_zip_json(zf, f"{sample_id}/spatial/scalefactors_json.json")

        obs = _build_obs_frame(
            sample_id,
            positions,
            scalefactors,
            adata.obs_names.astype(str),
            pseudo_fov_tile_um=pseudo_fov_tile_um,
        )
        keep_barcodes = obs["barcode"].astype(str).tolist()
        adata = adata[keep_barcodes].copy()
        adata.obs = obs.loc[[sample_id + "_" + barcode for barcode in keep_barcodes]].copy()
        adata.obs_names = adata.obs["unique_cell_id"].astype(str)
        adata.obs["patient"] = adata.obs["patient"].astype("category")
        adata.obs["Disease_State"] = adata.obs["Disease_State"].astype("category")
        adata.obs["fov"] = adata.obs["fov"].astype("category")
        adata.obsm["spatial"] = adata.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(dtype="float64")

        image_dir = output_image_dir / sample_id
        extracted_paths: dict[str, str] = {}
        for member_suffix in IMAGE_MEMBERS:
            member_name = f"{sample_id}/{member_suffix}"
            output_path = image_dir / Path(member_suffix).name
            extracted = _extract_zip_member(zf, member_name, output_path)
            if extracted is not None:
                extracted_paths[Path(member_suffix).stem] = str(extracted)

    sample_manifest_row = {
        "sample_id": sample_id,
        "zip_path": str(zip_path),
        "disease_state": infer_disease_state(sample_id),
        "n_spots_in_tissue": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "spot_diameter_fullres": float(scalefactors.get("spot_diameter_fullres", 0.0)),
        "tissue_hires_scalef": float(scalefactors.get("tissue_hires_scalef", 0.0)),
        "tissue_lowres_scalef": float(scalefactors.get("tissue_lowres_scalef", 0.0)),
        "pseudo_fov_tile_um": pseudo_fov_tile_um if pseudo_fov_tile_um is not None else "",
        "pseudo_fov_count": int(adata.obs["field_of_view"].nunique()),
        **{f"{key}_path": value for key, value in extracted_paths.items()},
    }
    return adata, adata.obs.reset_index(drop=True), sample_manifest_row


def build_spatial_h5ad(
    input_dir: Path,
    output_dir: Path,
    sample_ids: list[str],
    dataset_id: str,
    pseudo_fov_tile_um: float | None = None,
) -> tuple[Path, Path, Path]:
    adatas: list[ad.AnnData] = []
    metadata_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    image_dir = output_dir / "images"

    for sample_id in sample_ids:
        zip_path = input_dir / f"{sample_id}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing ZIP for sample {sample_id}: {zip_path}")
        adata, metadata_frame, manifest_row = load_sample(
            zip_path,
            image_dir,
            pseudo_fov_tile_um=pseudo_fov_tile_um,
        )
        adatas.append(adata)
        metadata_frames.append(metadata_frame)
        manifest_rows.append(manifest_row)

    combined = ad.concat(adatas, axis=0, join="outer", merge="same", fill_value=0, index_unique=None)
    combined.obs_names = combined.obs["unique_cell_id"].astype(str)
    combined.var_names_make_unique()

    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_h5ad_path = output_dir / f"{dataset_id}_spatial.h5ad"
    metadata_csv_path = output_dir / f"{dataset_id}_metadata.csv"
    manifest_csv_path = output_dir / f"{dataset_id}_sample_manifest.csv"

    combined.write_h5ad(spatial_h5ad_path)
    pd.concat(metadata_frames, ignore_index=True).to_csv(metadata_csv_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_csv_path, index=False)
    return spatial_h5ad_path, metadata_csv_path, manifest_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a merged Visium spatial h5ad and metadata CSV from per-sample ZIPs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--samples", nargs="+", default=None, help="Optional explicit sample IDs to include.")
    parser.add_argument(
        "--pseudo-fov-tile-um",
        type=float,
        default=None,
        help="Optional pseudo-FOV tile size in microns. If set, assigns tiled pseudo-FOV IDs within each sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = args.samples or discover_sample_ids(args.input_dir, DEFAULT_EXCLUDE_PREFIXES)
    spatial_h5ad_path, metadata_csv_path, manifest_csv_path = build_spatial_h5ad(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_ids=sample_ids,
        dataset_id=args.dataset_id,
        pseudo_fov_tile_um=args.pseudo_fov_tile_um,
    )
    print(f"spatial_h5ad={spatial_h5ad_path}")
    print(f"cell_metadata={metadata_csv_path}")
    print(f"sample_manifest={manifest_csv_path}")
    print("reference_h5ad=missing_skin_reference_h5ad")


if __name__ == "__main__":
    main()
