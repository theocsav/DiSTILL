from __future__ import annotations

import argparse
import gzip
import io
import re
import tarfile
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import sparse


DEFAULT_RAW_TAR = Path("skin_dataset") / "GSE195452_RAW.tar"
DEFAULT_METADATA = Path("skin_dataset") / "GSE195452_Cell_metadata_v26_anno.txt.gz"
DEFAULT_OUTPUT = Path("skin_dataset") / "processed" / "skin_reference.h5ad"
DEFAULT_SUMMARY = Path("skin_dataset") / "processed" / "skin_reference_gse195452_summary.csv"


def load_metadata(path: Path) -> pd.DataFrame:
    columns = [
        "Well_ID",
        "well_coordinates",
        "Amp_batch_ID",
        "Cell_barcode",
        "Seq_batch_ID",
        "Pool_barcode",
        "Pool_barcode_i5",
        "Pool_barcode_i7",
        "Number_of_cells",
        "annotation",
    ]
    df = pd.read_csv(path, sep="\t", names=columns, header=0, low_memory=False)
    required = {"Well_ID", "Cell_barcode", "annotation"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Metadata file is missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["Well_ID"] = df["Well_ID"].astype(str)
    df["Cell_barcode"] = df["Cell_barcode"].astype(str)
    df["annotation"] = df["annotation"].astype(str)
    df = df[df["annotation"].notna() & (df["annotation"] != "_")].copy()
    df["cell_type"] = df["annotation"].astype(str)
    df["patient"] = df["Cell_barcode"].astype(str)
    df["original_sample_id"] = df["Cell_barcode"].astype(str)
    df["library"] = df["Cell_barcode"].astype(str)
    df = df.set_index("Well_ID", drop=False)
    return df


def extract_sample_code(member_name: str) -> str:
    match = re.search(r"_([A-Za-z0-9-]+)\.txt\.gz$", member_name)
    if not match:
        raise RuntimeError(f"Could not parse sample code from member name: {member_name}")
    return match.group(1)


def load_sample_matrix(payload: bytes) -> pd.DataFrame:
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as gz:
        df = pd.read_csv(gz, sep="\t", index_col=0)
    df.columns = df.columns.astype(str)
    df.index = df.index.astype(str)
    return df


def build_reference_h5ad(
    raw_tar: Path,
    metadata_path: Path,
    output_h5ad: Path,
    summary_csv: Path,
    sample_limit: int | None = None,
) -> tuple[Path, Path]:
    metadata = load_metadata(metadata_path)
    adatas: list[ad.AnnData] = []
    summary_rows: list[dict] = []

    with tarfile.open(raw_tar) as tf:
        members = [member for member in tf.getmembers() if member.isfile() and member.name.endswith(".txt.gz")]
        if sample_limit is not None:
            members = members[:sample_limit]

        for i, member in enumerate(members, start=1):
            sample_code = extract_sample_code(member.name)
            print(f"[{i}/{len(members)}] processing {sample_code} ({member.name})")
            sample_meta = metadata[metadata["Cell_barcode"] == sample_code]
            if sample_meta.empty:
                print(f"  skipped: no annotated metadata rows")
                continue

            payload = tf.extractfile(member).read()
            counts = load_sample_matrix(payload)

            matched_cells = [cell_id for cell_id in sample_meta.index if cell_id in counts.columns]
            if not matched_cells:
                print(f"  skipped: no overlapping Well_ID columns")
                continue

            counts = counts.loc[:, matched_cells]
            obs = sample_meta.loc[matched_cells].copy()
            obs["cell_barcode"] = obs["Well_ID"].astype(str)
            obs["sample_file"] = member.name
            obs_names = obs["Well_ID"].astype(str)

            matrix = sparse.csr_matrix(counts.transpose().to_numpy(dtype="int32"))
            var = pd.DataFrame(index=pd.Index(counts.index.astype(str), name="gene"))
            adata = ad.AnnData(X=matrix, obs=obs, var=var)
            adata.obs_names = obs_names
            adatas.append(adata)

            summary_rows.append(
                {
                    "sample_code": sample_code,
                    "sample_file": member.name,
                    "n_cells_total_in_metadata": int(len(sample_meta)),
                    "n_cells_matched": int(len(matched_cells)),
                    "n_genes": int(counts.shape[0]),
                }
            )
            print(f"  kept {len(matched_cells)} labeled cells across {counts.shape[0]} genes")

    if not adatas:
        raise RuntimeError("No annotated cells could be matched between GSE195452 metadata and raw count files.")

    combined = ad.concat(adatas, axis=0, join="outer", merge="same", fill_value=0, index_unique=None)
    combined.obs_names = combined.obs["Well_ID"].astype(str)
    combined.var_names_make_unique()

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(output_h5ad)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    return output_h5ad, summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a skin scRNA reference h5ad from GSE195452 raw counts and metadata.")
    parser.add_argument("--raw-tar", type=Path, default=DEFAULT_RAW_TAR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-h5ad", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional limit on number of raw samples to process for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_h5ad, summary_csv = build_reference_h5ad(
        raw_tar=args.raw_tar,
        metadata_path=args.metadata,
        output_h5ad=args.output_h5ad,
        summary_csv=args.summary_csv,
        sample_limit=args.sample_limit,
    )
    print(f"reference_h5ad={output_h5ad}")
    print(f"summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
