#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

DEFAULT_FIXED_TEMPLATE = "pipeline_assets/scripts/IBD_3000epochs_500samples_NMF-k4.py"
DEFAULT_ELBOW_TEMPLATE = "pipeline_assets/scripts/IBD_3000epochs_systematicNMFapproach.py"
DEFAULT_POISSON_TEMPLATE = "pipeline_assets/scripts/IBD_3000epochs_systematicNMFapproach.py"
DEFAULT_NMF_ONLY_SCRIPT = "pipeline_assets/scripts/IBD_Run_NMF_From_Cell2Loc.py"
DEFAULT_POST_NMF_NOTEBOOK = "pipeline_assets/IBD_Post_NMF_Analysis.ipynb"
DEFAULT_RCAUSAL_NOTEBOOK = "pipeline_assets/IBD_RCausalMGM_Preparation.ipynb"
DEFAULT_MLP_SCRIPT = "pipeline_assets/IBD_MLP_44Features.py"
DEFAULT_FOV_MLP_INPUT_BUILDER = "pipeline_assets/IBD_Build_FOV_MLP_Inputs.py"
DEFAULT_REPORT_TITLE = "NicheRunner Run Report"
ALLOWED_STAGES = ("cell2loc_nmf", "cell2loc", "nmf", "post_nmf", "rcausal_mgm", "mlp", "report")
ALLOWED_MODES = ("fixed_k", "elbow_k", "poisson_redundancy_k", "poisson_cumulative_improvement_k")


def default_template_for_mode(mode: str) -> str:
    if mode == "fixed_k":
        return DEFAULT_FIXED_TEMPLATE
    if mode in ("poisson_redundancy_k", "poisson_cumulative_improvement_k"):
        return DEFAULT_POISSON_TEMPLATE
    return DEFAULT_ELBOW_TEMPLATE


def norm_path(value):
    return value.replace("\\", "/")


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def replace_assignment(text, var_name, value_expr):
    pattern = r"^(?P<indent>\s*){}\s*=.*$".format(re.escape(var_name))
    replacement = rf"\g<indent>{var_name} = {value_expr}"
    return re.subn(pattern, replacement, text, flags=re.MULTILINE)


def apply_assignment(text, var_name, value_expr, warnings):
    text, count = replace_assignment(text, var_name, value_expr)
    if count == 0:
        warnings.append(f"Did not find assignment for '{var_name}' in template.")
    return text


def ensure_required(config, key):
    if key not in config or config[key] in (None, ""):
        raise ValueError(f"Missing required config key: {key}")


def resolve_template(root, template_path):
    template = Path(template_path)
    if not template.is_absolute():
        template = root / template
    return template


def render_value(value):
    if isinstance(value, str):
        return json.dumps(norm_path(value))
    return str(value)


def render_python_literal(value):
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return "None"
    return repr(value)


def write_text(path, content):
    path.write_text(content, encoding="utf-8")


def shell_quote(value):
    return shlex.quote(norm_path(str(value)))


def normalize_stages(config):
    stages = config.get("stages")
    if not stages:
        return ["cell2loc_nmf"]
    if isinstance(stages, str):
        stages = [item.strip() for item in stages.split(",") if item.strip()]
    if not isinstance(stages, list):
        raise ValueError("stages must be a list of stage names.")
    for stage in stages:
        if stage not in ALLOWED_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Allowed: {', '.join(ALLOWED_STAGES)}")
    return stages


def report_validation(errors, warnings):
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return
    print("Validation OK.")


def validate_cli_config(config, root, check_paths=True):
    errors = []
    warnings = []

    try:
        stages = normalize_stages(config)
    except ValueError as exc:
        errors.append(str(exc))
        stages = []

    needs_cell2loc_inputs = "cell2loc_nmf" in stages or "cell2loc" in stages
    needs_nmf_config = "cell2loc_nmf" in stages or "nmf" in stages

    if needs_cell2loc_inputs:
        for key in ("reference_h5ad_path", "cosmx_h5ad_path", "cell_metadata_path"):
            value = config.get(key)
            if not value:
                errors.append(f"Missing required config key: {key}")
                continue
            if check_paths and not Path(value).exists():
                errors.append(f"Path does not exist: {key} -> {value}")
    elif "nmf" in stages:
        value = config.get("cosmx_h5ad_path")
        if not value:
            warnings.append("cosmx_h5ad_path is not required for nmf-only stage generation.")

    mode = config.get("mode", "fixed_k")
    if needs_nmf_config:
        if mode not in ALLOWED_MODES:
            errors.append("mode must be one of: 'fixed_k', 'elbow_k', 'poisson_redundancy_k', 'poisson_cumulative_improvement_k'.")
        elif mode == "fixed_k":
            if config.get("n_components") is None and config.get("k") is None:
                errors.append("Fixed-k mode requires 'n_components' or 'k'.")
        else:
            k_min = int(config.get("k_min", 2))
            k_max = int(config.get("k_max", 20))
            if k_max < k_min:
                errors.append("k_max must be >= k_min.")
            if mode == "poisson_cumulative_improvement_k":
                target = float(config.get("poisson_cumulative_improvement_target", 0.95))
                if target <= 0 or target > 1:
                    errors.append("poisson_cumulative_improvement_target must be in the interval (0, 1].")

    template_path = config.get("template_path")
    if needs_cell2loc_inputs and not template_path:
        template_path = default_template_for_mode(mode)
    if needs_cell2loc_inputs:
        template_path = resolve_template(root, template_path)
        if check_paths and not template_path.exists():
            errors.append(f"Template not found: {template_path}")

    if "nmf" in stages:
        nmf_script_path = resolve_template(root, config.get("nmf_script_path", DEFAULT_NMF_ONLY_SCRIPT))
        if check_paths and not nmf_script_path.exists():
            errors.append(f"NMF-only script not found: {nmf_script_path}")

    if "post_nmf" in stages:
        post_nmf_mode = config.get("post_nmf_mode", "papermill")
        if post_nmf_mode not in ("papermill", "python"):
            errors.append("post_nmf_mode must be 'papermill' or 'python'.")
        elif post_nmf_mode == "papermill":
            notebook_path = resolve_template(root, config.get("post_nmf_notebook_path", DEFAULT_POST_NMF_NOTEBOOK))
            if check_paths and not notebook_path.exists():
                errors.append(f"Post-NMF notebook not found: {notebook_path}")
        else:
            script_path = config.get("post_nmf_script_path")
            if not script_path:
                errors.append("post_nmf_script_path is required when post_nmf_mode=python.")
            else:
                script_path = resolve_template(root, script_path)
                if check_paths and not script_path.exists():
                    errors.append(f"Post-NMF script not found: {script_path}")

    if "rcausal_mgm" in stages:
        rcausal_mode = config.get("rcausal_mode", "papermill")
        if rcausal_mode not in ("papermill", "python"):
            errors.append("rcausal_mode must be 'papermill' or 'python'.")
        elif rcausal_mode == "papermill":
            notebook_path = resolve_template(root, config.get("rcausal_notebook_path", DEFAULT_RCAUSAL_NOTEBOOK))
            if check_paths and not notebook_path.exists():
                errors.append(f"RCausalMGM notebook not found: {notebook_path}")
        else:
            script_path = config.get("rcausal_script_path")
            if not script_path:
                errors.append("rcausal_script_path is required when rcausal_mode=python.")
            else:
                script_path = resolve_template(root, script_path)
                if check_paths and not script_path.exists():
                    errors.append(f"RCausalMGM script not found: {script_path}")
            for support_path in config.get("rcausal_support_files", []):
                resolved_support = resolve_template(root, support_path)
                if check_paths and not resolved_support.exists():
                    errors.append(f"RCausalMGM support file not found: {resolved_support}")

    if "mlp" in stages:
        script_path = resolve_template(root, config.get("mlp_script_path", DEFAULT_MLP_SCRIPT))
        if check_paths and not script_path.exists():
            errors.append(f"MLP script not found: {script_path}")

    slurm = config.get("slurm", {})
    if slurm.get("enabled") and not slurm.get("conda_env"):
        warnings.append("slurm.enabled is true but slurm.conda_env is missing.")

    return errors, warnings


def copy_resource(source: Path, run_dir: Path) -> Path:
    destination = run_dir / source.name
    shutil.copy2(source, destination)
    return destination


def copy_support_resources(root: Path, run_dir: Path, resource_paths: list[str]) -> list[Path]:
    copied = []
    for resource_path in resource_paths:
        source = resolve_template(root, resource_path)
        if not source.exists():
            raise FileNotFoundError(f"Support resource not found: {source}")
        copied.append(copy_resource(source, run_dir))
    return copied


def build_papermill_command(input_path: Path, output_path: Path, parameters: dict) -> str:
    cmd = f"papermill {shell_quote(input_path)} {shell_quote(output_path)} -k python3"
    for key, value in parameters.items():
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value)
        cmd += f" -p {shell_quote(key)} {shell_quote(rendered)}"
    return cmd


def build_rcausal_args(config: dict, output_dir: str) -> list[str]:
    args = config.get("rcausal_args")
    if args:
        return list(args)
    defaults = []
    output_base = config.get("rcausal_output_dir") or str(Path(output_dir) / "rcausal_mgm")
    defaults.extend(["--output-dir", output_base])
    default_h5ad = config.get("rcausal_h5ad_path") or str(Path(output_dir) / "cosmx_with_nmf.h5ad")
    niche_h5ad = config.get("rcausal_niche_h5ad_path") or default_h5ad or config.get("cosmx_h5ad_path")
    neighborhood_h5ad = (
        config.get("rcausal_neighborhood_h5ad_path") or default_h5ad or config.get("cosmx_h5ad_path")
    )
    if niche_h5ad:
        defaults.extend(["--niche-h5ad", niche_h5ad])
    if neighborhood_h5ad:
        defaults.extend(["--neighborhood-h5ad", neighborhood_h5ad])
    return defaults


def build_run_script(run_dir: Path, output_dir: Path, stage_commands) -> str:
    run_dir_str = norm_path(str(run_dir))
    output_dir_str = norm_path(str(output_dir))
    logs_dir_str = norm_path(str(Path(output_dir) / "logs"))
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "umask 007",
        f"RUN_DIR={shell_quote(run_dir_str)}",
        f"OUTPUT_DIR={shell_quote(output_dir_str)}",
        f"LOGS_DIR={shell_quote(logs_dir_str)}",
        "mkdir -p \"$OUTPUT_DIR\" \"$LOGS_DIR\"",
        f"cd {shell_quote(run_dir_str)}",
        "",
    ]
    for stage, command in stage_commands:
        out_path = f"{logs_dir_str}/{stage}.out"
        err_path = f"{logs_dir_str}/{stage}.err"
        lines.append(f"echo \">>> Stage {stage} started: $(date)\"")
        lines.append(f"{command} > {shell_quote(out_path)} 2> {shell_quote(err_path)}")
        lines.append(f"echo \">>> Stage {stage} finished: $(date)\"")
        lines.append("")
    return "\n".join(lines)


def build_report_script(output_dir: Path, stages, report_title: str, report_notes: str, run_name: str) -> str:
    return f"""#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path({json.dumps(str(output_dir))})
STAGES = {json.dumps(stages)}
REPORT_TITLE = {json.dumps(report_title)}
REPORT_NOTES = {json.dumps(report_notes)}
RUN_NAME = {json.dumps(run_name)}

FIGURE_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".pdf")
TABLE_EXTS = (".csv", ".tsv", ".parquet", ".xlsx", ".xls")
SKIP_PREFIXES = ("report/", "artifacts/", "logs/")

GROUP_RULES = [
    ("cell2location", ["w_sf", "inferred_cell_type", "spatial_model", "ref_model", "inf_aver", "training_history", "QC"]),
    ("niches", ["NMF_", "nmf_", "niche"]),
    ("post_nmf", ["post_nmf", "Post-NMF", "post-nmf", "feature", "enrichment", "niche_gene"]),
    ("rcausal_mgm", ["RCausalMGM", "rcausal", "NeighborhoodInteractions", "NicheCompositions"]),
    ("models", ["MLP", "mlp", "confusion", "metrics", "predictions", "best_params"]),
    ("logs", ["logs/"]),
    ("report", ["report/"]),
]


def classify(path: str) -> str:
    for group, needles in GROUP_RULES:
        for needle in needles:
            if needle.endswith("/") and path.startswith(needle):
                return group
            if needle in path:
                return group
    return "other"


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def is_skipped(rel_path: str) -> bool:
    return any(rel_path.startswith(prefix) for prefix in SKIP_PREFIXES)


def collect_sources():
    figure_sources = []
    table_sources = []
    for path in OUTPUT_DIR.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(OUTPUT_DIR).as_posix()
        if is_skipped(rel):
            continue
        ext = path.suffix.lower()
        if ext in FIGURE_EXTS:
            figure_sources.append((rel, path))
        elif ext in TABLE_EXTS:
            table_sources.append((rel, path))
    return figure_sources, table_sources


def unique_asset_path(dest_dir: Path, rel: str) -> Path:
    safe_name = rel.replace("/", "__")
    candidate = dest_dir / safe_name
    if not candidate.exists():
        return candidate
    base, ext = os.path.splitext(safe_name)
    index = 1
    while True:
        candidate = dest_dir / f"{{base}}__{{index}}{{ext}}"
        if not candidate.exists():
            return candidate
        index += 1


def copy_assets(sources, dest_dir: Path):
    assets = []
    for rel, path in sources:
        dest = unique_asset_path(dest_dir, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        assets.append({{"path": dest.relative_to(OUTPUT_DIR).as_posix(), "source": rel}})
    return assets


def main() -> int:
    report_dir = OUTPUT_DIR / "report"
    artifacts_dir = OUTPUT_DIR / "artifacts"
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    if tables_dir.exists():
        shutil.rmtree(tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    figure_sources, table_sources = collect_sources()
    figure_assets = copy_assets(figure_sources, figures_dir)
    table_assets = copy_assets(table_sources, tables_dir)

    artifacts = []
    groups = {{}}
    for path in OUTPUT_DIR.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(OUTPUT_DIR).as_posix()
        artifacts.append({{"path": rel, "size": path.stat().st_size}})
        group = classify(rel)
        groups.setdefault(group, []).append(rel)

    for group in groups:
        groups[group] = sorted(groups[group])

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {{
        "generated_at": generated_at,
        "output_dir": str(OUTPUT_DIR),
        "stages": STAGES,
        "groups": groups,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "report_assets": {{
            "figures": figure_assets,
            "tables": table_assets,
        }},
    }}
    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_summary = {{
        "generated_at": generated_at,
        "run_name": RUN_NAME,
        "output_dir": str(OUTPUT_DIR),
        "stages": STAGES,
        "report_title": REPORT_TITLE,
        "report_notes": REPORT_NOTES,
        "report_path": "report/report.html",
        "manifest_path": "artifacts/manifest.json",
        "figures_count": len(figure_assets),
        "tables_count": len(table_assets),
    }}
    run_summary_path = artifacts_dir / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("<!doctype html>")
    lines.append("<html lang=\\"en\\">")
    lines.append("<head>")
    lines.append("<meta charset=\\"utf-8\\">")
    lines.append("<meta name=\\"viewport\\" content=\\"width=device-width,initial-scale=1\\">")
    lines.append(f"<title>{{html_escape(REPORT_TITLE)}}</title>")
    lines.append("<style>")
    lines.append("body{{font-family:Arial,Helvetica,sans-serif;margin:32px;color:#1f2933}}")
    lines.append("h1,h2{{margin-bottom:8px}}")
    lines.append("p{{max-width:960px;line-height:1.5}}")
    lines.append(".meta{{color:#52606d;font-size:14px}}")
    lines.append("ul{{padding-left:20px}}")
    lines.append("li{{margin:4px 0}}")
    lines.append("</style>")
    lines.append("</head>")
    lines.append("<body>")
    lines.append(f"<h1>{{html_escape(REPORT_TITLE)}}</h1>")
    lines.append(f"<div class=\\"meta\\">Generated {{html_escape(manifest['generated_at'])}}</div>")
    lines.append(f"<div class=\\"meta\\">Stages: {{html_escape(', '.join(STAGES))}}</div>")
    if REPORT_NOTES:
        lines.append(f"<p>{{html_escape(REPORT_NOTES)}}</p>")
    if figure_assets:
        lines.append("<h2>Figures</h2>")
        for asset in figure_assets:
            link = os.path.relpath((OUTPUT_DIR / asset["path"]), report_dir).replace(os.sep, "/")
            label = asset.get("source") or asset["path"]
            ext = os.path.splitext(asset["path"])[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".svg"):
                lines.append("<div style=\\"margin-bottom:16px\\">")
                lines.append(f"<div class=\\"meta\\">{{html_escape(label)}}</div>")
                lines.append(f"<img src=\\"{{html_escape(link)}}\\" style=\\"max-width:100%;height:auto;\\">")
                lines.append("</div>")
            else:
                lines.append(f"<div><a href=\\"{{html_escape(link)}}\\">{{html_escape(label)}}</a></div>")
    if table_assets:
        lines.append("<h2>Tables</h2>")
        lines.append("<ul>")
        for asset in table_assets:
            link = os.path.relpath((OUTPUT_DIR / asset["path"]), report_dir).replace(os.sep, "/")
            label = asset.get("source") or asset["path"]
            lines.append(f"<li><a href=\\"{{html_escape(link)}}\\">{{html_escape(label)}}</a></li>")
        lines.append("</ul>")
    for group_name, paths in groups.items():
        lines.append(f"<h2>{{html_escape(group_name)}}</h2>")
        if not paths:
            lines.append("<p class=\\"meta\\">No artifacts found.</p>")
            continue
        lines.append("<ul>")
        for rel in paths:
            link = os.path.relpath((OUTPUT_DIR / rel), report_dir).replace(os.sep, "/")
            lines.append(f"<li><a href=\\"{{html_escape(link)}}\\">{{html_escape(rel)}}</a></li>")
        lines.append("</ul>")
    lines.append("</body></html>")

    report_path = report_dir / "report.html"
    report_path.write_text("\\n".join(lines), encoding="utf-8")

    pdf_path = report_dir / "report.pdf"
    if shutil.which("pandoc"):
        subprocess.run(["pandoc", str(report_path), "-o", str(pdf_path)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""" 


def build_sbatch(run_dir, run_command, slurm, output_dir, run_name):
    job_name = slurm.get("job_name", run_name)
    output_path = slurm.get("output") or os.path.join(output_dir, f"{job_name}.out")
    error_path = slurm.get("error") or os.path.join(output_dir, f"{job_name}.err")
    cpus = slurm.get("cpus_per_task", 32)
    time_limit = slurm.get("time", "72:00:00")
    mem = slurm.get("mem", "500gb")
    account = slurm.get("account")
    partition = slurm.get("partition")
    qos = slurm.get("qos")
    gres = slurm.get("gres")
    gpus = slurm.get("gpus")
    gpus_per_node = slurm.get("gpus_per_node")
    constraint = slurm.get("constraint")
    mail_user = slurm.get("mail_user")
    mail_type = slurm.get("mail_type")
    conda_env = slurm.get("conda_env")
    extra_modules = slurm.get("modules", [])
    if isinstance(extra_modules, str):
        extra_modules = [extra_modules]
    use_module_conda = slurm.get("use_module_conda")
    if use_module_conda is None:
        env_flag = os.getenv("SLURM_USE_MODULE_CONDA") or os.getenv("USE_MODULE_CONDA") or ""
        use_module_conda = env_flag.strip().lower() in ("1", "true", "yes", "on")

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={norm_path(output_path)}",
        f"#SBATCH --error={norm_path(error_path)}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        "#SBATCH --nodes=1",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={mem}",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    if gpus is not None:
        lines.append(f"#SBATCH --gpus={gpus}")
    if gpus_per_node is not None:
        lines.append(f"#SBATCH --gpus-per-node={gpus_per_node}")
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    if mail_user:
        lines.append(f"#SBATCH --mail-user={mail_user}")
    if mail_type:
        lines.append(f"#SBATCH --mail-type={mail_type}")

    lines += [
        "",
        "set -euo pipefail",
        "umask 007",
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
        "",
    ]
    if use_module_conda:
        lines.append("module load conda")
    for module_name in extra_modules:
        if module_name and module_name != "conda":
            lines.append(f"module load {module_name}")
    if conda_env:
        lines += [
            "set +u",
            'export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"',
        ]
        if use_module_conda:
            lines.append(f"conda activate {conda_env}")
        else:
            lines += [
                'if [ -z "${CONDA_BASE:-}" ]; then',
                '  if [ -n "${CONDA_EXE:-}" ]; then',
                '    CONDA_BASE=$(dirname "$(dirname "$CONDA_EXE")")',
                "  elif command -v conda >/dev/null 2>&1; then",
                "    CONDA_BASE=$(conda info --base)",
                "  fi",
                "fi",
                'if [ -z "${CONDA_BASE:-}" ]; then',
                '  echo "Conda base not found; set CONDA_BASE or enable slurm.use_module_conda"',
                "  exit 1",
                "fi",
                'source "$CONDA_BASE/etc/profile.d/conda.sh"',
                f"conda activate {conda_env}",
            ]
        lines.append("set -u")
    lines += [
        'echo "CONDA_BASE=${CONDA_BASE:-}"',
        'if command -v conda >/dev/null 2>&1; then conda info --base; conda info --envs; fi',
        'echo "PYTHON_BIN=$(command -v python || echo not-found)"',
        "python --version || true",
    ]
    lines += [
        f"cd {norm_path(str(run_dir))}",
        f"mkdir -p {norm_path(output_dir)}",
        run_command,
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate a patched pipeline script and optional SLURM wrapper.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--emit-sbatch", action="store_true", help="Write a submit.sh in the run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and generate scripts without running.")
    parser.add_argument("--validate", action="store_true", help="Validate the config and exit without writing files.")
    parser.add_argument("--skip-path-checks", action="store_true", help="Skip existence checks for input paths.")
    parser.add_argument("--run", action="store_true", help="Run the patched script locally.")
    parser.add_argument("--submit", action="store_true", help="Run sbatch on the generated submit.sh.")
    args = parser.parse_args()

    if args.validate and args.dry_run:
        raise ValueError("Use --validate or --dry-run, not both.")
    if args.validate and (args.run or args.submit):
        raise ValueError("--validate cannot be combined with --run or --submit.")
    if args.dry_run and (args.run or args.submit):
        raise ValueError("--dry-run cannot be combined with --run or --submit.")

    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    config = load_config(config_path)
    run_name = config.get("run_name")
    if not run_name:
        raise ValueError("Config must include 'run_name'.")

    if args.validate or args.dry_run:
        errors, warnings = validate_cli_config(config, root, check_paths=not args.skip_path_checks)
        report_validation(errors, warnings)
        if errors:
            raise SystemExit(1)
        if args.validate:
            return

    mode = config.get("mode", "fixed_k")
    if mode not in ALLOWED_MODES:
        raise ValueError("mode must be one of: 'fixed_k', 'elbow_k', 'poisson_redundancy_k', 'poisson_cumulative_improvement_k'.")

    stages = normalize_stages(config)
    if "cell2loc_nmf" in stages or "cell2loc" in stages:
        ensure_required(config, "reference_h5ad_path")
        ensure_required(config, "cosmx_h5ad_path")
        ensure_required(config, "cell_metadata_path")
    elif "post_nmf" in stages or "rcausal_mgm" in stages:
        ensure_required(config, "cosmx_h5ad_path")

    template_path = config.get("template_path")
    if "cell2loc_nmf" in stages or "cell2loc" in stages:
        if not template_path:
            template_path = default_template_for_mode(mode)
        template_path = resolve_template(root, template_path)
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
    else:
        template_path = resolve_template(root, config.get("template_path", default_template_for_mode(mode)))

    run_dir = config.get("run_dir") or str(root / "runs" / run_name)
    run_dir_path = Path(run_dir)
    if not run_dir_path.is_absolute():
        run_dir_path = root / run_dir_path
    run_dir_path.mkdir(parents=True, exist_ok=True)

    output_dir = config.get("output_dir") or str(run_dir_path / "outputs")
    ref_model_dir = config.get("ref_model_dir") or str(Path(output_dir) / "reference")
    ref_model_name = config.get("ref_model_name") or f"cell2location_reference_model_{run_name}"
    inf_aver_name = config.get("inf_aver_name") or f"inf_aver_{run_name}.csv"

    warnings = []

    patched_script_path = run_dir_path / f"{run_name}_pipeline.py"
    if "cell2loc_nmf" in stages or "cell2loc" in stages:
        text = template_path.read_text(encoding="utf-8")

        text = apply_assignment(text, "reference_h5ad_path", render_value(config["reference_h5ad_path"]), warnings)
        text = apply_assignment(text, "cosmx_h5ad_path", render_value(config["cosmx_h5ad_path"]), warnings)
        text = apply_assignment(text, "ref_model_dir", render_value(ref_model_dir), warnings)
        text = apply_assignment(text, "ref_model_path", f"os.path.join(ref_model_dir, {json.dumps(ref_model_name)})", warnings)
        text = apply_assignment(text, "inf_aver_csv_path", f"os.path.join(ref_model_dir, {json.dumps(inf_aver_name)})", warnings)

        text = apply_assignment(text, "output_dir", render_value(output_dir), warnings)
        text = apply_assignment(text, "nmf_output_dir", render_value(output_dir), warnings)
        text = apply_assignment(
            text,
            "c2l_train_accelerator",
            render_python_literal(config.get("cell2loc_train_accelerator", "cpu")),
            warnings,
        )
        text = apply_assignment(
            text,
            "c2l_train_devices",
            render_python_literal(config.get("cell2loc_train_devices", "auto")),
            warnings,
        )
        text = apply_assignment(
            text,
            "c2l_export_accelerator",
            render_python_literal(config.get("cell2loc_export_accelerator", config.get("cell2loc_train_accelerator", "cpu"))),
            warnings,
        )

        cell_metadata_path = norm_path(config["cell_metadata_path"])
        cell_metadata_file = os.path.basename(cell_metadata_path)
        text = apply_assignment(text, "cell_metadata_file_name", json.dumps(cell_metadata_file), warnings)
        text = apply_assignment(text, "spatial_metadata_path", render_value(cell_metadata_path), warnings)

        if mode == "fixed_k":
            n_components = config.get("n_components") or config.get("k")
            if n_components is None:
                raise ValueError("Fixed-k mode requires 'n_components' or 'k'.")
            text = apply_assignment(text, "n_components", str(int(n_components)), warnings)
        else:
            k_min = int(config.get("k_min", 2))
            k_max = int(config.get("k_max", 20))
            if k_max < k_min:
                raise ValueError("k_max must be >= k_min.")
            text = apply_assignment(text, "K_range", f"range({k_min}, {k_max + 1})", warnings)
            if mode == "poisson_redundancy_k":
                selection_method = "poisson_redundancy_k"
            elif mode == "poisson_cumulative_improvement_k":
                selection_method = "poisson_cumulative_improvement_k"
            else:
                selection_method = "elbow_k"
            text = apply_assignment(text, "nmf_selection_method", render_value(selection_method), warnings)
            if mode in ("poisson_redundancy_k", "poisson_cumulative_improvement_k"):
                if config.get("poisson_n_runs") is not None:
                    text = apply_assignment(text, "poisson_n_runs", str(int(config["poisson_n_runs"])), warnings)
                if config.get("poisson_max_iter") is not None:
                    text = apply_assignment(text, "poisson_max_iter", str(int(config["poisson_max_iter"])), warnings)
                if config.get("poisson_normalize_rows_to_sum1") is not None:
                    text = apply_assignment(
                        text,
                        "poisson_normalize_rows_to_sum1",
                        str(bool(config["poisson_normalize_rows_to_sum1"])),
                        warnings,
                    )
            if mode == "poisson_cumulative_improvement_k" and config.get("poisson_cumulative_improvement_target") is not None:
                text = apply_assignment(
                    text,
                    "poisson_cumulative_improvement_target",
                    str(float(config["poisson_cumulative_improvement_target"])),
                    warnings,
                )

        write_text(patched_script_path, text)

    stage_commands = []

    if "cell2loc_nmf" in stages:
        stage_commands.append(("cell2loc_nmf", f"python {shell_quote(patched_script_path)}"))
    else:
        if "cell2loc" in stages:
            stage_commands.append(
                (
                    "cell2loc",
                    (
                        f"NICHERUNNER_PIPELINE_STAGE={shell_quote('cell2loc')} "
                        f"python {shell_quote(patched_script_path)}"
                    ).strip(),
                )
            )
        if "nmf" in stages:
            nmf_script_source = resolve_template(root, config.get("nmf_script_path", DEFAULT_NMF_ONLY_SCRIPT))
            if not nmf_script_source.exists():
                raise FileNotFoundError(f"NMF-only script not found: {nmf_script_source}")
            nmf_text = nmf_script_source.read_text(encoding="utf-8")
            nmf_text = apply_assignment(
                nmf_text,
                "input_h5ad_path",
                render_value(str(Path(output_dir) / "cosmx_cell2loc_only.h5ad")),
                warnings,
            )
            nmf_text = apply_assignment(nmf_text, "nmf_output_dir", render_value(output_dir), warnings)
            nmf_text = apply_assignment(
                nmf_text,
                "nmf_h5ad_path",
                render_value(str(Path(output_dir) / "cosmx_with_nmf.h5ad")),
                warnings,
            )
            if mode == "fixed_k":
                n_components = config.get("n_components") or config.get("k")
                if n_components is None:
                    raise ValueError("Fixed-k mode requires 'n_components' or 'k'.")
                nmf_text = apply_assignment(nmf_text, "K_range", f"range({int(n_components)}, {int(n_components) + 1})", warnings)
                nmf_text = apply_assignment(nmf_text, "nmf_selection_method", render_value("fixed_k"), warnings)
            else:
                k_min = int(config.get("k_min", 2))
                k_max = int(config.get("k_max", 20))
                nmf_text = apply_assignment(nmf_text, "K_range", f"range({k_min}, {k_max + 1})", warnings)
                if mode == "poisson_redundancy_k":
                    selection_method = "poisson_redundancy_k"
                elif mode == "poisson_cumulative_improvement_k":
                    selection_method = "poisson_cumulative_improvement_k"
                else:
                    selection_method = "elbow_k"
                nmf_text = apply_assignment(nmf_text, "nmf_selection_method", render_value(selection_method), warnings)
            if config.get("poisson_n_runs") is not None:
                nmf_text = apply_assignment(nmf_text, "poisson_n_runs", str(int(config["poisson_n_runs"])), warnings)
            if config.get("poisson_max_iter") is not None:
                nmf_text = apply_assignment(nmf_text, "poisson_max_iter", str(int(config["poisson_max_iter"])), warnings)
            if config.get("poisson_normalize_rows_to_sum1") is not None:
                nmf_text = apply_assignment(
                    nmf_text,
                    "poisson_normalize_rows_to_sum1",
                    str(bool(config["poisson_normalize_rows_to_sum1"])),
                    warnings,
                )
            if config.get("poisson_cumulative_improvement_target") is not None:
                nmf_text = apply_assignment(
                    nmf_text,
                    "poisson_cumulative_improvement_target",
                    str(float(config["poisson_cumulative_improvement_target"])),
                    warnings,
                )
            nmf_script_copy = run_dir_path / "run_nmf_from_cell2loc.py"
            write_text(nmf_script_copy, nmf_text)
            stage_commands.append(("nmf", f"python {shell_quote(nmf_script_copy)}"))

    if "post_nmf" in stages:
        post_nmf_mode = config.get("post_nmf_mode", "papermill")
        if post_nmf_mode not in ("papermill", "python"):
            raise ValueError("post_nmf_mode must be 'papermill' or 'python'.")
        if post_nmf_mode == "papermill":
            notebook_source = resolve_template(root, config.get("post_nmf_notebook_path", DEFAULT_POST_NMF_NOTEBOOK))
            if not notebook_source.exists():
                raise FileNotFoundError(f"Post-NMF notebook not found: {notebook_source}")
            notebook_copy = copy_resource(notebook_source, run_dir_path)
            output_notebook = run_dir_path / f"{run_name}_post_nmf.ipynb"
            cosmx_with_nmf_path = config.get("cosmx_with_nmf_path") or str(Path(output_dir) / "cosmx_with_nmf.h5ad")
            parameters = {
                "OUTPUT_DIR": output_dir,
                "RUN_DIR": str(run_dir_path),
                "REFERENCE_H5AD": config.get("reference_h5ad_path", ""),
                "COSMX_H5AD": config.get("cosmx_h5ad_path", ""),
                "COSMX_WITH_NMF_H5AD": cosmx_with_nmf_path,
                "CELL_METADATA_CSV": config.get("cell_metadata_path", ""),
            }
            parameters.update(config.get("post_nmf_parameters", {}))
            stage_commands.append(
                ("post_nmf", build_papermill_command(notebook_copy, output_notebook, parameters))
            )
        else:
            script_source = config.get("post_nmf_script_path")
            if not script_source:
                raise ValueError("post_nmf_script_path is required when post_nmf_mode=python.")
            script_source = resolve_template(root, script_source)
            if not script_source.exists():
                raise FileNotFoundError(f"Post-NMF script not found: {script_source}")
            script_copy = copy_resource(script_source, run_dir_path)
            args_list = config.get("post_nmf_args", [])
            extra_args = " ".join(shell_quote(arg) for arg in args_list)
            stage_commands.append(("post_nmf", f"python {shell_quote(script_copy)} {extra_args}".strip()))

    if "rcausal_mgm" in stages:
        rcausal_mode = config.get("rcausal_mode", "papermill")
        if rcausal_mode not in ("papermill", "python"):
            raise ValueError("rcausal_mode must be 'papermill' or 'python'.")
        if rcausal_mode == "papermill":
            notebook_source = resolve_template(root, config.get("rcausal_notebook_path", DEFAULT_RCAUSAL_NOTEBOOK))
            if not notebook_source.exists():
                raise FileNotFoundError(f"RCausalMGM notebook not found: {notebook_source}")
            notebook_copy = copy_resource(notebook_source, run_dir_path)
            output_notebook = run_dir_path / f"{run_name}_rcausal_mgm.ipynb"
            parameters = {"output_dir": output_dir, "run_dir": str(run_dir_path)}
            parameters.update(config.get("rcausal_parameters", {}))
            stage_commands.append(
                ("rcausal_mgm", build_papermill_command(notebook_copy, output_notebook, parameters))
            )
        else:
            script_source = config.get("rcausal_script_path")
            if not script_source:
                raise ValueError("rcausal_script_path is required when rcausal_mode=python.")
            script_source = resolve_template(root, script_source)
            if not script_source.exists():
                raise FileNotFoundError(f"RCausalMGM script not found: {script_source}")
            script_copy = copy_resource(script_source, run_dir_path)
            support_files = config.get("rcausal_support_files", [])
            if support_files:
                copy_support_resources(root, run_dir_path, list(support_files))
            args_list = build_rcausal_args(config, output_dir)
            extra_args = " ".join(shell_quote(arg) for arg in args_list)
            stage_commands.append(("rcausal_mgm", f"python {shell_quote(script_copy)} {extra_args}".strip()))

    if "mlp" in stages:
        script_source = resolve_template(root, config.get("mlp_script_path", DEFAULT_MLP_SCRIPT))
        if not script_source.exists():
            raise FileNotFoundError(f"MLP script not found: {script_source}")
        script_copy = copy_resource(script_source, run_dir_path)
        args_list = config.get("mlp_args", [])
        extra_args = " ".join(shell_quote(arg) for arg in args_list)
        mlp_feature_scope = str(config.get("mlp_feature_scope", "patient")).strip().lower()
        mlp_output_subdir = config.get("mlp_output_subdir")
        if not mlp_output_subdir:
            mlp_output_subdir = "MLP_FOVFeatures" if mlp_feature_scope == "fov" else "MLP_44Features"
        mlp_output_dir = str(Path(output_dir) / mlp_output_subdir)
        if mlp_feature_scope == "fov":
            builder_source = resolve_template(root, config.get("mlp_input_builder_script", DEFAULT_FOV_MLP_INPUT_BUILDER))
            if not builder_source.exists():
                raise FileNotFoundError(f"FOV MLP input builder script not found: {builder_source}")
            builder_copy = copy_resource(builder_source, run_dir_path)
            mlp_input_subdir = config.get("mlp_input_subdir", "MLP_FOVFeatures_inputs")
            mlp_input_dir = str(Path(output_dir) / mlp_input_subdir)
            niche_gene_count = int(config.get("mlp_niche_gene_count_per_group", 20))
            builder_cmd = (
                f"python {shell_quote(builder_copy)} "
                f"--output-dir {shell_quote(output_dir)} "
                f"--cosmx-with-nmf {shell_quote(str(Path(output_dir) / 'cosmx_with_nmf.h5ad'))} "
                f"--dest-dir {shell_quote(mlp_input_dir)} "
                f"--niche-gene-count {niche_gene_count}"
            )
            mlp_cmd = (
                f"NICHERUNNER_OUTPUT_DIR={shell_quote(mlp_input_dir)} "
                f"NICHERUNNER_SOURCE_OUTPUT_DIR={shell_quote(output_dir)} "
                f"NICHERUNNER_MLP_UNIT={shell_quote('fov')} "
                f"NICHERUNNER_MLP_OUTPUT_DIR={shell_quote(mlp_output_dir)} "
                f"python {shell_quote(script_copy)} {extra_args}"
            ).strip()
            mlp_stage_script = run_dir_path / "run_mlp_stage.sh"
            write_text(
                mlp_stage_script,
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        builder_cmd,
                        mlp_cmd,
                        "",
                    ]
                ),
            )
            stage_commands.append(("mlp", f"bash {shell_quote(mlp_stage_script)}"))
        else:
            stage_commands.append(
                (
                    "mlp",
                    (
                        f"NICHERUNNER_OUTPUT_DIR={shell_quote(output_dir)} "
                        f"NICHERUNNER_SOURCE_OUTPUT_DIR={shell_quote(output_dir)} "
                        f"NICHERUNNER_MLP_UNIT={shell_quote('patient')} "
                        f"NICHERUNNER_MLP_OUTPUT_DIR={shell_quote(mlp_output_dir)} "
                        f"python {shell_quote(script_copy)} {extra_args}"
                    ).strip(),
                )
            )

    report_script_path = None
    if "report" in stages:
        report_title = config.get("report_title", DEFAULT_REPORT_TITLE)
        report_notes = config.get("report_notes", "")
        report_script_path = run_dir_path / "generate_report.py"
        write_text(
            report_script_path,
            build_report_script(Path(output_dir), stages, report_title, report_notes, run_name),
        )
        stage_commands.append(("report", f"python {shell_quote(report_script_path)}"))

    if not stage_commands:
        raise ValueError("No stages selected; provide at least one stage.")

    run_script_path = run_dir_path / "run.sh"
    write_text(run_script_path, build_run_script(run_dir_path, Path(output_dir), stage_commands))

    resolved_config = dict(config)
    resolved_config.update(
        {
            "mode": mode,
            "run_dir": str(run_dir_path),
            "output_dir": output_dir,
            "ref_model_dir": ref_model_dir,
            "ref_model_name": ref_model_name,
            "inf_aver_name": inf_aver_name,
            "template_path": str(template_path),
            "patched_script": str(patched_script_path),
            "run_script": str(run_script_path),
            "stages": stages,
        }
    )
    if report_script_path:
        resolved_config["report_script"] = str(report_script_path)
    write_text(run_dir_path / "config.resolved.json", json.dumps(resolved_config, indent=2))

    emit_sbatch = args.emit_sbatch or bool(config.get("slurm", {}).get("enabled"))
    submit_path = run_dir_path / "submit.sh"
    if emit_sbatch:
        slurm = config.get("slurm", {})
        run_command = f"bash {shell_quote(run_script_path)}"
        sbatch_text = build_sbatch(run_dir_path, run_command, slurm, output_dir, run_name)
        write_text(submit_path, sbatch_text)

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    print(f"Patched script: {patched_script_path}")
    print(f"Run script: {run_script_path}")
    print(f"Resolved config: {run_dir_path / 'config.resolved.json'}")
    if emit_sbatch:
        print(f"SLURM script: {submit_path}")

    if args.run:
        os.system(f"bash {run_script_path}")

    if args.submit:
        if not emit_sbatch:
            raise ValueError("submit requested but SLURM script was not created. Use --emit-sbatch or enable slurm in config.")
        os.system(f"sbatch {submit_path}")


if __name__ == "__main__":
    main()
