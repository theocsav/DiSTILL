from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import subprocess

import pandas as pd
import pydot

DISEASE_NODE = "Disease.Health.State"
SUPPORT_COLS = [
    "undir",
    "right.dir",
    "left.dir",
    "bidir",
    "right.partdir",
    "left.partdir",
    "nondir",
]


@dataclass
class GraphSpec:
    csv_path: Path
    output_dir: Path
    stem: str
    disease_min_support: float
    edge_min_support: float
    neighborhood_mode: bool = False
    niche_gene_mode: bool = False


def canonical_interaction(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def edge_support(row: pd.Series) -> float:
    return float(row[SUPPORT_COLS].max())


def is_disease_edge(row: pd.Series) -> bool:
    return row["var1"] == DISEASE_NODE or row["var2"] == DISEASE_NODE


def clean_label(name: str, neighborhood_mode: bool, niche_gene_mode: bool = False) -> str:
    if name == DISEASE_NODE:
        return "Disease State"
    if neighborhood_mode and name.startswith("enrichment_"):
        return name.replace("enrichment_", "E").replace(".", "-")
    if niche_gene_mode and name.startswith("niche_") and "_gene_" in name:
        prefix, gene = name.split("_gene_", 1)
        niche_id = prefix.replace("niche_", "N")
        return f"{niche_id}: {gene}"
    if name.startswith("Niche_"):
        return name.replace("Niche_", "N")
    return name


def select_edges(df: pd.DataFrame, spec: GraphSpec) -> pd.DataFrame:
    df = df.copy()
    df["interaction"] = df["interaction"].apply(canonical_interaction)
    df["max_support"] = df.apply(edge_support, axis=1)

    disease_edges = (
        df[df.apply(is_disease_edge, axis=1) & (df["max_support"] >= spec.disease_min_support)]
        .sort_values("max_support", ascending=False)
        .copy()
    )

    selected_indices: list[int] = list(disease_edges.index)
    seen_index = set(selected_indices)
    frontier = {
        node
        for _, row in disease_edges.iterrows()
        for node in (row["var1"], row["var2"])
        if node != DISEASE_NODE
    }

    candidate_pool = (
        df[(df["interaction"] != "") & (df["max_support"] >= spec.edge_min_support)]
        .sort_values("max_support", ascending=False)
        .copy()
    )

    while frontier:
        hop_candidates = candidate_pool[
            ~candidate_pool.index.isin(seen_index)
            & (
                candidate_pool["var1"].isin(frontier)
                | candidate_pool["var2"].isin(frontier)
            )
        ]
        if hop_candidates.empty:
            break

        next_frontier: set[str] = set()
        for idx, row in hop_candidates.iterrows():
            selected_indices.append(idx)
            seen_index.add(idx)
            for node in (row["var1"], row["var2"]):
                if node != DISEASE_NODE and node not in frontier:
                    next_frontier.add(node)
        frontier = next_frontier

    rows = (
        df.loc[selected_indices, ["var1", "interaction", "var2", "max_support"]]
        .drop_duplicates(subset=["var1", "interaction", "var2"])
        .sort_values(["max_support", "var1", "var2"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return rows


def build_layers(rows: pd.DataFrame) -> dict[str, int]:
    adjacency: dict[str, set[str]] = {}
    nodes = set(rows["var1"]).union(set(rows["var2"]))
    for node in nodes:
        adjacency[node] = set()
    for _, row in rows.iterrows():
        adjacency[row["var1"]].add(row["var2"])
        adjacency[row["var2"]].add(row["var1"])

    layers = {DISEASE_NODE: 0}
    queue = deque([DISEASE_NODE])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor not in layers:
                layers[neighbor] = layers[node] + 1
                queue.append(neighbor)

    for node in nodes:
        layers.setdefault(node, 99)
    return layers


def edge_style(interaction: str, support: float) -> dict[str, str]:
    width = f"{max(1.8, 1.2 + 3.6 * support):.2f}"
    base = {
        "color": "#5B6470",
        "penwidth": width,
        "arrowsize": "0.9",
        "fontname": "Helvetica",
    }
    mapping = {
        "-->": {"dir": "forward", "arrowhead": "normal", "arrowtail": "none", "color": "#385E8A"},
        "<--": {"dir": "back", "arrowhead": "normal", "arrowtail": "none", "color": "#385E8A"},
        "<->": {"dir": "both", "arrowhead": "normal", "arrowtail": "normal", "color": "#5A3E8C"},
        "o->": {"dir": "both", "arrowhead": "normal", "arrowtail": "odot", "color": "#8A6B16"},
        "<-o": {"dir": "both", "arrowhead": "odot", "arrowtail": "normal", "color": "#8A6B16"},
        "o-o": {"dir": "both", "arrowhead": "odot", "arrowtail": "odot", "color": "#7D7D7D", "style": "dashed"},
        "": {
            "dir": "none",
            "arrowhead": "none",
            "arrowtail": "none",
            "color": "#6F7782",
            "style": "dashed",
            "penwidth": f"{max(4.6, 3.2 + 4.2 * support):.2f}",
        },
    }
    base.update(mapping.get(interaction, mapping[""]))
    return base


def add_rank_subgraphs(graph: pydot.Dot, rows: pd.DataFrame) -> None:
    layers = build_layers(rows)
    max_layer = max((layer for layer in layers.values() if layer < 99), default=0)
    for level in range(0, max_layer + 1):
        nodes = sorted([node for node, layer in layers.items() if layer == level])
        if not nodes:
            continue
        subgraph = pydot.Subgraph(rank="same")
        for node in nodes:
            subgraph.add_node(pydot.Node(node))
        graph.add_subgraph(subgraph)


def build_dot(rows: pd.DataFrame, spec: GraphSpec) -> pydot.Dot:
    graph = pydot.Dot("G", graph_type="digraph")
    graph.set_rankdir("LR")
    graph.set_splines("spline")
    graph.set_bgcolor("white")
    graph.set_overlap("false")
    graph.set_outputorder("edgesfirst")
    graph.set_nodesep("0.62")
    graph.set_ranksep("1.15")
    graph.set_pad("0.35")
    graph.set_margin("0.16")
    graph.set_fontname("Helvetica")

    nodes = sorted(set(rows["var1"]).union(set(rows["var2"])))
    for node in nodes:
        if node == DISEASE_NODE:
            attrs = dict(
                label="Disease State",
                shape="box",
                style="filled,rounded,bold",
                fillcolor="#F7E29A",
                color="#8A5A00",
                penwidth="2.8",
                fontsize="26",
                fontname="Helvetica-Bold",
                margin="0.22,0.14",
                width="2.35",
            )
        elif spec.neighborhood_mode:
            attrs = dict(
                label=clean_label(node, True, False),
                shape="box",
                style="filled,rounded",
                fillcolor="#E8F3E2",
                color="#3F6A3A",
                penwidth="1.9",
                fontsize="18",
                fontname="Helvetica-Bold",
                margin="0.18,0.12",
            )
        elif spec.niche_gene_mode:
            attrs = dict(
                label=clean_label(node, False, True),
                shape="box",
                style="filled,rounded",
                fillcolor="#E5EEFB",
                color="#355F95",
                penwidth="1.9",
                fontsize="17",
                fontname="Helvetica-Bold",
                margin="0.18,0.12",
            )
        else:
            attrs = dict(
                label=clean_label(node, False, False),
                shape="circle",
                style="filled",
                fillcolor="#E5EEFB",
                color="#355F95",
                penwidth="2.0",
                fontsize="21",
                fontname="Helvetica-Bold",
                width="1.10",
                height="1.10",
                fixedsize="true",
            )
        graph.add_node(pydot.Node(node, **attrs))

    add_rank_subgraphs(graph, rows)

    for _, row in rows.iterrows():
        attrs = edge_style(row["interaction"], row["max_support"])
        graph.add_edge(pydot.Edge(row["var1"], row["var2"], **attrs))

    return graph


def render_graphviz(dot_path: Path, png_path: Path, svg_path: Path, pdf_path: Path) -> None:
    dot_bin = Path(r"C:\Program Files\Graphviz\bin\dot.exe")
    if not dot_bin.exists():
        raise FileNotFoundError("Graphviz dot.exe was not found.")

    subprocess.run([str(dot_bin), "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
    subprocess.run([str(dot_bin), "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    subprocess.run([str(dot_bin), "-Tpdf", str(dot_path), "-o", str(pdf_path)], check=True)


def render(spec: GraphSpec) -> None:
    df = pd.read_csv(spec.csv_path)
    rows = select_edges(df, spec)
    dot = build_dot(rows, spec)

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    dot_path = spec.output_dir / f"{spec.stem}.dot"
    png_path = spec.output_dir / f"{spec.stem}.png"
    svg_path = spec.output_dir / f"{spec.stem}.svg"
    pdf_path = spec.output_dir / f"{spec.stem}.pdf"
    csv_path = spec.output_dir / f"{spec.stem}_selected_edges.csv"

    dot_path.write_text(dot.to_string(), encoding="utf-8")
    rows.to_csv(csv_path, index=False)
    render_graphviz(dot_path, png_path, svg_path, pdf_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render prettier filtered RCausal graphs.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--disease-min-support", type=float, required=True)
    parser.add_argument("--edge-min-support", type=float, required=True)
    parser.add_argument("--neighborhood-mode", action="store_true")
    parser.add_argument("--niche-gene-mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = GraphSpec(
        csv_path=Path(args.csv_path),
        output_dir=Path(args.output_dir),
        stem=args.stem,
        disease_min_support=args.disease_min_support,
        edge_min_support=args.edge_min_support,
        neighborhood_mode=args.neighborhood_mode,
        niche_gene_mode=args.niche_gene_mode,
    )
    render(spec)


if __name__ == "__main__":
    main()
