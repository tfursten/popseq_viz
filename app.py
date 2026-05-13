#!/usr/bin/env python3

import io
import math
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from Bio import Phylo
from plotly.colors import qualitative


st.set_page_config(page_title="Node Frequency Tree Viewer", layout="wide")
st.title("Node Frequency Tree Viewer")

REQUIRED_FREQ_COLUMNS = [
    "Sample", "node", "child_node", "true_pos", "relative_pos", "node_site",
    "derived_allele", "derived_allele_frequency", "edge_state1", "edge_state2",
    "chrom_matrix", "chrom", "pos", "ref", "depth", "A", "C", "G", "T",
    "in_lofreq_vcf",
]

PALETTE = (
    qualitative.Plotly
    + qualitative.Dark24
    + qualitative.Light24
    + qualitative.Alphabet
)


def read_table(uploaded_file):
    name = uploaded_file.name.lower()
    sep = "\t" if name.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(uploaded_file, sep=sep)


def guess_tree_format(text, source_name=""):
    name = str(source_name or "").lower()
    body = str(text or "").lstrip().lower()
    if name.endswith((".nex", ".nexus")) or body.startswith("#nexus") or "begin trees;" in body:
        return "nexus"
    return "newick"


def parse_tree_text(text, source_name=""):
    if not str(text or "").strip():
        raise ValueError("Tree input is empty.")

    preferred = guess_tree_format(text, source_name)
    formats = [preferred] + [fmt for fmt in ["newick", "nexus"] if fmt != preferred]
    errors = []
    for fmt in formats:
        try:
            trees = list(Phylo.parse(io.StringIO(text), fmt))
        except Exception as exc:
            errors.append(f"{fmt}: {exc}")
            continue
        if trees:
            return trees[0], fmt
    raise ValueError("Could not parse tree as Newick or Nexus. " + " | ".join(errors))


def read_tree(uploaded_file, pasted_text):
    if str(pasted_text or "").strip():
        return parse_tree_text(pasted_text, "pasted tree")
    text = uploaded_file.getvalue().decode("utf-8")
    return parse_tree_text(text, uploaded_file.name)


def clade_name(clade):
    return str(clade.name) if getattr(clade, "name", None) else ""


def normalize_name(value):
    return str(value).strip()


def normalize_position_value(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""

    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric) and np.isfinite(float(numeric)) and float(numeric).is_integer():
        return str(int(float(numeric)))
    return text


def parse_position_filter(text):
    cleaned = str(text or "").replace(",", " ").replace(";", " ")
    return {
        pos
        for pos in (normalize_position_value(token) for token in cleaned.split())
        if pos
    }


def default_metadata_columns(columns):
    columns = list(columns)
    preferred = ["week", "Person", "BodySite", "ST"]
    selected = []
    used = set()
    lower_to_column = {str(col).lower(): col for col in columns}

    for name in preferred:
        match = lower_to_column.get(name.lower())
        if match is not None and match not in used:
            selected.append(match)
            used.add(match)

    for col in columns:
        if len(selected) >= len(preferred):
            break
        if col not in used:
            selected.append(col)
            used.add(col)

    return selected[:len(preferred)]


def branch_length(clade):
    val = getattr(clade, "branch_length", None)
    try:
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


def capped_branch_length(clade, cap):
    val = branch_length(clade)
    if cap is None or cap <= 0:
        return val
    return min(val, float(cap))


def transform_x(value, max_x, mode, boost):
    value = float(value)
    if mode == "log1p":
        value = math.log1p(value)
    elif mode == "sqrt":
        value = math.sqrt(max(value, 0.0))

    if boost and boost > 0 and max_x > 0:
        k = float(boost)
        value = max_x * (math.log1p(k * (value / max_x)) / math.log1p(k))
    return value


def assign_tree_layout(tree, mode="phylogram", branch_cap=30.0, x_transform="linear", boost=0.0, y_spacing=1.0):
    tips = tree.get_terminals()
    y_pos = {tip: float(len(tips) - i) * float(y_spacing) for i, tip in enumerate(tips)}

    def assign_y(clade):
        for child in clade.clades:
            assign_y(child)
        if clade.clades:
            y_pos[clade] = sum(y_pos[c] for c in clade.clades) / len(clade.clades)

    assign_y(tree.root)

    raw_x = {}

    def walk_x(clade, parent_x=0.0):
        if clade is tree.root:
            x = 0.0
        elif mode == "cladogram":
            x = parent_x + 1.0
        else:
            x = parent_x + capped_branch_length(clade, branch_cap)
        raw_x[clade] = x
        for child in clade.clades:
            walk_x(child, x)

    walk_x(tree.root)
    max_x = max(raw_x.values()) if raw_x else 1.0
    x_pos = {clade: transform_x(x, max_x, x_transform, boost) for clade, x in raw_x.items()}
    return x_pos, y_pos


def node_label(clade, internal_index):
    return clade_name(clade) or f"internal_{internal_index}"


def build_tree_records(tree, x_pos, y_pos):
    rows = []
    internal_idx = 0
    root_label = node_label(tree.root, internal_idx)
    if not tree.root.is_terminal():
        internal_idx += 1
    rows.append({
        "node": root_label,
        "node_norm": normalize_name(root_label),
        "is_terminal": tree.root.is_terminal(),
        "x": x_pos[tree.root],
        "bar_x": x_pos[tree.root],
        "y": y_pos[tree.root],
        "distance": 0.0,
    })

    for parent in tree.find_clades(order="preorder"):
        for child in parent.clades:
            label = node_label(child, internal_idx)
            if not child.is_terminal():
                internal_idx += 1
            rows.append({
                "node": label,
                "node_norm": normalize_name(label),
                "is_terminal": child.is_terminal(),
                "x": x_pos[child],
                "bar_x": (x_pos[parent] + x_pos[child]) / 2.0,
                "y": y_pos[child],
                "distance": branch_length(child),
            })
    return pd.DataFrame(rows)


def branch_segments(tree, x_pos, y_pos):
    horiz = []
    vert = []
    labels = []
    internal_idx = 0
    for parent in tree.find_clades(order="preorder"):
        if parent.clades:
            ys = [y_pos[c] for c in parent.clades]
            vert.append((x_pos[parent], min(ys), x_pos[parent], max(ys)))
        for child in parent.clades:
            label = node_label(child, internal_idx)
            if not child.is_terminal():
                internal_idx += 1
            dist = branch_length(child)
            horiz.append((x_pos[parent], y_pos[child], x_pos[child], y_pos[child], label, dist))
            labels.append(((x_pos[parent] + x_pos[child]) / 2, y_pos[child], dist, label))
    return horiz, vert, labels


def leaf_table(tree, x_pos, y_pos):
    return pd.DataFrame([
        {"tip": clade_name(tip), "x": x_pos[tip], "y": y_pos[tip]}
        for tip in tree.get_terminals()
    ])


def color_map(values):
    vals = sorted(pd.Series(values).dropna().astype(str).unique())
    return {val: PALETTE[i % len(PALETTE)] for i, val in enumerate(vals)}


def merge_tip_metadata(tips_df, metadata_df, key_col):
    out = tips_df.copy()
    if metadata_df is None or key_col is None:
        return out
    md = metadata_df.copy()
    md[key_col] = md[key_col].astype(str)
    out["tip"] = out["tip"].astype(str)
    return out.merge(md, left_on="tip", right_on=key_col, how="left")


def metadata_hover(row, metadata_cols):
    parts = [f"Tip: {row.get('tip', '')}"]
    for col in metadata_cols:
        if col in row.index:
            val = row.get(col)
            parts.append(f"{col}: {'' if pd.isna(val) else val}")
    return "<br>".join(parts)


def colored_tip_label(row, label_cols, value_color_maps):
    if not label_cols:
        return str(row.get("tip", ""))
    pieces = []
    for col in label_cols:
        value = row.get(col, "") if col in row.index else ""
        if pd.isna(value) or str(value) == "":
            continue
        color = value_color_maps.get(col, {}).get(str(value), "#111")
        pieces.append(f"<span style='color:{color}'>{value}</span>")
    return " | ".join(pieces) if pieces else str(row.get("tip", ""))


def metadata_label_from_row(row, label_cols, fallback, value_color_maps=None):
    value_color_maps = value_color_maps or {}
    pieces = []
    for col in label_cols:
        if col not in row.index:
            continue
        value = row.get(col)
        if pd.isna(value) or str(value) == "":
            continue
        color = value_color_maps.get(col, {}).get(str(value), "#111")
        pieces.append(f"<span style='color:{color}'>{html.escape(str(value))}</span>")
    return " | ".join(pieces) if pieces else html.escape(str(fallback))


def validate_frequency_table(df):
    missing = [c for c in REQUIRED_FREQ_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Frequency table is missing required columns: " + ", ".join(missing))


def normalize_frequency_table(df):
    validate_frequency_table(df)
    out = df.copy()
    out["Sample"] = out["Sample"].astype(str)
    out["node"] = out["node"].astype(str)
    out["node_norm"] = out["node"].map(normalize_name)
    out["true_pos"] = out["true_pos"].map(normalize_position_value)
    out["relative_pos_num"] = pd.to_numeric(out["relative_pos"], errors="coerce")
    out["frequency"] = pd.to_numeric(out["derived_allele_frequency"], errors="coerce")
    out["depth_num"] = pd.to_numeric(out["depth"], errors="coerce")
    out["lofreq"] = pd.to_numeric(out["in_lofreq_vcf"], errors="coerce").fillna(0).astype(int)
    out["derived_reads"] = out.apply(
        lambda row: pd.to_numeric(row.get(str(row.get("derived_allele", "")).upper()), errors="coerce"),
        axis=1,
    )
    return out


def build_sample_table(freq_df, metadata_df=None, sample_key=None):
    samples = pd.DataFrame({"Sample": sorted(freq_df["Sample"].dropna().astype(str).unique())})
    if metadata_df is None or sample_key is None or sample_key not in metadata_df.columns:
        return samples
    md = metadata_df.copy()
    md[sample_key] = md[sample_key].astype(str)
    return samples.merge(md, left_on="Sample", right_on=sample_key, how="left")


def is_categorical_filter_column(series):
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False

    non_missing = series.dropna().astype(str).str.strip()
    non_missing = non_missing[non_missing != ""]
    if non_missing.empty:
        return True

    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    return not numeric_values.notna().all()


def apply_frequency_filters(
    freq_df,
    samples,
    hidden_nodes,
    overlap_all,
    lofreq_only,
    min_depth,
    position_filter_mode="All",
    true_positions=None,
):
    out = freq_df[freq_df["Sample"].isin(samples)].copy()
    if hidden_nodes:
        hidden = {normalize_name(n) for n in hidden_nodes}
        out = out[~out["node_norm"].isin(hidden)]
    true_positions = {normalize_position_value(pos) for pos in (true_positions or set())}
    true_positions = {pos for pos in true_positions if pos}
    if true_positions and position_filter_mode == "Include":
        out = out[out["true_pos"].isin(true_positions)]
    elif true_positions and position_filter_mode == "Exclude":
        out = out[~out["true_pos"].isin(true_positions)]
    out = out[out["depth_num"].fillna(-1) >= float(min_depth)]
    if lofreq_only:
        out = out[out["lofreq"] == 1]
    out = out[out["frequency"].notna()]

    if overlap_all and samples:
        needed = len(set(samples))
        keep = out.groupby(["node_norm", "true_pos"])["Sample"].nunique()
        keep = keep[keep == needed].reset_index()[["node_norm", "true_pos"]]
        out = out.merge(keep, on=["node_norm", "true_pos"], how="inner")
    return out


def position_order_for_node(
    node_df,
    selected_samples,
    sort_mode,
    sort_samples,
    consensus_min_contrast=0.0,
    consensus_power=1.0,
):
    pos = node_df[["true_pos", "relative_pos_num"]].drop_duplicates()
    if pos.empty:
        return []

    if sort_mode == "relative position":
        return pos.sort_values(["relative_pos_num", "true_pos"], kind="stable")["true_pos"].astype(str).tolist()

    if sort_mode == "sample priority":
        sort_samples = [s for s in sort_samples if s in selected_samples]
        if not sort_samples:
            sort_samples = list(selected_samples)
        pivot = node_df.pivot_table(index="true_pos", columns="Sample", values="frequency", aggfunc="mean")
        pivot = pivot.reindex(columns=sort_samples)
        order = pos.copy()
        for sample in sort_samples:
            order[sample] = order["true_pos"].map(pivot[sample])
        return order.sort_values(
            sort_samples + ["relative_pos_num", "true_pos"],
            ascending=[False] * len(sort_samples) + [True, True],
            na_position="last",
            kind="stable",
        )["true_pos"].astype(str).tolist()

    if sort_mode == "mean frequency":
        summary = node_df.groupby("true_pos", as_index=False).agg(
            mean_frequency=("frequency", "mean"),
            value_count=("frequency", "count"),
        )
        order = pos.merge(summary, on="true_pos", how="left")
        return order.sort_values(
            ["mean_frequency", "value_count", "relative_pos_num", "true_pos"],
            ascending=[False, False, True, True],
            na_position="last",
            kind="stable",
        )["true_pos"].astype(str).tolist()

    if sort_mode == "consensus seriation":
        freq_pivot = node_df.pivot_table(index="true_pos", columns="Sample", values="frequency", aggfunc="mean")
        depth_pivot = node_df.pivot_table(index="true_pos", columns="Sample", values="depth_num", aggfunc="mean")
        freq_pivot = freq_pivot.reindex(columns=selected_samples)
        depth_pivot = depth_pivot.reindex(columns=selected_samples)
        transformed = np.arcsin(np.sqrt(freq_pivot.clip(lower=0.0, upper=1.0)))
        weights = depth_pivot.where(depth_pivot > 0, 1.0).fillna(0.0)
        weighted_sum = (transformed.fillna(0.0) * weights).sum(axis=1)
        weight_sum = weights.where(transformed.notna(), 0.0).sum(axis=1)
        weighted_score = weighted_sum / weight_sum.replace(0.0, np.nan)
        value_count = transformed.notna().sum(axis=1)
        rel_pos = pos.set_index("true_pos")["relative_pos_num"]
        order_df = pd.DataFrame({
            "true_pos": transformed.index.astype(str),
            "weighted_score": weighted_score.fillna(-1.0),
            "value_count": value_count,
            "relative_pos_num": transformed.index.map(rel_pos),
        }).reset_index(drop=True)
        order = order_df.sort_values(
            ["weighted_score", "value_count", "relative_pos_num", "true_pos"],
            ascending=[False, False, True, True],
            kind="stable",
            na_position="last",
        )["true_pos"].tolist()

        x_lookup = transformed.to_dict(orient="index")
        n_lookup = depth_pivot.to_dict(orient="index")
        min_contrast = float(consensus_min_contrast)
        contrast_power = float(consensus_power)

        def contrast_value(diff):
            sign = 1.0 if diff >= 0 else -1.0
            magnitude = abs(float(diff))
            if magnitude <= min_contrast:
                return 0.0
            return sign * ((magnitude - min_contrast) ** contrast_power)

        def pair_support(left_pos, right_pos):
            support = 0.0
            left_x = x_lookup.get(left_pos, {})
            right_x = x_lookup.get(right_pos, {})
            left_n = n_lookup.get(left_pos, {})
            right_n = n_lookup.get(right_pos, {})
            for sample in selected_samples:
                x_left = left_x.get(sample)
                x_right = right_x.get(sample)
                if pd.isna(x_left) or pd.isna(x_right):
                    continue
                n_left = left_n.get(sample)
                n_right = right_n.get(sample)
                if pd.isna(n_left) or float(n_left) <= 0:
                    n_left = 1.0
                if pd.isna(n_right) or float(n_right) <= 0:
                    n_right = 1.0
                weight = 1.0 / ((1.0 / float(n_left)) + (1.0 / float(n_right)))
                support += weight * contrast_value(float(x_left) - float(x_right))
            return support

        for _ in range(max(1, len(order))):
            swapped = False
            for idx in range(len(order) - 1):
                if pair_support(order[idx], order[idx + 1]) < 0:
                    order[idx], order[idx + 1] = order[idx + 1], order[idx]
                    swapped = True
            if not swapped:
                break
        return order

    return pos.sort_values(["relative_pos_num", "true_pos"], kind="stable")["true_pos"].astype(str).tolist()


def add_metadata_tracks(fig, merged_tips, metadata_cols, start_x, spacing, marker_size):
    legend_seen = set()
    header_y = float(merged_tips["y"].max()) + 1.2 if not merged_tips.empty else 1.0
    for idx, col in enumerate(metadata_cols):
        x = start_x + idx * spacing
        cmap = color_map(merged_tips[col])
        fig.add_annotation(x=x, y=header_y, text=f"<b>{col}</b>", showarrow=False, textangle=-30, font=dict(size=11))
        for value, color in cmap.items():
            sub = merged_tips[merged_tips[col].astype(str) == value]
            key = (col, value)
            fig.add_trace(go.Scatter(
                x=[x] * len(sub),
                y=sub["y"],
                mode="markers",
                marker=dict(size=marker_size, color=color, line=dict(color="#333", width=0.4)),
                name=f"{col}: {value}",
                legendgroup=col,
                showlegend=key not in legend_seen,
                customdata=[metadata_hover(row, metadata_cols) for _, row in sub.iterrows()],
                hovertemplate="%{customdata}<extra></extra>",
            ))
            legend_seen.add(key)


def add_frequency_bars(fig, freq_df, node_df, selected_samples, sample_colors, sample_labels,
                       sort_mode, sort_samples, row_gap, bar_gap, bar_width,
                       bar_height, min_frequency_to_plot, x_shift,
                       consensus_min_contrast=0.0, consensus_power=1.0):
    max_x = None
    if freq_df.empty or not selected_samples:
        return max_x

    node_lookup = node_df.set_index("node_norm").to_dict("index")
    for node_norm, node_freq in freq_df.groupby("node_norm", sort=False):
        if node_norm not in node_lookup:
            continue
        anchor = node_lookup[node_norm]
        x_anchor = float(anchor.get("bar_x", anchor["x"])) + float(x_shift)
        y_anchor = float(anchor["y"])
        order = position_order_for_node(
            node_freq,
            selected_samples,
            sort_mode,
            sort_samples,
            consensus_min_contrast=consensus_min_contrast,
            consensus_power=consensus_power,
        )
        if not order:
            continue
        pos_to_idx = {pos: i for i, pos in enumerate(order)}
        x_start = x_anchor - (len(order) - 1) * bar_gap / 2.0
        node_right = x_start + (len(order) - 1) * bar_gap + bar_width
        max_x = node_right if max_x is None else max(max_x, node_right)

        plotted_sample_idx = 0
        for sample in selected_samples:
            sample_df = node_freq[node_freq["Sample"] == sample].copy()
            if sample_df.empty:
                continue
            sample_df = sample_df[sample_df["frequency"] >= float(min_frequency_to_plot)].copy()
            if sample_df.empty:
                continue
            sample_df["_order"] = sample_df["true_pos"].astype(str).map(pos_to_idx)
            sample_df = sample_df[sample_df["_order"].notna()].sort_values("_order", kind="stable")
            baseline = y_anchor - (plotted_sample_idx + 1) * row_gap
            plotted_sample_idx += 1
            xs = x_start + sample_df["_order"].astype(float) * bar_gap
            heights = sample_df["frequency"].apply(
                lambda v: 0.0 if float(v) == 0.0 else max(float(v), 0.01) * bar_height
            ).clip(upper=bar_height)
            color = sample_colors.get(sample, "#1f77b4")

            fig.add_trace(go.Bar(
                x=xs,
                y=[bar_height] * len(sample_df),
                base=[baseline] * len(sample_df),
                width=bar_width,
                marker=dict(color="rgba(0,0,0,0)", line=dict(color="#8a8a8a", width=0.8)),
                showlegend=False,
                customdata=sample_df[["node", "Sample", "true_pos", "relative_pos", "derived_allele", "frequency", "depth", "derived_reads", "lofreq"]].values,
                hovertemplate=(
                    "Node: %{customdata[0]}<br>Sample: %{customdata[1]}<br>"
                    "Position: %{customdata[2]} (relative %{customdata[3]})<br>"
                    "Allele: %{customdata[4]}<br>Frequency: %{customdata[5]:.4f}<br>"
                    "Depth: %{customdata[6]}<br>Derived reads: %{customdata[7]}<br>"
                    "LoFreq support: %{customdata[8]}<extra></extra>"
                ),
            ))
            fig.add_trace(go.Bar(
                x=xs,
                y=heights,
                base=[baseline] * len(sample_df),
                width=bar_width,
                marker=dict(color=color, line=dict(color="#222", width=0.4)),
                name=sample,
                showlegend=False,
                customdata=sample_df[["node", "Sample", "true_pos", "relative_pos", "derived_allele", "frequency", "depth", "derived_reads", "lofreq"]].values,
                hovertemplate=(
                    "Node: %{customdata[0]}<br>Sample: %{customdata[1]}<br>"
                    "Position: %{customdata[2]} (relative %{customdata[3]})<br>"
                    "Allele: %{customdata[4]}<br>Frequency: %{customdata[5]:.4f}<br>"
                    "Depth: %{customdata[6]}<br>Derived reads: %{customdata[7]}<br>"
                    "LoFreq support: %{customdata[8]}<extra></extra>"
                ),
            ))
            fig.add_annotation(
                x=x_start - bar_gap,
                y=baseline + 0.5,
                text=f"{anchor['node']} | {sample_labels.get(sample, sample)}",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(size=10, color=color),
            )
    return max_x


with st.sidebar:
    st.header("Inputs")
    tree_file = st.file_uploader("Newick/Nexus tree with labeled nodes", type=["nwk", "newick", "nex", "nexus", "tree", "txt"])
    tree_text = st.text_area("Or paste tree text", height=140)
    metadata_file = st.file_uploader("Metadata table", type=["csv", "tsv", "txt"])
    freq_file = st.file_uploader("Population frequency table", type=["csv", "tsv", "txt"])

if tree_file is None and not str(tree_text or "").strip():
    st.info("Upload or paste a Newick/Nexus tree to begin.")
    st.stop()

try:
    tree, parsed_format = read_tree(tree_file, tree_text)
except Exception as exc:
    st.error(f"Could not parse tree: {exc}")
    st.stop()

metadata_df = read_table(metadata_file) if metadata_file is not None else None
freq_df = None
if freq_file is not None:
    try:
        freq_df = normalize_frequency_table(read_table(freq_file))
    except Exception as exc:
        st.error(f"Could not parse frequency table: {exc}")
        st.stop()

with st.sidebar:
    st.header("Tree")
    display_mode = st.radio("Tree mode", ["phylogram", "cladogram"], horizontal=True)
    branch_cap = st.number_input("Cap plotted branch lengths over", min_value=0.0, value=30.0, step=1.0)
    x_transform = st.selectbox("X transform", ["linear", "log1p", "sqrt"])
    short_branch_boost = st.slider("Short branch boost", 0.0, 50.0, 0.0, 0.5)
    vertical_spacing = st.slider("Vertical node spacing", 0.5, 5.0, 1.0, 0.05)
    show_tip_labels = st.checkbox("Show tip labels", value=True)
    show_node_labels = st.checkbox("Show node labels", value=True)
    show_branch_distances = st.checkbox("Show branch distances", value=True)
    font_size = st.slider("Label font size", 8, 24, 11)
    tip_label_offset = st.slider("Tip label offset", 0.0, 5.0, 0.25, 0.05)
    node_label_x_offset = st.slider("Node label horizontal shift", -10.0, 10.0, 0.0, 0.05)
    node_label_y_offset = st.slider("Node label vertical offset", -2.0, 2.0, 0.25, 0.05)
    branch_distance_x_offset = st.slider("Branch distance horizontal shift", -10.0, 10.0, 0.0, 0.05)
    branch_distance_y_offset = st.slider("Branch distance vertical offset", -2.0, 2.0, 0.18, 0.05)

x_pos, y_pos = assign_tree_layout(
    tree,
    display_mode,
    branch_cap,
    x_transform,
    short_branch_boost,
    vertical_spacing,
)
leaves = leaf_table(tree, x_pos, y_pos)
nodes = build_tree_records(tree, x_pos, y_pos)
horiz, vert, branch_labels = branch_segments(tree, x_pos, y_pos)
tree_span = max(float(nodes["x"].max()), 1.0) if not nodes.empty else 1.0

metadata_key = None
sample_key = None
selected_dot_cols = []
label_cols = []
merged_tips = leaves.copy()
if metadata_df is not None:
    with st.sidebar:
        st.header("Metadata")
        metadata_key = st.selectbox("Tip lookup column", metadata_df.columns.tolist(), index=0)
        sample_key = st.selectbox("Sample lookup column", metadata_df.columns.tolist(), index=0)
        available_meta = [c for c in metadata_df.columns if c != metadata_key]
        selected_dot_cols = st.multiselect(
            "Metadata dot columns",
            available_meta,
            default=default_metadata_columns(available_meta),
        )
        label_cols = st.multiselect(
            "Tip label metadata",
            metadata_df.columns.tolist(),
            default=default_metadata_columns(metadata_df.columns),
        )
        metadata_left_pad = st.slider("Metadata dot left pad", 0.0, 10.0, 1.0, 0.05)
        dot_spacing = st.slider("Metadata column spacing", 0.05, 5.0, 0.45, 0.05)
        dot_size = st.slider("Metadata dot size", 4, 18, 9)
else:
    metadata_left_pad = 1.0
    dot_spacing = 0.45
    dot_size = 9

merged_tips = merge_tip_metadata(leaves, metadata_df, metadata_key) if metadata_df is not None else leaves.copy()
label_value_colors = {
    col: color_map(merged_tips[col])
    for col in label_cols
    if col in merged_tips.columns
}

selected_samples = []
sample_colors = {}
filtered_freq = pd.DataFrame()
selected_sample_table = pd.DataFrame()
if freq_df is not None:
    sample_table = build_sample_table(freq_df, metadata_df, sample_key)
    sample_table = sample_table.copy()
    sample_table.insert(0, "Select", False)
    default_n = min(3, len(sample_table))
    if default_n:
        sample_table.loc[:default_n - 1, "Select"] = True

    with st.sidebar:
        st.header("Frequencies")
        all_freq_nodes = sorted(freq_df["node"].dropna().astype(str).unique())
        hidden_nodes = st.multiselect("Hide frequency nodes", all_freq_nodes, default=[])
        overlap_all = st.checkbox("Only positions shared by selected samples", value=False)
        lofreq_only = st.checkbox("Only positions with LoFreq support", value=False)
        min_depth = st.number_input("Minimum coverage", min_value=0.0, value=0.0, step=1.0)
        position_filter_mode = st.radio(
            "True position filter",
            ["All", "Include", "Exclude"],
            horizontal=True,
        )
        position_filter_text = ""
        if position_filter_mode != "All":
            position_filter_text = st.text_area(
                "True positions",
                help="Use true_pos values from the frequency table. Separate positions with commas, spaces, or new lines.",
            )
            parsed_position_filter = parse_position_filter(position_filter_text)
            st.caption(f"{len(parsed_position_filter)} true position(s) entered")
        else:
            parsed_position_filter = set()
        row_gap = st.slider("Sample row spacing", 0.2, 3.0, 1.15, 0.05)
        bar_x_shift = st.slider("Bar chart horizontal shift", -10.0, 10.0, 0.0, 0.05)
        bar_gap = st.slider("Bar spacing", 0.01, 1.0, 0.08, 0.01)
        bar_width = st.slider("Bar width", 0.005, 0.5, 0.045, 0.005)
        bar_height = st.slider("Bar height", 0.2, 2.0, 1.0, 0.05)
        min_frequency_slider = st.slider("Minimum frequency quick set", 0.0, 0.1, 0.0, 0.001)
        min_frequency_to_plot = st.number_input(
            "Minimum frequency to plot",
            min_value=0.0,
            max_value=1.0,
            value=float(min_frequency_slider),
            step=0.001,
            format="%.4f",
        )
        sort_mode = st.selectbox(
            "Genome position order",
            ["relative position", "sample priority", "mean frequency", "consensus seriation"],
            index=3,
        )
        consensus_min_contrast = 0.0
        consensus_power = 1.0
        if sort_mode == "consensus seriation":
            consensus_min_contrast = st.slider(
                "Consensus minimum contrast",
                0.0,
                0.25,
                0.02,
                0.005,
                help="Differences this small on the arcsine-sqrt scale count as zero.",
            )
            consensus_power = st.slider(
                "Consensus contrast power",
                1.0,
                3.0,
                1.5,
                0.1,
                help="Higher values make large differences dominate small differences more strongly.",
            )

    st.subheader("Population Samples")
    sample_table_view = sample_table.copy()
    filter_cols = [
        c
        for c in sample_table.columns
        if c != "Select" and is_categorical_filter_column(sample_table[c])
    ]
    if filter_cols:
        st.caption("Column filters")
        filter_widgets = st.columns(len(filter_cols))
        for widget_col, col in zip(filter_widgets, filter_cols):
            vals = sorted(sample_table[col].dropna().astype(str).unique())
            with widget_col:
                selected_vals = st.multiselect(
                    col,
                    vals,
                    default=vals,
                    key=f"sample_filter_{col}",
                )
            sample_table_view = sample_table_view[
                sample_table_view[col].astype(str).isin(selected_vals)
            ]
    edited_samples = st.data_editor(
        sample_table_view,
        hide_index=True,
        use_container_width=True,
        disabled=[c for c in sample_table.columns if c != "Select"],
        key="sample_selector",
    )
    selected_sample_table = edited_samples[edited_samples["Select"]].copy()
    selected_samples = selected_sample_table["Sample"].astype(str).tolist()
    sample_colors = {sample: PALETTE[i % len(PALETTE)] for i, sample in enumerate(selected_samples)}
    sample_labels = {}
    for _, row in selected_sample_table.iterrows():
        sample = str(row["Sample"])
        sample_labels[sample] = metadata_label_from_row(
            row,
            label_cols,
            sample,
            label_value_colors,
        )

    sort_samples = list(selected_samples)
    if sort_mode == "sample priority" and selected_samples:
        sort_samples = st.sidebar.multiselect(
            "Sample sort priority",
            options=selected_samples,
            default=selected_samples,
            help="Positions are sorted by the first selected sample, then the second, and so on.",
        )

    filtered_freq = apply_frequency_filters(
        freq_df,
        selected_samples,
        hidden_nodes,
        overlap_all,
        lofreq_only,
        min_depth,
        position_filter_mode,
        parsed_position_filter,
    )
else:
    hidden_nodes = []
    sort_mode = "relative position"
    sort_samples = []
    consensus_min_contrast = 0.0
    consensus_power = 1.0
    sample_labels = {}
    row_gap = 1.15
    bar_x_shift = 0.0
    bar_gap = 0.08
    bar_width = 0.045
    bar_height = 1.0
    min_frequency_to_plot = 0.0

fig = go.Figure()
for x0, y0, x1, y1 in vert:
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color="#111", width=2), hoverinfo="skip", showlegend=False))
for x0, y0, x1, y1, label, dist in horiz:
    fig.add_trace(go.Scatter(
        x=[x0, x1],
        y=[y0, y1],
        mode="lines",
        line=dict(color="#111", width=2),
        showlegend=False,
        customdata=[[label, dist], [label, dist]],
        hovertemplate="Node: %{customdata[0]}<br>Distance: %{customdata[1]:.6g}<extra></extra>",
    ))

if show_branch_distances:
    for x, y, dist, label in branch_labels:
        fig.add_annotation(
            x=x + branch_distance_x_offset,
            y=y + branch_distance_y_offset,
            text=f"{dist:.6g}",
            showarrow=False,
            font=dict(size=font_size, color="#444"),
        )

if show_tip_labels and not merged_tips.empty:
    for _, row in merged_tips.iterrows():
        fig.add_annotation(
            x=float(row["x"]) + tip_label_offset,
            y=float(row["y"]),
            text=colored_tip_label(row, label_cols, label_value_colors),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=font_size, color="#111"),
        )
    hover_cols = metadata_df.columns if metadata_df is not None else []
    fig.add_trace(go.Scatter(
        x=merged_tips["x"] + tip_label_offset,
        y=merged_tips["y"],
        mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0.01)", line=dict(color="rgba(0,0,0,0)", width=0)),
        customdata=[metadata_hover(row, hover_cols) for _, row in merged_tips.iterrows()],
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ))

if show_node_labels and not nodes.empty:
    internal_nodes = nodes[~nodes["is_terminal"]].copy()
    fig.add_trace(go.Scatter(
        x=internal_nodes["x"] + node_label_x_offset,
        y=internal_nodes["y"] + node_label_y_offset,
        mode="text",
        text=internal_nodes["node"],
        textfont=dict(size=font_size, color="#222"),
        hovertemplate="Node: %{text}<extra></extra>",
        showlegend=False,
    ))

metadata_right = float(leaves["x"].max()) + tip_label_offset + metadata_left_pad if not leaves.empty else tree_span + metadata_left_pad
if metadata_df is not None and selected_dot_cols:
    add_metadata_tracks(fig, merged_tips, selected_dot_cols, metadata_right, dot_spacing, dot_size)

freq_right = None
if freq_df is not None and selected_samples:
    freq_right = add_frequency_bars(
        fig,
        filtered_freq,
        nodes,
        selected_samples,
        sample_colors,
        sample_labels,
        sort_mode,
        sort_samples,
        row_gap,
        bar_gap,
        bar_width,
        bar_height,
        min_frequency_to_plot,
        bar_x_shift,
        consensus_min_contrast,
        consensus_power,
    )

right_extent = float(nodes["x"].max()) + tip_label_offset + 2.0 if not nodes.empty else 10.0
if metadata_df is not None and selected_dot_cols:
    right_extent = max(right_extent, metadata_right + len(selected_dot_cols) * dot_spacing + 1.0)
if freq_right is not None:
    right_extent = max(right_extent, freq_right + 1.0)

fig.update_layout(
    template="simple_white",
    height=max(650, 24 * len(leaves) + 200),
    barmode="overlay",
    bargap=0,
    xaxis=dict(showgrid=False, zeroline=False, range=[0, right_extent], title=f"{display_mode} x-position"),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(title="Metadata"),
)

st.plotly_chart(fig, use_container_width=True)

if freq_df is not None and selected_samples:
    st.subheader("Selected Sample Colors")
    color_rows = []
    for sample in selected_samples:
        row = {"Sample": sample, "Color": sample_colors[sample]}
        if not selected_sample_table.empty:
            match = selected_sample_table[selected_sample_table["Sample"] == sample]
            if not match.empty:
                for col in match.columns:
                    if col not in {"Select", "Sample"}:
                        row[col] = match.iloc[0][col]
        color_rows.append(row)
    color_df = pd.DataFrame(color_rows)
    if not color_df.empty:
        display_cols = ["Swatch"] + [c for c in color_df.columns if c != "Color"]
        header = "".join(f"<th>{html.escape(str(col))}</th>" for col in display_cols)
        body_rows = []
        for _, row in color_df.iterrows():
            color = html.escape(str(row["Color"]))
            cells = [
                (
                    "<td>"
                    f"<div style='width:28px;height:16px;background:{color};"
                    "border:1px solid #333;border-radius:3px'></div>"
                    "</td>"
                )
            ]
            for col in display_cols[1:]:
                val = "" if pd.isna(row.get(col)) else str(row.get(col))
                cells.append(f"<td>{html.escape(val)}</td>")
            body_rows.append("<tr>" + "".join(cells) + "</tr>")
        st.markdown(
            "<table style='border-collapse:collapse;width:100%'>"
            "<thead><tr>"
            + header
            + "</tr></thead><tbody>"
            + "".join(body_rows)
            + "</tbody></table>"
            "<style>table th, table td {border-bottom:1px solid #e5e7eb;"
            "padding:0.35rem 0.5rem;text-align:left;}</style>",
            unsafe_allow_html=True,
        )

    st.subheader("Selected Frequency Rows")
    table_cols = [
        "Sample", "node", "child_node", "true_pos", "relative_pos", "derived_allele",
        "frequency", "depth", "lofreq", "edge_state1", "edge_state2", "ref", "A", "C", "G", "T",
    ]
    st.dataframe(filtered_freq[[c for c in table_cols if c in filtered_freq.columns]], use_container_width=True)

st.subheader("Tip Metadata")
st.dataframe(merged_tips, use_container_width=True)
