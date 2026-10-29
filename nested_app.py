"""Run with: streamlit run viz/nested_app.py"""
import io
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from nested_core import build_matrix_hierarchy, group_frequency_lookup, read_rows, layout_nested_blocks, filter_isolates, metadata_value, shared_frequency_order, update_population_selection
from nested_plot import overview_figure

st.set_page_config(page_title="Nested SNP Explorer", layout="wide")
st.title("Nested SNP Explorer")
st.caption("One row per selected isolate, ordered so nested SNP groups stay together. Labels and metadata appear on the right.")
BASE = Path(__file__).resolve().parents[1]


@st.cache_data
def load_rows(data, name):
    return read_rows(io.StringIO(data.decode("utf-8-sig")), name)


def input_rows(label, default, key, use_local):
    upload = st.sidebar.file_uploader(label, type=["tsv", "csv", "txt"], key=key)
    if upload:
        return load_rows(upload.getvalue(), upload.name)
    path = BASE / default
    if use_local and path.exists():
        return load_rows(path.read_bytes(), path.name)
    return None


with st.sidebar:
    st.header("Data")
    local = st.checkbox("Use bundled Latest data", value=True)
    st.caption("Uploads replace the corresponding local input. SNP groups and their alleles are read directly from the matrix.")
matrix = input_rows("SNP matrix (CHROM, POS, isolate columns)", "Latest/Colony/all_runs.filtered_matrix.tsv", "matrix", local)
freq_rows = input_rows("Population frequencies (optional)", "Latest/app_data/paup_node_and_freq_matrix_all.tsv", "frequencies", local)
metadata = input_rows("Metadata (optional)", "Latest/MetaData/all_metadata.tsv", "metadata", local)
if not matrix:
    st.info("Provide a SNP matrix to begin.")
    st.stop()
if not {"CHROM", "POS"} <= matrix[0].keys():
    st.error("The matrix must have CHROM and POS columns followed by isolate allele columns.")
    st.stop()
isolates = [c for c in matrix[0] if c not in {"CHROM", "POS", "REF"}]
md = pd.DataFrame(metadata or [])
key = None
sample_metadata = {}
filters = {}
if not md.empty:
    with st.sidebar:
        st.header("Isolate filters")
        key = st.selectbox("Metadata sample ID column", md.columns, index=list(md.columns).index("sample_uid") if "sample_uid" in md else 0)
        if md[key].duplicated().any():
            st.warning("Metadata has duplicate sample IDs; the first row per ID is used.")
        sample_metadata = {str(row[key]): row for row in md.drop_duplicates(key).fillna("").to_dict("records")}
        normalized = {"".join(c for c in col.lower() if c.isalnum()): col for col in md.columns}
        for field, label in [("person", "Person"), ("week", "Week"), ("bodysite", "Body site")]:
            column = normalized.get(field)
            if column is None:
                st.caption(f"{label} filter unavailable: metadata column missing.")
                continue
            values = {metadata_value(sample_metadata.get(sample, {}), column) for sample in isolates}
            def value_sort(value):
                if field == "week":
                    try:
                        return (0, float(value), value)
                    except ValueError:
                        pass
                return (1, 0, value)
            filters[column] = st.multiselect(f"Isolate {label}", sorted(values, key=value_sort), key=f"isolate_filter_{field}")
        st.caption("Empty filters include all values. Values within a filter are combined; Person, Week and Body site filters must all match. Missing metadata can be selected as Missing.")
else:
    st.sidebar.caption("Upload metadata to filter isolates by person, week and body site.")
eligible = filter_isolates(isolates, sample_metadata, filters)
st.sidebar.caption(f"{len(eligible)} of {len(isolates)} isolates match the filters.")
selected = st.sidebar.multiselect("Included isolates", eligible, default=eligible)
reference = next((sample for sample in isolates if sample.startswith(("GCF_", "GCA_"))), isolates[0])
orientation_sample = st.sidebar.selectbox("Orient nesting relative to isolate", isolates, index=isolates.index(reference))
st.sidebar.caption("This isolate orients the grouping; its allele is not assumed ancestral. The allele shared by a group defines that group's SNP frequency.")
if not eligible:
    st.info("No isolates match these metadata filters. Clear or change the filters to continue.")
    st.stop()
if not selected:
    st.info("Select isolates to build the nesting.")
    st.stop()
try:
    nodes, audit = build_matrix_hierarchy(matrix, selected, orientation_sample)
    layout_nested_blocks(nodes)
except (ValueError, KeyError) as exc:
    st.error(f"Cannot build hierarchy: {exc}")
    st.stop()
sites = [dict(site, carriers=", ".join(sorted(node["carriers"]))) for node in nodes for site in node["sites"]]
st.write(f"**{len(selected)} isolates · {len(sites):,} placed SNP alleles · {len(nodes)} nested groups · {len(audit):,} unplaced SNPs**")
st.caption("Each block spans exactly the isolate rows carrying its SNPs. SNP counts appear as labels. Samples are ordered recursively by nested group, with sample ID breaking ties within indistinguishable groups. The first block contains matrix alleles shared by all displayed isolates. Other blocks contain group-defining alleles relative to the orientation isolate; these are not labeled ancestral or derived.")
metadata_fields = []
label_color_field = None
if not md.empty:
    with st.sidebar:
        st.header("Sample metadata colors")
        available = [c for c in md.columns if c != key]
        defaults = [c for c in available if c.lower() in {"person", "week", "bodysite"}]
        metadata_fields = st.multiselect("Metadata color strips", available, default=defaults)
        choices = ["None"]+available
        preferred = next((c for c in available if c.lower() == "person"), "None")
        color_choice = st.selectbox("Color sample labels by", choices, index=choices.index(preferred))
        label_color_field = None if color_choice == "None" else color_choice
        unmatched = sum(s not in sample_metadata for s in selected)
        if unmatched:
            st.caption(f"{unmatched} selected isolates have no matching metadata; shown in gray.")
if not sites:
    st.warning("No SNPs can be placed for this selection.")

freq = pd.DataFrame(freq_rows or [])
populations = []
if not freq.empty:
    required = {"Sample", "true_pos"}
    has_counts = "depth" in freq and bool({"A", "C", "G", "T"} & set(freq.columns))
    has_reported = {"derived_allele", "derived_allele_frequency"} <= set(freq.columns)
    if not required <= set(freq.columns) or not (has_counts or has_reported):
        st.error("Frequency input needs Sample and true_pos, plus A/C/G/T counts with depth or an allele-labeled frequency column.")
        st.stop()
    table = pd.DataFrame({"Sample": sorted(freq.Sample.unique())})
    if metadata:
        table = table.merge(md.drop_duplicates(key), left_on="Sample", right_on=key, how="left")
    st.subheader("Population samples")
    population_ids = table.Sample.tolist()
    selection_key = "selected_population_ids"
    revision_key = "population_selection_revision"
    st.session_state.setdefault(selection_key, [])
    st.session_state.setdefault(revision_key, 0)
    st.session_state[selection_key] = [sample for sample in st.session_state[selection_key] if sample in population_ids]
    meta_columns = [column for column in table.columns if column != "Sample"]
    default_columns = [column for column in meta_columns if column.lower() in {"person", "week", "bodysite", "run"}]
    with st.expander("Filter and arrange population table", expanded=True):
        search = st.text_input("Search population table", help="Matches sample IDs or any metadata value.")
        display_columns = st.multiselect("Population metadata columns", meta_columns, default=default_columns)
        filter_columns = st.multiselect("Filter populations by", meta_columns,
                                       default=[column for column in meta_columns if column.lower() in {"person", "week", "bodysite"}])
        view = table.copy()
        if search.strip():
            matches = table.astype(str).apply(lambda column: column.str.contains(search.strip(), case=False, regex=False, na=False)).any(axis=1)
            view = view[matches]
        filter_widgets = st.columns(max(1, min(3, len(filter_columns))))
        for index, column in enumerate(filter_columns):
            values = sorted(table[column].fillna("Missing").replace("", "Missing").astype(str).unique())
            with filter_widgets[index % len(filter_widgets)]:
                chosen = st.multiselect(f"Population {column}", values, key=f"population_filter_{column}")
            if chosen:
                view = view[view[column].fillna("Missing").replace("", "Missing").astype(str).isin(chosen)]
        sort_column = st.selectbox("Sort population table by", ["Sample"]+display_columns)
        descending = st.checkbox("Sort populations descending", value=False)
        if sort_column.lower() == "week":
            view = view.assign(_week_sort=pd.to_numeric(view[sort_column], errors="coerce")).sort_values("_week_sort", ascending=not descending, kind="stable", na_position="last").drop(columns="_week_sort")
        else:
            view = view.sort_values(sort_column, ascending=not descending, kind="stable", na_position="last")
    visible_ids = tuple(view.Sample.tolist())

    def population_table_edited(widget_key, sample_ids):
        st.session_state[selection_key] = update_population_selection(
            st.session_state[selection_key], sample_ids,
            st.session_state[widget_key].get("edited_rows", {}))
        st.session_state[revision_key] += 1

    def population_bulk_selection(sample_ids, clear=False):
        st.session_state[selection_key] = [] if clear else list(dict.fromkeys(st.session_state[selection_key]+list(sample_ids)))
        st.session_state[revision_key] += 1

    buttons = st.columns([1, 1, 3])
    buttons[0].button("Select visible", disabled=not visible_ids, on_click=population_bulk_selection, args=(visible_ids,))
    buttons[1].button("Clear selection", on_click=population_bulk_selection, args=((), True))
    display = view[["Sample"]+display_columns].copy()
    for column in display_columns:
        if column.lower() == "week":
            numeric = pd.to_numeric(display[column], errors="coerce")
            if display[column].fillna("").astype(str).str.strip().eq("").all() or numeric.notna().sum()==display[column].fillna("").astype(str).str.strip().ne("").sum():
                display[column] = numeric
    display.insert(0, "Select", display.Sample.isin(st.session_state[selection_key]))
    view_signature = hashlib.sha1(repr(visible_ids).encode()).hexdigest()[:12]
    editor_key = f"population_table_{st.session_state[revision_key]}_{view_signature}"
    st.data_editor(display, hide_index=True, use_container_width=True, num_rows="fixed",
                   disabled=[column for column in display.columns if column != "Select"],
                   column_config={"Select": st.column_config.CheckboxColumn("Show", width="small")},
                   key=editor_key, on_change=population_table_edited, args=(editor_key, visible_ids))
    populations = list(st.session_state[selection_key])
    hidden_selected = len(set(populations)-set(visible_ids))
    st.caption(f"{len(view)} of {len(table)} populations shown · {len(populations)} selected" +
               (f" ({hidden_selected} hidden by table filters)" if hidden_selected else "") +
               ". Check Show to plot a population. Click a column header to sort. Filtering keeps selections; uncheck a row or clear selection to remove it. P1, P2, … follow selection order.")
    min_depth = st.sidebar.number_input("Minimum frequency coverage", min_value=0, value=0)
    if "row_type" in freq:
        summaries = ~freq.row_type.fillna("site").isin(["site", ""])
        if summaries.any():
            st.warning(f"Ignoring {summaries.sum():,} summary rows: per-SNP frequencies are required for alignment.")
        freq = freq[~summaries].copy()
    if "depth" in freq and min_depth > 0:
        freq = freq[pd.to_numeric(freq.depth, errors="coerce") >= min_depth].copy()

mode = st.sidebar.selectbox("Frequency display", ["Auto", "Bar plots", "Boxen plots"])
threshold = st.sidebar.slider("Use boxen plots above this SNP count", 2, 100, 15)
st.sidebar.caption("Small blocks show one bar per SNP in shared descending mean-frequency order. Large blocks show boxen plots on a fixed 0 → 1 scale. Each outlined bar spans 0–1; bar widths are constant. Missing values are excluded from means and summaries.")
missing = 0
ambiguous = 0
frequency_lookup = {}
chrom_col = None
if populations:
    chrom_col = "chrom_matrix" if "chrom_matrix" in freq else "chrom" if "chrom" in freq else None
    if chrom_col is None and len({s["chrom"] for s in sites}) > 1:
        st.error("Multiple chromosomes require chrom_matrix or chrom in the frequency table for unambiguous matching.")
        st.stop()
    frequency_lookup = group_frequency_lookup(freq[freq.Sample.isin(populations)].to_dict("records"), set(populations), use_chromosome=chrom_col is not None)

if populations:
    coverage_rows = []
    for node in nodes:
        node["sites"] = shared_frequency_order(node["sites"], populations, frequency_lookup, chrom_col)
    sites = [dict(site, carriers=", ".join(sorted(node["carriers"]))) for node in nodes for site in node["sites"]]
    for node in nodes:
        for sample in populations:
            for site in node["sites"]:
                lookup_key = (sample, site["pos"], site["allele"]) + ((site["chrom"],) if chrom_col else ())
                value, conflict = frequency_lookup.get(lookup_key, (None, False))
                missing += int(value is None)
                ambiguous += int(conflict)
    for index, sample in enumerate(populations, 1):
        available = 0
        for site in sites:
            lookup_key = (sample, site["pos"], site["allele"]) + ((site["chrom"],) if chrom_col else ())
            available += frequency_lookup.get(lookup_key, (None, False))[0] is not None
        row = {"Track": f"P{index}", "Population": sample, "Available SNPs": available,
               "Missing SNPs": len(sites)-available}
        row.update({field: sample_metadata.get(sample, {}).get(field, "") for field in metadata_fields})
        coverage_rows.append(row)
    st.write(f"**Showing {len(populations)} population sample(s)**")
    with st.expander("Selected population SNP coverage"):
        st.dataframe(pd.DataFrame(coverage_rows), hide_index=True, use_container_width=True)
    if not any(row["Available SNPs"] for row in coverage_rows):
        st.warning("The selected populations have no usable frequencies matching the displayed SNPs. Check that the matrix and frequency table use matching chromosomes, positions and group-defining alleles, and check the coverage cutoff.")
overview = overview_figure(nodes, sample_metadata, metadata_fields, label_color_field,
                           populations, frequency_lookup, chrom_col, mode, threshold)
st.plotly_chart(overview, use_container_width=True, key="nesting_overview")
st.download_button("Download interactive nesting overview", overview.to_html(include_plotlyjs=True), "nested_overview.html", mime="text/html")
sample_order = pd.DataFrame({"row": range(1, len(selected)+1), "isolate": nodes[0]["sample_order"]})
with st.expander("Ordered isolate rows"):
    st.dataframe(sample_order, hide_index=True, use_container_width=True)
    st.download_button("Download isolate order", sample_order.to_csv(index=False), "nested_isolate_order.csv")
with st.expander("How the nesting is calculated"):
    st.write("For each matrix row, isolates carrying the same allele form a candidate group. If all displayed isolates agree, that allele belongs to the first block. Otherwise, alleles differing from the orientation isolate define the candidate groups. Equal carrier sets are combined; each group nests under its smallest strict superset. The orientation isolate sets the diagram's root without implying ancestry. Overlapping sets that cannot nest are reported as unresolved. Population frequencies use the group's own allele count divided by depth. If counts are unavailable, an allele-labeled reported frequency is used only when its allele matches the group allele.")
if populations:
    st.caption("Each block has one horizontal row per population (P1, P2, …). Bar plots show one frequency per SNP for small blocks; boxen plots summarize available SNP frequencies for large blocks. Small-block SNPs use one shared left-to-right order: decreasing mean frequency across selected populations, excluding missing values; all-missing SNPs go last. Bar outlines show full height 1; filled height gives frequency. All SNP bars have the same width. Boxen plots share a fixed 0 → 1 scale, labeled 0, 0.5 and 1. Missing values never enter the calculations.")
    st.caption(f"{missing:,} SNP/sample slots have no usable frequency (including {ambiguous:,} conflicting duplicate values). Missing SNP boxes are gray; measured zero retains an unfilled outlined box with a colored baseline tick. Hover anywhere in a SNP box for population metadata and frequency or missing status. Missing values are omitted from summaries.")
else:
    st.info("Check Show in the population table above to add frequency rows beside each SNP block.")
with st.expander("Branch membership and SNP order"):
    st.dataframe(pd.DataFrame(sites), use_container_width=True)
    st.download_button("Download SNP order", pd.DataFrame(sites).to_csv(index=False), "nested_snp_order.csv")
with st.expander("Unplaced SNPs and reasons"):
    st.dataframe(pd.DataFrame(audit), use_container_width=True)
    st.download_button("Download unplaced SNPs", pd.DataFrame(audit).to_csv(index=False), "unplaced_snps.csv")
