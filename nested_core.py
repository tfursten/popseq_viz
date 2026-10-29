"""Directed SNP carrier sets and an exact, laminar interval layout (no tree needed)."""
import csv
import math
import statistics
from collections import defaultdict


def shared_frequency_order(sites, populations, lookup, chrom_col=None):
    """Highest mean first; ignore missing values and retain all-missing SNPs last."""
    ordered = []
    for site in sites:
        values = []
        for population in populations:
            key = (population, site["pos"], site["allele"]) + ((site["chrom"],) if chrom_col else ())
            value = lookup.get(key, (None, False))[0]
            if value is not None and math.isfinite(value):
                values.append(value)
        site = dict(site, mean_frequency=statistics.mean(values) if values else None)
        ordered.append(site)
    ordered.sort(key=lambda site: (site["mean_frequency"] is None,
                                  -site["mean_frequency"] if site["mean_frequency"] is not None else 0,
                                  site["site"]))
    for rank, site in enumerate(ordered, 1):
        site["frequency_order"] = rank
    return ordered


def metadata_value(row, column):
    value = row.get(column)
    return str(value).strip() if value is not None and str(value).strip() else "Missing"


def filter_isolates(isolates, metadata, filters):
    """OR within metadata fields, AND between fields; preserve input order."""
    return [sample for sample in isolates if all(
        not values or metadata_value(metadata.get(sample, {}), column) in values
        for column, values in filters.items()
    )]


def update_population_selection(selected, visible_samples, edited_rows):
    """Apply checkbox edits by sample ID; leave filtered-out selections intact."""
    selected = list(selected)
    for row_index, changes in edited_rows.items():
        if "Select" not in changes:
            continue
        sample = visible_samples[int(row_index)]
        if changes["Select"] and sample not in selected:
            selected.append(sample)
        elif not changes["Select"] and sample in selected:
            selected.remove(sample)
    return selected


def read_rows(handle, name):
    return list(csv.DictReader(handle, delimiter="," if name.lower().endswith(".csv") else "\t"))


def directed_changes(rows):
    """Map one-based matrix characters to unambiguous single-change alleles."""
    grouped = defaultdict(list)
    for row in rows:
        site = int(row.get("site") or row.get("character"))
        if "acctran_edges" in row:
            edges = [e.strip() for e in row["acctran_edges"].split(";") if e.strip()]
            allele = edges[0].rsplit("->", 1)[-1] if len(edges) == 1 else ""
            count = int(row.get("acctran_n_changes") or len(edges))
            grouped[site].append(allele if count == 1 else "")
        else:
            allele = row.get("to_state", row.get("state2", ""))
            if float(row.get("steps") or 1) != 1:
                allele = ""
            grouped[site].append(allele)
    return {site: values[0].upper() if len(values) == 1 else "" for site, values in grouped.items()}


def build_hierarchy(matrix, changes, samples):
    if not samples:
        raise ValueError("Select at least one isolate.")
    groups = defaultdict(list)
    audit = []
    for index, row in enumerate(matrix, 1):
        allele = changes.get(index, "")
        record = {"site": index, "chrom": row.get("CHROM", ""), "pos": str(row["POS"]), "allele": allele}
        calls = {s: str(row[s]).upper() for s in samples}
        reason = ""
        if allele not in "ACGT" or len(allele) != 1:
            reason = "Missing or multiple directed changes"
        elif any(v not in {"A", "C", "G", "T"} for v in calls.values()):
            reason = "Missing or ambiguous isolate call"
        carriers = frozenset(s for s, v in calls.items() if v == allele)
        if not reason and not carriers:
            reason = "Derived allele absent from selected isolates"
        if reason:
            audit.append(dict(record, status=reason))
        else:
            groups[carriers].append(record)
    return _assemble_hierarchy(groups, samples, audit)


def _assemble_hierarchy(groups, samples, audit):
    keys = list(groups)
    conflicts = set()
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a & b and not (a <= b or b <= a):
                conflicts.update((a, b))
    for group in conflicts:
        audit.extend(dict(r, status="Incompatible carrier sets (non-clonal nesting)") for r in groups.pop(group))
    root = frozenset(samples)
    groups.setdefault(root, [])
    nodes = {key: {"carriers": key, "sites": sorted(values, key=lambda r: r["site"]), "children": []} for key, values in groups.items()}
    for key in groups:
        if key != root:
            parent = min((k for k in groups if key < k), key=len)
            nodes[parent]["children"].append(key)
    ordered = []
    cursor = 0

    def walk(key, depth):
        nonlocal cursor
        node = nodes[key]
        node.update(start=cursor, depth=depth, label=f"B{len(ordered) + 1}")
        ordered.append(node)
        for record in node["sites"]:
            record["x"] = cursor + 0.5
            record["branch"] = node["label"]
            cursor += 1
        node["own_end"] = cursor
        for child in sorted(node["children"], key=lambda k: tuple(sorted(k))):
            walk(child, depth + 1)
        node["end"] = cursor
    walk(root, 0)
    return ordered, audit


def build_matrix_hierarchy(matrix, samples, orientation_sample):
    """Group observed alleles relative to a matrix anchor, without ancestral labels.

    Shared calls become root markers. Variable calls differing from the anchor
    define carrier clades. An anchor orients splits; it does not infer ancestry.
    """
    if not samples:
        raise ValueError("Select at least one isolate.")
    groups, audit = defaultdict(list), []
    for index, row in enumerate(matrix, 1):
        record = dict(site=index, chrom=row.get("CHROM", ""), pos=str(row["POS"]))
        calls = {sample: str(row[sample]).strip().upper() for sample in samples}
        if any(call not in {"A","C","G","T"} for call in calls.values()):
            audit.append(dict(record, allele="", status="Missing or ambiguous isolate call"))
            continue
        alleles = set(calls.values())
        if len(alleles)==1:
            allele = next(iter(alleles))
            groups[frozenset(samples)].append(dict(record, allele=allele, allele_rule="shared_matrix_allele"))
            continue
        anchor = str(row.get(orientation_sample, "")).strip().upper()
        if anchor not in {"A","C","G","T"}:
            audit.append(dict(record, allele="", status="Missing or ambiguous orientation-isolate call"))
            continue
        for allele in sorted(alleles-{anchor}):
            carriers = frozenset(sample for sample, call in calls.items() if call==allele)
            groups[carriers].append(dict(record, allele=allele, allele_rule="group_defining_matrix_allele"))
    return _assemble_hierarchy(groups, samples, audit)


def group_frequency_lookup(rows, populations, use_chromosome=True):
    """Use each base's count/depth; fall back only to a matching reported allele.

    Distinct duplicate values remain missing. Missing base counts are never
    inferred as zero, and opposite-allele frequencies are never complemented.
    """
    values = defaultdict(list)
    for row in rows:
        sample = str(row.get("Sample", ""))
        if sample not in populations:
            continue
        position = str(row.get("true_pos", "")).removesuffix(".0")
        chrom = str(row.get("chrom_matrix") or row.get("chrom") or "")
        try:
            depth = float(row.get("depth", ""))
        except (TypeError, ValueError):
            depth = float("nan")
        if math.isfinite(depth) and depth <= 0:
            continue
        for allele in "ACGT":
            value = None
            if math.isfinite(depth) and depth>0:
                try:
                    count = float(row.get(allele, ""))
                    if math.isfinite(count) and 0<=count<=depth:
                        value = count/depth
                except (TypeError, ValueError):
                    pass
            if value is None and str(row.get("derived_allele", "")).upper()==allele:
                try:
                    reported = float(row.get("derived_allele_frequency", ""))
                    if math.isfinite(reported) and 0<=reported<=1:
                        value = reported
                except (TypeError, ValueError):
                    pass
            if value is not None:
                key = (sample, position, allele) + ((chrom,) if use_chromosome else ())
                values[key].append(value)
    result = {}
    for key, observed in values.items():
        conflict = not all(math.isclose(observed[0], value, rel_tol=1e-6, abs_tol=1e-9) for value in observed[1:])
        result[key] = (None if conflict else observed[0], conflict)
    return result


def layout_nested_blocks(nodes):
    """Order samples by recursive clade traversal; each sample gets one column.

    Child clades are indivisible contiguous groups. Samples with no further
    distinguishing SNPs retain individual columns, ordered by sample ID.
    """
    by_carriers = {node["carriers"]: node for node in nodes}

    def order_samples(node):
        children = [by_carriers[k] for k in node["children"]]
        covered = set().union(*(c["carriers"] for c in children))
        residual = node["carriers"] - covered
        # Keep each child intact, while giving unsplit samples their own slots.
        units = [(min(c["carriers"]), order_samples(c)) for c in children]
        units.extend((sample, [sample]) for sample in residual)
        node["sample_order"] = [sample for _, unit in sorted(units) for sample in unit]
        node["display_width"] = len(node["sample_order"])
        return node["sample_order"]

    def place(node, left):
        node["start"] = left
        node["end"] = node["own_end"] = left + node["display_width"]
        count = len(node["sites"])
        for i, site in enumerate(node["sites"]):
            site["x"] = left + (i + .5) * node["display_width"] / count
        positions = {sample: left+i for i, sample in enumerate(node["sample_order"])}
        for key in node["children"]:
            child = by_carriers[key]
            place(child, min(positions[sample] for sample in child["carriers"]))

    order_samples(nodes[0])
    place(nodes[0], 0)
    return nodes
