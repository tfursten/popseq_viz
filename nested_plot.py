"""Sample-aligned nested hierarchy and categorical metadata annotations."""
import html
import math
import statistics
from nested_core import shared_frequency_order
import plotly.graph_objects as go
from plotly.colors import qualitative


def letter_value_intervals(values):
    """Tukey-depth letter values: quartiles, eighths, sixteenths, ... ."""
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return []
    def quantile(p):
        index = (len(values)-1)*p
        lower = int(index)
        upper = min(lower+1, len(values)-1)
        return values[lower]+(values[upper]-values[lower])*(index-lower)
    depth = max(int(math.log2(len(values)))-3, 1)
    return [dict(level=i, probability=2**(-i-2),
                 low=quantile(2**(-i-2)), high=quantile(1-2**(-i-2)))
            for i in range(depth)]


def metadata_colors(metadata, fields):
    """Stable category colors, with explicit missing values."""
    colors = {}
    for field in fields:
        values = sorted({str(row.get(field) or "Missing") for row in metadata.values()})
        colors[field] = {value: ("#b8b8b8" if value == "Missing" else qualitative.Dark24[i % len(qualitative.Dark24)]) for i, value in enumerate(values)}
        colors[field].setdefault("Missing", "#b8b8b8")
    return colors


def overview_figure(nodes, metadata=None, metadata_fields=None, label_color_field=None,
                    populations=None, frequency_lookup=None, chrom_col=None,
                    frequency_mode="Auto", box_threshold=15):
    metadata = metadata or {}
    metadata_fields = metadata_fields or []
    fields = list(dict.fromkeys(metadata_fields + ([label_color_field] if label_color_field else [])))
    colors = metadata_colors(metadata, fields)
    populations = populations or []
    frequency_lookup = frequency_lookup or {}
    bar_pitch, bar_width = .16, .12
    bar_counts = [len(n["sites"]) for n in nodes
                  if frequency_mode == "Bar plots" or (frequency_mode == "Auto" and len(n["sites"]) <= box_threshold)]
    panel_width = max(2.85, max(bar_counts, default=0)*bar_pitch)
    stride = panel_width+1.75 if populations else 1
    row_scale = max(1, .85*len(populations))
    figure_height = max(700, int(len(nodes[0]["sample_order"])*row_scale*26+230))
    # Plotly's drawable height excludes the 100px top and 150px bottom margins.
    one_pixel_height = len(nodes[0]["sample_order"])*row_scale/(figure_height-250)
    population_colors = {sample: qualitative.Plotly[i % len(qualitative.Plotly)] for i, sample in enumerate(populations)}
    fig = go.Figure()
    for node in nodes:
        count = len(node["sites"])
        top, bottom = node["start"]*row_scale, node["end"]*row_scale
        fig.add_trace(go.Bar(
            x=[node["depth"]*stride], y=[bottom-top], base=[top], width=.88,
            marker=dict(color=qualitative.Safe[node["depth"] % len(qualitative.Safe)],
                        line=dict(color="white", width=2)),
            text=[f"{node['label']}<br>{count:,} SNPs"], textposition="inside",
            insidetextanchor="middle", textfont=dict(size=11, color="white"),
            customdata=[[len(node["carriers"]), "<br>".join(html.escape(s) for s in node["sample_order"])]],
            hovertemplate=f"{node['label']} · {count:,} SNPs<br>" + "%{customdata[0]} isolates<br>%{customdata[1]}<extra></extra>",
            showlegend=False,
        ))
        if not populations:
            continue
        panel_start = node["depth"]*stride + 1.05
        use_box = frequency_mode == "Boxen plots" or (frequency_mode == "Auto" and count > box_threshold)
        scale_y = (top+bottom)/2-(len(populations)-1)*.4-.32
        for fraction, label in ([(0, "0"), (.5, "0.5"), (1, "1")] if use_box else []):
            fig.add_annotation(x=panel_start+fraction*panel_width, y=scale_y, text=label, showarrow=False, font=dict(size=8, color="#666"), xanchor="left" if not use_box else "center")
        center = (top+bottom)/2
        for j, sample in enumerate(populations):
            row_y = center + (j-(len(populations)-1)/2)*.8
            valid = []
            for site in node["sites"]:
                key = (sample, site["pos"], site["allele"]) + ((site["chrom"],) if chrom_col else ())
                value = frequency_lookup.get(key, (None, False))[0]
                if value is not None and math.isfinite(value):
                    valid.append((site, value))
            fig.add_shape(type="line", x0=panel_start, x1=panel_start+panel_width,
                          y0=row_y+.24, y1=row_y+.24, line=dict(color="#ddd", width=.6))
            fig.add_annotation(x=panel_start-.08, y=row_y, text=f"P{j+1}", showarrow=False,
                               xanchor="right", font=dict(size=9, color=population_colors[sample]),
                               hovertext=html.escape(sample))
            missing = count-len(valid)
            if not valid and (use_box or not count):
                fig.add_annotation(x=panel_start+panel_width/2, y=row_y,
                                   text="0 SNPs" if not count else f"No data ({missing} missing)",
                                   showarrow=False, font=dict(size=9, color="#999"))
                continue
            color = population_colors[sample]
            if use_box:
                values = [value for _, value in valid]
                fig.add_shape(type="rect", x0=panel_start, x1=panel_start+panel_width,
                              y0=row_y-.24, y1=row_y+.24,
                              line=dict(color="#bbb", width=.7), fillcolor="rgba(0,0,0,0)", layer="below")
                for frequency in (0, .5, 1):
                    x = panel_start+frequency*panel_width
                    fig.add_shape(type="line", x0=x, x1=x, y0=row_y-.24, y1=row_y+.24,
                                  line=dict(color="#ddd", width=.6), layer="below")
                intervals = letter_value_intervals(values)
                metadata_text = "<br>".join(f"{html.escape(str(field))}: {html.escape(str(val))}"
                                           for field, val in metadata.get(sample, {}).items()
                                           if val is not None and str(val).strip())
                for band in reversed(intervals):
                    left, right = panel_start+band["low"]*panel_width, panel_start+band["high"]*panel_width
                    half_height = .16*2**(-band["level"])
                    tooltip = (f"{html.escape(sample)} · {node['label']}<br>"
                               f"Percentiles: {100*band['probability']:.3g}–{100*(1-band['probability']):.3g}<br>"
                               f"Frequency interval: {band['low']:.6g}–{band['high']:.6g}<br>"
                               f"Available: {len(values)} · Missing: {missing}" + ("<br>"+metadata_text if metadata_text else ""))
                    fig.add_trace(go.Scatter(
                        x=[left,right,right,left,left],
                        y=[row_y-half_height,row_y-half_height,row_y+half_height,row_y+half_height,row_y-half_height],
                        mode="lines", line=dict(color=color, width=1), fill="toself", fillcolor=color,
                        opacity=max(.25,.8-.1*band["level"]), name=sample,
                        legendgroup=f"population:{sample}", showlegend=False,
                        hoveron="fills", text=tooltip, hoverinfo="text",
                        meta=dict(kind="boxen_band", **band),
                    ))
                outer = intervals[-1]
                tails = sorted(set(v for v in values if v < outer["low"] or v > outer["high"]))
                if tails:
                    fig.add_trace(go.Scatter(x=[panel_start+v*panel_width for v in tails], y=[row_y]*len(tails),
                                            mode="markers", marker=dict(color=color, size=3), showlegend=False,
                                            legendgroup=f"population:{sample}",
                                            text=[f"{html.escape(sample)} · {node['label']}<br>Frequency: {v:.8g}" for v in tails], hoverinfo="text"))
                q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive") if len(values)>1 else [values[0]]*3
                fig.add_shape(type="line", x0=panel_start+median*panel_width, x1=panel_start+median*panel_width,
                              y0=row_y-.18, y1=row_y+.18, line=dict(color=color, width=2))
                fig.add_trace(go.Scatter(x=[panel_start+median*panel_width], y=[row_y], mode="markers",
                    marker=dict(size=12, color="rgba(0,0,0,0)"), showlegend=False,
                    customdata=[[sample, node["label"], len(values), missing, min(values), q1, median, q3, max(values)]],
                    hovertemplate="%{customdata[0]} · %{customdata[1]}<br>Available: %{customdata[2]} · Missing: %{customdata[3]}<br>Min: %{customdata[4]:.4f}<br>Q1: %{customdata[5]:.4f}<br>Median: %{customdata[6]:.4f}<br>Q3: %{customdata[7]:.4f}<br>Max: %{customdata[8]:.4f}<extra></extra>"))
            else:
                # One SNP order across populations, computed without missing values.
                ordered = shared_frequency_order(node["sites"], populations, frequency_lookup, chrom_col)
                point_x, point_y, point_data = [], [], []
                hover_boxes = []
                for rank, site in enumerate(ordered):
                    point = panel_start+(rank+.5)*bar_pitch
                    key = (sample, site["pos"], site["allele"]) + ((site["chrom"],) if chrom_col else ())
                    value = frequency_lookup.get(key, (None, False))[0]
                    is_missing = value is None or not math.isfinite(value)
                    fig.add_shape(type="rect", x0=point-bar_width/2, x1=point+bar_width/2,
                                  y0=row_y-.24, y1=row_y+.24,
                                  line=dict(color="#999", width=.8),
                                  fillcolor="#c4c4c4" if is_missing else "rgba(0,0,0,0)", layer="below")
                    status = "Missing data (excluded from ordering and summaries)" if is_missing else f"Measured frequency: {value:.8g}"
                    metadata_text = "<br>".join(f"{html.escape(str(field))}: {html.escape(str(val))}"
                                               for field, val in metadata.get(sample, {}).items()
                                               if val is not None and str(val).strip())
                    tooltip = (f"{html.escape(sample)} · {node['label']}<br>"
                               f"{html.escape(str(site['chrom']))}:{html.escape(str(site['pos']))} → {html.escape(site['allele'])}<br>"
                               f"{status}<br>Shared SNP rank: {rank+1}" + ("<br>"+metadata_text if metadata_text else "<br>No population metadata available"))
                    hover_boxes.append((point, tooltip, is_missing))
                    if is_missing:
                        continue
                    point_x.append(point)
                    point_y.append(-max(.48*value, one_pixel_height) if value > 0 else 0)
                    point_data.append([sample, node["label"], site["chrom"], site["pos"], site["allele"], value, missing, site["mean_frequency"], rank+1])
                fig.add_trace(go.Bar(
                    x=point_x, y=point_y, base=[row_y+.24]*len(point_x),
                    width=bar_width, marker=dict(color=color),
                    name=sample, legendgroup=f"population:{sample}", showlegend=False,
                    customdata=point_data,
                    hoverinfo="skip",
                ))
                zeros = [x for x, data in zip(point_x, point_data) if data[5] == 0]
                if zeros:
                    fig.add_trace(go.Scatter(x=zeros, y=[row_y+.24]*len(zeros), mode="markers",
                                             marker=dict(symbol="line-ew", color=color, size=6),
                                             showlegend=False, hoverinfo="skip"))
                # Filled polygons provide hit areas over the entire 0–1 box.
                for point, tooltip, is_missing in hover_boxes:
                    left, right = point-bar_width/2, point+bar_width/2
                    fig.add_trace(go.Scatter(
                        x=[left, right, right, left, left],
                        y=[row_y-.24, row_y-.24, row_y+.24, row_y+.24, row_y-.24],
                        mode="lines", line=dict(width=0), fill="toself",
                        fillcolor="rgba(0,0,0,0.001)", hoveron="fills",
                        text=tooltip, hoverinfo="text", showlegend=False,
                        legendgroup=f"population:{sample}",
                    ))
    depth = max(n["depth"] for n in nodes)
    right_edge = depth*stride + (panel_width+1.15 if populations else .44)
    samples = nodes[0]["sample_order"]
    for i, sample in enumerate(populations):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                marker=dict(color=population_colors[sample], size=8),
                                name=f"P{i+1}: {sample}", legendgroup=f"population:{sample}"))
    for j, field in enumerate(metadata_fields):
        column_x = right_edge + .41 + .35*j
        for value, color in colors[field].items():
            indices = [i for i, sample in enumerate(samples) if str(metadata.get(sample, {}).get(field) or "Missing") == value]
            if not indices:
                continue
            fig.add_trace(go.Scatter(
                x=[column_x]*len(indices), y=[(i+.5)*row_scale for i in indices], mode="markers",
                marker=dict(symbol="square", size=13, color=color),
                name=f"{field}: {value}", legendgroup=field,
                customdata=[[samples[i], field, value] for i in indices],
                hovertemplate="%{customdata[0]}<br>%{customdata[1]}: %{customdata[2]}<extra></extra>",
            ))
    labels = []
    for sample in samples:
        label = html.escape(sample)
        if label_color_field:
            value = str(metadata.get(sample, {}).get(label_color_field) or "Missing")
            color = colors[label_color_field].get(value, "#b8b8b8")
            label = f'<span style="color:{color}">{label}</span>'
        labels.append(label)
    if label_color_field and label_color_field not in metadata_fields:
        for value, color in colors[label_color_field].items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color=color, size=10), name=f"{label_color_field}: {value}"))
    fig.update_layout(
        template="simple_white", barmode="overlay", height=figure_height,
        margin=dict(l=35, r=370, t=100, b=150), bargap=0,
        xaxis=dict(range=[-.5, right_edge+.16+.35*len(metadata_fields)],
                   tickvals=[i*stride for i in range(depth+1)]+[right_edge+.41+.35*j for j in range(len(metadata_fields))],
                   ticktext=["All selected"]+[f"Level {i}" for i in range(1, depth+1)]+[html.escape(f) for f in metadata_fields],
                   side="top", tickangle=-45, showgrid=False, zeroline=False,
                   title="Nested SNP groups → · bars: outlined 0–1 · boxen: fixed 0 → 1 scale" if populations else "Nested SNP groups →"),
        yaxis=dict(range=[len(samples)*row_scale, 0], tickmode="array", side="right",
                   tickvals=[(i+.5)*row_scale for i in range(len(samples))], ticktext=labels,
                   tickfont=dict(size=11), showgrid=True,
                   gridcolor="rgba(100,100,100,0.12)", zeroline=False),
        boxmode="overlay",
        legend=dict(orientation="h", yanchor="top", y=-.03, x=0, font=dict(size=10)),
    )
    return fig
