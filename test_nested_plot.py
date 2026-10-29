import unittest

from nested_core import build_hierarchy, layout_nested_blocks
from nested_plot import overview_figure, letter_value_intervals


class OverviewTests(unittest.TestCase):
    def test_missing_boxes_are_gray_and_full_boxes_have_metadata_hover(self):
        matrix = [dict(CHROM="c", POS=str(i), a="T") for i in range(1,4)]
        nodes, _ = build_hierarchy(matrix, {i:"T" for i in range(1,4)}, ["a"])
        layout_nested_blocks(nodes)
        fig = overview_figure(nodes, metadata={"p": {"Person":"Tal"}, "q": {"Person":"Kaleo"}},
                              populations=["p","q"], frequency_lookup={("p","1","T"):(0,False)},
                              frequency_mode="Bar plots")
        gray = [shape for shape in fig.layout.shapes if shape.type=="rect" and shape.fillcolor=="#c4c4c4"]
        self.assertEqual(len(gray),5)
        hover = [t for t in fig.data if t.type=="scatter" and t.hoveron=="fills"]
        self.assertEqual(len(hover),6)
        self.assertEqual(sum("Missing data" in t.text for t in hover),5)
        self.assertEqual(sum("Measured frequency: 0" in t.text for t in hover),1)

        for t in hover:
            self.assertAlmostEqual(max(t.y)-min(t.y), .48)
            self.assertIn("Person:", t.text)
            self.assertIn("Shared SNP rank:", t.text)
            self.assertEqual(t.hoverinfo,"text")
        self.assertEqual(sum("Kaleo" in t.text for t in hover),3)

    def test_tiny_positive_bar_has_one_pixel_minimum_and_retains_true_frequency(self):
        matrix = [dict(CHROM="c", POS=str(i), a="T") for i in range(1,4)]
        nodes, _ = build_hierarchy(matrix, {i:"T" for i in range(1,4)}, ["a"])
        layout_nested_blocks(nodes)
        fig = overview_figure(nodes, populations=["p"], frequency_mode="Bar plots",
                              frequency_lookup={("p","1","T"):(1e-10,False), ("p","2","T"):(0,False)})
        bar = next(t for t in fig.data if t.type=="bar" and t.name=="p")
        pixels_per_unit = (fig.layout.height-250)/abs(fig.layout.yaxis.range[1]-fig.layout.yaxis.range[0])
        self.assertAlmostEqual(abs(bar.y[0])*pixels_per_unit,1)
        self.assertEqual(bar.y[1],0)
        self.assertEqual(bar.customdata[0][5],1e-10)

    def test_letter_values_expand_into_tails_and_exclude_missing(self):
        intervals = letter_value_intervals(list(range(256))+[None,float("nan")])
        self.assertEqual(len(intervals),5)
        self.assertAlmostEqual(intervals[0]["low"],63.75)
        self.assertAlmostEqual(intervals[0]["high"],191.25)
        for inner,outer in zip(intervals,intervals[1:]):
            self.assertLess(outer["low"],inner["low"])
            self.assertGreater(outer["high"],inner["high"])
            self.assertLess(outer["probability"],inner["probability"])
        self.assertEqual(letter_value_intervals([]),[])
        self.assertEqual(letter_value_intervals([.5])[0]["low"],.5)

    def test_population_rows_bar_frequency_scale_and_missing(self):
        matrix = [dict(CHROM="c", POS="1", a="T", b="A"),
                  dict(CHROM="c", POS="2", a="T", b="A")]
        nodes, _ = build_hierarchy(matrix, {1: "T", 2: "T"}, ["a", "b"])
        layout_nested_blocks(nodes)
        lookup = {("p1", "1", "T", "c"): (.2, False),
                  ("p1", "2", "T", "c"): (0, False),
                  ("p2", "1", "T", "c"): (.8, False)}
        fig = overview_figure(nodes, populations=["p1", "p2"],
                              frequency_lookup=lookup, chrom_col="chrom",
                              frequency_mode="Bar plots")
        points = [t for t in fig.data if t.type == "bar" and t.name is not None]
        self.assertEqual(len(points), 2)
        self.assertNotEqual(points[0].y[0], points[1].y[0])
        self.assertEqual(len(points[0].x), 2)
        self.assertEqual(len(points[1].x), 1)
        # Both populations use identical SNP ranks even when one is missing.
        self.assertEqual(points[1].x[0], points[0].x[0])
        self.assertLess(points[0].x[0], points[0].x[1])
        self.assertLess(points[1].y[0], points[0].y[0]+.8)
        self.assertEqual(points[0].customdata[1][5], 0)
        self.assertEqual(points[1].customdata[0][6], 1)
        self.assertFalse(any(t.type == "box" for t in fig.data))
        self.assertAlmostEqual(points[0].y[0], -.48*.2)
        self.assertEqual(points[0].y[1], 0)
        self.assertAlmostEqual(points[1].y[0], -.48*.8)
        self.assertEqual(points[0].width, points[1].width)

    def test_boxenplots_share_fixed_zero_to_one_scale(self):
        matrix = [dict(CHROM="c", POS=str(i), a="T") for i in range(1, 4)]
        nodes, _ = build_hierarchy(matrix, {i: "T" for i in range(1, 4)}, ["a"])
        layout_nested_blocks(nodes)
        lookup = {("p", "1", "T"): (1.0, False), ("p", "2", "T"): (0.0, False),
                  ("q", "1", "T"): (.25, False), ("q", "2", "T"): (.75, False)}
        fig = overview_figure(nodes, populations=["p", "q"], frequency_lookup=lookup,
                              frequency_mode="Boxen plots")
        boxes = [t for t in fig.data if isinstance(t.meta,dict) and t.meta.get("kind")=="boxen_band"]
        self.assertEqual(len(boxes), 2)
        # Common affine transform, with 0 at the left and 1 at the right.
        left, right = min(boxes[0].x), max(boxes[0].x)
        self.assertAlmostEqual(boxes[1].x[0], left+.25*(right-left))
        self.assertAlmostEqual(boxes[1].x[1], left+.75*(right-left))
        stats = [t.customdata[0] for t in fig.data if t.type=="scatter" and t.customdata is not None]
        self.assertEqual(stats[0][2:4], [2, 1])
        self.assertAlmostEqual(stats[0][6], .5)
        self.assertTrue({"0", "0.5", "1"} <= {a.text for a in fig.layout.annotations})

    def test_boxenplots_exclude_absent_none_and_nan_but_keep_zero(self):
        matrix = [dict(CHROM="c", POS=str(i), a="T") for i in range(1, 6)]
        nodes, _ = build_hierarchy(matrix, {i:"T" for i in range(1,6)}, ["a"])
        layout_nested_blocks(nodes)
        lookup = {("p", "1", "T"): (0, False), ("p", "2", "T"): (.8, False),
                  ("p", "3", "T"): (None, True), ("p", "4", "T"): (float("nan"), False)}
        fig = overview_figure(nodes, populations=["p"], frequency_lookup=lookup, frequency_mode="Boxen plots")
        box = next(t for t in fig.data if isinstance(t.meta,dict) and t.meta.get("kind")=="boxen_band")
        self.assertAlmostEqual(box.meta["low"], .2)
        self.assertAlmostEqual(box.meta["high"], .6)
        stats = next(t.customdata[0] for t in fig.data if t.type=="scatter" and t.customdata is not None)
        self.assertEqual(stats[2:4], [2,3])
        self.assertAlmostEqual(stats[6], .4)
        self.assertNotIn("Population frequencies", fig.layout.xaxis.ticktext)

    def test_bar_widths_and_unit_outlines_across_different_snp_counts(self):
        matrix = [dict(CHROM="c", POS="1", a="T", b="T"),
                  dict(CHROM="c", POS="2", a="T", b="T"),
                  dict(CHROM="c", POS="3", a="T", b="A")]
        nodes, _ = build_hierarchy(matrix, {1:"T", 2:"T", 3:"T"}, ["a", "b"])
        layout_nested_blocks(nodes)
        lookup = {("p", str(i), "T"): (.2*i, False) for i in range(1,4)}
        fig = overview_figure(nodes, populations=["p"], frequency_lookup=lookup,
                              frequency_mode="Bar plots")
        bars = [t for t in fig.data if t.type=="bar" and t.name=="p"]
        self.assertEqual([len(t.x) for t in bars], [2,1])
        self.assertEqual(bars[0].width, bars[1].width)
        outlines = [shape for shape in fig.layout.shapes if shape.type=="rect"]
        self.assertFalse(any("SNP rank" in a.text or "outline height" in a.text for a in fig.layout.annotations))
        self.assertNotIn("Population frequencies", fig.layout.xaxis.ticktext)
        self.assertEqual(len(outlines),3)
        for outline in outlines:
            self.assertAlmostEqual(abs(outline.y1-outline.y0), .48)
            self.assertAlmostEqual(outline.x1-outline.x0, bars[0].width)

    def test_rotated_rows_and_metadata_alignment(self):
        matrix = [dict(CHROM="c", POS="1", a="T", b="A", c="T")]
        nodes, _ = build_hierarchy(matrix, {1: "T"}, ["a", "b", "c"])
        layout_nested_blocks(nodes)
        metadata = {"a": {"Person": "One"}, "b": {"Person": "Two"}}
        fig = overview_figure(nodes, metadata, ["Person"], "Person")
        self.assertEqual(fig.layout.yaxis.side, "right")
        self.assertEqual(list(fig.layout.yaxis.range), [3, 0])
        self.assertEqual(list(fig.layout.yaxis.tickvals), [.5, 1.5, 2.5])
        self.assertEqual(list(fig.data[0].x), [0])
        self.assertEqual(list(fig.data[0].y), [3])
        missing = next(trace for trace in fig.data if trace.name == "Person: Missing")
        index = nodes[0]["sample_order"].index("c")
        self.assertEqual(list(missing.y), [index+.5])
        self.assertEqual(missing.marker.color, "#b8b8b8")
        self.assertIn("One", [t.name for t in fig.data if t.name][0])

    def test_without_metadata(self):
        nodes, _ = build_hierarchy([], {}, ["a"])
        layout_nested_blocks(nodes)
        fig = overview_figure(nodes)
        self.assertEqual(list(fig.layout.yaxis.ticktext), ["a"])
        self.assertEqual(len(fig.data), 1)


if __name__ == "__main__":
    unittest.main()
