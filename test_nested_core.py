import unittest
from nested_core import build_hierarchy, build_matrix_hierarchy, group_frequency_lookup, directed_changes, layout_nested_blocks, filter_isolates, shared_frequency_order, update_population_selection


class HierarchyTests(unittest.TestCase):
    def test_population_checkbox_edits_follow_sample_ids_after_filter_and_sort(self):
        selected = update_population_selection(["hidden","a"], ["c","b","a"], {0:{"Select":True},2:{"Select":False}})
        self.assertEqual(selected,["hidden","c"])
        self.assertEqual(update_population_selection(selected,["b"],{}),selected)
        self.assertEqual(update_population_selection(selected,["c"],{"0":{"Select":False}}),["hidden"])

    def test_matrix_groups_use_observed_group_alleles_and_include_shared_states(self):
        matrix = [dict(CHROM="c", POS="1", a="C", b="C", c="T", ref="T"),
                  dict(CHROM="c", POS="2", a="G", b="G", c="G", ref="G"),
                  dict(CHROM="c", POS="3", a="A", b="C", c="C", ref="C")]
        nodes, audit = build_matrix_hierarchy(matrix,["a","b","c"],"ref")
        self.assertEqual(audit,[])
        self.assertEqual([(n["carriers"], n["sites"][0]["allele"]) for n in nodes],
                         [(frozenset(["a","b","c"]),"G"),(frozenset(["a","b"]),"C"),(frozenset(["a"]),"A")])
        subset, _ = build_matrix_hierarchy(matrix,["a","b"],"ref")
        self.assertEqual({s["allele"] for s in subset[0]["sites"]},{"C","G"})

    def test_group_frequency_uses_counts_for_opposite_allele_and_keeps_zero(self):
        rows = [dict(Sample="p",true_pos="1",chrom_matrix="c",depth="10",A="0",C="8",G="0",T="2",
                     derived_allele="T",derived_allele_frequency="0.2"),
                dict(Sample="p",true_pos="2",chrom_matrix="c",derived_allele="T",derived_allele_frequency="0.3"),
                dict(Sample="p",true_pos="3",chrom_matrix="c",depth="0",C="0",derived_allele="C",derived_allele_frequency="0")]
        lookup = group_frequency_lookup(rows,{"p"})
        self.assertEqual(lookup[("p","1","C","c")],(.8,False))
        self.assertEqual(lookup[("p","1","A","c")],(0,False))
        self.assertEqual(lookup[("p","2","T","c")],(.3,False))
        self.assertNotIn(("p","2","C","c"),lookup)
        self.assertNotIn(("p","3","C","c"),lookup)

    def test_group_frequency_conflicting_duplicates_remain_missing(self):
        rows = [dict(Sample="p",true_pos="1",depth=10,C=count) for count in (2,3)]
        self.assertEqual(group_frequency_lookup(rows,{"p"},False)[("p","1","C")],(None,True))

    def test_shared_frequency_order_excludes_missing_from_denominator(self):
        sites = [dict(site=i, chrom="c", pos=str(i), allele="T") for i in range(1, 4)]
        lookup = {("a", "1", "T"): (.4, False),
                  ("a", "2", "T"): (.3, False), ("b", "2", "T"): (.3, False)}
        ordered = shared_frequency_order(sites, ["a", "b"], lookup)
        self.assertEqual([s["site"] for s in ordered], [1, 2, 3])
        self.assertEqual(ordered[0]["mean_frequency"], .4)
        self.assertIsNone(ordered[-1]["mean_frequency"])

    def test_metadata_filters_intersect_and_keep_missing_explicit(self):
        samples = ["a", "b", "c", "d"]
        metadata = {"a": dict(Person="One", week="0", BodySite="nose"),
                    "b": dict(Person="One", week="1", BodySite="nose"),
                    "c": dict(Person="Two", week="0", BodySite="mouth")}
        self.assertEqual(filter_isolates(samples, metadata, {}), samples)
        self.assertEqual(filter_isolates(samples, metadata, {"Person": ["One"], "week": ["0"], "BodySite": ["nose"]}), ["a"])
        self.assertEqual(filter_isolates(samples, metadata, {"Person": ["One", "Two"]}), ["a", "b", "c"])
        self.assertEqual(filter_isolates(samples, metadata, {"week": ["Missing"]}), ["d"])
        self.assertEqual(filter_isolates(samples, metadata, {"Person": ["One"], "BodySite": ["mouth"]}), [])

    def test_sample_order_preserves_residual_and_identical_samples(self):
        matrix = [dict(CHROM="c", POS="1", z="T", a="T", m="A", b="T"),
                  dict(CHROM="c", POS="2", z="T", a="T", m="A", b="A")]
        nodes, _ = build_hierarchy(matrix, {1: "T", 2: "T"}, ["m", "z", "b", "a"])
        layout_nested_blocks(nodes)
        order = nodes[0]["sample_order"]
        self.assertEqual(order, ["a", "z", "b", "m"])
        self.assertEqual(nodes[0]["end"], 4)
        self.assertEqual(len(set(order)), 4)
        for node in nodes:
            covered = order[node["start"]:node["end"]]
            self.assertEqual(set(covered), set(node["carriers"]))
            self.assertEqual(node["display_width"], len(node["carriers"]))
        subset, _ = build_hierarchy(matrix, {1: "T", 2: "T"}, ["m", "z"])
        layout_nested_blocks(subset)
        self.assertEqual(set(subset[0]["sample_order"]), {"m", "z"})
        self.assertEqual(subset[0]["end"], 2)

    def test_display_width_ignores_snp_count_and_preserves_nesting(self):
        matrix = [dict(CHROM="c", POS=str(i), a="T", b="A") for i in range(100)]
        matrix.append(dict(CHROM="c", POS="101", a="A", b="T"))
        nodes, _ = build_hierarchy(matrix, {i: "T" for i in range(1, 102)}, ["a", "b"])
        layout_nested_blocks(nodes)
        self.assertEqual(nodes[0]["end"], 2)
        self.assertEqual([n["display_width"] for n in nodes[1:]], [1, 1])
        for node in nodes[1:]:
            self.assertTrue(all(node["start"] < s["x"] < node["end"] for s in node["sites"]))
        self.assertLessEqual(nodes[1]["end"], nodes[2]["start"])

    def test_nested_counts_and_reselection(self):
        matrix = [dict(CHROM="c", POS=str(i), **calls) for i, calls in enumerate([
            dict(a="T", b="T", c="T"), dict(a="T", b="T", c="A"),
            dict(a="T", b="A", c="A"), dict(a="A", b="A", c="N")], 1)]
        changes = {i: "T" for i in range(1, 5)}
        nodes, audit = build_hierarchy(matrix, changes, ["a", "b", "c"])
        self.assertEqual([len(n["sites"]) for n in nodes], [1, 1, 1])
        self.assertEqual([n["depth"] for n in nodes], [0, 1, 2])
        self.assertEqual(nodes[0]["end"], 3)
        self.assertEqual(len(audit), 1)
        nodes, _ = build_hierarchy(matrix, changes, ["a", "b"])
        self.assertEqual(len(nodes[0]["sites"]), 2)

    def test_crossing_groups_are_reported(self):
        matrix = [dict(CHROM="c", POS="1", a="T", b="T", c="A"),
                  dict(CHROM="c", POS="2", a="A", b="T", c="T")]
        nodes, audit = build_hierarchy(matrix, {1: "T", 2: "T"}, ["a", "b", "c"])
        self.assertEqual(nodes[0]["end"], 0)
        self.assertEqual(len(audit), 2)

    def test_change_formats_and_recurrent_sites(self):
        self.assertEqual(directed_changes([dict(site="1", acctran_n_changes="1", acctran_edges="node_2->node_1:A->T")]), {1: "T"})
        self.assertEqual(directed_changes([dict(character="1", to_state="T"), dict(character="1", to_state="A")]), {1: ""})


if __name__ == "__main__":
    unittest.main()
