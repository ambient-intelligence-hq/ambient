import unittest

from ambient.tools.citations import _extract_temporal_citations


class ExtractTemporalCitationsTests(unittest.TestCase):
    def assert_citation(self, text, kind, start, end):
        citations = _extract_temporal_citations(f"x <citation>{text}</citation> y")
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["type"], kind)
        self.assertAlmostEqual(citations[0]["start"], start)
        self.assertAlmostEqual(citations[0]["end"], end)

    def test_parses_mmss_frame(self):
        self.assert_citation("12:05", "frame", 725.0, 725.0)

    def test_parses_mmss_range(self):
        self.assert_citation("12:05::18:20", "range", 725.0, 1100.0)

    def test_parses_hhmmss_frame(self):
        self.assert_citation("01:00:58", "frame", 3658.0, 3658.0)

    def test_parses_hhmmss_range(self):
        self.assert_citation("00:00:58::00:01:02", "range", 58.0, 62.0)

    def test_parses_several_citations_in_order(self):
        text = "one <citation>00:10</citation> two <citation>00:20::00:30</citation>"
        citations = _extract_temporal_citations(text)
        self.assertEqual([c["type"] for c in citations], ["frame", "range"])

    def test_text_without_citations_returns_nothing(self):
        self.assertEqual(_extract_temporal_citations("no citations here"), [])


if __name__ == "__main__":
    unittest.main()
