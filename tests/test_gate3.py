import unittest

from experiments.gate3_temporal_context_reinstatement import run_seed


class Gate3Tests(unittest.TestCase):
    def test_episodic_reinstatement_beats_cue_only(self):
        result = run_seed(0)
        self.assertGreater(
            result["episodic_reinstatement_cosine"],
            result["cue_only_cosine"] + 0.8,
        )

    def test_shuffling_temporal_links_kills_reinstatement(self):
        result = run_seed(0)
        self.assertGreater(
            result["episodic_reinstatement_cosine"],
            result["shuffled_temporal_link_cosine"] + 0.8,
        )

    def test_index_strength_tracks_reinstatement(self):
        result = run_seed(0)
        self.assertGreater(
            result["index_weight_vs_reinstatement_corr"],
            0.7,
        )


if __name__ == "__main__":
    unittest.main()
