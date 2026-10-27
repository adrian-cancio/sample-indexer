import unittest

from sample_indexer import labels as L


def interpret(rel, ext=".wav", duration=None):
    return L.interpret_resource([rel], ext, duration)


class BpmParsingTests(unittest.TestCase):
    def test_explicit_markers_only(self):
        for name, value in [
            ("Beat17_125BPM(Full).wav", 125.0),
            ("Bongo shot 140 BPM 001.wav", 140.0),
            ("loop 128bpm.wav", 128.0),
            ("Loop 99Bpm.wav", 99.0),
            ("Thing BPM_150.wav", 150.0),
            ("Stereo Delay Hats 21 (128BPM)(1).wav", 128.0),
            ("odd 127.5BPM.wav", 127.5),
        ]:
            self.assertEqual(L.parse_bpm(name)["value"], value, name)

    def test_bare_numbers_are_not_tempo(self):
        for name in ["TR909 Kick.wav", "SVE2 Perc 39.wav", "Kick 160.wav", "GV_VocPad_120-Fmin.wav",
                     "HiSynthK05-C5.wav", "BORKICK 01.wav", "PE_CrunchCellos-Cmin2-95.wav", "808.wav"]:
            self.assertIsNone(L.parse_bpm(name)["value"], name)

    def test_kick_160_vs_160bpm(self):
        self.assertIsNone(L.parse_bpm("Kick 160.wav")["value"])
        self.assertEqual(L.parse_bpm("Kick 160BPM.wav")["value"], 160.0)

    def test_out_of_range_is_rejected_not_stored(self):
        for name in ["Loop 59BPM.wav", "Loop 241BPM.wav", "Loop 999BPM.wav", "x 20 BPM.wav"]:
            r = L.parse_bpm(name)
            self.assertIsNone(r["value"], name)
            self.assertTrue(r["rejected"], name)
        self.assertEqual(L.parse_bpm("a 60BPM.wav")["value"], 60.0)
        self.assertEqual(L.parse_bpm("a 240BPM.wav")["value"], 240.0)

    def test_conflicting_markers_are_ambiguous(self):
        r = L.parse_bpm("Loop 120BPM to 130BPM.wav")
        self.assertIsNone(r["value"])
        self.assertTrue(r["ambiguous"])

    def test_raw_token_is_kept(self):
        self.assertEqual(L.parse_bpm("Beat_125bpm.wav")["raw"], "125bpm")


class KeyParsingTests(unittest.TestCase):
    def test_only_fmin_in_voc_pad_example(self):
        name = "GV_VocPad_120-Fmin.wav"
        self.assertEqual(L.parse_key(name)["value"], "Fmin")
        self.assertEqual(L.parse_key(name)["raw"], "Fmin")
        self.assertIsNone(L.parse_bpm(name)["value"])  # 120 is a bare number here

    def test_accepted_tokens(self):
        for name, value in [("Pad_F#min.wav", "F#min"), ("Stab Cmaj.wav", "Cmaj"),
                            ("x-Bbminor-y.wav", "Bbmin"), ("Abmaj vox.wav", "Abmaj"),
                            ("Lead Emajor.wav", "Emaj")]:
            self.assertEqual(L.parse_key(name)["value"], value, name)

    def test_not_keys(self):
        for name in ["TR909 Kick.wav", "SVE2 Perc 39.wav", "A Bass.wav", "Kick C.wav", "Part A.wav",
                     "Cabinet Bass 1.wav", "Beat17_B.wav", "Amazing.wav", "Cmin7th.wav",
                     "PE_CrunchCellos-Cmin2-95.wav", "PE_AgedViolins-Emaj7min9-95.wav",
                     "Bmaj7.wav", "ABmin.wav", "HiSynthK05-C5.wav", "DIRT_ToneWheelChord_Gmin9.wav",
                     "gmin.wav"]:
            self.assertIsNone(L.parse_key(name)["value"], name)

    def test_variant_tokens_recorded_as_rejected_evidence(self):
        r = L.parse_key("PE_CrunchCellos-Cmin2-95.wav")
        self.assertIsNone(r["value"])
        self.assertEqual(r["rejected"], ["Cmin2"])


class BpmDurationTests(unittest.TestCase):
    def test_whole_bars(self):
        beats, check = L.bpm_duration_check(7.5, 128.0)  # 2 bars at 128 bpm = 7.5 s
        self.assertEqual((beats, check), (16.0, "whole_bars"))

    def test_tolerance_and_off_grid(self):
        self.assertEqual(L.bpm_duration_check(7.52, 128.0)[1], "whole_bars")  # inside tolerance
        self.assertEqual(L.bpm_duration_check(7.9, 128.0)[1], "off_grid")
        self.assertEqual(L.bpm_duration_check(3.75, 128.0), (8.0, "whole_bars"))  # 8 beats = 2 bars
        self.assertEqual(L.bpm_duration_check(1.40625, 128.0)[1], "whole_beats")  # 3 beats

    def test_long_file_with_fractional_beats_is_off_grid(self):
        # Real pilot case: 140bpm_break_02.aif, 31.1452 s -> 72.67 beats. A duration-proportional
        # tolerance of 0.5% wrongly reported it as whole beats.
        self.assertEqual(L.bpm_duration_check(31.1452, 140.0), (72.67, "off_grid"))
        self.assertEqual(L.bpm_duration_check(43.039, 72.5), (52.01, "whole_bars"))  # 52 beats = 13 bars, 4.5 ms off

    def test_no_bpm_no_beats(self):
        self.assertEqual(L.bpm_duration_check(4.0, None), (None, None))
        self.assertEqual(L.bpm_duration_check(None, 120.0), (None, None))

    def test_beats_hint_is_rounded_not_precise(self):
        beats, _ = L.bpm_duration_check(1.2345678, 133.0)
        self.assertEqual(beats, round(1.2345678 * 133.0 / 60, 2))


class KindCategoryTests(unittest.TestCase):
    def test_explicit_folders(self):
        r = interpret("Samples/Drums - One Shots/Kicks/BORKICK 01.wav")
        self.assertEqual((r.kind, r.kind_source, r.category), ("one_shot", "folder", "kick"))
        r = interpret("Samples/Loops/Beat Loops/DD_Beat_Valve_110-05.wav")
        self.assertEqual((r.kind, r.category), ("loop", "drums"))
        r = interpret("Samples/Synths/Synth Shots/Stab 1.wav")
        self.assertEqual(r.kind, "one_shot")
        r = interpret("Samples/Synths/Synth Loops/Worp A-03.wav")
        self.assertEqual(r.kind, "loop")

    def test_general_synth_and_fx_folders_do_not_set_kind(self):
        for rel in ["Samples/Synths/Leads/Lead 1.wav", "Samples/Synths/Misc/Thing.wav",
                    "Samples/Synths/Hooks/Hook.wav", "Samples/FX & Atmo/Impacts/Hit 01.wav",
                    "Samples/FX & Atmo/FX Loops/Whoosh 01.wav", "Samples/FX & Atmo/FX Shots/Zap 01.wav",
                    "Samples/FX & Atmo/Textures/Crackle_10.wav"]:
            self.assertEqual(interpret(rel).kind, "unknown", rel)

    def test_loop_name_under_one_shots_is_unknown(self):
        r = interpret("Samples/Drums - One Shots/Kicks/Big Loop 01.wav")
        self.assertEqual((r.kind, r.kind_source, r.kind_confidence), ("unknown", "folder_name_conflict", None))
        self.assertTrue(r.evidence["paths"][0]["name"]["kinds"] == ["loop"])  # raw hint preserved

    def test_name_agreeing_with_folder_raises_confidence(self):
        r = interpret("Samples/Loops/Kick Loops/Rolling Kick Loop 150BPM 5.wav")
        self.assertEqual((r.kind, r.kind_confidence, r.category, r.category_confidence), ("loop", "high", "kick", "high"))

    def test_unsorted_and_generic_hihats_and_dirty_fat_layer_are_not_specific(self):
        r = interpret("Samples/Unsorted/warehouse-techno-samples/Thing 01.wav")
        self.assertEqual((r.kind, r.category), ("unknown", "unknown"))
        r = interpret("Samples/Unsorted/industrial-samples/Kick 01.wav")  # name hint only, low confidence
        self.assertEqual((r.category, r.category_confidence, r.kind), ("kick", "low", "unknown"))
        r = interpret("Samples/Drums - One Shots/Hi-Hats/Hat 05.wav")
        self.assertEqual(r.category, "hihat")  # generic: neither open nor closed
        for sub in ("Dirty Loops", "Fat Loops", "Layer Loops"):
            r = interpret(f"Samples/Loops/{sub}/Dirty 12 (130BPM).wav")
            self.assertEqual((r.kind, r.category), ("loop", "unknown"), sub)
            self.assertNotIn(r.category, ("kick", "hihat", "hihat_open", "hihat_closed"))

    def test_open_closed_hats_come_from_specific_folders_only(self):
        self.assertEqual(interpret("Samples/Drums - One Shots/Hi-Hats Open/KHats Open-01.wav").category, "hihat_open")
        self.assertEqual(interpret("Samples/Drums - One Shots/Hi-Hats Closed/x 1.wav").category, "hihat_closed")

    def test_name_denying_the_folder_category(self):
        r = interpret("Samples/Loops/Kick Loops/Lucifer TR-666 - processed no kick.wav")
        self.assertEqual(r.category, "unknown")
        self.assertEqual(r.kind, "loop")
        self.assertIn("negated_categories", r.evidence["paths"][0]["name"])

    def test_conflicting_category_is_unknown(self):
        r = interpret("Samples/Drums - One Shots/Kicks/Snare 05.wav")
        self.assertEqual((r.category, r.category_source), ("unknown", "folder_name_conflict"))

    def test_broad_folder_categories_stay_broad(self):
        r = interpret("Samples/Loops/Beat Loops/Kick Top 01.wav")
        self.assertEqual(r.category, "drums")  # a name token never narrows a broad folder category
        self.assertEqual(r.category_confidence, "low")

    def test_midi_kind_is_ext_based_and_valid_untouched(self):
        r = interpret("MIDI/Basslines/Acid Bass 303.mid", ".mid")
        self.assertEqual((r.kind, r.kind_source, r.category), ("midi", "ext", "bass"))

    def test_bpm_duration_contradiction_makes_kind_unknown(self):
        # "one shot" folder but a 16-beat duration at the explicit 128BPM
        r = interpret("Samples/Drums - One Shots/Percussion/Perc 128BPM.wav", duration=7.5)
        self.assertEqual((r.kind, r.kind_source), ("unknown", "bpm_duration_conflict"))
        self.assertEqual(r.bpm_check, "whole_bars")
        # loop shorter than one beat
        r = interpret("Samples/Loops/Beat Loops/Beat 120BPM.wav", duration=0.1)
        self.assertEqual(r.kind, "unknown")
        # consistent: kept
        r = interpret("Samples/Loops/Beat Loops/Beat 128BPM.wav", duration=7.5)
        self.assertEqual(r.kind, "loop")

    def test_one_shot_with_bpm_in_name_and_short_duration_is_kept(self):
        r = interpret("Samples/Drums - One Shots/Kicks/Kick 160BPM.wav", duration=0.3)
        self.assertEqual((r.kind, r.bpm_name), ("one_shot", 160.0))

    def test_paths_that_disagree_preserve_ambiguity(self):
        r = L.interpret_resource(
            ["Samples/Drums - One Shots/Kicks/a.wav", "Samples/Drums - One Shots/Snares/b.wav"], ".wav", None
        )
        self.assertEqual((r.category, r.category_source), ("unknown", "path_conflict"))
        self.assertEqual(r.kind, "one_shot")  # agreeing part is kept
        self.assertEqual(len(r.evidence["paths"]), 2)
        self.assertTrue(r.evidence["conflicts"])
        r = L.interpret_resource(
            ["Samples/Drums - One Shots/Kicks/a.wav", "Samples/Loops/Kick Loops/a.wav"], ".wav", None
        )
        self.assertEqual((r.kind, r.category), ("unknown", "kick"))

    def test_evidence_separates_source_hint_from_interpretation(self):
        r = interpret("Samples/Drums - One Shots/Kicks/Kick 160BPM.wav")
        entry = r.evidence["paths"][0]
        self.assertIn("rules", entry["folder"])
        self.assertEqual(entry["kind"][0], "one_shot")
        self.assertEqual(entry["bpm"]["matches"], ["160BPM"])
        self.assertEqual(r.evidence["basis"], "path_name_provisional")


if __name__ == "__main__":
    unittest.main()
