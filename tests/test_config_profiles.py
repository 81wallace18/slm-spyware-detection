import unittest

from src.utils import load_config


class ConfigProfileTests(unittest.TestCase):
    def test_16gb_profile_overrides_nested_slm_values(self):
        cfg = load_config(profile="16gb")

        self.assertEqual(cfg["active_profile"], "16gb")
        self.assertEqual(cfg["slm"]["batch_size"], 1)
        self.assertEqual(cfg["slm"]["max_seq_len"], 512)
        self.assertEqual(cfg["slm"]["quantization"]["quant_type"], "nf4")
        self.assertTrue(cfg["slm"]["quantization"]["use_double_quant"])
        self.assertEqual(
            cfg["slm"]["benign_only"]["gradient_accumulation_steps"], 16
        )


if __name__ == "__main__":
    unittest.main()
