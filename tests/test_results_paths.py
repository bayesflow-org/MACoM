"""Check that launchers and standalone scripts select the same study from any cwd."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ResultsPathsTests(unittest.TestCase):
    def resolve(self, cwd, study=None):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        for name in ("STUDY_ROOT", "RESULTS_ROOT", "DESIGN_GENERATOR",
                     "INTERACTION_POLICY", "INTERACTION_PROB",
                     "ENCODER_STRUCTURE_CONDITIONING"):
            env.pop(name, None)
        if study is not None:
            env["STUDY_ROOT"] = str(study)
        code = """
import json
from experiments.paths import RESULTS_ROOT, INPUT_DATA_ROOT
from experiments import shared_config
print(json.dumps(dict(results=str(RESULTS_ROOT), inputs=str(INPUT_DATA_ROOT),
                     checkpoint=shared_config.get('MC_CKPT'),
                     tag=shared_config.get('MC_TAG'))))
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                                text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_default_paths_and_checkpoint_ignore_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = self.resolve(ROOT)
            self.assertEqual(self.resolve(tmp), expected)
            self.assertEqual(self.resolve(ROOT / "experiments/experiment_4"), expected)
        self.assertEqual(expected["results"], str(ROOT / "outputs"))
        self.assertEqual(expected["inputs"], str(ROOT / "data"))
        self.assertIn("_gaussian_softmax_intoptional_beta22_encstruct", expected["tag"])
        self.assertEqual(Path(expected["checkpoint"]).parent,
                         ROOT / "outputs/checkpoints/model_class/cf" / expected["tag"])

    def test_relative_override_is_repository_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.resolve(tmp, "outputs"), self.resolve(ROOT))

    def test_alternative_study_controls_both_paths_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp) / "study with spaces"
            study.mkdir()
            (study / "study.env").write_text("DESIGN_GENERATOR=mixture\n")
            resolved = self.resolve(ROOT, study)
            self.assertEqual(resolved, self.resolve(tmp, study))
            self.assertEqual(resolved["results"], str(study))
            self.assertNotIn("_gaussian_softmax", resolved["tag"])
            self.assertTrue(Path(resolved["checkpoint"]).is_relative_to(study))
            self.assertNotIn("outputs", Path(resolved["checkpoint"]).relative_to(study).parts)


if __name__ == "__main__":
    unittest.main()
