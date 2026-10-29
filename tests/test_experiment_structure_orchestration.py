"""Ensure study settings survive the launcher and resolve matching checkpoints."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StructureOrchestrationTests(unittest.TestCase):
    def dry_run(self, settings, *stages):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'study.env').write_text(settings)
            env = dict(os.environ, STUDY_ROOT=tmp, DRY_RUN='1', USE_WANDB='0', PY=sys.executable)
            for key in ('INTERACTION_POLICY', 'INTERACTION_PROB', 'ENCODER_STRUCTURE_CONDITIONING',
                        'DESIGN_GENERATOR'):
                env.pop(key, None)
            result = subprocess.run(['bash', str(ROOT / 'run_experiments.sh'), *stages],
                                    env=env, text=True, capture_output=True)
            return result

    def commands(self, output):
        return [shlex.split(line.strip()) for line in output.splitlines()
                if ' -m experiments.' in line]

    def test_exp1_structure_flags_and_checkpoint_agreement(self):
        result = self.dry_run('DESIGN_GENERATOR=gaussian_softmax\nINTERACTION_POLICY=optional\n'
                              'ENCODER_STRUCTURE_CONDITIONING=true\n', 'exp1')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        commands = self.commands(result.stdout)
        train = next(c for c in commands if 'experiments.experiment_1.model_class_cf_train' in c)
        ensemble = next(c for c in commands if 'experiments.experiment_1.model_class_cf_ensemble' in c)
        for c in (train, ensemble):
            self.assertEqual(c[c.index('--interaction_policy') + 1], 'optional')
        for c in commands:
            if '--checkpoint' in c:
                path = c[c.index('--checkpoint') + 1]
                self.assertTrue(path.endswith('_intoptional_beta22_encstruct.pt'), path)
                self.assertEqual(str(Path(path).parent), train[train.index('--checkpoint_dir') + 1].removeprefix('./'))
                self.assertIn('--encoder_structure_conditioning', c)
        self.assertIn('--encoder_structure_conditioning', train)
        self.assertEqual(train[train.index('--train_batch_size') + 1], '128')
        self.assertEqual(train[train.index('--epochs') + 1], '15000')

    def test_legacy_study_does_not_enable_new_mode(self):
        result = self.dry_run('DESIGN_GENERATOR=mixture\n', 'exp1')
        self.assertEqual(result.returncode, 0, result.stderr)
        for c in self.commands(result.stdout):
            self.assertNotIn('--encoder_structure_conditioning', c)
            if '--checkpoint' in c:
                self.assertNotIn('_intoptional', c[c.index('--checkpoint') + 1])

    def test_exp4_loads_conditioned_checkpoint_in_every_probe(self):
        result = self.dry_run('ENCODER_STRUCTURE_CONDITIONING=true\n', 'exp4')
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in self.commands(result.stdout):
            if '--cogformer_checkpoint' in command:
                self.assertEqual(command.count('--encoder_structure_conditioning'), 1)

    def test_lba_realdata_apply_preserves_checkpoint_conditioning(self):
        for enabled in (True, False):
            with self.subTest(conditioned=enabled):
                result = self.dry_run(
                    f'ENCODER_STRUCTURE_CONDITIONING={str(enabled).lower()}\n',
                    'e3_ppc_apply')
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                command, = self.commands(result.stdout)
                self.assertIn('experiments.experiment_2.lba_mc_cf_apply', command)
                self.assertEqual(command.count('--encoder_structure_conditioning'), int(enabled))
                checkpoint = command[command.index('--checkpoint') + 1]
                self.assertEqual('_encstruct' in checkpoint, enabled)

    def test_probability_format_and_invalid_settings(self):
        result = self.dry_run('INTERACTION_POLICY=optional\nINTERACTION_PROB=0.50\n', 'e1_train')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('_intoptional_0.5', result.stdout)
        for settings in ('INTERACTION_POLICY=invalid\n', 'INTERACTION_PROB=0.5\n',
                         'INTERACTION_POLICY=optional\nINTERACTION_PROB=2\n'):
            self.assertNotEqual(self.dry_run(settings, 'e1_train').returncode, 0)


if __name__ == '__main__':
    unittest.main()
