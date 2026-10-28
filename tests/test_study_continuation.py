"""Exercise dependency ordering, GPU isolation and independent failure handling."""
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments import continue_study as scheduler


class SchedulerTests(unittest.TestCase):
    def run_schedule(self, failing='none'):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root / 'fake.sh'
            launcher.write_text('''#!/bin/bash
echo "stage=$1 gpu=$CUDA_VISIBLE_DEVICES"
if [[ "$1" == "$FAILING_STAGE" ]]; then exit 7; fi
''')
            env = dict(CONTINUATION_DIR=str(root / 'run'), BF_REUSE_CHECKPOINTS='1',
                       CUDA_VISIBLE_DEVICES='', FAILING_STAGE=failing)
            tasks = {'train': (), 'baseline': (), 'compare': ('train', 'baseline')}
            with open(os.devnull, 'w') as sink, \
                    patch.dict(os.environ, env), patch.object(scheduler, 'TASKS', tasks), \
                    patch.object(scheduler, 'CPU_TASKS', {'compare'}), \
                    patch.object(sys, 'argv', ['scheduler', '--launcher', str(launcher)]), \
                    patch.object(scheduler.time, 'sleep', lambda _: None), \
                    contextlib.redirect_stdout(sink):
                try:
                    scheduler.main()
                except SystemExit as e:
                    self.assertEqual(failing, 'baseline')
                    self.assertIn('Incomplete stages', str(e))
            data = json.loads((root / 'run/status.json').read_text())
            logs = {p.stem: p.read_text() for p in (root / 'run').glob('*.log')}
            return data, logs

    def test_two_cards_and_dependencies(self):
        data, logs = self.run_schedule()
        self.assertEqual(data['status'], 'complete')
        tasks = data['tasks']
        self.assertNotEqual(tasks['train']['gpu'], tasks['baseline']['gpu'])
        self.assertGreaterEqual(tasks['compare']['started_at'], tasks['train']['finished_at'])
        self.assertGreaterEqual(tasks['compare']['started_at'], tasks['baseline']['finished_at'])
        for name, task in tasks.items():
            self.assertIn(f"gpu={'' if task['gpu'] == 'cpu' else task['gpu']}", logs[name])
        self.assertEqual(tasks['compare']['gpu'], 'cpu')

    def test_failure_blocks_dependents_but_finishes_independent_task(self):
        data, logs = self.run_schedule('baseline')
        self.assertEqual(data['status'], 'failed')
        self.assertEqual(data['tasks']['train']['status'], 'complete')
        self.assertEqual(data['tasks']['baseline']['returncode'], 7)
        self.assertEqual(data['tasks']['compare']['status'], 'blocked')
        self.assertNotIn('compare', logs)


if __name__ == '__main__':
    unittest.main()
