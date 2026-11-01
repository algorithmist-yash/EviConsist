import json, unittest
from pathlib import Path
from unittest.mock import patch
from baseline_supplement import reliability, completed

class SupplementTests(unittest.TestCase):
    def test_reliability_keeps_probability_one(self):
        bins=reliability([{'probability':0.,'gold':0},{'probability':1.,'gold':1}])
        self.assertEqual(sum(b['n'] for b in bins),2)
        self.assertEqual(bins[-1]['bin'],9)
        self.assertEqual(bins[-1]['positive_fraction'],1.)

    def test_wait_handles_concurrent_progress_write(self):
        directory=Path('simulated-progress-directory')
        with patch.object(Path,'read_text',side_effect=FileNotFoundError):
            self.assertFalse(completed(directory))
        with patch.object(Path,'read_text',return_value='{'):
            self.assertFalse(completed(directory))
        with patch.object(Path,'read_text',return_value=json.dumps({'status':'running'})):
            self.assertFalse(completed(directory))
        with patch.object(Path,'read_text',return_value=json.dumps({'status':'complete'})):
            self.assertTrue(completed(directory))

if __name__=='__main__': unittest.main()
