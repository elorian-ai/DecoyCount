import json
from pathlib import Path
import unittest
from unittest.mock import patch

from check_release import check


ROOT = Path(__file__).resolve().parents[2] / 'dataset'


class ReleaseTests(unittest.TestCase):
    def test_candidate_is_consistent_but_not_cleared(self):
        issues = check(ROOT, metadata_only=True)
        self.assertEqual(sum('redistribution not confirmed' in item for item in issues), 161)
        self.assertTrue(any('Task-3dn4bb24' in item for item in issues))

    def test_full_release_cannot_pass_without_images(self):
        self.assertTrue(any('missing or symlinked image' in item for item in check(ROOT)))

    def test_retained_blur_cannot_be_replaced(self):
        real_loads = json.loads

        def changed(text):
            value = real_loads(text)
            if isinstance(value, dict) and 'retained_secondary_blurs' in value:
                path = next(iter(value['retained_secondary_blurs']))
                value['retained_secondary_blurs'][path] = '0' * 40
            return value

        with patch('check_release.json.loads', side_effect=changed):
            with self.assertRaises(ValueError):
                check(ROOT, metadata_only=True)


if __name__ == '__main__':
    unittest.main()
