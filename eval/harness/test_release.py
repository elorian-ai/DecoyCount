import json
from pathlib import Path
import unittest
from unittest.mock import patch

from check_release import check


ROOT = Path(__file__).resolve().parents[2] / 'dataset'


class ReleaseTests(unittest.TestCase):
    def test_v6_selection_checks_both_image_roles(self):
        issues = check(ROOT, metadata_only=True)
        self.assertFalse(any('redistribution not confirmed' in item for item in issues))
        self.assertEqual(len(issues), 6, 'Preserve the five incomplete v6 primary records')
        manifest = json.loads((ROOT / 'media_manifest.json').read_text())
        self.assertEqual((manifest['tasks'], manifest['secondary_images']), (1586, 783))
        excluded = json.loads((ROOT / 'exclusions.json').read_text())['excluded_tasks']
        self.assertEqual(len(excluded), 74)
        self.assertIn('Task-42zn064e', excluded, 'A secondary-image failure excludes the task')

    def test_full_release_cannot_pass_without_images(self):
        self.assertTrue(any('missing or symlinked image' in item for item in check(ROOT)))

    def test_media_mode_does_not_override_attribution(self):
        self.assertFalse(any('redistribution' in item for item in check(ROOT, media_only=True)))
        self.assertEqual(len(check(ROOT, metadata_only=True)), 6)

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
