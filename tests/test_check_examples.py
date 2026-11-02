"""tools/check_examples.py: the README quote it checks, and pass/fail on a look.md without running look."""
import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_examples', REPO/'tools'/'check_examples.py')
check_examples = importlib.util.module_from_spec(spec); spec.loader.exec_module(check_examples)


class Examples(unittest.TestCase):
    def test_readme_quote_is_the_dancer_window(self):
        readme = (REPO/'README.md').read_text()
        pieces = check_examples.readme_quote(readme)
        self.assertTrue(pieces[0].startswith(check_examples.EXAMPLES['dancer']['window']+' |'))
        self.assertGreater(len(pieces), 1)
        for other in ('docs/README.en.md', 'docs/README.ja.md'):  # the same Korean line is quoted in every README
            self.assertEqual(check_examples.readme_quote((REPO/other).read_text()), pieces, other)

    def test_pass_and_fail(self):
        readme = (REPO/'README.md').read_text()
        line = '- '+' X '.join(check_examples.readme_quote(readme))
        self.assertTrue(all(found for _, found in check_examples.check('dancer', 'header\n'+line+'\n', readme)))
        self.assertFalse(all(found for _, found in check_examples.check('dancer', line.replace('147px', '140px'), readme)))
        self.assertFalse(any(found for _, found in check_examples.check('dancer', 'no window line', readme)))
        good = '- 0.00–1.50s f0..44 | 전체: 카메라 모름(47%만 맞춤) | 몸 기준: 왼팔(화면 오른쪽: 손목) 오른쪽 124px(…) → 오른쪽 68px(…)'
        self.assertTrue(all(found for _, found in check_examples.check('live2d', good, readme)))
        self.assertFalse(all(found for _, found in check_examples.check('live2d', good.replace('카메라 모름', '회전 반시계 방향 3.0°'), readme)))


if __name__ == '__main__':
    unittest.main()
