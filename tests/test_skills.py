"""Agent skills under skills/: valid frontmatter, and every image_lens command they show exists with those flags."""
import re
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKILLS = sorted((REPO/'skills').glob('*/SKILL.md'))


def frontmatter(text):
    head = text.split('---\n')[1]
    return dict(line.split(': ', 1) for line in head.strip().splitlines())


def commands(text):
    """[(command, [--flags])] of each `python -m image_lens ...` line, after --workspace W."""
    out = []
    for line in text.splitlines():
        if line.startswith('python -m image_lens '):
            words = shlex.split(line)[3:]
            if words[:1] == ['--workspace']:
                words = words[2:]
            out.append((words[0], [w for w in words[1:] if w.startswith('--')]))
    return out


class Skills(unittest.TestCase):
    def test_frontmatter(self):
        self.assertTrue(SKILLS)
        for path in SKILLS:
            meta = frontmatter(path.read_text())
            self.assertEqual(meta['name'], path.parent.name, path)
            self.assertTrue(re.fullmatch(r'[a-z0-9-]{1,64}', meta['name']), path)
            self.assertTrue(0 < len(meta['description']) <= 1024, path)

    def test_commands_match_the_cli(self):
        seen = 0
        for path in SKILLS:
            for command, flags in commands(path.read_text()):
                help_text = subprocess.run([sys.executable, '-B', '-m', 'image_lens', command, '--help'], cwd=REPO,
                                           capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
                self.assertEqual(help_text.returncode, 0, (path.parent.name, command, help_text.stderr[-300:]))
                for flag in flags:
                    self.assertIn(flag, help_text.stdout, (path.parent.name, command))
                seen += 1
        self.assertGreater(seen, 0)


if __name__ == '__main__':
    unittest.main()
