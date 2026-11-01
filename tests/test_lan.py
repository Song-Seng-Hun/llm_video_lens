"""LAN API server on 127.0.0.1 (test mode) with a throwaway certificate and a fake image_lens CLI."""
import hashlib
import http.client
import json
import os
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'lan'))
import server  # noqa: E402
import sign  # noqa: E402

FAKE_CLI = r'''
import json, os, sys
a = sys.argv[1:]; ws = a[a.index('--workspace')+1]; cmd = a[2]; out = a[a.index('--out')+1]
if cmd == 'look':
    if a[a.index('--max-shots')+1] == '13':
        print(json.dumps(dict(status='failed', error_type='ValueError', error='no shots'))); sys.exit(1)
    os.makedirs(out)
    open(os.path.join(out, 'look.md'), 'w').write('# look\n- shot-0000 window\n')
    open(os.path.join(out, 'frame-sheet-1.jpg'), 'wb').write(b'\xff\xd8fake')
    open(os.path.join(out, 'steps.log'), 'w').write('not listed\n')
    print(json.dumps(dict(status='written', output=out, shots=1, windows=1, errors=0, elapsed_seconds=0.1)))
elif cmd == 'hang':
    import signal, time
    signal.signal(signal.SIGTERM, lambda *_: (open(out, 'w').write('term'), sys.exit(143)))
    time.sleep(60)
elif cmd == 'at':
    os.makedirs(out)
    open(os.path.join(out, 'at.md'), 'w').write('# ' + a[a.index('--span')+1] + '\n')
    open(os.path.join(out, 'at-frames-1.jpg'), 'wb').write(b'\xff\xd8at')
    print(json.dumps(dict(status='written', output=out)))
'''
VIDEO = b'\x00\x00\x00\x18ftypmp42' + bytes(range(256))*40


def private(path, text):
    path.write_text(text); path.chmod(0o600)
    return str(path)


class Lan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name)
        subprocess.run(['/usr/bin/openssl', 'req', '-x509', '-newkey', 'ec', '-pkeyopt', 'ec_paramgen_curve:P-256', '-pkeyopt', 'ec_param_enc:named_curve',
                        '-nodes', '-days', '1', '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
                        '-keyout', str(d/'tls.key'), '-out', str(d/'tls.pem')], check=True, capture_output=True)
        (d/'tls.key').chmod(0o600)
        fake = d/'repo'/'image_lens'; fake.mkdir(parents=True); (fake/'__main__.py').write_text(FAKE_CLI)
        for name in ('det.onnx', 'pose.onnx', 'ck.pth'):
            (d/name).write_bytes(b'x')
        cls.keys = {'claude-window': os.urandom(32), 'codex-window': os.urandom(32)}
        clients = {c: private(d/f'{c}.key', k.hex()+'\n') for c, k in cls.keys.items()}
        cls.cfg_raw = dict(bind=['127.0.0.1'], port=0, allow='127.0.0.0/8', clients=clients, cert=str(d/'tls.pem'),
                           key=str(d/'tls.key'), root=str(d/'root'), python=sys.executable, repo=str(d/'repo'),
                           detector=str(d/'det.onnx'), pose_model=str(d/'pose.onnx'), checkpoint=str(d/'ck.pth'),
                           test_mode=True, limits=dict(max_queue=2, auth_deadline=2.0, term_grace=5))
        cls.cfg_path = private(d/'api.json', json.dumps(cls.cfg_raw))
        cls.app = server.App(server.load_config(cls.cfg_path))
        cls.listeners = server.start(cls.app, ['127.0.0.1'], 0)
        cls.port = cls.listeners[0].server_address[1]
        cls.ctx = ssl.create_default_context(cafile=str(d/'tls.pem'))

    @classmethod
    def tearDownClass(cls):
        for s in cls.listeners:
            s.shutdown(); s.server_close()
        cls.app.close(); cls.tmp.cleanup()

    def call(self, method, target, body=b'', client='claude-window', headers=None, sign_as=None):
        h, _ = sign.sign_headers(client, sign_as or self.keys[client], method, target, body)
        h.update(headers or {})
        conn = http.client.HTTPSConnection('127.0.0.1', self.port, context=self.ctx, timeout=20)
        conn.request(method, target, body=body if method == 'POST' else None, headers=h)
        r = conn.getresponse(); data = r.read(); conn.close()
        return r.status, (json.loads(data) if r.getheader('Content-Type', '').startswith('application/json') else data), r

    def finished(self, jid):
        for _ in range(100):
            status, job, _ = self.call('GET', f'/jobs/{jid}')
            if job['state'] in ('done', 'failed'):
                return job
            time.sleep(.05)
        self.fail('job did not finish')

    def test_status_needs_a_valid_signature_once(self):
        self.assertEqual(self.call('GET', '/status')[0], 200)
        self.assertEqual(self.call('GET', '/status', sign_as=os.urandom(32))[0], 401)
        h, _ = sign.sign_headers('claude-window', self.keys['claude-window'], 'GET', '/status')
        for expect in (200, 401):  # the same nonce twice is a replay
            conn = http.client.HTTPSConnection('127.0.0.1', self.port, context=self.ctx, timeout=20)
            conn.request('GET', '/status', headers=h); self.assertEqual(conn.getresponse().status, expect); conn.close()
        old, _ = sign.sign_headers('claude-window', self.keys['claude-window'], 'GET', '/status', time_s=int(time.time())-600)
        conn = http.client.HTTPSConnection('127.0.0.1', self.port, context=self.ctx, timeout=20)
        conn.request('GET', '/status', headers=old); self.assertEqual(conn.getresponse().status, 401); conn.close()

    def test_look_then_files_and_at(self):
        status, job, _ = self.call('POST', '/look?name=' + sign.encode_name('테스트 clip') + '&max_shots=2', VIDEO)
        self.assertEqual(status, 202); self.assertEqual(job['name'], '테스트 clip')
        job = self.finished(job['id'])
        self.assertEqual(job['state'], 'done'); self.assertEqual(job['result']['windows'], 1)
        self.assertEqual([f['name'] for f in job['files']], ['frame-sheet-1.jpg', 'look.md'])  # steps.log is not served
        status, data, r = self.call('GET', f"/jobs/{job['id']}/files/look.md")
        self.assertEqual(data, b'# look\n- shot-0000 window\n'); self.assertEqual(r.getheader('X-Content-SHA256'), hashlib.sha256(data).hexdigest())
        self.assertEqual(self.call('GET', f"/jobs/{job['id']}/files/steps.log")[0], 404)
        self.assertEqual(self.call('GET', f"/jobs/{job['id']}/files/..%2Fapi.json")[0], 404)
        status, at, _ = self.call('POST', f"/jobs/{job['id']}/at?span=1-3")
        self.assertEqual(status, 200); self.assertEqual(at['text'], '# 1-3\n')
        self.assertEqual([f['name'] for f in at['files']], ['at-1-3/at-frames-1.jpg', 'at-1-3/at.md'])
        self.assertEqual(self.call('POST', f"/jobs/{job['id']}/at?span=1-3")[1]['text'], '# 1-3\n')  # cached
        self.assertEqual(self.call('GET', f"/jobs/{job['id']}/files/at-1-3/at.md")[1], b'# 1-3\n')
        self.assertEqual(self.call('POST', f"/jobs/{job['id']}/at?span=abc")[0], 400)
        # Another client does not see the job.
        self.assertEqual(self.call('GET', f"/jobs/{job['id']}", client='codex-window')[0], 404)
        self.assertNotIn(job['id'], [j['id'] for j in self.call('GET', '/jobs', client='codex-window')[1]['jobs']])

    def test_failed_look_reports_its_error(self):
        status, job, _ = self.call('POST', '/look?max_shots=13', VIDEO)
        job = self.finished(job['id'])
        self.assertEqual(job['state'], 'failed'); self.assertEqual(job['error'], dict(code='look_failed', message='no shots'))
        self.assertEqual(self.call('POST', f"/jobs/{job['id']}/at?span=1-3")[0], 409)

    def test_bad_uploads_are_refused(self):
        self.assertEqual(self.call('POST', '/look', b'not a video at all')[1]['error']['code'], 'invalid_video')
        self.assertEqual(self.call('POST', '/look?max_shots=0', VIDEO)[1]['error']['code'], 'invalid_argument')
        self.assertEqual(self.call('POST', '/look?colour=red', VIDEO)[0], 400)
        # The signed hash is of other bytes than the ones sent.
        h, _ = sign.sign_headers('claude-window', self.keys['claude-window'], 'POST', '/look', VIDEO)
        conn = http.client.HTTPSConnection('127.0.0.1', self.port, context=self.ctx, timeout=20)
        conn.request('POST', '/look', body=VIDEO[:-1]+b'!', headers=h); r = conn.getresponse()
        self.assertEqual(json.loads(r.read())['error']['code'], 'body_mismatch'); conn.close()
        self.assertEqual(sorted(p.suffix for p in Path(self.app.uploads).iterdir() if p.suffix == '.part'), [])

    def test_restart_marks_unfinished_jobs_and_cleanup_removes_old_ones(self):
        state = Path(self.app.state_dir)
        (state/('a'*32+'.json')).write_text(json.dumps(dict(id='a'*32, client='claude-window', state='running', created=1.)))
        old_look = Path(self.app.looks)/('b'*32); old_look.mkdir()
        (state/('b'*32+'.json')).write_text(json.dumps(dict(id='b'*32, client='claude-window', state='done', created=1., finished=1.)))
        again = server.App(server.load_config(self.cfg_path))
        try:
            self.assertEqual(again.jobs['a'*32]['error']['code'], 'interrupted')
            self.assertNotIn('b'*32, again.jobs); self.assertFalse(old_look.exists())
            # An `at` lock goes with its job.
            again.jobs['c'*32] = dict(id='c'*32, state='done', finished=1.); again.at_locks[('c'*32, '1-3')] = object()
            again.cleanup(); self.assertEqual(again.at_locks, {})
        finally:
            again.close()
            with self.app.lock:
                self.app.jobs.pop('a'*32, None)
            (state/('a'*32+'.json')).unlink()

    def test_status_reports_the_snapshot_commit_file(self):
        commit = Path(self.cfg_raw['repo'])/'COMMIT'
        commit.write_text('abc1234\n')
        again = server.App(server.load_config(self.cfg_path))
        try:
            self.assertEqual(again.commit, 'abc1234')
        finally:
            again.close(); commit.unlink()

    def test_a_peer_trickling_headers_is_cut_at_the_auth_deadline(self):
        raw = socket.create_connection(('127.0.0.1', self.port), timeout=10)
        s = self.ctx.wrap_socket(raw, server_hostname='127.0.0.1')
        s.sendall(b'GET /status HTTP/1.1\r\nHost: x\r\n')
        started = time.monotonic(); closed = False
        while time.monotonic() - started < 8:
            try:
                s.sendall(b'X-Slow: 1\r\n')  # each byte restarts read_timeout, not the auth deadline
                s.settimeout(.5)
                if s.recv(1) == b'':
                    closed = True; break
            except socket.timeout:
                continue
            except (OSError, ssl.SSLError):
                closed = True; break
        s.close()
        self.assertTrue(closed)
        self.assertLess(time.monotonic() - started, 6)

    def test_timeout_sends_sigterm_before_sigkill(self):
        marker = Path(self.tmp.name)/'hang-term'
        code, _, _ = self.app.run_cli(['hang', '--out', str(marker)], 1)
        self.assertIsNone(code)
        self.assertEqual(marker.read_text(), 'term')

    def test_production_config_refuses_loopback_and_overrides(self):
        d = Path(self.tmp.name)
        for change in (dict(test_mode=False), dict(test_mode=False, bind=['en0'], limits={}, allow='127.0.0.0/8', port=8766),
                       dict(test_mode=False, bind=['en0'], allow='192.0.2.0/24', port=8766)):
            with self.assertRaises(server.ConfigError):
                server.load_config(private(d/'c.json', json.dumps(dict(self.cfg_raw, **change))))


if __name__ == '__main__':
    unittest.main()
