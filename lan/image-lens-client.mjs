#!/usr/bin/env node
// Image Lens LAN API client (Node 22, no dependencies). Signs requests like the Music Lens client (lan/sign.py).
// Config: %USERPROFILE%\.image-lens\client.json (or $IMAGE_LENS_CLIENT_CONFIG):
//   {"host": "<mac-host>.local", "port": 8766, "client": "claude-window",
//    "key_path": "...\\.music-lens\\keys\\mac.key", "ca_path": "...\\.music-lens\\mac-cert.pem",
//    "pin_sha256": "<sha256 of the server certificate DER, hex>", "addresses": ["192.0.2.x"]}
// `addresses` is optional: connect to these IPs (TLS name stays `host`) when the .local name does not resolve.
// Output: one JSON line on stdout. Exit 0 ok, 1 HTTP/runtime error, 2 usage/config, 3 queue busy.
import { createHash, createHmac, randomBytes } from 'node:crypto';
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, statSync, unlinkSync } from 'node:fs';
import { request } from 'node:https';
import { checkServerIdentity } from 'node:tls';
import { homedir } from 'node:os';
import { join, basename, dirname } from 'node:path';

const USAGE = `usage:
  image-lens-client status
  image-lens-client look <video> [--name LABEL] [--max-shots N] [--wait] [--out NEW_DIR]
  image-lens-client job <job_id> [--wait] [--out NEW_DIR]
  image-lens-client jobs
  image-lens-client get <job_id> <file name> --out <new file>
  image-lens-client at <job_id> <start-end seconds> [--out NEW_DIR]`;
const WAIT_S = 45 * 60, POLL_S = 10;

function out(obj, code) { process.stdout.write(JSON.stringify(obj) + '\n'); process.exit(code); }
function usage(message) { out({ error: { code: 'usage', message }, usage: USAGE }, 2); }

function config() {
  const path = process.env.IMAGE_LENS_CLIENT_CONFIG || join(homedir(), '.image-lens', 'client.json');
  let c;
  try { c = JSON.parse(readFileSync(path, 'utf8')); } catch (e) { out({ error: { code: 'config', message: `cannot read ${path}` } }, 2); }
  for (const k of ['host', 'port', 'client', 'key_path', 'ca_path']) if (!c[k]) out({ error: { code: 'config', message: `client.json needs ${k}` } }, 2);
  const keyText = readFileSync(c.key_path, 'ascii');
  if (!/^[0-9a-f]{64}\n?$/.test(keyText)) out({ error: { code: 'config', message: 'key must be 64 lowercase hex' } }, 2);
  return { ...c, key: Buffer.from(keyText.slice(0, 64), 'hex'), ca: readFileSync(c.ca_path) };
}

const EMPTY = createHash('sha256').digest('hex');
function headers(c, method, target, length, bodySha) {
  const time = String(Math.floor(Date.now() / 1000)), nonce = randomBytes(16).toString('hex');
  const ct = method === 'GET' ? '' : 'application/octet-stream', cl = method === 'GET' ? 0 : length;
  const canon = [c.client, method, target, ct, String(cl), time, nonce, bodySha].join('\n');
  const h = { 'X-ML-Client': c.client, 'X-ML-Time': time, 'X-ML-Nonce': nonce, 'X-ML-Body-SHA256': bodySha,
    'X-ML-Sig': createHmac('sha256', c.key).update(canon, 'utf8').digest('hex') };
  if (method !== 'GET') { h['Content-Type'] = ct; h['Content-Length'] = String(cl); }
  return h;
}

// One request on one of the candidate addresses; body is null, a Buffer or a file path. Resolves {status, headers, body}.
function call(c, method, target, body = null, saveTo = null) {
  const hosts = c.addresses?.length ? c.addresses : [c.host];
  let length = 0, sha = EMPTY;
  if (typeof body === 'string') { length = statSync(body).size; sha = fileSha(body); }
  else if (body) { length = body.length; sha = createHash('sha256').update(body).digest('hex'); }
  const attempt = (i) => new Promise((resolve, reject) => {
    const req = request({ host: hosts[i], port: c.port, method, path: target, servername: c.host, ca: c.ca,
      headers: headers(c, method, target, length, sha), timeout: 240000,  // above the server's 180 s `at` limit
      checkServerIdentity: (name, cert) => {  // CA and host name as usual, then the pin
        const bad = checkServerIdentity(name, cert);
        if (bad) return bad;
        if (c.pin_sha256 && createHash('sha256').update(cert.raw).digest('hex') !== c.pin_sha256.toLowerCase())
          return new Error('certificate pin mismatch');
        return undefined;
      } }, (res) => {
      const done = (data) => resolve({ status: res.statusCode, headers: res.headers, body: data });
      if (saveTo && res.statusCode === 200) {
        const hash = createHash('sha256'), file = createWriteStream(saveTo, { flags: 'wx' });
        const fail = (e) => { file.destroy(); try { unlinkSync(saveTo); } catch {} reject(e); };  // no half file left
        res.on('data', (d) => hash.update(d)); res.on('error', fail); res.on('aborted', () => fail(new Error('download cut off')));
        res.pipe(file);
        file.on('finish', () => done({ saved: saveTo, sha256: hash.digest('hex') })); file.on('error', fail);
      } else {
        const parts = []; res.on('data', (d) => parts.push(d)); res.on('error', reject);
        res.on('end', () => { const b = Buffer.concat(parts); try { done(JSON.parse(b.toString('utf8'))); } catch { done(b.toString('utf8')); } });
      }
    });
    req.on('timeout', () => req.destroy(new Error('timeout')));
    req.on('error', (e) => (e.code === 'ENOTFOUND' || e.code === 'ECONNREFUSED' || e.code === 'EHOSTUNREACH') && i + 1 < hosts.length ? attempt(i + 1).then(resolve, reject) : reject(e));
    if (typeof body === 'string') createReadStream(body).pipe(req); else req.end(body || undefined);
  });
  return attempt(0);
}

function fileSha(path) {  // whole file in memory: fine up to the server's 512 MiB upload limit
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

async function job(c, id, wait) {
  const until = Date.now() + WAIT_S * 1000;
  for (;;) {
    const r = await call(c, 'GET', `/jobs/${id}`);
    if (r.status !== 200 || !wait || ['done', 'failed'].includes(r.body.state)) return r;
    if (Date.now() > until) return { status: 0, body: { ...r.body, wait: 'gave up after 45 min; run job again' } };
    await new Promise((s) => setTimeout(s, POLL_S * 1000));
  }
}

async function download(c, id, files, dir, prefix = '') {
  if (existsSync(dir)) usage(`--out ${dir} already exists`);
  mkdirSync(dir, { recursive: true });
  const saved = [];
  // What a reader opens: Markdown, pictures and look.json; tracks, freezes and pose data stay on the Mac (`get` fetches one).
  for (const f of files.filter((f) => f.name.startsWith(prefix) && (/\.(md|jpg|png)$/.test(f.name) || f.name === 'look.json'))) {
    const path = join(dir, ...f.name.split('/'));
    mkdirSync(dirname(path), { recursive: true });
    const r = await call(c, 'GET', `/jobs/${id}/files/${f.name}`, null, path);
    if (r.status !== 200 || r.body.sha256 !== r.headers['x-content-sha256']) throw new Error(`download failed: ${f.name}`);
    saved.push(f.name);
  }
  return saved;
}

function finish(r, extra = {}) {
  const code = r.status >= 200 && r.status < 300 ? 0 : r.body?.error?.code === 'busy' ? 3 : 1;
  out({ http_status: r.status, response: r.body, ...extra }, code);
}

async function main() {
  const args = process.argv.slice(2), flags = {}, pos = [];
  for (let i = 0; i < args.length; i++) {
    if (!args[i].startsWith('--')) { pos.push(args[i]); continue; }
    const k = args[i].slice(2);
    if (k in flags) usage(`--${k} given twice`);
    if (k === 'wait') flags.wait = true;
    else if (['name', 'max-shots', 'out'].includes(k) && i + 1 < args.length) flags[k] = args[++i];
    else usage(`unknown flag --${k}`);
  }
  const [cmd, a, b] = pos, c = config();
  const id = (x) => (/^[0-9a-f]{32}$/.test(x || '') ? x : usage('job_id is 32 hex'));
  if (cmd === 'status' && pos.length === 1) return finish(await call(c, 'GET', '/status'));
  if (cmd === 'jobs' && pos.length === 1) return finish(await call(c, 'GET', '/jobs'));
  if (cmd === 'look' && pos.length === 2) {
    if (!existsSync(a)) usage(`no such file ${a}`);
    const q = [`name=${encodeURIComponent(flags.name ?? basename(a))}`];
    if (flags['max-shots']) q.push(`max_shots=${encodeURIComponent(flags['max-shots'])}`);
    const r = await call(c, 'POST', `/look?${q.join('&')}`, a);
    if (r.status !== 202 || !flags.wait) return finish(r);
    return showJob(c, r.body.id, true, flags.out);
  }
  if (cmd === 'job' && pos.length === 2) return showJob(c, id(a), flags.wait, flags.out);
  if (cmd === 'get' && pos.length === 3 && flags.out) {
    if (existsSync(flags.out)) usage(`--out ${flags.out} already exists`);
    const r = await call(c, 'GET', `/jobs/${id(a)}/files/${b}`, null, flags.out);
    return finish(r, r.status === 200 && r.body.sha256 !== r.headers['x-content-sha256'] ? { error: 'sha256 mismatch' } : {});
  }
  if (cmd === 'at' && pos.length === 3) {
    const r = await call(c, 'POST', `/jobs/${id(a)}/at?span=${encodeURIComponent(b)}`, Buffer.alloc(0));
    const saved = r.status === 200 && flags.out ? await download(c, a, r.body.files, flags.out, `at-${b}/`) : undefined;
    return finish(r, saved ? { saved_to: flags.out, saved } : {});
  }
  usage('unknown command or wrong arguments');
}

async function showJob(c, jid, wait, dir) {
  const r = await job(c, jid, wait);
  let extra = {};
  if (r.status === 200 && r.body.state === 'done') {
    const look = await call(c, 'GET', `/jobs/${jid}/files/look.md`);
    extra.look_md = look.status === 200 ? look.body : null;
    if (dir) extra = { ...extra, saved_to: dir, saved: await download(c, jid, r.body.files, dir) };
  }
  finish(r, extra);
}

main().catch((e) => out({ error: { code: 'runtime', message: String(e.message || e) } }, 1));
