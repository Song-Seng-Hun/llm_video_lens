#!/usr/bin/python3 -I
"""Image Lens LAN API server: lets Windows teammates run `look` and `at` on this Mac over the home LAN.
Stdlib only, runs on /usr/bin/python3 (3.9). The listener loads no model and decodes no video: each job runs the
image_lens CLI as a child process of the configured runtime python, by argv (no shell). Same TLS certificate, HMAC
signing (sign.py) and client keys as the Music Lens LAN API, on its own port, so one Windows client setup serves both.
Usage: server.py [--config PATH]   (default ~/.image-lens/api.json, mode 0600)
"""
import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import shutil
import signal
import socket
import ssl
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:  # -I drops the script directory from sys.path; this directory is the trusted install dir
    sys.path.insert(0, HERE)
import sign  # noqa: E402

VERSION = "0.1.0"
HOME = os.path.expanduser("~")
MiB = 1024 * 1024
DEFAULT_LIMITS = {
    "skew": 120, "max_conns": 8, "read_timeout": 30.0, "handshake_timeout": 10.0, "upload_deadline": 900.0,
    "max_body": 512 * MiB, "disk_floor": 4 * 1024 * MiB, "max_queue": 3, "look_seconds": 1800, "kill_grace": 300,
    "at_timeout": 180, "retention_days": 7, "jobs_listed": 20, "stdout_bytes": 4 * MiB, "log_bytes": 5 * MiB,
}
CONFIG_KEYS = {"bind", "port", "allow", "clients", "cert", "key", "root", "python", "repo", "detector", "pose_model",
               "checkpoint", "test_mode", "limits"}
CHILD_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
IPCONFIG = "/usr/sbin/ipconfig"
ERRORS = {
    "bad_request": (400, "Malformed request"),
    "invalid_argument": (400, "Invalid argument"),
    "invalid_video": (400, "Body is not an MP4/MOV, WebM/MKV or AVI file"),
    "body_mismatch": (400, "Body does not match X-ML-Body-SHA256"),
    "truncated_body": (400, "Request body ended early"),
    "unauthorized": (401, None),
    "not_found": (404, "Not found"),
    "method_not_allowed": (405, "Method not allowed"),
    "not_ready": (409, "The look job has not finished"),
    "busy": (409, "The job queue is full; try later"),
    "request_timeout": (408, "Upload exceeded its deadline"),
    "payload_too_large": (413, "Body exceeds the upload limit"),
    "look_failed": (422, "image_lens returned an error"),
    "internal_error": (500, "Internal error"),
    "insufficient_storage": (503, "Not enough free disk space"),
    "shutting_down": (503, "Server is stopping"),
    "timeout": (504, "Operation timed out"),
}
RE_IFACE = re.compile(r"\A[a-z]{1,8}[0-9]{1,3}\Z")
RE_TIME = re.compile(r"\A(0|[1-9][0-9]{0,11})\Z")
RE_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")
RE_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
RE_CL = re.compile(r"\A(0|[1-9][0-9]{0,11})\Z")
RE_SPAN = re.compile(r"\A[0-9]{1,5}(\.[0-9]{1,2})?-[0-9]{1,5}(\.[0-9]{1,2})?\Z")
RE_FILE = re.compile(r"\A(at-[0-9.]{1,8}-[0-9.]{1,8}/)?[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
ROUTE_JOB = re.compile(r"\A/jobs/([0-9a-f]{32})\Z")
ROUTE_FILE = re.compile(r"\A/jobs/([0-9a-f]{32})/files/(.+)\Z")
ROUTE_AT = re.compile(r"\A/jobs/([0-9a-f]{32})/at\Z")
FILE_TYPES = {".md": "text/markdown; charset=utf-8", ".json": "application/json", ".jpg": "image/jpeg",
              ".png": "image/png"}


class ConfigError(Exception):
    pass


class HttpError(Exception):
    def __init__(self, code, message=None):
        super().__init__(code)
        self.code, self.message = code, message


def _private_file(path, what):
    try:
        st = os.stat(path)
    except OSError:
        raise ConfigError(what + " not readable")
    if not stat.S_ISREG(st.st_mode):
        raise ConfigError(what + " is not a regular file")
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise ConfigError(what + " must be owned by this user with mode 0600")


def _abs(v, what):
    if not isinstance(v, str) or not os.path.isabs(v) or "\x00" in v:
        raise ConfigError(what + " must be an absolute path")
    return os.path.normpath(v)


def _private_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path, 0o700)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise ConfigError(path + " must be a 0700 directory owned by this user")
    return path


def load_config(path):
    _private_file(path, "config")
    try:
        with open(path, "rb") as f:
            raw = json.loads(f.read().decode("utf-8"))
    except (ValueError, OSError) as e:
        raise ConfigError("config is not valid JSON: " + type(e).__name__)
    if not isinstance(raw, dict):
        raise ConfigError("config must be an object")
    unknown = set(raw) - CONFIG_KEYS
    if unknown:
        raise ConfigError("unknown config keys: " + ", ".join(sorted(unknown)))
    test_mode = raw.get("test_mode", False)
    if test_mode not in (True, False):
        raise ConfigError("test_mode must be boolean")
    cfg = {"test_mode": test_mode}
    bind = raw.get("bind")
    if not isinstance(bind, list) or not bind or len(bind) > 4 or not all(isinstance(b, str) for b in bind):
        raise ConfigError("bind must be a non-empty list of interface names")
    for b in bind:
        if not RE_IFACE.match(b) and not (test_mode and b == "127.0.0.1"):
            raise ConfigError("bind entries must be interface names (en0, en1)")
    cfg["bind"] = list(bind)
    port = raw.get("port", 8766)
    if not isinstance(port, int) or isinstance(port, bool) or not ((0 if test_mode else 1024) <= port <= 65535):
        raise ConfigError("port out of range")
    cfg["port"] = port
    try:
        allow = ipaddress.ip_network(raw.get("allow"), strict=True)
    except (TypeError, ValueError):
        raise ConfigError("allow must be an IPv4 network like 192.0.2.0/24")
    if allow.version != 4 or (not test_mode and (allow.prefixlen < 24 or not allow.is_private or allow.is_loopback)):
        raise ConfigError("allow must be a private LAN /24 or narrower, never loopback")
    cfg["allow"] = allow
    clients = raw.get("clients")
    if not isinstance(clients, dict) or not clients or len(clients) > 16:
        raise ConfigError("clients must map client id to key path")
    keys = {}
    for cid, kpath in clients.items():
        if not sign.CLIENT_RE.match(cid):
            raise ConfigError("client id invalid")
        kpath = _abs(kpath, "client key path")
        _private_file(kpath, "client key file")
        try:
            with open(kpath, "rb") as f:
                keys[cid] = sign.parse_key_text(f.read(200).decode("ascii"))
        except (UnicodeDecodeError, sign.KeyFormatError, OSError):
            raise ConfigError("client key for " + cid + " must be 64 lowercase hex + at most one LF")
    if len(set(keys.values())) != len(keys):
        raise ConfigError("each client needs its own key")
    cfg["clients"] = keys
    cfg["cert"], cfg["key"] = _abs(raw.get("cert"), "cert"), _abs(raw.get("key"), "key")
    if not os.path.isfile(cfg["cert"]):
        raise ConfigError("cert missing")
    _private_file(cfg["key"], "TLS key")
    cfg["root"] = _abs(raw.get("root", HOME + "/AI-Look"), "root")
    cfg["python"] = _abs(raw.get("python"), "python")
    if not (os.path.isfile(cfg["python"]) and os.access(cfg["python"], os.X_OK)):
        raise ConfigError("python missing or not executable")
    cfg["repo"] = _abs(raw.get("repo"), "repo")
    if not os.path.isfile(os.path.join(cfg["repo"], "image_lens", "__main__.py")):
        raise ConfigError("repo must contain image_lens/__main__.py")
    for k in ("detector", "pose_model", "checkpoint"):
        cfg[k] = _abs(raw.get(k), k)
        if not os.path.isfile(cfg[k]):
            raise ConfigError(k + " missing")
    limits = dict(DEFAULT_LIMITS)
    over = raw.get("limits", {})
    if over:
        if not test_mode:
            raise ConfigError("limits overrides are test-only")
        if not isinstance(over, dict) or set(over) - set(DEFAULT_LIMITS):
            raise ConfigError("unknown limits")
        for k, v in over.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
                raise ConfigError("limit " + k + " must be positive")
            limits[k] = v
    cfg["limits"] = limits
    return cfg


def iface_addr(name):
    try:
        p = subprocess.run([IPCONFIG, "getifaddr", name], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=5, env={"PATH": CHILD_PATH})
        ip = ipaddress.ip_address(p.stdout.decode("ascii").strip()) if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError, ValueError):
        return None
    return ip if ip is not None and ip.version == 4 else None


def discover_addresses(cfg, lookup=iface_addr):
    """One address per distinct current IPv4 of the bind interfaces, inside allow. Never 0.0.0.0."""
    out = []
    for b in cfg["bind"]:
        ip = lookup(b) if RE_IFACE.match(b) else ipaddress.ip_address(b)
        if ip is None or ip.is_unspecified or ip not in cfg["allow"] or (ip.is_loopback and not cfg["test_mode"]):
            continue
        if str(ip) not in out:
            out.append(str(ip))
    return out


def sniff(head):
    """File extension for the container the first bytes name, or None."""
    if head[4:8] == b"ftyp":
        return ".mp4"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return ".mkv"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return ".avi"
    return None


class Log:
    """One JSON line per event; no query strings, labels or file contents."""

    def __init__(self, path, max_bytes):
        self.path, self.max_bytes, self.lock = path, max_bytes, threading.Lock()

    def event(self, name, **kw):
        line = json.dumps(dict(t=round(time.time(), 3), event=name, **kw), ensure_ascii=False) + "\n"
        with self.lock:
            try:
                if os.path.exists(self.path) and os.path.getsize(self.path) > self.max_bytes:
                    os.replace(self.path, self.path + ".1")
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError:
                pass


class Nonces:
    """Nonces seen within the clock skew. In memory: a restart forgets them, so a captured request could be replayed
    once within `skew` seconds of a restart; it would only repeat a read or queue the same video again."""

    def __init__(self):
        self.seen, self.lock = {}, threading.Lock()

    def admit(self, client, nonce, expiry, now):
        with self.lock:
            for k in [k for k, e in self.seen.items() if e < now]:
                del self.seen[k]
            if (client, nonce) in self.seen:
                return False
            self.seen[(client, nonce)] = expiry
            return True


def _write_json(path, obj):
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


class App:
    def __init__(self, cfg, clock=time.time):
        self.cfg, self.limits, self.clock = cfg, cfg["limits"], clock
        root = _private_dir(cfg["root"])
        self.state_dir = _private_dir(root + "/jobs")
        self.workspace = _private_dir(root + "/workspace")
        self.uploads = _private_dir(self.workspace + "/uploads")
        self.looks = _private_dir(self.workspace + "/looks")
        self.log = Log(_private_dir(root + "/logs") + "/api.log", self.limits["log_bytes"])
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.options |= getattr(ssl, "OP_NO_COMPRESSION", 0) | getattr(ssl, "OP_NO_RENEGOTIATION", 0)
        ctx.load_cert_chain(cfg["cert"], cfg["key"])
        self.ssl_ctx = ctx
        self.nonces = Nonces()
        self.lock = threading.Condition()
        self.jobs, self.queue, self.running, self.stopping = {}, [], None, False
        self.child = None
        self.conns = 0
        self.at_locks = {}
        self.commit = self._commit()
        for name in os.listdir(self.uploads):  # an upload cut short by a restart
            if name.endswith(".part"):
                os.unlink(os.path.join(self.uploads, name))
        for name in sorted(os.listdir(self.state_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.state_dir, name), encoding="utf-8") as f:
                    job = json.load(f)
            except (OSError, ValueError):
                continue
            if job.get("state") in ("queued", "running"):
                job.update(state="failed", finished=self.clock(),
                           error=dict(code="interrupted", message="the server restarted before the job finished"))
                _write_json(self.state_path(job["id"]), job)
            self.jobs[job["id"]] = job
        self.cleanup()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def _commit(self):
        try:  # a `git archive` snapshot has no .git: the install step writes COMMIT
            with open(os.path.join(self.cfg["repo"], "COMMIT"), encoding="ascii") as f:
                return f.read().strip() or None
        except (OSError, UnicodeDecodeError):
            pass
        try:
            p = subprocess.run(["/usr/bin/git", "-C", self.cfg["repo"], "rev-parse", "--short", "HEAD"], timeout=5,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               env={"PATH": CHILD_PATH, "HOME": HOME})
            return p.stdout.decode("ascii").strip() or None
        except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError):
            return None

    def state_path(self, jid):
        return os.path.join(self.state_dir, jid + ".json")

    def save(self, job):
        _write_json(self.state_path(job["id"]), job)

    def peer_allowed(self, ip):
        try:
            return ipaddress.ip_address(ip) in self.cfg["allow"]
        except ValueError:
            return False

    def acquire_conn(self):
        with self.lock:
            if self.conns >= self.limits["max_conns"]:
                return False
            self.conns += 1
            return True

    def release_conn(self):
        with self.lock:
            self.conns -= 1

    # ---- jobs
    def reserve(self, job):
        """Add the job as uploading, or raise busy when queued plus uploading jobs fill the queue."""
        with self.lock:
            if self.stopping:
                raise HttpError("shutting_down")
            waiting = len(self.queue) + sum(1 for j in self.jobs.values() if j["state"] == "uploading")
            if waiting >= self.limits["max_queue"]:
                raise HttpError("busy")
            self.jobs[job["id"]] = job

    def enqueue(self, job):
        with self.lock:
            job["state"] = "queued"
            self.save(job)
            self.queue.append(job["id"])
            self.lock.notify_all()

    def _work(self):
        while True:
            with self.lock:
                while not self.queue and not self.stopping:
                    self.lock.wait()
                if self.stopping:
                    return
                job = self.jobs[self.queue.pop(0)]
                job.update(state="running", started=self.clock())
                self.running = job["id"]
            try:  # nothing here may end the only worker thread (a full disk fails the save, not the queue)
                self.save(job)
                self._run_look(job)
            except Exception as e:
                job.update(state="failed", error=dict(code="internal_error", message=type(e).__name__))
            with self.lock:
                job["finished"] = self.clock()
                self.running = None
            try:
                self.save(job)
                self.cleanup()
            except Exception as e:
                self.log.event("worker_error", kind=type(e).__name__)
            self.log.event("job_done", job=job["id"], state=job["state"],
                           seconds=round(job["finished"] - job["started"], 1))

    def child_env(self):
        return {"PATH": CHILD_PATH, "HOME": HOME, "TMPDIR": os.environ.get("TMPDIR", "/tmp"), "LANG": "en_US.UTF-8"}

    def run_cli(self, args, timeout):
        """(returncode or None on timeout, last JSON object printed on stdout or None)."""
        argv = [self.cfg["python"], "-B", "-m", "image_lens", "--workspace", self.workspace] + args
        out_path = os.path.join(self.cfg["root"], "logs", "child-%s.out" % secrets.token_hex(4))
        with open(out_path, "w+b") as out:
            p = subprocess.Popen(argv, cwd=self.cfg["repo"], env=self.child_env(), stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True)
            with self.lock:
                self.child = p
            try:
                code = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                code = None
            finally:
                with self.lock:
                    self.child = None
            out.seek(0)
            text = out.read(self.limits["stdout_bytes"]).decode("utf-8", "replace")
        os.unlink(out_path)
        last = None
        for line in text.splitlines():
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                last = obj
        return code, last, text

    def _run_look(self, job):
        out = os.path.join(self.looks, job["id"])
        args = ["look", "--video", job["upload"], "--out", out, "--detector", self.cfg["detector"],
                "--pose-model", self.cfg["pose_model"], "--checkpoint", self.cfg["checkpoint"],
                "--max-shots", str(job["max_shots"]), "--max-seconds", str(int(self.limits["look_seconds"]))]
        code, result, _ = self.run_cli(args, self.limits["look_seconds"] + self.limits["kill_grace"])
        if code is None:
            job.update(state="failed", error=dict(code="timeout", message="look ran past its wall budget"))
        elif code != 0 or not result or result.get("status") == "failed":
            job.update(state="failed", error=dict(code="look_failed", message=str((result or {}).get("error", ""))[:600]))
        else:
            job.update(state="done", result={k: result[k] for k in result if k in
                                             ("status", "shots", "windows", "errors", "elapsed_seconds")})

    def run_at(self, job, span):
        out = os.path.join(self.looks, job["id"], "at-" + span)
        with self.lock:
            lock = self.at_locks.setdefault((job["id"], span), threading.Lock())
        with lock:
            if os.path.isfile(os.path.join(out, "at.md")):
                return out
            if os.path.isdir(out):  # left by a failed or killed run
                shutil.rmtree(out)
            code, result, _ = self.run_cli(["at", "--look", os.path.join(self.looks, job["id"]), "--span", span,
                                            "--out", out], self.limits["at_timeout"])
            if code is None:
                raise HttpError("timeout")
            if code != 0 or not result or result.get("status") != "written":
                raise HttpError("look_failed", str((result or {}).get("error", ""))[:600])
            return out

    def files(self, job):
        """[{name, bytes}] the client may fetch: top-level look files and `at` results, by extension."""
        root = os.path.join(self.looks, job["id"])
        found = []
        if not os.path.isdir(root):
            return found
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isfile(path) and os.path.splitext(name)[1] in FILE_TYPES and RE_FILE.match(name):
                found.append(dict(name=name, bytes=os.path.getsize(path)))
            elif os.path.isdir(path) and name.startswith("at-"):
                for sub in sorted(os.listdir(path)):
                    full = name + "/" + sub
                    if os.path.isfile(os.path.join(path, sub)) and os.path.splitext(sub)[1] in FILE_TYPES and RE_FILE.match(full):
                        found.append(dict(name=full, bytes=os.path.getsize(os.path.join(path, sub))))
        return found

    def cleanup(self):
        """Remove jobs finished more than retention_days ago: state, upload and look output."""
        cutoff = self.clock() - self.limits["retention_days"] * 86400
        with self.lock:
            old = [j for j in self.jobs.values() if j.get("finished") and j["finished"] < cutoff]
            for j in old:
                del self.jobs[j["id"]]
            for key in [k for k in self.at_locks if k[0] not in self.jobs]:
                del self.at_locks[key]
        for j in old:
            shutil.rmtree(os.path.join(self.looks, j["id"]), ignore_errors=True)
            for path in (j.get("upload"), self.state_path(j["id"])):
                if path and os.path.isfile(path):
                    os.unlink(path)
            self.log.event("job_removed", job=j["id"])

    def view(self, job):
        out = {k: job.get(k) for k in ("id", "name", "state", "created", "started", "finished", "bytes", "max_shots",
                                       "error", "result")}
        if job["state"] == "done":
            out["files"] = self.files(job)
        return out

    def close(self):
        with self.lock:
            self.stopping = True
            self.lock.notify_all()
            if self.child is not None:
                try:
                    os.killpg(self.child.pid, signal.SIGTERM)
                except OSError:
                    pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "image-lens-lan/" + VERSION
    sys_version = ""

    def log_message(self, format, *args):  # no request line (labels, spans) in any log
        pass

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def dispatch(self, method):
        app = self.server.app
        self.close_connection = True
        route = "-"
        try:
            client, target, cl = self.authenticate(app, method)
            path, _, query = target.partition("?")
            route = re.sub(r"[0-9a-f]{32}", "<id>", path.split("/files/")[0])
            self.route(app, client, method, path, query, cl)
            status = "ok"
        except HttpError as e:
            status = e.code
            self.send_error_code(e.code, e.message)
        except Exception as e:
            status = "internal_error"
            app.log.event("handler_error", kind=type(e).__name__)
            self.send_error_code("internal_error")
        app.log.event("request", ip=self.client_address[0], method=method, route=route[:64], status=status)

    def authenticate(self, app, method):
        lim = app.limits
        target = self.path
        if len(target) > 2048 or not target.startswith("/") or any(ord(c) < 0x21 or ord(c) > 0x7E for c in target):
            raise HttpError("bad_request")
        h = self.headers
        if h.get("Transfer-Encoding") is not None or h.get("Content-Encoding", "identity").lower() != "identity":
            raise HttpError("bad_request")
        cl_text = h.get("Content-Length")
        if cl_text is not None and not RE_CL.match(cl_text):
            raise HttpError("bad_request")
        if method == "GET":
            if cl_text not in (None, "0") or h.get("Content-Type") is not None:
                raise HttpError("bad_request")
            ct, cl = "", 0
        else:
            if cl_text is None or h.get("Content-Type") != "application/octet-stream":
                raise HttpError("bad_request")
            ct, cl = h["Content-Type"], int(cl_text)
        vals = {}
        for name in ("X-ML-Client", "X-ML-Time", "X-ML-Nonce", "X-ML-Body-SHA256", "X-ML-Sig"):
            if len(h.get_all(name) or []) != 1:
                raise HttpError("bad_request")
            vals[name] = h[name]
        if not (RE_TIME.match(vals["X-ML-Time"]) and RE_HEX32.match(vals["X-ML-Nonce"])
                and RE_HEX64.match(vals["X-ML-Body-SHA256"]) and RE_HEX64.match(vals["X-ML-Sig"])):
            raise HttpError("bad_request")
        client = vals["X-ML-Client"]
        key = app.cfg["clients"].get(client)
        if key is None:
            raise HttpError("unauthorized")
        now = app.clock()
        t = int(vals["X-ML-Time"])
        if abs(now - t) > lim["skew"]:
            raise HttpError("unauthorized")
        canon = sign.canonical(client, method, target, ct, cl, vals["X-ML-Time"], vals["X-ML-Nonce"],
                               vals["X-ML-Body-SHA256"])
        if not hmac.compare_digest(sign.signature(key, canon), vals["X-ML-Sig"]):
            raise HttpError("unauthorized")
        if cl == 0 and vals["X-ML-Body-SHA256"] != sign.EMPTY_SHA256:
            raise HttpError("unauthorized")
        if not app.nonces.admit(client, vals["X-ML-Nonce"], t + lim["skew"], now):
            raise HttpError("unauthorized")
        self.body_sha256 = vals["X-ML-Body-SHA256"]
        return client, target, cl

    def query(self, query, allowed):
        try:
            pairs = urllib.parse.parse_qsl(query, keep_blank_values=True, strict_parsing=bool(query))
        except ValueError:
            raise HttpError("bad_request")
        out = {}
        for k, v in pairs:
            if k not in allowed or k in out:
                raise HttpError("invalid_argument", "unknown or repeated parameter: " + k[:32])
            out[k] = v
        return out

    def own(self, app, client, jid):
        job = app.jobs.get(jid)
        if job is None or job["client"] != client:
            raise HttpError("not_found")
        return job

    def route(self, app, client, method, path, query, cl):
        if method == "POST" and path != "/look" and cl != 0:
            raise HttpError("bad_request")
        if path == "/status":
            self.need(method, "GET")
            with app.lock:
                queued, running = len(app.queue), app.running is not None
            return self.send_json(200, dict(service="image-lens-lan", version=VERSION, image_lens_commit=app.commit,
                                            running=running, queued=queued, max_queue=app.limits["max_queue"],
                                            max_body=app.limits["max_body"], retention_days=app.limits["retention_days"]))
        if path == "/look":
            self.need(method, "POST")
            return self.look(app, client, self.query(query, ("name", "max_shots")), cl)
        if path == "/jobs":
            self.need(method, "GET")
            mine = sorted((j for j in app.jobs.values() if j["client"] == client), key=lambda j: -j["created"])
            return self.send_json(200, dict(jobs=[{k: j.get(k) for k in ("id", "name", "state", "created", "finished")}
                                                  for j in mine[:app.limits["jobs_listed"]]]))
        m = ROUTE_JOB.match(path)
        if m:
            self.need(method, "GET")
            return self.send_json(200, app.view(self.own(app, client, m.group(1))))
        m = ROUTE_AT.match(path)
        if m:
            self.need(method, "POST")
            job = self.own(app, client, m.group(1))
            span = self.query(query, ("span",)).get("span", "")
            if not RE_SPAN.match(span):
                raise HttpError("invalid_argument", "span must be start-end seconds, e.g. 18-22")
            if job["state"] != "done":
                raise HttpError("not_ready")
            out = app.run_at(job, span)
            with open(os.path.join(out, "at.md"), encoding="utf-8") as f:
                text = f.read(256 * 1024)
            prefix = "at-" + span + "/"
            return self.send_json(200, dict(span=span, text=text,
                                            files=[f for f in app.files(job) if f["name"].startswith(prefix)]))
        m = ROUTE_FILE.match(path)
        if m:
            self.need(method, "GET")
            job = self.own(app, client, m.group(1))
            name = urllib.parse.unquote(m.group(2))
            if not RE_FILE.match(name) or name not in {f["name"] for f in app.files(job)}:
                raise HttpError("not_found")
            return self.send_file(os.path.join(app.looks, job["id"], name))
        raise HttpError("not_found")

    def need(self, method, want):
        if method != want:
            raise HttpError("method_not_allowed")

    def look(self, app, client, q, cl):
        name = q.get("name", "")
        if len(name.encode("utf-8")) > 120 or any(ord(c) < 0x20 or ord(c) == 0x7F for c in name):
            raise HttpError("invalid_argument", "name: at most 120 UTF-8 bytes, no control characters")
        try:
            max_shots = int(q.get("max_shots", "16"))
        except ValueError:
            raise HttpError("invalid_argument", "max_shots must be an integer")
        if not 1 <= max_shots <= 256:
            raise HttpError("invalid_argument", "max_shots must be 1..256")
        if cl == 0:
            raise HttpError("invalid_video")
        if cl > app.limits["max_body"]:
            raise HttpError("payload_too_large")
        if shutil.disk_usage(app.uploads).free < cl + app.limits["disk_floor"]:
            raise HttpError("insufficient_storage")
        jid = secrets.token_hex(16)
        job = dict(id=jid, client=client, name=name, state="uploading", created=app.clock(), bytes=cl,
                   max_shots=max_shots, upload_sha256=self.body_sha256)
        app.reserve(job)
        part = os.path.join(app.uploads, jid + ".part")
        try:
            ext = self.receive(app, part, cl)
        except BaseException:
            with app.lock:
                app.jobs.pop(jid, None)
            if os.path.exists(part):
                os.unlink(part)
            raise
        job["upload"] = os.path.join(app.uploads, jid + ext)
        os.replace(part, job["upload"])
        app.enqueue(job)
        app.log.event("job_queued", job=jid, client=client, bytes=cl)
        return self.send_json(202, app.view(job))

    def receive(self, app, part, cl):
        """Stream the body to `part`, check its hash and container; return the extension."""
        deadline = time.monotonic() + app.limits["upload_deadline"]
        digest, left, head = hashlib.sha256(), cl, b""
        fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            while left:
                if time.monotonic() > deadline:
                    raise HttpError("request_timeout")
                try:
                    block = self.rfile.read(min(left, MiB))
                except (socket.timeout, OSError):
                    raise HttpError("truncated_body")
                if not block:
                    raise HttpError("truncated_body")
                if len(head) < 16:
                    head += block[:16 - len(head)]
                digest.update(block)
                f.write(block)
                left -= len(block)
        if digest.hexdigest() != self.body_sha256:
            raise HttpError("body_mismatch")
        ext = sniff(head)
        if ext is None:
            raise HttpError("invalid_video")
        return ext

    def send_json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_error_code(self, code, message=None):
        status, text = ERRORS[code]
        try:
            self.send_json(status, dict(error=dict(code=code, message=message or text)) if text else dict(error=dict(code=code)))
        except OSError:
            pass

    def send_file(self, path):
        size = os.path.getsize(path)
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(MiB), b""):
                digest.update(block)
        self.send_response(200)
        self.send_header("Content-Type", FILE_TYPES[os.path.splitext(path)[1]])
        self.send_header("Content-Length", str(size))
        self.send_header("X-Content-SHA256", digest.hexdigest())
        self.send_header("Connection", "close")
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, MiB)


class Listener(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(self, addr, app):
        self.app = app
        super().__init__(addr, Handler)

    def server_bind(self):
        TCPServer.server_bind(self)  # skip getfqdn reverse lookup
        self.server_name, self.server_port = self.server_address[:2]

    def verify_request(self, request, client_address):
        # Peer outside allow: close before reading anything.
        return self.app.peer_allowed(client_address[0]) and self.app.acquire_conn()

    def finish_request(self, request, client_address):
        lim = self.app.limits
        request.settimeout(lim["handshake_timeout"])
        try:
            sock = self.app.ssl_ctx.wrap_socket(request, server_side=True)
        except (OSError, ValueError):
            return
        sock.settimeout(lim["read_timeout"])
        try:
            Handler(sock, client_address, self)
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def handle_error(self, request, client_address):
        self.app.log.event("listener_error")

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.app.release_conn()


def start(app, addresses, port):
    listeners = []
    for a in addresses:
        try:
            srv = Listener((a, port), app)
        except OSError as e:
            app.log.event("bind_failed", addr=a, errno=e.errno)
            continue
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
        listeners.append(srv)
        app.log.event("listening", addr=a, port=srv.server_address[1])
    return listeners


def main(argv=None):
    ap = argparse.ArgumentParser(description="Image Lens LAN API server")
    ap.add_argument("--config", default=HOME + "/.image-lens/api.json")
    ap.add_argument("--ready-file", help=argparse.SUPPRESS)  # tests: write the bound ports here
    ns = ap.parse_args(argv)
    try:
        cfg = load_config(ns.config)
        app = App(cfg)
    except (ConfigError, OSError, ssl.SSLError) as e:
        sys.stderr.write("image-lens-lan: config error: %s\n" % (e if isinstance(e, ConfigError) else type(e).__name__))
        return 2
    if ns.ready_file and not cfg["test_mode"]:
        sys.stderr.write("image-lens-lan: --ready-file is test-only\n")
        return 2
    addrs = discover_addresses(cfg)
    listeners = start(app, addrs, cfg["port"]) if addrs else []
    if not listeners:
        app.log.event("no_listener")
        sys.stderr.write("image-lens-lan: no address bound inside allow\n")
        return 1
    if ns.ready_file:
        _write_json(ns.ready_file, {"ports": [s.server_address[1] for s in listeners], "pid": os.getpid()})
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stopped.set())
    signal.signal(signal.SIGINT, lambda *a: stopped.set())
    while not stopped.wait(1):
        pass
    for s in listeners:
        s.shutdown()
        s.server_close()
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
