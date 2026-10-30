# Image Lens LAN API (Mac server)

A small HTTPS listener for the home LAN that lets Windows teammates run `look` and `at` on this Mac. It uses the
same TLS certificate, request signing and client keys as the Music Lens LAN API (`music-lens-lan`), on its own
port (8766), so one Windows setup reaches both. The listener runs on `/usr/bin/python3` (3.9) with the standard
library only; it loads no model and decodes no video. Each job runs the image_lens CLI as a child process of the
configured runtime python, by argv.

Nothing here is installed. No LaunchAgent, firewall rule, certificate or key is created by this directory.

## Files

| File | Purpose |
| --- | --- |
| `server.py` | Config, en0/en1 address discovery, TLS 1.2+, HMAC auth, upload with hash and container check, one-at-a-time job queue, `at`, file downloads, 7-day cleanup, JSON log |
| `sign.py` | Canonical string and HMAC helper, copied from music-lens-lan at a5592bc |
| `image-lens-client.mjs` | Windows client (Node 22, no dependencies) |
| `api.example.json` | Example config (`~/.image-lens/api.json`, mode 0600) |
| `com.image-lens.lan-api.plist.template` | LaunchAgent **template** |
| `skill/image-lens-remote/SKILL.md` | Skill for the Windows AIs |

Tests: `tests/test_lan.py` (127.0.0.1 test mode, throwaway certificate, fake image_lens CLI). Run them under both
Pythons: `/usr/bin/python3 -B -m unittest tests.test_lan` and the runtime python.

## API

Every request is signed like Music Lens: headers `X-ML-Client`, `X-ML-Time` (skew 120 s), `X-ML-Nonce` (32 hex, once),
`X-ML-Body-SHA256`, `X-ML-Sig` = hex HMAC-SHA256 over the 8-field canonical string (see `sign.py`). POST bodies are
`application/octet-stream`. Errors are `{"error": {"code", "message"}}`. A client sees only its own jobs.

| Request | Does |
| --- | --- |
| `GET /status` | Service version, image_lens commit, whether a job is running, queue length and limits |
| `POST /look?name=LABEL&max_shots=N` + video body | Upload (MP4/MOV, WebM/MKV or AVI, at most 512 MiB) and queue a `look`; 202 with the job. `name` at most 120 UTF-8 bytes, `max_shots` 1..256 (default 16) |
| `GET /jobs` | The client's last 20 jobs |
| `GET /jobs/<id>` | State `uploading`, `queued`, `running`, `done` or `failed` (`error.code`: `look_failed`, `timeout`, `interrupted`); when done, `result` and the file list |
| `GET /jobs/<id>/files/<name>` | One listed file (`.md`, `.json`, `.jpg`, `.png`; `at-<span>/` files too) with `X-Content-SHA256` |
| `POST /jobs/<id>/at?span=18-22` (empty body) | Run `at` on a done job (at most 4 s); returns `at.md` text and its files. The same span again returns the stored result |

Jobs run one at a time; the queue holds 3 (queued plus uploading), then 409 `busy`. A `look` gets the CLI's
1800 s wall budget and is killed 300 s after it. A job and its files are removed 7 days after it finished (checked
at start and after each job).

## Install steps (user only; not done)

Do these only after approving an always-on listener (RunAtLoad + KeepAlive, en0/en1, port 8766, 172.30.1.0/24).

1. Copy the server and a snapshot of image_lens, so a branch checked out in `~/dev` does not change the service:
   ```sh
   mkdir -p ~/tools/image-lens/api ~/tools/image-lens/app && chmod 700 ~/tools/image-lens
   cp ~/dev/llm_image_lens/lan/{server.py,sign.py} ~/tools/image-lens/api/
   git -C ~/dev/llm_image_lens archive master | tar -x -C ~/tools/image-lens/app
   ```
2. Write the config: `mkdir -p ~/.image-lens ~/AI-Look && chmod 700 ~/.image-lens ~/AI-Look` (launchd writes its
   output files into `~/AI-Look`), copy `api.example.json` to
   `~/.image-lens/api.json`, `chmod 600` it, and fill in `python` and the three model paths (the values in the
   workspace `models.json`). It points at the Music Lens certificate and client keys; nothing new is created.
3. Firewall: `/usr/bin/python3` is already allowed for Music Lens; nothing to add.
4. Load the LaunchAgent:
   ```sh
   cp ~/dev/llm_image_lens/lan/com.image-lens.lan-api.plist.template ~/Library/LaunchAgents/com.image-lens.lan-api.plist
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.image-lens.lan-api.plist
   ```
   Unload: `launchctl bootout gui/$(id -u)/com.image-lens.lan-api`.
5. Windows (Claude-Window): copy `image-lens-client.mjs` and write `%USERPROFILE%\.image-lens\client.json` with
   port 8766 and `key_path`, `ca_path` and `pin_sha256` taken from the existing Music Lens client setup.

To update: unload, replace `~/tools/image-lens/app` with a fresh archive of master and copy the server files again
(step 1), then load. `/status` shows the image_lens commit the service runs.

## Runtime layout

- `~/AI-Look/jobs/<id>.json`: job state (0600). At start, jobs left `queued` or `running` become `failed`
  with `interrupted`; nothing is retried.
- `~/AI-Look/workspace/uploads/<id>.<ext>`: the upload; `.part` files from a cut-off upload are removed at start.
- `~/AI-Look/workspace/looks/<id>/`: the look output, `at-<span>/` inside it.
- `~/AI-Look/logs/api.log`: one JSON line per request or job event (no query strings or labels), rotated at 5 MiB.
- Nonces are kept in memory: a request captured on the LAN could be replayed once within 120 s of a restart. It
  would repeat a read or queue the same video again.

## Stopping

SIGTERM closes the listeners and sends SIGTERM to the running look's process group. Its state file stays `running`
and reads `interrupted` after the next start. Whether every tracking grandchild ends with it is untested.
