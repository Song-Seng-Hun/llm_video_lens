"""Hard POSIX parent deadline for one experimental tracker process group."""
import json
import os
import select
import signal
import subprocess
import time


class ChildTimeout(TimeoutError):
    def __init__(self, pid, stdout, stderr):
        super().__init__('tracking parent wall budget exhausted; child process group killed and reaped; no retry')
        self.child_pid = pid
        self.stdout = stdout
        self.stderr = stderr


def run(command, max_seconds):
    if os.name != 'posix':
        raise ValueError('experimental tracking currently requires a POSIX parent watchdog')
    if not 0 < max_seconds <= 600:
        raise ValueError('invalid parent wall budget')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=max_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The child exited at the deadline; it must still be reaped.
        stdout, stderr = process.communicate()
        raise ChildTimeout(process.pid, stdout, stderr)
    except BaseException:
        # Interrupted callers still own this child's process group.
        _kill(process)
        raise
    return process.returncode, stdout, stderr


def _kill(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate()


class Worker:
    """One long-lived tracker child for many jobs (`tracking_child.py --serve`): each job keeps its own hard deadline,
    and a timeout or interruption kills and reaps the whole group; the next job starts a fresh child.
    submit() queues a job and returns at once, so the parent can work while the child tracks; result() takes the
    replies in submit order. A failed job ends the child, and the jobs queued behind it are sent again to a fresh one."""
    def __init__(self):
        self.process = None
        self.command = None
        self.pending = []
        self.buffer = b''

    def run(self, command, arguments, max_seconds):
        return self.result(self.submit(command, arguments, max_seconds))

    def submit(self, command, arguments, max_seconds):
        if os.name != 'posix':
            raise ValueError('experimental tracking currently requires a POSIX parent watchdog')
        if not 0 < max_seconds <= 600:
            raise ValueError('invalid parent wall budget')
        if self.process is not None and self.command != command:
            self.close()
        job = (command, [str(a) for a in arguments], max_seconds)
        self.pending.append(job)
        self._send(job)
        return job

    def _send(self, job):
        if self.process is not None and self.process.poll() is not None:
            self.process = None
        if self.process is None:
            self.command = job[0]; self.buffer = b''
            self.process = subprocess.Popen([*job[0], '--serve'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            bufsize=0, start_new_session=True)
        try:
            self.process.stdin.write((json.dumps(job[1])+'\n').encode())
            self.process.stdin.flush()
        except BaseException:
            self._drop()
            raise

    def result(self, job):
        if not self.pending or self.pending[0] is not job:
            raise ValueError('tracking replies are taken in submit order')
        process = self.process
        try:
            # The deadline starts when the parent waits for this job: never sooner than the job could have started.
            # Replies are read raw into self.buffer: a buffered readline could take the next reply too, where select()
            # no longer sees it (clip B: the parent then waited out the 300 s budget on a reply it already had).
            deadline = time.monotonic()+job[2]
            while b'\n' not in self.buffer:
                ready, _, _ = select.select([process.stdout], [], [], max(0, deadline-time.monotonic()))
                if not ready:
                    self._drop()
                    raise ChildTimeout(process.pid, '', '')
                chunk = os.read(process.stdout.fileno(), 1 << 16)
                if not chunk:
                    break
                self.buffer += chunk
            line, newline, self.buffer = self.buffer.partition(b'\n')
            if not newline:
                line = b''
        except ChildTimeout:
            raise
        except BaseException:
            self._drop()
            raise
        self.pending.pop(0)
        if not line:  # the child died without answering
            self.process = None
            process.stdin.close()
            code = process.wait() or 1
            self._resend()
            return code, '', 'tracking worker exited without a reply'
        reply = json.loads(line)
        if reply['returncode']:  # the child exits after a failed job
            self.process = None
            process.stdin.close()
            process.wait()
            self._resend()
        return reply['returncode'], reply['stdout'], reply['stderr']

    def _resend(self):
        for job in self.pending:
            self._send(job)

    def _drop(self):
        """Kill the child; the jobs still queued are dropped, so a later result() for them fails."""
        process, self.process, self.pending = self.process, None, []
        if process is not None:
            _kill(process)

    def close(self):
        process, self.process, self.pending = self.process, None, []
        if process is None:
            return
        process.stdin.close()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _kill(process)


shared = None  # set by look.run so every tracking window reuses one child
