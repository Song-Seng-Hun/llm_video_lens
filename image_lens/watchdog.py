"""Hard POSIX parent deadline for one experimental tracker process group."""
import json
import os
import select
import signal
import subprocess


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
    and a timeout or interruption kills and reaps the whole group; the next job starts a fresh child."""
    def __init__(self):
        self.process = None
        self.command = None

    def run(self, command, arguments, max_seconds):
        if os.name != 'posix':
            raise ValueError('experimental tracking currently requires a POSIX parent watchdog')
        if not 0 < max_seconds <= 600:
            raise ValueError('invalid parent wall budget')
        if self.process is not None and (self.command != command or self.process.poll() is not None):
            self.close()
        if self.process is None:
            self.command = command
            self.process = subprocess.Popen([*command, '--serve'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            text=True, start_new_session=True)
        process = self.process
        try:
            process.stdin.write(json.dumps([str(a) for a in arguments])+'\n')
            process.stdin.flush()
            ready, _, _ = select.select([process.stdout], [], [], max_seconds)
            if not ready:
                self.process = None
                _kill(process)
                raise ChildTimeout(process.pid, '', '')
            line = process.stdout.readline()
        except ChildTimeout:
            raise
        except BaseException:
            self.process = None
            _kill(process)
            raise
        if not line:  # the child died without answering
            self.process = None
            process.stdin.close()
            return process.wait() or 1, '', 'tracking worker exited without a reply'
        reply = json.loads(line)
        if reply['returncode']:  # the child exits after a failed job
            self.close()
        return reply['returncode'], reply['stdout'], reply['stderr']

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        process.stdin.close()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _kill(process)


shared = None  # set by look.run so every tracking window reuses one child
