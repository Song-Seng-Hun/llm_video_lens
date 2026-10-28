"""Hard POSIX parent deadline for one experimental tracker process group."""
import os
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
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    return process.returncode, stdout, stderr
