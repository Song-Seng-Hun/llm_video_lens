import signal
from image_lens.cli import main


def _terminated(*_):
    # LAN server timeout or stop: raise SystemExit so finally blocks stop the children that run in their own sessions.
    raise SystemExit(143)


signal.signal(signal.SIGTERM, _terminated)
raise SystemExit(main())
