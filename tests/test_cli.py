from subprocess import run, PIPE
import sys


def test_smoke_exits_ok():
    r = run(
        [sys.executable, "-m", "francis", "--smoke"],
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    assert r.returncode == 0
    assert "ok" in r.stdout
