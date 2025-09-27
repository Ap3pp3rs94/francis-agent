from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="francis", description="Francis agent")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument(
        "--smoke", action="store_true", help="Quick start/exit smoke test"
    )
    args = parser.parse_args(argv)

    if args.version:
        try:
            print(version("francis-agent"))
        except PackageNotFoundError:
            print("unknown")
        return 0

    if args.smoke:
        print("ok")
        return 0

    # TODO: start the real agent loop here
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
