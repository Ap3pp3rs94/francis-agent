"""
Francis package.
"""


def entrypoint():
    """Console entrypoint: run the module as if `python -m francis`."""
    import runpy

    runpy.run_module("francis.__main__", run_name="__main__")


__all__ = ["entrypoint"]
