"""Tactile maze paradigm (2-level binary decision tree with servo gratings).

``simplerCode.py`` is still a top-level script (it runs on import), so the
``amaze-tactile`` entry point executes it through ``runpy`` rather than
importing a function. Phase 5 of the v2 roadmap rebuilds it on the shared
modules; until then this wrapper keeps the command line stable.
"""

import runpy


def main() -> None:
    runpy.run_module("amazeing.simplermaze.simplerCode", run_name="__main__")
