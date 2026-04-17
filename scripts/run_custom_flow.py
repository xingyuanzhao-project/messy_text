"""Entry point for configuration-driven MessyText flows.

This script is the generic replacement for the model-specific runners under
``scripts/run_summary_conversation_{70b,qwen,mistral,...}.py``. It reads a
single flow YAML (the path is held in the module-level variable
``flow_config``), hands it to :func:`src.flow_builder.build_flow`, and
executes the returned :class:`src.flow_builder.FlowRunner`.

The script follows the same no-CLI-flags convention used by
:mod:`scripts.run_summary_conversation`: edit the ``flow_config`` variable
in this file before running to switch which flow executes. This keeps flow
selection explicit and reproducible in version control.

Contents
--------

- ``flow_config``: Module-level path string pointing at the flow YAML.
- :func:`main`: Builds and runs the flow.

Usage
-----

From the project root::

    python scripts/run_custom_flow.py

To switch flows, change ``flow_config`` to another file under
``config/flows/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.flow_builder import build_flow  # noqa: E402


flow_config: str = "config/flows/conversation_summary.yml"
"""Path to the flow YAML to execute.

Edit this variable to switch flows. Mirrors the
``settings = "config/settings.yaml"`` module-level variable used by the
existing ``run_summary_conversation.py``.
"""


def main() -> None:
    """Build the flow from :data:`flow_config` and run it synchronously.

    Returns:
        None.
    """
    flow_runner = build_flow(Path(flow_config))
    flow_runner.run()


if __name__ == "__main__":
    main()
