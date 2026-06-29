# Copyright 2024 GenesisAI Contributors
# Licensed under the Apache License, Version 2.0

The `core` package provides the foundational, dependency-free (aside
from the Python standard library) primitives shared by every other
GenesisAI subsystem: constants, the exception hierarchy, configuration
loading and validation, dependency injection, runtime/execution/session
context objects, state management, eventing, scheduling, resource and
process management, health checking, serialization, validation, logging,
profiling, and generic utilities.

No symbol is re-exported here. Every consumer must import directly from
the specific submodule it needs (e.g. `from core.exceptions import
ConfigurationError`), never from `core` itself. This keeps import
graphs explicit and avoids accidental circular imports, which matters
enormously in a codebase with this many interdependent packages.
"""

from __future__ import annotations
