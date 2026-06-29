# Copyright 2024 GenesisAI Contributors
# Licensed under the Apache License, Version 2.0

"""
core/exceptions.py

The canonical exception hierarchy for GenesisAI.

Every error raised by any first-party GenesisAI module (app, core,
engine, runtime, brain, cognition, memory, llm, reasoning, cli,
execution, ai_security, benchmarking_suite, updater_server,
crash_reporting, and all other subsystems) MUST derive, directly or
indirectly, from `GenesisAIError`. Raising a bare `Exception`, a bare
`ValueError`, or any other non-GenesisAI exception type from first-party
code is prohibited, because it breaks the structured-error contract
that the CLI's top-level handler (cli/dispatcher.py), the
crash_reporting subsystem, and the ai_security audit pipeline all rely
on to introspect failures uniformly.

This module MUST NOT import from any other first-party GenesisAI
package except `core.constants`, to avoid circular imports during
interpreter start-up.
"""

from __future__ import annotations

from typing import Any

from core.constants import ExitCode


class GenesisAIError(Exception):
    """
    Base class for every exception raised by GenesisAI.

    Attributes:
        message: A human-readable description of the failure.
        error_code: A short, stable, machine-readable identifier for
            this error category (e.g. "CONFIGURATION_ERROR"). Subclasses
            should override the `default_error_code` class attribute
            rather than passing `error_code` explicitly at every call
            site.
        details: Arbitrary structured context relevant to the failure
            (e.g. {"config_path": "/etc/genesisai/config.toml"}). Must
            only contain values that are safe to log and safe to include
            in crash reports — never place secrets, API keys, or raw
            credentials in `details`.
        exit_code: The process exit code (see core.constants.ExitCode)
            the CLI should return when this error propagates, uncaught,
            to the top-level command dispatcher.
        cause: The original exception that triggered this error, if any.
            Mirrors `__cause__` but is also explicitly retained so that
            `to_dict()` can serialize it even after exception-chaining
            context has been lost (for example, across a process
            boundary or when re-raised from a worker thread).
    """

    default_error_code: str = "GENESISAI_ERROR"
    default_exit_code: ExitCode = ExitCode.GENERAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        exit_code: ExitCode | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.error_code: str = error_code or self.default_error_code
        self.details: dict[str, Any] = details or {}
        self.exit_code: ExitCode = exit_code or self.default_exit_code
        self.cause: BaseException | None = cause
        if cause is not None and self.__cause__ is None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """Serialize this error into a structured, log/report-safe mapping."""
        payload: dict[str, Any] = {
            "error_type": type(self).__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "exit_code": int(self.exit_code),
        }
        if self.cause is not None:
            payload["cause"] = {
                "error_type": type(self.cause).__name__,
                "message": str(self.cause),
            }
        return payload

    def __str__(self) -> str:
        if self.details:
            return f"[{self.error_code}] {self.message} | details={self.details!r}"
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(message={self.message!r}, "
            f"error_code={self.error_code!r}, details={self.details!r})"
        )


# --------------------------------------------------------------------------
# Configuration & validation
# --------------------------------------------------------------------------


class ConfigurationError(GenesisAIError):
    """Configuration is missing, malformed, or otherwise unusable."""

    default_error_code = "CONFIGURATION_ERROR"
    default_exit_code = ExitCode.CONFIGURATION_ERROR


class ConfigurationNotFoundError(ConfigurationError):
    """No configuration source could be located."""

    default_error_code = "CONFIGURATION_NOT_FOUND"


class ConfigurationValidationError(ConfigurationError):
    """Configuration values failed schema or semantic validation."""

    default_error_code = "CONFIGURATION_VALIDATION_ERROR"


class ValidationError(GenesisAIError):
    """An input value failed validation outside of the configuration
    subsystem (CLI arguments, structured LLM output, etc.)."""

    default_error_code = "VALIDATION_ERROR"
    default_exit_code = ExitCode.VALIDATION_ERROR


# --------------------------------------------------------------------------
# Dependency injection / component lifecycle
# --------------------------------------------------------------------------


class DependencyResolutionError(GenesisAIError):
    """A requested dependency could not be resolved or instantiated."""

    default_error_code = "DEPENDENCY_RESOLUTION_ERROR"
    default_exit_code = ExitCode.DEPENDENCY_ERROR


class CircularDependencyError(DependencyResolutionError):
    """The dependency graph contains a cycle."""

    default_error_code = "CIRCULAR_DEPENDENCY_ERROR"


class InitializationError(GenesisAIError):
    """A component, service, or the application as a whole failed to
    initialize."""

    default_error_code = "INITIALIZATION_ERROR"


class ShutdownError(GenesisAIError):
    """A component failed to shut down cleanly. Must never suppress the
    original shutdown failure; callers should continue shutting down
    remaining components after catching this."""

    default_error_code = "SHUTDOWN_ERROR"


class ComponentNotRegisteredError(GenesisAIError):
    """A requested component identifier has no registered
    implementation."""

    default_error_code = "COMPONENT_NOT_REGISTERED"
    default_exit_code = ExitCode.NOT_FOUND_ERROR


class DuplicateRegistrationError(GenesisAIError):
    """An identifier is already registered and `overwrite=True` was not
    supplied."""

    default_error_code = "DUPLICATE_REGISTRATION_ERROR"


# --------------------------------------------------------------------------
# State management
# --------------------------------------------------------------------------


class StateError(GenesisAIError):
    """Generic state-handling failure."""

    default_error_code = "STATE_ERROR"


class InvalidStateTransitionError(StateError):
    """A state machine was asked to transition to a state unreachable
    via any defined transition."""

    default_error_code = "INVALID_STATE_TRANSITION"


# --------------------------------------------------------------------------
# Concurrency / scheduling
# --------------------------------------------------------------------------


class ConcurrencyError(GenesisAIError):
    """Generic concurrency failure (detected race condition, invalid
    cross-thread access, etc.)."""

    default_error_code = "CONCURRENCY_ERROR"


class LockAcquisitionError(ConcurrencyError):
    """A required lock could not be acquired within the configured
    timeout."""

    default_error_code = "LOCK_ACQUISITION_ERROR"
    default_exit_code = ExitCode.TIMEOUT_ERROR


class OperationTimeoutError(GenesisAIError):
    """An operation exceeded its allotted time budget. Named to avoid
    shadowing the builtin `TimeoutError`."""

    default_error_code = "OPERATION_TIMEOUT"
    default_exit_code = ExitCode.TIMEOUT_ERROR


class TaskCancelledError(GenesisAIError):
    """A scheduled task was cancelled before or during execution."""

    default_error_code = "TASK_CANCELLED"


# --------------------------------------------------------------------------
# Resources / I/O / persistence
# --------------------------------------------------------------------------


class ResourceError(GenesisAIError):
    """Generic resource-handling failure (files, sockets, handles)."""

    default_error_code = "RESOURCE_ERROR"


class ResourceNotFoundError(ResourceError):
    """A required resource (file, model, plugin, project, workspace,
    session, etc.) could not be located."""

    default_error_code = "RESOURCE_NOT_FOUND"
    default_exit_code = ExitCode.NOT_FOUND_ERROR


class ResourceExhaustedError(ResourceError):
    """A resource limit (memory, disk, quota, concurrency slots) has
    been exhausted."""

    default_error_code = "RESOURCE_EXHAUSTED"


class StorageError(GenesisAIError):
    """Generic persistence failure."""

    default_error_code = "STORAGE_ERROR"


class SerializationError(GenesisAIError):
    """An object could not be serialized to its target format."""

    default_error_code = "SERIALIZATION_ERROR"


class DeserializationError(GenesisAIError):
    """Serialized data could not be parsed back into a valid in-memory
    object."""

    default_error_code = "DESERIALIZATION_ERROR"


# --------------------------------------------------------------------------
# Memory subsystem
# --------------------------------------------------------------------------


class MemorySubsystemError(GenesisAIError):
    """Generic failure in the `memory` package. Named to avoid shadowing
    the builtin `MemoryError`, which specifically denotes interpreter
    out-of-memory conditions."""

    default_error_code = "MEMORY_SUBSYSTEM_ERROR"


class MemoryNotFoundError(MemorySubsystemError):
    """A requested memory record could not be located."""

    default_error_code = "MEMORY_NOT_FOUND"
    default_exit_code = ExitCode.NOT_FOUND_ERROR


class MemoryCapacityExceededError(MemorySubsystemError):
    """A bounded memory store (short-term, working memory) is full and
    cannot accept new entries without eviction."""

    default_error_code = "MEMORY_CAPACITY_EXCEEDED"


# --------------------------------------------------------------------------
# LLM / provider subsystem
# --------------------------------------------------------------------------


class ProviderError(GenesisAIError):
    """Generic LLM provider failure."""

    default_error_code = "PROVIDER_ERROR"
    default_exit_code = ExitCode.PROVIDER_ERROR


class ProviderUnavailableError(ProviderError):
    """A provider's API is unreachable or returns a server-side outage
    signal."""

    default_error_code = "PROVIDER_UNAVAILABLE"


class ProviderAuthenticationError(ProviderError):
    """A provider rejected the configured API credentials."""

    default_error_code = "PROVIDER_AUTHENTICATION_ERROR"
    default_exit_code = ExitCode.SECURITY_ERROR


class ModelNotFoundError(ProviderError):
    """The requested model identifier is not recognized by the selected
    provider."""

    default_error_code = "MODEL_NOT_FOUND"
    default_exit_code = ExitCode.NOT_FOUND_ERROR


class InferenceError(ProviderError):
    """A provider accepted a request but the inference call itself
    failed (malformed completion, content-filter rejection,
    context-length exceeded, etc.)."""

    default_error_code = "INFERENCE_ERROR"


class RateLimitExceededError(ProviderError):
    """A provider reported the caller exceeded its rate limit. `details`
    should include a `retry_after_seconds` key when the provider
    supplies one."""

    default_error_code = "RATE_LIMIT_EXCEEDED"


# --------------------------------------------------------------------------
# Security
# --------------------------------------------------------------------------


class SecurityError(GenesisAIError):
    """Generic security-policy violation detected by ai_security."""

    default_error_code = "SECURITY_ERROR"
    default_exit_code = ExitCode.SECURITY_ERROR


class AuthenticationError(SecurityError):
    """A user, service, or token failed authentication."""

    default_error_code = "AUTHENTICATION_ERROR"


class AuthorizationError(SecurityError):
    """An authenticated principal lacks permission to perform the
    requested action."""

    default_error_code = "AUTHORIZATION_ERROR"


class PermissionDeniedError(SecurityError):
    """A filesystem, sandbox, or capability-based permission check
    failed."""

    default_error_code = "PERMISSION_DENIED"
    default_exit_code = ExitCode.PERMISSION_ERROR


class UntrustedContentError(SecurityError):
    """Content (plugin code, downloaded model artifact, remote prompt
    template) failed an integrity or trust-verification check before
    being loaded or executed."""

    default_error_code = "UNTRUSTED_CONTENT_ERROR"


# --------------------------------------------------------------------------
# Networking
# --------------------------------------------------------------------------


class NetworkError(GenesisAIError):
    """Generic network failure not already classified as a provider,
    timeout, or rate-limit error."""

    default_error_code = "NETWORK_ERROR"
    default_exit_code = ExitCode.NETWORK_ERROR


# --------------------------------------------------------------------------
# Plugins / tools / extensibility
# --------------------------------------------------------------------------


class PluginError(GenesisAIError):
    """Generic plugin-subsystem failure."""

    default_error_code = "PLUGIN_ERROR"
    default_exit_code = ExitCode.PLUGIN_ERROR


class PluginNotFoundError(PluginError):
    """A referenced plugin identifier is not installed or not
    discoverable on the configured plugin path."""

    default_error_code = "PLUGIN_NOT_FOUND"
    default_exit_code = ExitCode.NOT_FOUND_ERROR


class PluginLoadError(PluginError):
    """A plugin was found but failed to import, initialize, or pass its
    manifest validation."""

    default_error_code = "PLUGIN_LOAD_ERROR"


class ToolError(GenesisAIError):
    """Generic failure in the agent tool-calling subsystem."""

    default_error_code = "TOOL_ERROR"


class ToolExecutionError(ToolError):
    """A tool was invoked successfully but its execution failed."""

    default_error_code = "TOOL_EXECUTION_ERROR"


class ToolNotFoundError(ToolError):
    """An agent or workflow referenced a tool name that is not
    registered."""

    default_error_code = "TOOL_NOT_FOUND"
    default_exit_code = ExitCode.NOT_FOUND_ERROR


# --------------------------------------------------------------------------
# Workflow / agent / reasoning
# --------------------------------------------------------------------------


class WorkflowError(GenesisAIError):
    """Generic workflow-engine failure."""

    default_error_code = "WORKFLOW_ERROR"


class WorkflowStepError(WorkflowError):
    """A single step within a workflow failed. `details` should include
    the `step_id` of the failing step."""

    default_error_code = "WORKFLOW_STEP_ERROR"


class AgentError(GenesisAIError):
    """Generic agent-subsystem failure (brain/, cognition/,
    reasoning/)."""

    default_error_code = "AGENT_ERROR"


class ReasoningError(AgentError):
    """A reasoning strategy (chain, tree, graph) failed to produce a
    usable result."""

    default_error_code = "REASONING_ERROR"


class PlanningError(AgentError):
    """The planning subsystem could not produce a valid plan for the
    given goal and constraints."""

    default_error_code = "PLANNING_ERROR"


# --------------------------------------------------------------------------
# Scheduling / execution engine
# --------------------------------------------------------------------------


class SchedulerError(GenesisAIError):
    """Generic scheduler failure across core/, engine/, runtime/, and
    execution/."""

    default_error_code = "SCHEDULER_ERROR"


class ExecutionError(GenesisAIError):
    """Generic execution-pipeline failure."""

    default_error_code = "EXECUTION_ERROR"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class CLIError(GenesisAIError):
    """Generic CLI-layer failure. The top-level command dispatcher
    (cli/dispatcher.py) catches `CLIError` before the broader
    `GenesisAIError` so CLI-specific exit codes are preserved."""

    default_error_code = "CLI_ERROR"


class CommandNotFoundError(CLIError):
    """The user invoked a command name not registered in
    core.command_registry / cli.command_registry."""

    default_error_code = "COMMAND_NOT_FOUND"
    default_exit_code = ExitCode.NOT_FOUND_ERROR


class CommandExecutionError(CLIError):
    """A recognized command failed during execution."""

    default_error_code = "COMMAND_EXECUTION_ERROR"


class ArgumentParsingError(CLIError):
    """CLI argument parsing failed (malformed flags, missing required
    arguments, invalid argument values)."""

    default_error_code = "ARGUMENT_PARSING_ERROR"
    default_exit_code = ExitCode.VALIDATION_ERROR


__all__ = [
    "GenesisAIError",
    "ConfigurationError",
    "ConfigurationNotFoundError",
    "ConfigurationValidationError",
    "ValidationError",
    "DependencyResolutionError",
    "CircularDependencyError",
    "InitializationError",
    "ShutdownError",
    "ComponentNotRegisteredError",
    "DuplicateRegistrationError",
    "StateError",
    "InvalidStateTransitionError",
    "ConcurrencyError",
    "LockAcquisitionError",
    "OperationTimeoutError",
    "TaskCancelledError",
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceExhaustedError",
    "StorageError",
    "SerializationError",
    "DeserializationError",
    "MemorySubsystemError",
    "MemoryNotFoundError",
    "MemoryCapacityExceededError",
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderAuthenticationError",
    "ModelNotFoundError",
    "InferenceError",
    "RateLimitExceededError",
    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    "PermissionDeniedError",
    "UntrustedContentError",
    "NetworkError",
    "PluginError",
    "PluginNotFoundError",
    "PluginLoadError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "WorkflowError",
    "WorkflowStepError",
    "AgentError",
    "ReasoningError",
    "PlanningError",
    "SchedulerError",
    "ExecutionError",
    "CLIError",
    "CommandNotFoundError",
    "CommandExecutionError",
    "ArgumentParsingError",
  ]
