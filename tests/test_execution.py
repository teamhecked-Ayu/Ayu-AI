"""Tests for immutable execution contracts and default-deny authorization."""
import tempfile
from dataclasses import replace
import unittest
from pathlib import Path

from ayu.execution import (
    ExecutionContract,
    ExecutionPolicy,
    ExecutionPolicyError,
    ResourceBudget,
)


class ExecutionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name).resolve()
        self.policy = ExecutionPolicy()

    def tearDown(self):
        self.temp.cleanup()

    def test_local_agent_contract_grants_only_read_and_local_inference(self):
        contract = ExecutionContract.for_agent(self.workspace, mode="local")
        self.assertEqual(set(contract.capabilities), {"workspace.read", "model.inference.local"})
        self.policy.authorize(contract, ("workspace.read",), workspace=self.workspace)
        self.policy.authorize(contract, ("model.inference.local",))
        with self.assertRaises(ExecutionPolicyError):
            self.policy.authorize(contract, ("workspace.write.task_worktree",), workspace=self.workspace)
        with self.assertRaises(ExecutionPolicyError):
            self.policy.authorize(contract, ("process.execute.sandbox",), workspace=self.workspace)

    def test_remote_context_requires_explicit_approval(self):
        with self.assertRaises(ExecutionPolicyError):
            ExecutionContract.for_agent(self.workspace, mode="remote")
        contract = ExecutionContract.for_agent(
            self.workspace, mode="remote", remote_context_approved=True
        )
        self.policy.authorize(contract, ("model.inference.remote", "remote_context.send"))
        self.assertTrue(contract.public_dict()["remote_context_approved"])

    def test_workspace_scope_cannot_be_changed(self):
        contract = ExecutionContract.for_agent(self.workspace, mode="local")
        other = self.workspace / "other"
        with self.assertRaisesRegex(ExecutionPolicyError, "outside the workspace"):
            self.policy.authorize(contract, ("workspace.read",), workspace=other)

    def test_unknown_and_unlisted_capabilities_are_denied(self):
        contract = ExecutionContract.for_agent(self.workspace, mode="local")
        with self.assertRaises(ExecutionPolicyError):
            self.policy.authorize(contract, ("shell.execute",))
        with self.assertRaises(ExecutionPolicyError):
            self.policy.authorize(contract, ())

    def test_resource_budgets_are_validated_and_enforced(self):
        with self.assertRaises(ValueError):
            ResourceBudget(100, 1, 0, 60, 4096)
        contract = ExecutionContract.for_agent(self.workspace, mode="local", max_tool_calls=2)
        self.policy.check_budget(contract, elapsed_seconds=1, model_calls=2, tool_calls=2)
        with self.assertRaisesRegex(ExecutionPolicyError, "tool-call budget"):
            self.policy.check_budget(contract, elapsed_seconds=1, tool_calls=3)
        with self.assertRaisesRegex(ExecutionPolicyError, "wall-clock budget"):
            self.policy.check_budget(contract, elapsed_seconds=181)
        with self.assertRaisesRegex(ExecutionPolicyError, "model-output budget"):
            self.policy.check_budget(contract, elapsed_seconds=1, output_bytes=64 * 1024 + 1)

    def test_contract_is_immutable(self):
        contract = ExecutionContract.for_agent(self.workspace, mode="local")
        with self.assertRaises(AttributeError):
            contract.mode = "remote"


    def test_forged_contract_cannot_add_capabilities_or_raise_budgets(self):
        contract = ExecutionContract.for_agent(self.workspace, mode="local")
        forged_capabilities = replace(
            contract,
            capabilities=contract.capabilities + ("workspace.write.task_worktree",),
        )
        with self.assertRaisesRegex(ExecutionPolicyError, "capabilities do not match"):
            self.policy.authorize(forged_capabilities, ("workspace.read",))
        forged_budget = replace(
            contract,
            budget=ResourceBudget(8, 4, 0, 180, 64 * 1024),
        )
        with self.assertRaisesRegex(ExecutionPolicyError, "resource ceilings"):
            self.policy.authorize(forged_budget, ("workspace.read",))

if __name__ == "__main__":
    unittest.main()
