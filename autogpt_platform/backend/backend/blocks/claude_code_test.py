"""Claude Code only resumes a sandbox that was created for the same user, and
bills at least a minimum for every run that started the CLI."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.claude_code import ClaudeCodeBlock, ClaudeCodeExecutionError
from backend.executor.utils import ExecutionContext


async def test_resuming_another_users_sandbox_is_refused():
    block = ClaudeCodeBlock()
    theirs = AsyncMock()
    theirs.get_info = AsyncMock(
        return_value=AsyncMock(
            metadata={
                "service": "autogpt-platform",
                "autogpt_owner": "user:user-a",
                "autogpt_kind": "claude_code",
                "autogpt_source": "block",
                "autogpt_env": "dev",
                "autogpt_user": "user-a",
            }
        )
    )
    with patch("backend.blocks.claude_code.BaseAsyncSandbox") as cls:
        # The stamp is read through the static ``get_info`` before any connect.
        cls.get_info = AsyncMock(return_value=theirs.get_info.return_value)
        cls.connect = AsyncMock(return_value=theirs)
        # The block wraps every failure in its own error; the reason survives.
        with pytest.raises(ClaudeCodeExecutionError, match="does not belong"):
            await block.execute_claude_code(
                e2b_api_key="k",
                anthropic_api_key="a",
                prompt="hi",
                timeout=60,
                setup_commands=[],
                working_directory="/home/user",
                session_id="s1",
                existing_sandbox_id="sb-theirs",
                conversation_history="",
                dispose_sandbox=True,
                execution_context=ExecutionContext(user_id="user-b"),
            )
    cls.get_info.assert_awaited_once_with("sb-theirs", api_key="k")
    # Connecting would resume the other user's box on their bill.
    cls.connect.assert_not_awaited()
    theirs.commands.run.assert_not_awaited()
    theirs.kill.assert_not_awaited()


def _command_result(stdout: str = "", exit_code: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr="", exit_code=exit_code)


async def _run_new_sandbox(block: ClaudeCodeBlock, run_command, *, minimum=0.02):
    """Run ``execute_claude_code`` in a fresh fake sandbox.

    ``run_command(command)`` answers the CLI invocation; the install and the
    timestamp lookups succeed. ``minimum`` is the configured minimum charge.
    """

    async def run(command: str, **_kwargs):
        if command.startswith("npm install"):
            return _command_result()
        if command.startswith("date "):
            return _command_result("2026-01-01T00:00:00")
        return await run_command(command)

    sandbox = MagicMock(sandbox_id="sb-1")
    sandbox.commands.run = AsyncMock(side_effect=run)
    settings = MagicMock()
    settings.config.claude_code_min_run_cost_usd = minimum
    with (
        patch(
            "backend.blocks.claude_code.create_sandbox", AsyncMock(return_value=sandbox)
        ),
        patch("backend.blocks.claude_code.kill_sandbox", AsyncMock()),
        patch(
            "backend.blocks.claude_code.extract_and_store_sandbox_files",
            AsyncMock(return_value=[]),
        ),
        patch("backend.blocks.claude_code.Settings", return_value=settings),
    ):
        return await block.execute_claude_code(
            e2b_api_key="k",
            anthropic_api_key="a",
            prompt="hi",
            timeout=60,
            setup_commands=[],
            working_directory="/home/user",
            session_id="",
            existing_sandbox_id="",
            conversation_history="",
            dispose_sandbox=True,
            execution_context=ExecutionContext(user_id="user-a"),
        )


async def test_a_run_whose_cli_never_finishes_is_billed_the_minimum():
    block = ClaudeCodeBlock()

    async def timed_out(_command):
        raise Exception("Command ended without an end event")

    with pytest.raises(ClaudeCodeExecutionError):
        await _run_new_sandbox(block, timed_out)

    assert block.execution_stats.provider_cost == pytest.approx(0.02)
    assert block.execution_stats.provider_cost_type == "cost_usd"


async def test_a_run_with_a_nonzero_exit_is_billed_the_minimum():
    block = ClaudeCodeBlock()

    async def killed(_command):
        return _command_result(exit_code=137)

    with pytest.raises(ClaudeCodeExecutionError):
        await _run_new_sandbox(block, killed)

    assert block.execution_stats.provider_cost == pytest.approx(0.02)


@pytest.mark.parametrize(
    "reported, billed",
    [
        (0.50, 0.50),  # above the minimum: exactly what the CLI reported
        (0.01, 0.02),  # below it: topped up to the minimum
        (0.0, 0.02),  # reported as nothing: the minimum
        (None, 0.02),  # not reported at all: the minimum
    ],
)
async def test_a_clean_run_is_billed_the_larger_of_reported_and_minimum(
    reported, billed
):
    block = ClaudeCodeBlock()
    payload = {"result": "done"}
    if reported is not None:
        payload["total_cost_usd"] = reported

    async def clean(_command):
        return _command_result(json.dumps(payload))

    await _run_new_sandbox(block, clean)

    assert block.execution_stats.provider_cost == pytest.approx(billed)
    assert block.execution_stats.provider_cost_type == "cost_usd"


async def test_a_run_that_failed_before_the_cli_started_is_not_charged_a_minimum():
    block = ClaudeCodeBlock()
    sandbox = MagicMock(sandbox_id="sb-1")
    sandbox.commands.run = AsyncMock(return_value=_command_result(exit_code=1))
    with (
        patch(
            "backend.blocks.claude_code.create_sandbox", AsyncMock(return_value=sandbox)
        ),
        patch("backend.blocks.claude_code.kill_sandbox", AsyncMock()),
        pytest.raises(ClaudeCodeExecutionError, match="Failed to install"),
    ):
        await block.execute_claude_code(
            e2b_api_key="k",
            anthropic_api_key="a",
            prompt="hi",
            timeout=60,
            setup_commands=[],
            working_directory="/home/user",
            session_id="",
            existing_sandbox_id="",
            conversation_history="",
            dispose_sandbox=True,
            execution_context=ExecutionContext(user_id="user-a"),
        )

    assert block.execution_stats.provider_cost is None


async def test_a_minimum_of_zero_bills_only_what_the_cli_reports():
    block = ClaudeCodeBlock()

    async def timed_out(_command):
        raise Exception("Command ended without an end event")

    with pytest.raises(ClaudeCodeExecutionError):
        await _run_new_sandbox(block, timed_out, minimum=0)

    assert block.execution_stats.provider_cost is None
