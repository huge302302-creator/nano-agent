import importlib.util
import json
import sys
import tempfile
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_agent_module(relative_path: str, module_name: str):
    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))

    setattr(fake_openai, "OpenAI", FakeOpenAI)
    previous_openai = sys.modules.get("openai")
    sys.modules["openai"] = fake_openai
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            REPO_ROOT / relative_path,
        )
        assert spec is not None
        module = importlib.util.module_from_spec(cast(ModuleSpec, spec))
        loader = spec.loader
        assert loader is not None
        loader.exec_module(module)
        return module
    finally:
        if previous_openai is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous_openai


def make_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AgentRegressionTests(unittest.TestCase):
    def setUp(self):
        self.agent = load_agent_module("agent.py", "nanoagent_agent")

    def test_parse_tool_arguments_reports_invalid_json(self):
        parsed = self.agent.parse_tool_arguments('{"command":')
        self.assertIn("_argument_error", parsed)
        self.assertIn("Invalid JSON arguments", parsed["_argument_error"])

    def test_parse_tool_arguments_ignores_non_object_payloads(self):
        self.assertEqual(self.agent.parse_tool_arguments('["ls"]'), {})

    def test_run_agent_returns_unknown_tool_error_to_model_loop(self):
        captured_messages = []

        def fake_create(*, model, messages, tools):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return make_response(
                    SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="tc-1",
                                function=SimpleNamespace(
                                    name="missing_tool", arguments="{}"
                                ),
                            )
                        ],
                    )
                )
            return make_response(SimpleNamespace(content="done", tool_calls=[]))

        setattr(
            self.agent,
            "client",
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        result = self.agent.run_agent("test unknown tool", max_iterations=2)

        self.assertEqual(result, "done")
        tool_messages = [
            m
            for m in captured_messages[1]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("Unknown tool 'missing_tool'", tool_messages[0]["content"])

    def test_run_agent_returns_argument_errors_to_model_loop(self):
        captured_messages = []

        def fake_create(*, model, messages, tools):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return make_response(
                    SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="tc-1",
                                function=SimpleNamespace(
                                    name="read_file",
                                    arguments='{"path":',
                                ),
                            )
                        ],
                    )
                )
            return make_response(SimpleNamespace(content="done", tool_calls=[]))

        setattr(
            self.agent,
            "client",
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        result = self.agent.run_agent("test invalid args", max_iterations=2)

        self.assertEqual(result, "done")
        tool_messages = [
            m
            for m in captured_messages[1]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("Invalid JSON arguments", tool_messages[0]["content"])

    def _fake_write_file_loop(self, target, captured_messages):
        """让假模型第一轮请求 write_file，第二轮返回最终回答。"""

        def fake_create(*, model, messages, tools):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return make_response(
                    SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="tc-1",
                                function=SimpleNamespace(
                                    name="write_file",
                                    arguments=json.dumps(
                                        {"path": str(target), "content": "hello"}
                                    ),
                                ),
                            )
                        ],
                    )
                )
            return make_response(SimpleNamespace(content="done", tool_calls=[]))

        setattr(
            self.agent,
            "client",
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

    def test_run_agent_denies_write_file_when_user_rejects(self):
        """SAFE_MODE 开启且用户拒绝时，write_file 不应真正落盘。"""
        captured_messages = []
        self.agent.SAFE_MODE = True

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "should_not_exist.txt"
            self._fake_write_file_loop(target, captured_messages)

            with mock.patch("builtins.input", return_value="n"):
                result = self.agent.run_agent("write a file", max_iterations=2)

            self.assertEqual(result, "done")
            self.assertFalse(target.exists())
            tool_messages = [
                m
                for m in captured_messages[1]
                if isinstance(m, dict) and m.get("role") == "tool"
            ]
            self.assertEqual(len(tool_messages), 1)
            self.assertIn("用户拒绝执行", tool_messages[0]["content"])

    def test_run_agent_writes_file_after_user_confirms(self):
        """用户输入 y 后才真正执行 write_file。"""
        captured_messages = []
        self.agent.SAFE_MODE = True

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "created.txt"
            self._fake_write_file_loop(target, captured_messages)

            with mock.patch("builtins.input", return_value="y"):
                result = self.agent.run_agent("write a file", max_iterations=2)

            self.assertEqual(result, "done")
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "hello")


class AgentPlusRegressionTests(unittest.TestCase):
    def setUp(self):
        self.agent = load_agent_module("agent-plus.py", "nanoagent_agent_plus")

    def test_run_agent_step_returns_unknown_tool_error(self):
        captured_messages = []

        def fake_create(*, model, messages, tools):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return make_response(
                    SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="tc-1",
                                function=SimpleNamespace(
                                    name="missing_tool", arguments="{}"
                                ),
                            )
                        ],
                    )
                )
            return make_response(SimpleNamespace(content="done", tool_calls=[]))

        setattr(
            self.agent,
            "client",
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        result, actions, _messages = self.agent.run_agent_step(
            "test unknown tool",
            [{"role": "system", "content": "hi"}],
            max_iterations=2,
        )

        self.assertEqual(result, "done")
        self.assertEqual(actions, [])
        tool_messages = [
            m
            for m in captured_messages[1]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("Unknown tool 'missing_tool'", tool_messages[0]["content"])

    def test_run_agent_step_returns_argument_errors(self):
        captured_messages = []

        def fake_create(*, model, messages, tools):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return make_response(
                    SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="tc-1",
                                function=SimpleNamespace(
                                    name="read_file", arguments='{"path":'
                                ),
                            )
                        ],
                    )
                )
            return make_response(SimpleNamespace(content="done", tool_calls=[]))

        setattr(
            self.agent,
            "client",
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        result, actions, _messages = self.agent.run_agent_step(
            "test invalid args",
            [{"role": "system", "content": "hi"}],
            max_iterations=2,
        )

        self.assertEqual(result, "done")
        self.assertEqual(actions, [])
        tool_messages = [
            m
            for m in captured_messages[1]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("Invalid JSON arguments", tool_messages[0]["content"])


class AgentClaudeCodeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.agent = load_agent_module(
            "agent-claudecode.py", "nanoagent_agent_claudecode"
        )

    def test_run_agent_step_returns_unknown_tool_error(self):
        captured_messages = []

        def fake_create(*, model, messages, tools):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return make_response(
                    SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="tc-1",
                                function=SimpleNamespace(
                                    name="missing_tool", arguments="{}"
                                ),
                            )
                        ],
                    )
                )
            return make_response(SimpleNamespace(content="done", tool_calls=[]))

        setattr(
            self.agent,
            "client",
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        result, _messages = self.agent.run_agent_step(
            [{"role": "system", "content": "hi"}, {"role": "user", "content": "test"}],
            self.agent.base_tools,
            max_iterations=2,
        )

        self.assertEqual(result, "done")
        tool_messages = [
            m
            for m in captured_messages[1]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("Unknown tool 'missing_tool'", tool_messages[0]["content"])

    def test_run_agent_step_returns_argument_errors(self):
        captured_messages = []

        def fake_create(*, model, messages, tools):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return make_response(
                    SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="tc-1",
                                function=SimpleNamespace(
                                    name="read", arguments='{"path":'
                                ),
                            )
                        ],
                    )
                )
            return make_response(SimpleNamespace(content="done", tool_calls=[]))

        setattr(
            self.agent,
            "client",
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        result, _messages = self.agent.run_agent_step(
            [{"role": "system", "content": "hi"}, {"role": "user", "content": "test"}],
            self.agent.base_tools,
            max_iterations=2,
        )

        self.assertEqual(result, "done")
        tool_messages = [
            m
            for m in captured_messages[1]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("Invalid JSON arguments", tool_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
