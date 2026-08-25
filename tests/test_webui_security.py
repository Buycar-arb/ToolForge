"""Security regressions for browser-facing helpers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from toolforge.webui import toollist
from toolforge.webui.security import (
    WEBUI_PASSWORD_ENV,
    WEBUI_USERNAME_ENV,
    is_loopback_host,
    launch_security,
)


def test_write_tool_list_quotes_untrusted_bank_text(
    tmp_path: Path, monkeypatch,
) -> None:
    target = tmp_path / "tool_selection.py"
    target.write_text(
        'TOOL_LIST = """\nold：description\n"""\nUNCHANGED = True\n',
        encoding="utf-8",
    )
    name = 'odd"\\name'
    description = 'safe"""\nINJECTED = True\n"""\\g<1>\\n\x00'
    monkeypatch.setattr(toollist, "describe_bank", lambda _path: {name: description})

    ok, _message = toollist.write_tool_list([name], target)

    assert ok
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert assignments == {
        "TOOL_LIST": f"\n{name}：{description}\n",
        "UNCHANGED": True,
    }
    assert toollist.read_tool_list(target) == [name]

    # A second edit must still locate the complete block; an escaped triple
    # quote in the first value must not truncate the regex match.
    ok, _message = toollist.write_tool_list([name], target)
    assert ok
    assert ast.parse(target.read_text(encoding="utf-8"))


def test_compare_view_renders_untrusted_role_as_text() -> None:
    compare_html = Path(__file__).parents[1] / "viewer" / "compare.html"
    source = compare_html.read_text(encoding="utf-8")

    assert "${esc(t(role) || role)}" in source
    assert '${t(role) || role}' not in source


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.12.34.56", "::1", "[::1]", "localhost", "LOCALHOST."],
)
def test_loopback_hosts_are_local(host: str) -> None:
    assert is_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "toolforge.local", ""])
def test_ambiguous_or_external_hosts_are_remote(host: str) -> None:
    assert not is_loopback_host(host)


def test_default_loopback_launch_remains_anonymous() -> None:
    assert launch_security("127.0.0.1", False, environ={}) == (None, True)


@pytest.mark.parametrize(
    ("host", "share"),
    [("0.0.0.0", False), ("127.0.0.1", True), ("::", True)],
)
def test_remote_launch_without_authentication_is_rejected(host: str, share: bool) -> None:
    with pytest.raises(RuntimeError, match="requires authentication"):
        launch_security(host, share, environ={})


@pytest.mark.parametrize(
    "environment",
    [
        {WEBUI_USERNAME_ENV: "operator"},
        {WEBUI_PASSWORD_ENV: "correct horse battery staple"},
        {WEBUI_USERNAME_ENV: "operator", WEBUI_PASSWORD_ENV: "   "},
    ],
)
def test_remote_launch_rejects_incomplete_environment_auth(environment: dict[str, str]) -> None:
    with pytest.raises(RuntimeError, match="requires authentication"):
        launch_security("0.0.0.0", False, environ=environment)


def test_remote_launch_uses_environment_auth_and_hides_errors() -> None:
    environment = {
        WEBUI_USERNAME_ENV: "operator",
        WEBUI_PASSWORD_ENV: "correct horse battery staple",
    }

    assert launch_security("0.0.0.0", False, environ=environment) == (
        ("operator", "correct horse battery staple"),
        False,
    )
