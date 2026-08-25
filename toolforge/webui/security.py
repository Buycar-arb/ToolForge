"""Security policy for exposing the Web UI beyond the local machine.

The views intentionally accept local filesystem paths and can start paid API
work.  Those capabilities are convenient on a developer's loopback interface,
but they must not be exposed to anonymous remote users.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping

WEBUI_USERNAME_ENV = "TOOLFORGE_WEBUI_USERNAME"
WEBUI_PASSWORD_ENV = "TOOLFORGE_WEBUI_PASSWORD"


def is_loopback_host(host: str) -> bool:
    """Return whether *host* unambiguously names the local machine only.

    Hostnames other than ``localhost`` are deliberately treated as remote.  A
    fail-closed check avoids DNS rebinding and keeps startup independent of DNS.
    """
    candidate = str(host).strip().lower()
    if candidate.rstrip(".") == "localhost":
        return True

    # IPv6 URLs may spell the address as ``[::1]``.  Strip only those brackets;
    # arbitrary hostnames containing brackets remain invalid/non-local.
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def launch_security(
    host: str,
    share: bool,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[tuple[str, str] | None, bool]:
    """Resolve the launch policy without importing Gradio.

    Loopback-only launches preserve the existing anonymous local workflow.
    Public-share links and non-loopback listeners require authentication before
    Gradio starts, because callbacks can read/write paths and make API calls.
    """
    remote = bool(share) or not is_loopback_host(host)
    if not remote:
        return None, True

    environment = os.environ if environ is None else environ
    username = environment.get(WEBUI_USERNAME_ENV, "").strip()
    password = environment.get(WEBUI_PASSWORD_ENV, "")
    if not username or not password.strip():
        raise RuntimeError(
            "Remote Web UI access requires authentication. Set both "
            f"{WEBUI_USERNAME_ENV} and {WEBUI_PASSWORD_ENV} before using "
            "--share or a non-loopback --host."
        )
    return (username, password), False
