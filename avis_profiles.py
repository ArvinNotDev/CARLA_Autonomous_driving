from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class AvisGitHubProfileSync:
    """Public profile reader and authenticated writer for the AVIS profile folder."""

    def __init__(
        self,
        repository: str = "ArvinNotDev/CARLA_Autonomous_driving",
        branch: str = "main",
        directory: str = "avis_profiles",
        timeout_seconds: float = 12.0,
    ) -> None:
        self.repository = repository
        self.branch = branch
        self.directory = directory.strip("/")
        self.timeout_seconds = max(2.0, float(timeout_seconds))

    @staticmethod
    def _token() -> str | None:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token and token.strip():
            return token.strip()

        gh = shutil.which("gh")
        if not gh:
            return None
        try:
            result = subprocess.run(
                [gh, "auth", "token", "--hostname", "github.com"],
                check=True,
                capture_output=True,
                text=True,
                timeout=4.0,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        token = result.stdout.strip()
        return token or None

    def _url(self, path: str) -> str:
        encoded_path = urllib.parse.quote(path, safe="/")
        return (
            f"https://api.github.com/repos/{self.repository}/contents/"
            f"{encoded_path}?ref={urllib.parse.quote(self.branch, safe='')}"
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        authenticated: bool = False,
        allow_missing: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "AVIS-Profile-Sync",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if authenticated:
            token = self._token()
            if not token:
                raise RuntimeError(
                    "GitHub publishing is not authenticated. Run 'gh auth login' "
                    "on this PC, or set GITHUB_TOKEN / GH_TOKEN for this process."
                )
            headers["Authorization"] = f"Bearer {token}"

        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._url(path),
            data=data,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if allow_missing and exc.code == 404:
                return None
            if exc.code in {401, 403} and authenticated:
                raise RuntimeError(
                    "GitHub rejected the credentials or they lack Contents "
                    f"write permission (HTTP {exc.code})."
                ) from exc
            raise RuntimeError(
                f"GitHub profile request failed (HTTP {exc.code}): {detail[:300]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Could not reach GitHub: {exc}") from exc

        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("GitHub returned an invalid JSON response.") from exc

    @staticmethod
    def _filename(name: str) -> str:
        # Profile names are validated by AvisDashboardState and cannot contain
        # path separators. Percent-encoding keeps non-ASCII names portable.
        return urllib.parse.quote(name.strip(), safe="-_. ") + ".json"

    def list_profiles(self) -> list[str]:
        result = self._request(
            "GET",
            f"{self.directory}",
            allow_missing=True,
        )
        if result is None:
            return []
        if not isinstance(result, list):
            raise RuntimeError("The GitHub AVIS profile folder is not a directory.")
        names = []
        for item in result:
            if item.get("type") == "file" and str(item.get("name", "")).lower().endswith(".json"):
                names.append(str(item["name"])[:-5])
        return sorted(names, key=str.casefold)

    @staticmethod
    def _decode_profile(content_item: dict[str, Any]) -> dict[str, Any]:
        encoded = content_item.get("content")
        if not isinstance(encoded, str):
            raise RuntimeError("The selected GitHub profile has no readable content.")
        try:
            raw = base64.b64decode(encoded, validate=False).decode("utf-8")
            profile = json.loads(raw)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("The selected GitHub profile is not valid JSON.") from exc
        if not isinstance(profile, dict) or not isinstance(profile.get("settings"), dict):
            raise RuntimeError("The selected GitHub file is not an AVIS profile.")
        return profile

    def load_profile(self, name: str) -> dict[str, Any]:
        path = f"{self.directory}/{self._filename(name)}"
        result = self._request("GET", path, allow_missing=True)
        if result is None:
            raise RuntimeError(f"GitHub profile '{name}' was not found.")
        if not isinstance(result, dict):
            raise RuntimeError("GitHub returned an unexpected profile response.")
        return self._decode_profile(result)

    def publish_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        name = str(profile.get("name", "")).strip()
        settings = profile.get("settings")
        if not name or not isinstance(settings, dict):
            raise ValueError("A named AVIS profile with settings is required.")

        path = f"{self.directory}/{self._filename(name)}"
        existing = self._request(
            "GET",
            path,
            authenticated=True,
            allow_missing=True,
        )
        payload: dict[str, Any] = {
            "message": f"AVIS: save profile {name}",
            "content": base64.b64encode(
                (json.dumps(profile, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            ).decode("ascii"),
            "branch": self.branch,
        }
        if isinstance(existing, dict) and isinstance(existing.get("sha"), str):
            payload["sha"] = existing["sha"]
        result = self._request(
            "PUT",
            path,
            payload=payload,
            authenticated=True,
        )
        if not isinstance(result, dict):
            raise RuntimeError("GitHub did not confirm the profile save.")
        content = result.get("content") or {}
        commit = result.get("commit") or {}
        return {
            "name": name,
            "path": path,
            "sha": content.get("sha"),
            "commit_sha": commit.get("sha"),
            "html_url": content.get("html_url"),
        }
