"""Explicit remote and local inference adapters; neither is selected silently."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
from urllib.parse import urlsplit
import urllib.request
from pathlib import Path

from .settings import Settings


class ProviderError(RuntimeError):
    """An inference configuration, execution, transport, or response error."""


def remote_endpoint_error(base_url: str) -> str | None:
    try: endpoint=urlsplit(base_url)
    except ValueError: return "AYU_BASE_URL is not a valid URL."
    if endpoint.scheme!="https" or not endpoint.hostname:
        return "Remote mode requires an HTTPS AYU_BASE_URL. Plain HTTP would expose the API key and conversation."
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        return "AYU_BASE_URL must not embed credentials, query parameters, or a fragment; configure the API key with AYU_API_KEY."
    return None


class ChatProvider:
    """Optional OpenAI-compatible remote adapter. Used only when explicitly selected."""
    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.settings.api_key:
            raise ProviderError("Remote mode needs AYU_API_KEY. No remote fallback was attempted.")
        if not self.settings.model:
            raise ProviderError("Remote mode needs AYU_MODEL set to a provider-supported model ID.")
        endpoint_error=remote_endpoint_error(self.settings.base_url)
        if endpoint_error: raise ProviderError(endpoint_error)
        payload = {"model": self.settings.model, "messages": messages,
                   "max_tokens": self.settings.max_output_tokens}
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.settings.api_key}",
                     "Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": "ayu-ai/0.2.0"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.settings.request_timeout) as response:
                raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ProviderError("Provider response exceeded the 8 MB safety limit.")
            data = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Provider bodies may echo prompts or other sensitive request data.
            raise ProviderError(f"Remote provider returned HTTP {exc.code}; response body suppressed to protect privacy.") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"Could not reach configured remote provider: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError("Provider returned invalid JSON.") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            if isinstance(data, dict) and data.get("error"):
                raise ProviderError("Provider returned an error response; body suppressed to protect privacy.") from exc
            raise ProviderError("Provider response did not contain a chat completion.") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Provider returned an empty or non-text answer.")
        return content.strip()


class LlamaCppProvider:
    """Local inference via an already-installed llama.cpp `llama-cli` executable."""
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate(self) -> tuple[Path, str]:
        model = self.settings.local_model_path
        if model is None:
            raise ProviderError("No local model selected. Set AYU_LOCAL_MODEL to a compatible GGUF file; Ayu never downloads weights automatically.")
        if not model.is_file():
            raise ProviderError(f"Configured local model does not exist: {model}")
        if model.suffix.lower() != ".gguf":
            raise ProviderError("The llama.cpp adapter currently accepts GGUF model files only.")
        executable = shutil.which(self.settings.llama_cli)
        if executable is None and Path(self.settings.llama_cli).is_file():
            executable = str(Path(self.settings.llama_cli).expanduser().resolve())
        if executable is None:
            raise ProviderError(f"Local runtime not found: {self.settings.llama_cli}. Install llama.cpp and configure AYU_LLAMA_CLI; no model will be downloaded.")
        return model.resolve(), executable

    def complete(self, messages: list[dict[str, str]]) -> str:
        model, executable = self.validate()
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"
        prompt_file = None
        try:
            with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",prefix="ayu-prompt-",suffix=".txt",delete=False) as stream:
                prompt_file = Path(stream.name)
                stream.write(prompt)
            command = [executable, "-m", str(model), "-f", str(prompt_file),
                       "-n", str(self.settings.max_output_tokens), "--no-display-prompt"]
            # Preserve common runtime/path variables, not arbitrary secret-bearing
            # environment variables, when launching a native local inference binary.
            safe_env_keys = {"PATH","HOME","USERPROFILE","SYSTEMROOT","WINDIR","TMP","TEMP",
                             "LD_LIBRARY_PATH","DYLD_LIBRARY_PATH","LANG","LC_ALL",
                             "CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES"}
            child_env = {k:v for k,v in os.environ.items() if k.upper() in safe_env_keys}
            result = subprocess.run(command, check=False, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace",
                                    timeout=self.settings.local_timeout, shell=False, env=child_env)
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"Local inference exceeded {self.settings.local_timeout:g} seconds and was stopped.") from exc
        except OSError as exc:
            raise ProviderError(f"Could not start local llama.cpp runtime: {exc}") from exc
        finally:
            if prompt_file is not None:
                try: prompt_file.unlink(missing_ok=True)
                except OSError: pass
        if result.returncode != 0:
            message = result.stderr.strip()[-2000:] or result.stdout.strip()[-2000:]
            raise ProviderError(f"llama.cpp exited with code {result.returncode}: {message}")
        answer = result.stdout.strip()
        if not answer:
            raise ProviderError("Local runtime returned no text. Check model compatibility and llama.cpp version.")
        return answer
