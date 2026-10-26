"""Explicit remote and local inference adapters; neither is selected silently."""
from __future__ import annotations

import json
import hashlib
import copy
import http.client
import ipaddress
import os
import shutil
import socket
import subprocess
import threading
import tempfile
import time
import urllib.error
from urllib.parse import urlencode, urlsplit
import urllib.request
from pathlib import Path
from datetime import date
from collections.abc import Iterator

from .settings import Settings


class ProviderError(RuntimeError):
    """An inference configuration, execution, transport, or response error."""


def remote_endpoint_error(base_url: str) -> str | None:
    try: endpoint=urlsplit(base_url)
    except ValueError: return "AYU_BASE_URL is not a valid URL."
    if endpoint.scheme!="https" or not endpoint.hostname:
        return "Remote mode requires an HTTPS AYU_BASE_URL. Plain HTTP would expose the API key and conversation."
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        return "AYU_BASE_URL must not embed credentials, query parameters, or a fragment; put the key in the selected provider's credential environment variable."
    return None


class ChatProvider:
    """Optional OpenAI-compatible remote adapter. Used only when explicitly selected."""
    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.settings.api_key:
            key_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(self.settings.remote_provider, "AYU_API_KEY")
            raise ProviderError(f"Remote mode needs {key_name}. No remote fallback was attempted.")
        if not self.settings.model:
            model_name={"openai":"AYU_MODEL or OPENAI_MODEL","anthropic":"AYU_MODEL or ANTHROPIC_MODEL"}.get(self.settings.remote_provider,"AYU_MODEL")
            raise ProviderError(f"Remote mode needs {model_name} set to a provider-supported model ID.")
        endpoint_error=remote_endpoint_error(self.settings.base_url)
        if endpoint_error: raise ProviderError(endpoint_error)
        payload = {"model": self.settings.model, "messages": messages,
                   "max_tokens": self.settings.max_output_tokens}
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.settings.api_key}",
                     "Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": "ayu-ai/0.4.0"}, method="POST")
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
            raise ProviderError(f"Could not reach configured remote provider ({type(exc).__name__}); endpoint details suppressed.") from None
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


class AnthropicProvider:
    """Optional Anthropic Messages API adapter; credential is never shared elsewhere."""
    endpoint = "https://api.anthropic.com/v1/messages"

    def __init__(self, settings: Settings):
        self.settings = settings

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.settings.api_key:
            raise ProviderError("Anthropic mode needs ANTHROPIC_API_KEY. No fallback was attempted.")
        if not self.settings.model:
            raise ProviderError("Anthropic mode needs AYU_MODEL or ANTHROPIC_MODEL.")
        system_parts=[]; conversation=[]
        for message in messages:
            role=message.get("role"); content=message.get("content")
            if role not in {"system","user","assistant"} or not isinstance(content,str):
                raise ProviderError("Messages must have system/user/assistant roles and text content.")
            if role=="system": system_parts.append(content)
            else: conversation.append({"role":role,"content":content})
        if not conversation or conversation[-1]["role"]!="user":
            raise ProviderError("Anthropic Messages API requires a final user message.")
        payload={"model":self.settings.model,"max_tokens":self.settings.max_output_tokens,"messages":conversation}
        if system_parts: payload["system"]="\n\n".join(system_parts)
        request=urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"x-api-key":self.settings.api_key,"anthropic-version":"2023-06-01","Content-Type":"application/json","Accept":"application/json","User-Agent":"ayu-ai/0.4.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request,timeout=self.settings.request_timeout) as response:
                raw=response.read(8_000_001)
            if len(raw)>8_000_000: raise ProviderError("Anthropic response exceeded the 8 MB safety limit.")
            data=json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"Anthropic returned HTTP {exc.code}; response body suppressed to protect privacy.") from None
        except (urllib.error.URLError,TimeoutError,OSError):
            raise ProviderError("Could not reach Anthropic within the configured request timeout.") from None
        except (UnicodeDecodeError,json.JSONDecodeError):
            raise ProviderError("Anthropic returned invalid JSON.") from None
        blocks=data.get("content") if isinstance(data,dict) else None
        if not isinstance(blocks,list): raise ProviderError("Anthropic response did not contain a content array.")
        text="".join(block.get("text","") for block in blocks if isinstance(block,dict) and block.get("type")=="text" and isinstance(block.get("text"),str)).strip()
        if not text: raise ProviderError("Anthropic returned no text content.")
        return text


def build_remote_provider(settings: Settings):
    """Select only the explicitly configured remote adapter."""
    if settings.remote_provider=="anthropic": return AnthropicProvider(settings)
    if settings.remote_provider in {"openai","openai-compatible"}: return ChatProvider(settings)
    raise ProviderError(f"Unsupported remote provider: {settings.remote_provider}")


class NewsAPIClient:
    """Optional NewsAPI.org metadata search; never used as an inference provider."""
    endpoint="https://newsapi.org/v2/everything"
    _cache: dict[str,tuple[float,dict]]={}

    def __init__(self, settings: Settings):
        self.settings=settings

    def search(self, query: str, *, from_date: str|None=None, to_date: str|None=None, sources: str|None=None, sort_by: str="publishedAt", page_size: int=10) -> dict:
        if not self.settings.newsapi_enabled:
            raise ProviderError("NewsAPI is disabled. Set AYU_NEWSAPI_ENABLED=true and explicitly run `ayu news search --allow-network` to enable a search.")
        key=self.settings.newsapi_api_key
        if not key: raise ProviderError("NewsAPI search needs NEWSAPI_API_KEY; it is separate from AI-provider credentials.")
        if not isinstance(query,str) or not query.strip() or len(query)>500: raise ProviderError("NewsAPI query must contain 1..500 characters.")
        if isinstance(page_size,bool) or not isinstance(page_size,int) or not 1<=page_size<=100: raise ProviderError("NewsAPI page size must be between 1 and 100.")
        if not isinstance(sort_by,str) or sort_by not in {"relevancy","popularity","publishedAt"}: raise ProviderError("sort_by must be relevancy, popularity, or publishedAt.")
        params={"q":query.strip(),"sortBy":sort_by,"pageSize":str(page_size)}
        dates={}
        for name,value in (("from",from_date),("to",to_date)):
            if value is not None:
                try: dates[name]=date.fromisoformat(value).isoformat()
                except (TypeError,ValueError): raise ProviderError(f"{name}_date must use YYYY-MM-DD format.") from None
                if len(value)!=10 or value[4:5]!="-" or value[7:8]!="-": raise ProviderError(f"{name}_date must use YYYY-MM-DD format.")
        if "from" in dates and "to" in dates and dates["from"]>dates["to"]: raise ProviderError("from_date must not be after to_date.")
        params.update(dates)
        if sources is not None and not isinstance(sources,str): raise ProviderError("sources must be a comma-separated string of NewsAPI source IDs.")
        if sources:
            source_ids=[item.strip() for item in sources.split(",") if item.strip()]
            if not source_ids or len(source_ids)>20 or any(not all(c.isalnum() or c in "-_" for c in item) for item in source_ids):
                raise ProviderError("sources must be a comma-separated list of at most 20 NewsAPI source IDs.")
            params["sources"]=",".join(source_ids)
        cache_material=json.dumps({"params":params,"key_fingerprint":hashlib.sha256(key.encode()).hexdigest()},sort_keys=True)
        cache_key=hashlib.sha256(cache_material.encode()).hexdigest()
        cached=self._cache.get(cache_key)
        if cached and time.monotonic()-cached[0]<120: return copy.deepcopy(cached[1])
        request=urllib.request.Request(self.endpoint+"?"+urlencode(params),headers={"X-Api-Key":key,"Accept":"application/json","User-Agent":"ayu-ai/0.4.0"})
        try:
            with urllib.request.urlopen(request,timeout=self.settings.request_timeout) as response: raw=response.read(4_000_001)
            if len(raw)>4_000_000: raise ProviderError("NewsAPI response exceeded the 4 MB safety limit.")
            data=json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"NewsAPI returned HTTP {exc.code}; response body suppressed to protect privacy.") from None
        except (urllib.error.URLError,TimeoutError,OSError):
            raise ProviderError("Could not reach NewsAPI within the configured request timeout.") from None
        except (UnicodeDecodeError,json.JSONDecodeError):
            raise ProviderError("NewsAPI returned invalid JSON.") from None
        if not isinstance(data,dict) or data.get("status")!="ok" or not isinstance(data.get("articles"),list):
            error_code=data.get("code") if isinstance(data,dict) else None
            safe_code=error_code if isinstance(error_code,str) and error_code.replace("_","").isalnum() else "invalid_response"
            raise ProviderError(f"NewsAPI request failed ({safe_code}); provider message suppressed.")
        articles=[]
        for item in data["articles"][:page_size]:
            if not isinstance(item,dict): continue
            url=item.get("url"); published=item.get("publishedAt"); source=item.get("source")
            if not isinstance(url,str) or urlsplit(url).scheme not in {"http","https"} or not urlsplit(url).hostname or not isinstance(published,str): continue
            source_name=source.get("name") if isinstance(source,dict) else None
            articles.append({"source":source_name if isinstance(source_name,str) else None,"title":str(item.get("title") or "")[:1000],"description":str(item.get("description") or "")[:4000],"url":url,"publishedAt":published})
        result={"total_results":data.get("totalResults") if isinstance(data.get("totalResults"),int) else len(articles),"articles":articles,"provider":"NewsAPI","note":"Metadata and descriptions are untrusted external content; verify at source. This result was not sent to an AI provider."}
        if len(self._cache)>=100: self._cache.clear()
        self._cache[cache_key]=(time.monotonic(),copy.deepcopy(result))
        return copy.deepcopy(result)


def _validate_gguf_model(model: Path | None) -> Path:
    if model is None:
        raise ProviderError("No local model selected. Set AYU_LOCAL_MODEL to a compatible GGUF file; Ayu never downloads weights automatically.")
    if not model.is_file():
        raise ProviderError(f"Configured local model does not exist: {model}")
    try:
        model_size = model.stat().st_size
    except OSError:
        raise ProviderError("Could not inspect the configured local GGUF file.") from None
    if model_size <= 0:
        raise ProviderError("Configured local GGUF file is empty.")
    if not os.access(model, os.R_OK):
        raise ProviderError("Configured local GGUF file is not readable by this process.")
    if model.suffix.lower() != ".gguf":
        raise ProviderError("The llama.cpp adapter currently accepts GGUF model files only.")
    return model.resolve()


class LlamaCppProvider:
    """Local inference via an already-installed llama.cpp `llama-cli` executable."""
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate(self) -> tuple[Path, str]:
        model = _validate_gguf_model(self.settings.local_model_path)
        executable = shutil.which(self.settings.llama_cli)
        if executable is None and Path(self.settings.llama_cli).is_file():
            executable = str(Path(self.settings.llama_cli).expanduser().resolve())
        if executable is None:
            raise ProviderError(f"Local runtime not found: {self.settings.llama_cli}. Install llama.cpp and configure AYU_LLAMA_CLI; no model will be downloaded.")
        if os.name != "nt" and not os.access(executable, os.X_OK):
            raise ProviderError("Configured llama.cpp runtime is not executable by this process.")
        return model, executable

    def complete(self, messages: list[dict[str, str]]) -> str:
        model, executable = self.validate()
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"
        prompt_file = None
        process = None
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
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, env=child_env)
            limit=4*1024*1024
            overflow=threading.Event()
            output={"stdout":{"chunks":[],"size":0},"stderr":{"chunks":[],"size":0}}
            def drain(stream, name):
                state=output[name]
                while True:
                    chunk=stream.read(64*1024)
                    if not chunk: break
                    if state["size"]+len(chunk)<=limit: state["chunks"].append(chunk)
                    state["size"]+=len(chunk)
                    if state["size"]>limit:
                        overflow.set()
                        try: process.kill()
                        except OSError: pass
            readers=[threading.Thread(target=drain,args=(process.stdout,"stdout"),daemon=True),threading.Thread(target=drain,args=(process.stderr,"stderr"),daemon=True)]
            for reader in readers: reader.start()
            try:
                return_code=process.wait(timeout=self.settings.local_timeout)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(); raise ProviderError(f"Local inference exceeded {self.settings.local_timeout:g} seconds and was stopped.") from None
            for reader in readers: reader.join(timeout=2.0)
            if any(reader.is_alive() for reader in readers): raise ProviderError("Local runtime output stream did not close; output suppressed.")
            if overflow.is_set(): raise ProviderError("Local runtime exceeded the 4 MiB per-stream output limit and was stopped.")
            stdout=b"".join(output["stdout"]["chunks"]).decode("utf-8",errors="replace")
        except subprocess.TimeoutExpired:
            raise ProviderError(f"Local inference exceeded {self.settings.local_timeout:g} seconds and was stopped.") from None
        except OSError as exc:
            raise ProviderError(f"Could not start or read from local llama.cpp runtime ({type(exc).__name__}); details suppressed.") from None
        finally:
            if process is not None and process.poll() is None:
                try: process.kill(); process.wait(timeout=2)
                except (OSError,subprocess.TimeoutExpired): pass
            if process is not None:
                for stream in (process.stdout,process.stderr):
                    try: stream.close()
                    except (OSError,AttributeError): pass
            if prompt_file is not None:
                try: prompt_file.unlink(missing_ok=True)
                except OSError: pass
        if return_code != 0:
            raise ProviderError(f"llama.cpp exited with code {return_code}; runtime output suppressed to protect prompt content.")
        answer = stdout.strip()
        if not answer:
            raise ProviderError("Local runtime returned no text. Check model compatibility and llama.cpp version.")
        return answer


class LlamaServerProvider:
    """Local chat via a user-started llama-server bound to a numeric loopback address."""

    MAX_REQUEST_BYTES = 2 * 1024 * 1024
    MAX_RESPONSE_BYTES = 8 * 1024 * 1024
    MAX_SSE_LINE_BYTES = 1024 * 1024

    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_runtime_info: dict[str, str | int] = {}

    def _endpoint(self) -> tuple[str, int]:
        try:
            endpoint = urlsplit(self.settings.llama_server_url)
            host = endpoint.hostname
            port = endpoint.port
        except ValueError:
            raise ProviderError("AYU_LLAMA_SERVER_URL is not a valid loopback URL.") from None
        if port is None:
            port = 80
        if (endpoint.scheme != "http" or not host or endpoint.username or endpoint.password
                or endpoint.query or endpoint.fragment or endpoint.path not in {"", "/"}):
            raise ProviderError("AYU_LLAMA_SERVER_URL must be a plain HTTP URL to a loopback IP, with no credentials, path, query, or fragment.")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise ProviderError("AYU_LLAMA_SERVER_URL must use a numeric loopback IP, such as 127.0.0.1 or ::1; hostnames and remote addresses are refused.") from None
        if isinstance(address, ipaddress.IPv6Address) and address.scope_id is not None:
            raise ProviderError("Scoped IPv6 addresses are not accepted for the local llama-server URL.")
        if not address.is_loopback:
            raise ProviderError("Local llama-server connections are restricted to loopback addresses; no remote fallback or network request was attempted.")
        if not 1 <= port <= 65535:
            raise ProviderError("AYU_LLAMA_SERVER_URL port must be between 1 and 65535.")
        if not 1 <= self.settings.local_timeout <= 3600:
            raise ProviderError("AYU_LOCAL_TIMEOUT must be between 1 and 3600 seconds.")
        return address.compressed, port

    def _request(self, method: str, path: str, payload: dict | None = None) -> bytes:
        host, port = self._endpoint()
        body = None
        if payload is not None:
            try:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError, UnicodeError):
                raise ProviderError("Could not encode a local inference request.") from None
            if len(body) > self.MAX_REQUEST_BYTES:
                raise ProviderError("Local inference request exceeded the 2 MiB safety limit.")
        connection = http.client.HTTPConnection(host, port, timeout=self.settings.local_timeout)
        try:
            headers = {"Accept": "application/json", "User-Agent": "ayu-ai/0.4.0"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                raise ProviderError(f"Local llama-server returned HTTP {response.status} for {path}; response body suppressed.")
            raw = response.read(self.MAX_RESPONSE_BYTES + 1)
            if len(raw) > self.MAX_RESPONSE_BYTES:
                raise ProviderError("Local llama-server response exceeded the 8 MiB safety limit.")
            return raw
        except ProviderError:
            raise
        except (TimeoutError, socket.timeout):
            raise ProviderError(f"Local llama-server request exceeded {self.settings.local_timeout:g} seconds.") from None
        except (OSError, http.client.HTTPException):
            raise ProviderError("Could not connect to the configured local llama-server; details suppressed.") from None
        finally:
            connection.close()

    @staticmethod
    def _json(raw: bytes, label: str) -> dict:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise ProviderError(f"Local llama-server returned invalid {label} JSON; body suppressed.") from None
        if not isinstance(value, dict):
            raise ProviderError(f"Local llama-server returned an invalid {label} response.")
        return value

    def _server_metadata(self) -> tuple[str, int, str | None]:
        configured_model = _validate_gguf_model(self.settings.local_model_path)
        health = self._json(self._request("GET", "/health"), "health")
        if health.get("status") != "ok":
            raise ProviderError("Local llama-server is reachable but not ready; wait for model loading to finish.")
        listing = self._json(self._request("GET", "/v1/models"), "model-list")
        models = listing.get("data")
        if not isinstance(models, list) or not models:
            raise ProviderError("Local llama-server did not report a loaded model.")
        entries = [item for item in models if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]]
        if self.settings.local_model_id:
            selected = [item for item in entries if item["id"] == self.settings.local_model_id]
            if len(selected) != 1:
                raise ProviderError("AYU_LOCAL_MODEL_ID did not match exactly one model reported by the local server.")
        elif len(entries) == 1:
            selected = entries
        else:
            raise ProviderError("The local server reports multiple models; set AYU_LOCAL_MODEL_ID to an exact model ID.")
        properties = self._json(self._request("GET", "/props"), "properties")
        reported_path = properties.get("model_path")
        if not isinstance(reported_path, str) or not reported_path:
            raise ProviderError("Local llama-server did not report its loaded model path; model identity cannot be verified.")
        try:
            server_model = Path(reported_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            raise ProviderError("The model path reported by local llama-server is not readable/resolvable by Ayu.") from None
        if server_model != configured_model:
            raise ProviderError("The loaded llama-server model path does not match AYU_LOCAL_MODEL; no inference was sent.")
        context_size = properties.get("default_generation_settings", {}).get("n_ctx") if isinstance(properties.get("default_generation_settings"), dict) else None
        if isinstance(context_size, bool) or not isinstance(context_size, int) or context_size <= 0:
            raise ProviderError("Local llama-server did not report a valid effective context size at /props.")
        build_info = properties.get("build_info")
        build_info = build_info if isinstance(build_info, str) and build_info else None
        self.last_runtime_info = {"model_id": selected[0]["id"], "model_path": str(server_model),
                                  "context_size": context_size, "build_info": build_info or "not reported"}
        return selected[0]["id"], context_size, build_info

    def validate(self) -> str:
        return self._server_metadata()[0]

    def _checked_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        limit = self.settings.local_max_input_chars
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000_000:
            raise ProviderError("AYU_LOCAL_MAX_INPUT_CHARS must be between 1 and 1000000.")
        if not isinstance(messages, list) or not messages or len(messages) > 100:
            raise ProviderError("Local chat requires between 1 and 100 messages.")
        clean = []
        total_chars = 0
        for item in messages:
            if not isinstance(item, dict) or set(item) != {"role", "content"}:
                raise ProviderError("Local chat messages must contain only role and content fields.")
            role, content = item["role"], item["content"]
            if not isinstance(role, str) or role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise ProviderError("Local chat supports only system, user, and assistant text messages.")
            total_chars += len(content)
            if total_chars > limit:
                raise ProviderError(f"Local chat exceeds AYU_LOCAL_MAX_INPUT_CHARS ({limit}); shorten the prompt or reduce saved history.")
            clean.append({"role": role, "content": content})
        return clean

    def _prepare_chat_request(self, messages: list[dict[str, str]]) -> dict:
        clean = self._checked_messages(messages)
        if (isinstance(self.settings.max_output_tokens, bool)
                or not isinstance(self.settings.max_output_tokens, int)
                or not 1 <= self.settings.max_output_tokens <= 8192):
            raise ProviderError("AYU_MAX_OUTPUT_TOKENS must be between 1 and 8192.")
        model_id, context_size, _ = self._server_metadata()
        request = {"model": model_id, "messages": clean,
                   "max_tokens": self.settings.max_output_tokens, "stream": False}
        token_result = self._json(self._request("POST", "/v1/chat/completions/input_tokens", request), "token-count")
        input_tokens = token_result.get("input_tokens")
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0:
            raise ProviderError("Local llama-server returned an invalid chat-template token count.")
        total_tokens = input_tokens + self.settings.max_output_tokens
        if total_tokens > context_size:
            raise ProviderError(f"Local prompt budget exceeded: {input_tokens} input tokens + {self.settings.max_output_tokens} output tokens exceeds the server's {context_size}-token context; shorten history/prompt or restart llama-server with a larger -c value.")
        return request

    def complete(self, messages: list[dict[str, str]]) -> str:
        request = self._prepare_chat_request(messages)
        result = self._json(self._request("POST", "/v1/chat/completions", request), "chat-completions")
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError("Local llama-server returned no chat completion choices.")
        message = choices[0].get("message")
        answer = message.get("content") if isinstance(message, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise ProviderError("Local llama-server returned no text. Check the model chat template and runtime version.")
        return answer.strip()

    def complete_structured(self, messages: list[dict[str, str]], schema: dict, *, schema_name: str) -> str:
        """Request local JSON-Schema-constrained output; never falls back to a remote provider."""
        if (not isinstance(schema_name, str) or not schema_name or len(schema_name) > 64
                or not schema_name[0].isalpha()
                or any(not (char.isalnum() or char == "_") for char in schema_name)):
            raise ProviderError("Local structured-output schema name is invalid.")
        if not isinstance(schema, dict):
            raise ProviderError("Local structured output requires a JSON Schema object.")
        try:
            schema_bytes = json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise ProviderError("Local structured-output schema is not serializable JSON.") from None
        if len(schema_bytes) > 128 * 1024:
            raise ProviderError("Local structured-output schema exceeded the 128 KiB limit.")
        request = self._prepare_chat_request(messages)
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
        result = self._json(self._request("POST", "/v1/chat/completions", request), "structured chat-completions")
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError("Local llama-server returned no structured completion choices.")
        message = choices[0].get("message")
        answer = message.get("content") if isinstance(message, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise ProviderError("Local llama-server returned no structured text; check runtime support for JSON Schema output.")
        return answer.strip()

    def stream_complete(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield server-generated text chunks; closing/interruption closes the HTTP socket."""
        request = self._prepare_chat_request(messages)
        request["stream"] = True
        host, port = self._endpoint()
        try:
            body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise ProviderError("Could not encode a local streaming request.") from None
        if len(body) > self.MAX_REQUEST_BYTES:
            raise ProviderError("Local inference request exceeded the 2 MiB safety limit.")
        connection = http.client.HTTPConnection(host, port, timeout=self.settings.local_timeout)
        received_text = False
        total_bytes = 0
        deadline = time.monotonic() + self.settings.local_timeout
        try:
            connection.request("POST", "/v1/chat/completions", body=body,
                               headers={"Accept": "text/event-stream, application/json",
                                        "Content-Type": "application/json",
                                        "User-Agent": "ayu-ai/0.4.0"})
            response = connection.getresponse()
            if response.status != 200:
                raise ProviderError(f"Local llama-server returned HTTP {response.status} for /v1/chat/completions; response body suppressed.")
            content_type = (response.getheader("Content-Type") or "").lower()
            if "text/event-stream" not in content_type:
                raw = response.read(self.MAX_RESPONSE_BYTES + 1)
                if len(raw) > self.MAX_RESPONSE_BYTES:
                    raise ProviderError("Local llama-server response exceeded the 8 MiB safety limit.")
                result = self._json(raw, "stream-fallback")
                choices = result.get("choices")
                message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
                answer = message.get("content") if isinstance(message, dict) else None
                if not isinstance(answer, str) or not answer.strip():
                    raise ProviderError("Local llama-server returned no text in its streaming response.")
                yield answer
                return
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderError(f"Local llama-server stream exceeded {self.settings.local_timeout:g} seconds.")
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                line = response.readline(self.MAX_SSE_LINE_BYTES + 1)
                if not line:
                    break
                total_bytes += len(line)
                if len(line) > self.MAX_SSE_LINE_BYTES or total_bytes > self.MAX_RESPONSE_BYTES:
                    raise ProviderError("Local llama-server streaming response exceeded its safety limit.")
                line = line.strip()
                if not line or line.startswith(b":") or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError, RecursionError):
                    raise ProviderError("Local llama-server returned malformed streaming data; body suppressed.") from None
                if not isinstance(event, dict):
                    continue
                choices = event.get("choices")
                delta = choices[0].get("delta") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
                content = delta.get("content") if isinstance(delta, dict) else None
                if isinstance(content, str) and content:
                    received_text = True
                    yield content
            if not received_text:
                raise ProviderError("Local llama-server stream ended without generated text.")
        except ProviderError:
            raise
        except (TimeoutError, socket.timeout):
            raise ProviderError(f"Local llama-server stream exceeded {self.settings.local_timeout:g} seconds.") from None
        except (OSError, http.client.HTTPException):
            raise ProviderError("Local llama-server streaming connection failed; details suppressed.") from None
        finally:
            connection.close()


def build_local_provider(settings: Settings) -> LlamaCppProvider | LlamaServerProvider:
    if settings.local_backend == "llama-server":
        return LlamaServerProvider(settings)
    if settings.local_backend != "llama-cli":
        raise ProviderError("AYU_LOCAL_BACKEND must be llama-cli or llama-server.")
    return LlamaCppProvider(settings)
