"""Explicit remote and local inference adapters; neither is selected silently."""
from __future__ import annotations

import json
import hashlib
import copy
import os
import shutil
import subprocess
import threading
import tempfile
import time
import urllib.error
from urllib.parse import urlencode, urlsplit
import urllib.request
from pathlib import Path
from datetime import date

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
                     "User-Agent": "ayu-ai/0.3.0"}, method="POST")
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
            headers={"x-api-key":self.settings.api_key,"anthropic-version":"2023-06-01","Content-Type":"application/json","Accept":"application/json","User-Agent":"ayu-ai/0.3.0"},
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
        request=urllib.request.Request(self.endpoint+"?"+urlencode(params),headers={"X-Api-Key":key,"Accept":"application/json","User-Agent":"ayu-ai/0.3.0"})
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
        try: model_size=model.stat().st_size
        except OSError: raise ProviderError("Could not inspect the configured local GGUF file.") from None
        if model_size <= 0:
            raise ProviderError("Configured local GGUF file is empty.")
        if not os.access(model, os.R_OK):
            raise ProviderError("Configured local GGUF file is not readable by this process.")
        if model.suffix.lower() != ".gguf":
            raise ProviderError("The llama.cpp adapter currently accepts GGUF model files only.")
        executable = shutil.which(self.settings.llama_cli)
        if executable is None and Path(self.settings.llama_cli).is_file():
            executable = str(Path(self.settings.llama_cli).expanduser().resolve())
        if executable is None:
            raise ProviderError(f"Local runtime not found: {self.settings.llama_cli}. Install llama.cpp and configure AYU_LLAMA_CLI; no model will be downloaded.")
        if os.name != "nt" and not os.access(executable, os.X_OK):
            raise ProviderError("Configured llama.cpp runtime is not executable by this process.")
        return model.resolve(), executable

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
