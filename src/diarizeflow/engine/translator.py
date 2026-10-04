"""Pluggable LLM Translation Engine supporting vLLM, llama.cpp, OpenAI, Claude, and Ollama.

Translates transcribed speech text into the specified target language in real-time.
Handles reasoning models (<think>...</think>) and cleans output for subtitle HUDs.
"""

import asyncio
from functools import lru_cache
import json
import re
from typing import Dict, Optional
import aiohttp
import requests

from diarizeflow.config import LLMConfig


class LLMTranslator:
    """Unified LLM Translation client supporting multiple API backends."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._cache: Dict[str, str] = {}
        self._cache_max_size = 500
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_loop: Optional[asyncio.AbstractEventLoop] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_loop: Optional[asyncio.AbstractEventLoop] = None
        self._resolved_model: Optional[str] = None
        self._model_probe_failed: bool = False
        self.fallback_used: bool = False

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or initialize the concurrency semaphore bound to the current running event loop."""
        limit = getattr(self.config, "concurrency_limit", 3)
        limit = max(1, int(limit))
        try:
            curr_loop = asyncio.get_running_loop()
        except RuntimeError:
            curr_loop = None

        if self._semaphore is None or self._semaphore_loop != curr_loop:
            self._semaphore = asyncio.Semaphore(limit)
            self._semaphore_loop = curr_loop
        return self._semaphore

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or initialize the persistent aiohttp.ClientSession connection pool bound to the running event loop."""
        try:
            curr_loop = asyncio.get_running_loop()
        except RuntimeError:
            curr_loop = None

        if (
            self._session is None
            or self._session.closed
            or self._session_loop != curr_loop
        ):
            if self._session is not None and not self._session.closed:
                if self._session_loop and not self._session_loop.is_closed() and self._session_loop.is_running():
                    try:
                        asyncio.run_coroutine_threadsafe(self._session.close(), self._session_loop)
                    except Exception:
                        pass
                else:
                    try:
                        if curr_loop and not curr_loop.is_closed():
                            await self._session.close()
                    except Exception:
                        pass

            connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=30.0)
            timeout = aiohttp.ClientTimeout(total=8.0, connect=1.5, sock_read=6.0)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            self._session_loop = curr_loop

        return self._session

    async def close(self):
        """Gracefully close the persistent HTTP client session."""
        if self._session is not None and not self._session.closed:
            try:
                await self._session.close()
            except Exception:
                pass
        self._session = None
        self._session_loop = None

    def close_sync(self):
        """Synchronously close the persistent HTTP client session."""
        if self._session is not None and not self._session.closed:
            if self._session_loop and self._session_loop.is_running():
                try:
                    fut = asyncio.run_coroutine_threadsafe(self.close(), self._session_loop)
                    fut.result(timeout=1.0)
                except Exception:
                    pass
            else:
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed() and not loop.is_running():
                        loop.run_until_complete(self.close())
                    else:
                        asyncio.run(self.close())
                except Exception:
                    pass
        self._session = None
        self._session_loop = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def update_config(self, config: LLMConfig):
        """Update translator configuration and refresh semaphore/session if needed."""
        old_base_url = getattr(self.config, "base_url", "")
        old_api_key = getattr(self.config, "api_key", "")
        old_limit = getattr(self.config, "concurrency_limit", 3)
        old_model = getattr(self.config, "model_name", "")

        self.config = config
        new_limit = getattr(config, "concurrency_limit", 3)
        if new_limit != old_limit:
            self._semaphore = None
            self._semaphore_loop = None

        # Reset model discovery cache if endpoint or model changed
        if (
            getattr(config, "base_url", "") != old_base_url
            or getattr(config, "model_name", "") != old_model
        ):
            self._resolved_model = None
            self._model_probe_failed = False
            self.fallback_used = False

        # Reset persistent session if base_url or api_key changed
        if (
            getattr(config, "base_url", "") != old_base_url
            or getattr(config, "api_key", "") != old_api_key
        ):
            if self._session is not None and not self._session.closed:
                try:
                    if self._session_loop and self._session_loop.is_running():
                        asyncio.run_coroutine_threadsafe(self._session.close(), self._session_loop)
                    else:
                        try:
                            loop = asyncio.get_event_loop()
                            if not loop.is_closed() and not loop.is_running():
                                loop.run_until_complete(self._session.close())
                        except Exception:
                            pass
                except Exception:
                    pass
            self._session = None
            self._session_loop = None


    def _clean_output(self, raw_text: str) -> str:
        """Strip reasoning blocks, quotes, and meta prefixes."""
        if not raw_text:
            return ""

        text = raw_text.strip()

        # Remove <think>...</think> blocks or unclosed <think>... (DeepSeek / Qwen3.5)
        text = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", text, flags=re.IGNORECASE).strip()

        # Remove markdown quotes, code blocks, or bold wrappers
        text = re.sub(r"^```[\w]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip('"\'「」『』')

        # Remove prefixes like "翻譯：", "Translation:", "Translated text:"
        prefix_pattern = r"^(翻譯|翻译|譯文|译文|Translation|Target|結果|结果):\s*"
        text = re.sub(prefix_pattern, "", text, flags=re.IGNORECASE).strip()

        # Normalize multiple spaces or line breaks
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_content_or_reasoning(self, choice_message: dict, target_lang: str = "繁體中文") -> str:
        """Extract valid translation from response, discarding any meta reasoning/thinking text."""
        content = choice_message.get("content")
        if content:
            cleaned = self._clean_output(content)
            if cleaned:
                return cleaned

        # Some reasoning models put the text in 'reasoning' or think tags
        reasoning = choice_message.get("reasoning", "")
        if reasoning:
            # If target language is non-English (Chinese/Japanese/Korean), discard English thinking traces
            has_cjk_target = any(k in target_lang for k in ["中", "漢", "汉", "日", "韓", "韩"])
            lines = [line.strip() for line in reasoning.split("\n") if line.strip()]
            valid_lines = []
            for line in lines:
                if line.startswith("*") or line.startswith("#") or "thinking process" in line.lower() or "let's" in line.lower():
                    continue
                cleaned = self._clean_output(line)
                if cleaned:
                    if has_cjk_target:
                        if any('\u4e00' <= char <= '\u9fff' for char in cleaned):
                            valid_lines.append(cleaned)
                    else:
                        valid_lines.append(cleaned)
            if valid_lines:
                return " ".join(valid_lines)

        return ""

    async def translate(
        self,
        text: str,
        target_language: Optional[str] = None,
        source_language: Optional[str] = None,
    ) -> str:
        """Translate text asynchronously using the configured LLM API."""
        if not text or not text.strip():
            return ""

        target_lang = target_language or self.config.target_language
        cache_key = f"{text.strip()}->{target_lang}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        provider = self.config.provider.lower()

        # Bypass mode: return original text
        if provider in ["bypass", "none", "original"]:
            return text

        sys_prompt = self.config.system_prompt.format(target_language=target_lang)
        user_prompt = f"請將這段話直接翻譯成【{target_lang}】（只輸出譯文，勿加任何解釋）：\n{text}"

        translated = ""
        sem = self._get_semaphore()
        async with sem:
            if cache_key in self._cache:
                return self._cache[cache_key]

            try:
                if provider in ["vllm", "llama.cpp", "openai", "ollama"]:
                    translated = await self._call_openai_compatible(sys_prompt, user_prompt, target_lang=target_lang)
                elif provider in ["claude", "anthropic"]:
                    translated = await self._call_claude(sys_prompt, user_prompt)
                else:
                    # Fallback to OpenAI compatible
                    translated = await self._call_openai_compatible(sys_prompt, user_prompt, target_lang=target_lang)

                if not translated:
                    translated = text  # fallback to original if API returned empty
            except Exception as e:
                print(f"[!] LLM Translation error ({provider}): {e}")
                translated = text  # Graceful fallback to original text

            # Cache result
            if len(self._cache) >= self._cache_max_size:
                self._cache.clear()
            self._cache[cache_key] = translated

        return translated

    def translate_sync(
        self,
        text: str,
        target_language: Optional[str] = None,
        source_language: Optional[str] = None,
    ) -> str:
        """Synchronous wrapper for translate."""
        async def _run_and_cleanup():
            try:
                return await self.translate(text, target_language, source_language)
            finally:
                await self.close()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running inside existing event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, _run_and_cleanup()).result()
            else:
                return loop.run_until_complete(self.translate(text, target_language, source_language))
        except Exception:
            return asyncio.run(_run_and_cleanup())

    async def _resolve_model_name(self) -> str:
        """Resolve active model name dynamically from endpoint without leaking local filesystem paths."""
        configured = self.config.model_name
        if configured and configured.lower() not in ["auto", "default", ""]:
            return configured

        if self._resolved_model:
            return self._resolved_model

        if self._model_probe_failed or self.fallback_used:
            return "default"

        try:
            base_url = self.config.base_url.rstrip("/")
            if base_url.endswith("/chat/completions"):
                models_url = base_url[:-len("/chat/completions")].rstrip("/") + "/models"
            elif base_url.endswith("/v1"):
                models_url = f"{base_url}/models"
            else:
                models_url = f"{base_url}/v1/models"

            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            session = await self._get_session()
            async with session.get(models_url, headers=headers, timeout=aiohttp.ClientTimeout(total=2.0, connect=1.0, sock_read=2.0)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models_list = data.get("data", [])
                    if models_list and "id" in models_list[0]:
                        self._resolved_model = models_list[0]["id"]
                        return self._resolved_model
                else:
                    self._model_probe_failed = True
                    self.fallback_used = True
        except Exception:
            self._model_probe_failed = True
            self.fallback_used = True

        return "default"

    async def _call_openai_compatible(self, system_prompt: str, user_prompt: str, target_lang: str = "繁體中文") -> str:
        """Call OpenAI-compatible chat completion endpoint (vLLM, llama.cpp, OpenAI, Ollama)."""
        base_url = self.config.base_url.rstrip("/")
        if not base_url.endswith("/chat/completions"):
            endpoint = f"{base_url}/chat/completions"
        else:
            endpoint = base_url

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        model_name = await self._resolve_model_name()

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            # Disable thinking for vLLM / Qwen reasoning models for instant translation
            "chat_template_kwargs": {"enable_thinking": False},
        }

        session = await self._get_session()
        try:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                if resp.status == 400:
                    # Some endpoints might not accept chat_template_kwargs, retry without it
                    payload.pop("chat_template_kwargs", None)
                    async with session.post(endpoint, json=payload, headers=headers) as retry_resp:
                        if retry_resp.status != 200:
                            err_txt = await retry_resp.text()
                            raise RuntimeError(f"HTTP {retry_resp.status}: {err_txt[:200]}")
                        data = await retry_resp.json()
                        msg = data["choices"][0]["message"]
                        return self._extract_content_or_reasoning(msg, target_lang=target_lang)
                elif resp.status != 200:
                    err_txt = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {err_txt[:200]}")
                data = await resp.json()
                msg = data["choices"][0]["message"]
                return self._extract_content_or_reasoning(msg, target_lang=target_lang)
        except Exception as e:
            raise e

    async def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic Claude API."""
        endpoint = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.config.model_name if "claude" in self.config.model_name else "claude-3-5-haiku-20241022",
            "max_tokens": self.config.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": self.config.temperature,
        }

        session = await self._get_session()
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status != 200:
                err_txt = await resp.text()
                raise RuntimeError(f"Claude API HTTP {resp.status}: {err_txt[:200]}")
            data = await resp.json()
            content_blocks = data.get("content", [])
            text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
            return self._clean_output(text)
