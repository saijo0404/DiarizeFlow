"""Unit tests for persistent aiohttp.ClientSession connection pool and probe caching in LLMTranslator (Issue #33)."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp
from aiohttp import web

from diarizeflow.app.config import LLMConfig
from diarizeflow.app.backend.translator import LLMTranslator


class TestPersistentTranslatorSession(unittest.IsolatedAsyncioTestCase):
    """Test suite for persistent HTTP session management and probe caching."""

    async def test_session_lazily_created_with_expected_settings(self):
        """Verify session is lazily created with connection pooling (limit=10, keepalive=30s) and granular timeouts."""
        translator = LLMTranslator()
        self.assertIsNone(translator._session)

        session = await translator._get_session()
        self.assertIsNotNone(session)
        self.assertIsInstance(session, aiohttp.ClientSession)
        self.assertFalse(session.closed)

        # Verify TCPConnector configuration
        connector = session.connector
        self.assertIsInstance(connector, aiohttp.TCPConnector)
        self.assertEqual(connector.limit, 10)
        self.assertEqual(connector._keepalive_timeout, 30.0)

        # Verify ClientTimeout settings
        timeout = session.timeout
        self.assertEqual(timeout.total, 8.0)
        self.assertEqual(timeout.connect, 1.5)
        self.assertEqual(timeout.sock_read, 6.0)

        # Verify subsequent call returns the exact same session instance (Keep-Alive reuse)
        session2 = await translator._get_session()
        self.assertIs(session, session2)

        await translator.close()
        self.assertTrue(session.closed)
        self.assertIsNone(translator._session)

    async def test_session_reused_across_translation_requests(self):
        """Verify that multiple OpenAI-compatible translation calls reuse the persistent ClientSession."""
        cfg = LLMConfig(provider="openai", model_name="gpt-4o-mini", api_key="test-key")
        translator = LLMTranslator(cfg)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "你好世界"}}]
        })

        session = await translator._get_session()

        with patch.object(session, "post") as mock_post:
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            res1 = await translator.translate("Hello world 1")
            res2 = await translator.translate("Hello world 2")

            self.assertEqual(res1, "你好世界")
            self.assertEqual(res2, "你好世界")
            # Both calls went through the same session instance
            self.assertEqual(mock_post.call_count, 2)
            self.assertIs(translator._session, session)

        await translator.close()

    async def test_claude_session_reuse(self):
        """Verify Claude provider reuses the persistent session across calls."""
        cfg = LLMConfig(provider="claude", model_name="claude-3-5-haiku-20241022", api_key="sk-ant-test")
        translator = LLMTranslator(cfg)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "content": [{"type": "text", "text": "早安"}]
        })

        session = await translator._get_session()

        with patch.object(session, "post") as mock_post:
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            res1 = await translator.translate("Good morning 1")
            res2 = await translator.translate("Good morning 2")

            self.assertEqual(res1, "早安")
            self.assertEqual(res2, "早安")
            self.assertEqual(mock_post.call_count, 2)
            self.assertIs(translator._session, session)

        await translator.close()

    async def test_session_reset_on_config_change(self):
        """Verify update_config closes and resets the session when base_url or api_key changes."""
        cfg = LLMConfig(base_url="http://localhost:8000/v1", api_key="key-1")
        translator = LLMTranslator(cfg)

        session1 = await translator._get_session()
        self.assertFalse(session1.closed)

        # Non-network config change (e.g. concurrency_limit) does NOT close session
        new_cfg_same = LLMConfig(base_url="http://localhost:8000/v1", api_key="key-1", concurrency_limit=5)
        translator.update_config(new_cfg_same)
        self.assertIs(translator._session, session1)
        self.assertFalse(session1.closed)

        # Changing base_url resets session
        new_cfg_url = LLMConfig(base_url="http://remote-llm:8000/v1", api_key="key-1")
        translator.update_config(new_cfg_url)
        self.assertIsNone(translator._session)

        session2 = await translator._get_session()
        self.assertIsNotNone(session2)
        self.assertIsNot(session1, session2)

        # Changing api_key resets session
        new_cfg_key = LLMConfig(base_url="http://remote-llm:8000/v1", api_key="key-2")
        translator.update_config(new_cfg_key)
        self.assertIsNone(translator._session)

        await translator.close()

    async def test_probe_failure_caching(self):
        """Verify _resolve_model_name caches failure and avoids redundant 2-second timeout probes."""
        cfg = LLMConfig(base_url="http://localhost:8000/v1", model_name="auto")
        translator = LLMTranslator(cfg)

        session = await translator._get_session()

        mock_resp_404 = MagicMock()
        mock_resp_404.status = 404

        with patch.object(session, "get") as mock_get:
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_resp_404)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

            # First probe: fails with 404
            model1 = await translator._resolve_model_name()
            self.assertEqual(model1, "default")
            self.assertTrue(translator._model_probe_failed)
            self.assertTrue(translator.fallback_used)
            self.assertEqual(mock_get.call_count, 1)

            # Second call: must return "default" immediately without hitting session.get
            model2 = await translator._resolve_model_name()
            self.assertEqual(model2, "default")
            self.assertEqual(mock_get.call_count, 1)  # No additional network call

        # Changing base_url via update_config clears the probe failure cache
        new_cfg = LLMConfig(base_url="http://new-host:8000/v1", model_name="auto")
        translator.update_config(new_cfg)
        self.assertFalse(translator._model_probe_failed)
        self.assertFalse(translator.fallback_used)

        await translator.close()

    async def test_probe_exception_caching(self):
        """Verify network exceptions during model probe also set fallback_used and avoid retrying."""
        cfg = LLMConfig(base_url="http://unreachable-endpoint:1234/v1", model_name="auto")
        translator = LLMTranslator(cfg)

        session = await translator._get_session()

        with patch.object(session, "get", side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))):
            model1 = await translator._resolve_model_name()
            self.assertEqual(model1, "default")
            self.assertTrue(translator._model_probe_failed)
            self.assertTrue(translator.fallback_used)

            # Subsequent call does not raise or retry network probe
            model2 = await translator._resolve_model_name()
            self.assertEqual(model2, "default")

        await translator.close()

    async def test_async_context_manager(self):
        """Verify LLMTranslator works as an async context manager, closing its session on exit."""
        async with LLMTranslator() as translator:
            session = await translator._get_session()
            self.assertFalse(session.closed)

        self.assertTrue(session.closed)
        self.assertIsNone(translator._session)

    async def test_event_loop_migration_safety(self):
        """Verify _get_session handles event loop transitions without crashing."""
        translator = LLMTranslator()
        session1 = await translator._get_session()
        self.assertFalse(session1.closed)

        # Simulate loop migration by setting a stale loop
        dummy_loop = asyncio.new_event_loop()
        translator._session_loop = dummy_loop

        # Now calling _get_session in the current running loop must detect the mismatch and create a new session
        session2 = await translator._get_session()
        self.assertIsNot(session1, session2)
        self.assertEqual(translator._session_loop, asyncio.get_running_loop())

        dummy_loop.close()
        await translator.close()


class TestTranslatorSyncLifecycle(unittest.TestCase):
    """Test suite for synchronous close and sync translate lifecycles."""

    def test_close_sync(self):
        """Verify close_sync cleanly tears down persistent sessions."""
        translator = LLMTranslator()
        # Initialize session inside a quick run
        asyncio.run(translator._get_session())
        self.assertIsNotNone(translator._session)

        translator.close_sync()
        self.assertIsNone(translator._session)
