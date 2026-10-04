"""Unit and integration tests for concurrent LLM translation and semaphore limiting (Issue #15)."""

import asyncio
from pathlib import Path
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from diarizeflow.config import AppConfig, LLMConfig
from diarizeflow.engine.translator import LLMTranslator
from diarizeflow.engine.pipeline import DiarizeFlowPipeline, SubtitleEvent


class TestLLMTranslatorConcurrency(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying LLMTranslator concurrency limiting and semaphore protection."""

    async def test_semaphore_limits_parallel_api_calls(self):
        """Verify that at most concurrency_limit requests execute concurrently."""
        cfg = LLMConfig(provider="vllm", concurrency_limit=2)
        translator = LLMTranslator(cfg)

        active_concurrent = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def mock_call(sys_prompt, user_prompt, target_lang="繁體中文"):
            nonlocal active_concurrent, max_concurrent
            async with lock:
                active_concurrent += 1
                if active_concurrent > max_concurrent:
                    max_concurrent = active_concurrent
            try:
                await asyncio.sleep(0.04)
                return f"Translated: {user_prompt[-10:]}"
            finally:
                async with lock:
                    active_concurrent -= 1

        with patch.object(translator, "_call_openai_compatible", side_effect=mock_call):
            texts = [f"Unique Sentence {i} to translate" for i in range(6)]
            results = await asyncio.gather(*[translator.translate(t) for t in texts])

            self.assertEqual(len(results), 6)
            for res in results:
                self.assertTrue(res.startswith("Translated:"))

            # Assert concurrency never exceeded the configured limit of 2
            self.assertEqual(max_concurrent, 2, f"Expected max concurrency 2, but got {max_concurrent}")

    async def test_cache_hit_bypasses_semaphore(self):
        """Verify cached translations return immediately without calling backend."""
        cfg = LLMConfig(provider="vllm", concurrency_limit=2)
        translator = LLMTranslator(cfg)
        translator._cache["Hello->繁體中文"] = "哈囉"

        with patch.object(translator, "_call_openai_compatible") as mock_api:
            result = await translator.translate("Hello", target_language="繁體中文")
            self.assertEqual(result, "哈囉")
            mock_api.assert_not_called()

    async def test_update_config_refreshes_semaphore(self):
        """Verify update_config resets semaphore when concurrency_limit changes."""
        cfg = LLMConfig(provider="vllm", concurrency_limit=2)
        translator = LLMTranslator(cfg)
        sem1 = translator._get_semaphore()
        self.assertEqual(sem1._value, 2)

        new_cfg = LLMConfig(provider="vllm", concurrency_limit=5)
        translator.update_config(new_cfg)
        sem2 = translator._get_semaphore()
        self.assertEqual(sem2._value, 5)


class TestPipelineConcurrentTranslation(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying end-to-end concurrent translation inside DiarizeFlowPipeline."""

    async def test_multi_speaker_segments_translated_concurrently(self):
        """Verify multiple speaker segments from diarize_and_split are translated concurrently."""
        cfg = AppConfig()
        cfg.llm.provider = "vllm"
        cfg.llm.concurrency_limit = 4

        broadcasted_events = []

        def on_broadcast(evt: SubtitleEvent):
            broadcasted_events.append(evt)

        pipeline = DiarizeFlowPipeline(cfg, on_subtitle_broadcast=on_broadcast)

        # Mock diarizer to return 2 distinct speaker segments
        seg1_audio = np.ones(16000 * 2, dtype=np.float32) * 0.1  # 2.0s
        seg2_audio = np.ones(16000 * 2, dtype=np.float32) * 0.2  # 2.0s
        mock_segments = [
            (seg1_audio, "講者 1", 0.95),
            (seg2_audio, "講者 2", 0.92),
        ]
        pipeline.diarizer.diarize_and_split = MagicMock(return_value=mock_segments)

        # Mock ASR
        def mock_transcribe(audio, lang="auto"):
            if np.mean(audio) < 0.15:
                return "今天天氣真好", "zh"
            else:
                return "確實是個大晴天", "zh"

        pipeline.asr.transcribe = MagicMock(side_effect=mock_transcribe)

        # Mock Translator with a controlled delay to test concurrency
        active_translations = 0
        max_active = 0
        trans_lock = asyncio.Lock()

        async def mock_translate(text, target_language=None, source_language=None):
            nonlocal active_translations, max_active
            async with trans_lock:
                active_translations += 1
                if active_translations > max_active:
                    max_active = active_translations
            try:
                await asyncio.sleep(0.05)
                return f"譯: {text}"
            finally:
                async with trans_lock:
                    active_translations -= 1

        pipeline.translator.translate = AsyncMock(side_effect=mock_translate)

        # Start pipeline worker on current loop
        pipeline.start(asyncio.get_running_loop())

        # Feed 3.0s audio segment into utterance queue to trigger diarize_and_split
        full_audio = np.ones(16000 * 3, dtype=np.float32) * 0.1
        pipeline._on_speech_utterance(full_audio, 3.0)

        # Wait for processing to complete
        for _ in range(30):
            if len(broadcasted_events) >= 2:
                break
            await asyncio.sleep(0.05)

        pipeline.stop()

        self.assertEqual(len(broadcasted_events), 2)
        speakers = [e.speaker for e in broadcasted_events]
        self.assertIn("講者 1", speakers)
        self.assertIn("講者 2", speakers)

        # Assert translations executed concurrently (both tasks active at once)
        self.assertEqual(max_active, 2, f"Expected 2 concurrent translations, got {max_active}")

    async def test_translation_error_isolation(self):
        """Verify an error in one speaker's translation does not cancel or crash other translations."""
        cfg = AppConfig()
        cfg.llm.provider = "vllm"

        broadcasted_events = []

        def on_broadcast(evt: SubtitleEvent):
            broadcasted_events.append(evt)

        pipeline = DiarizeFlowPipeline(cfg, on_subtitle_broadcast=on_broadcast)

        seg1_audio = np.ones(16000 * 2, dtype=np.float32) * 0.1
        seg2_audio = np.ones(16000 * 2, dtype=np.float32) * 0.2
        mock_segments = [
            (seg1_audio, "講者 1", 0.95),
            (seg2_audio, "講者 2", 0.92),
        ]
        pipeline.diarizer.diarize_and_split = MagicMock(return_value=mock_segments)

        def mock_transcribe(audio, lang="auto"):
            if np.mean(audio) < 0.15:
                return "句一失敗測試", "zh"
            else:
                return "句二正常翻譯", "zh"

        pipeline.asr.transcribe = MagicMock(side_effect=mock_transcribe)

        async def mock_translate(text, target_language=None, source_language=None):
            if "失敗" in text:
                raise RuntimeError("Simulated network timeout error")
            await asyncio.sleep(0.02)
            return "成功譯文"

        pipeline.translator.translate = AsyncMock(side_effect=mock_translate)

        pipeline.start(asyncio.get_running_loop())

        full_audio = np.ones(16000 * 3, dtype=np.float32) * 0.1
        pipeline._on_speech_utterance(full_audio, 3.0)

        for _ in range(30):
            if len(broadcasted_events) >= 2:
                break
            await asyncio.sleep(0.05)

        pipeline.stop()

        self.assertEqual(len(broadcasted_events), 2)
        # Speaker 1 should fall back to original text gracefully on exception
        evt1 = next(e for e in broadcasted_events if e.speaker == "講者 1")
        self.assertEqual(evt1.translated_text, "句一失敗測試")

        # Speaker 2 should succeed normally
        evt2 = next(e for e in broadcasted_events if e.speaker == "講者 2")
        self.assertEqual(evt2.translated_text, "成功譯文")


if __name__ == "__main__":
    unittest.main()
