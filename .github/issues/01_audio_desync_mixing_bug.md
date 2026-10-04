# [Bug]: Audio desynchronization and temporal expansion when mixing microphone and loopback audio

**Labels**: `bug`, `audio`, `high-priority`

## Description
When `mix_sources: true` is enabled in `config.json` (capturing both microphone and system loopback simultaneously), the audio capture processing worker exhibits a critical desynchronization bug. 

Because `buffer_mic` and `buffer_loop` receive audio chunks asynchronously via separate driver callbacks, the condition `len(buffer_mic) >= chunk_samples or len(buffer_loop) >= chunk_samples` triggers even if only one stream has ready samples. The stream that is not yet ready is padded with zeros and immediately dispatched. In the subsequent cycle, the second stream's delayed samples arrive and the other stream is padded with zeros.

As a result, rather than being aligned and mixed sample-by-sample, the two streams are interleaved with alternating silence blocks. This expands the effective audio duration (nearly doubling the number of chunks), slows down apparent playback speed by 50%, and injects rhythmic stuttering and clicks that corrupt Voice Activity Detection (VAD) and Automatic Speech Recognition (ASR).

## Code Reference
Location: [`src/diarizeflow/app/audio/capture.py:209-223`](file:///home/yijun/Project/DiarizeFlow/src/diarizeflow/app/audio/capture.py#L209-L223)

```python
elif has_mic and has_loop:
    # Both streams active: independent sample draining without blocking
    while len(buffer_mic) >= self.chunk_samples or len(buffer_loop) >= self.chunk_samples:
        if len(buffer_mic) >= self.chunk_samples:
            m_chunk = buffer_mic[: self.chunk_samples]
            buffer_mic = buffer_mic[self.chunk_samples :]
        else:
            m_chunk = np.zeros(self.chunk_samples, dtype=np.float32)

        if len(buffer_loop) >= self.chunk_samples:
            l_chunk = buffer_loop[: self.chunk_samples]
            buffer_loop = buffer_loop[self.chunk_samples :]
        else:
            l_chunk = np.zeros(self.chunk_samples, dtype=np.float32)

        mixed = np.clip(m_chunk + l_chunk, -1.0, 1.0)
        self._dispatch_chunk(mixed)
```

## Steps to Reproduce
1. In `config.json`, configure both a valid microphone and system loopback device, with `"mix_sources": true`.
2. Start the desktop HUD (`python scripts/run_app.py --mode desktop`).
3. Play system audio (e.g. YouTube or music) while speaking into the microphone.
4. Record or log incoming utterances passed to the pipeline.

## Expected Behavior
The microphone and system loopback streams should be synchronously aligned on the timeline and added together (`mic + loopback`) without altering the total audio playback duration or inserting artificial silence frames.

## Actual Behavior
Audio is time-dilated (~2x length), stuttered with interleaved silence chunks, causing VAD to trigger false endpoints and ASR to output garbled phonemes or cut off recognition.

## Proposed Fix
Synchronize draining so that chunks are only popped and mixed when *both* buffers contain at least `self.chunk_samples`, with a timeout or drift-compensation fallback if one stream halts:

```python
elif has_mic and has_loop:
    # Synchronously drain and mix only when both streams have accumulated a full chunk
    while len(buffer_mic) >= self.chunk_samples and len(buffer_loop) >= self.chunk_samples:
        m_chunk = buffer_mic[: self.chunk_samples]
        buffer_mic = buffer_mic[self.chunk_samples :]
        l_chunk = buffer_loop[: self.chunk_samples]
        buffer_loop = buffer_loop[self.chunk_samples :]

        mixed = np.clip(m_chunk + l_chunk, -1.0, 1.0)
        self._dispatch_chunk(mixed)
```
