"""xAI speech cascade. Speech-to-text, then the graph, then text-to-speech.

This is not the realtime speech-to-speech API. The graph in the middle is
the same one chat uses.
"""

from __future__ import annotations

import os
import re

import httpx

STT_URL = "https://api.x.ai/v1/stt"
TTS_URL = "https://api.x.ai/v1/tts"


class SpeechError(RuntimeError):
    pass


def transcribe(audio: bytes, *, filename: str = "audio.wav", client: httpx.Client | None = None) -> str:
    """POST multipart STT. The file field is last, which the API requires."""

    if not audio:
        raise SpeechError("No audio to transcribe")
    fields = [
        ("language", (None, "en")),
        ("format", (None, "true")),
        ("file", (filename, audio, "audio/wav")),
    ]
    response = _post(STT_URL, client, files=fields)
    try:
        text = response.json()["text"]
    except (ValueError, KeyError) as exc:
        raise SpeechError("Speech-to-text returned no text") from exc
    if not isinstance(text, str) or not text.strip():
        raise SpeechError("Speech-to-text returned no text")
    return text.strip()


def synthesize(text: str, *, client: httpx.Client | None = None) -> bytes:
    """POST TTS. The body is MP3 bytes. Text normalization is on.

    The transcript keeps BK-10245. Only this call spells the id.
    """

    spoken = _spell_order_ids((text or "").strip())
    if not spoken:
        raise SpeechError("No text to speak")
    voice = os.environ.get("XAI_TTS_VOICE", "eve")
    response = _post(
        TTS_URL,
        client,
        json={
            "text": spoken,
            "voice_id": voice,
            "language": "en",
            "text_normalization": True,
        },
    )
    if not response.content:
        raise SpeechError("Text-to-speech returned no audio")
    return response.content


def _spell_order_ids(text: str) -> str:
    def spell(match: re.Match[str]) -> str:
        return " ".join(character for character in match.group(0) if character.isalnum())

    return re.sub(r"BK-\d+", spell, text, flags=re.I)


def _post(url: str, client: httpx.Client | None, **kwargs) -> httpx.Response:
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        raise SpeechError("XAI_API_KEY is not set")
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        if client is None:
            response = httpx.post(url, headers=headers, timeout=60, **kwargs)
        else:
            response = client.post(url, headers=headers, timeout=60, **kwargs)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SpeechError(f"Speech request failed: {type(exc).__name__}") from exc
    return response
