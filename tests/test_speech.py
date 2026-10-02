"""Speech request shape. These tests do not call xAI."""

import httpx

from voice.xai_speech import SpeechError, synthesize, transcribe


def test_transcribe_puts_the_file_last_and_reads_text(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        body = request.content.decode("latin1")
        seen["file_after_language"] = body.find("name=\"language\"") < body.find("name=\"file\"")
        seen["format"] = "name=\"format\"" in body
        return httpx.Response(200, json={"text": " where is my package "})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert transcribe(b"RIFFwav", client=client) == "where is my package"
    assert seen["url"] == "https://api.x.ai/v1/stt"
    assert seen["file_after_language"] is True
    assert seen["format"] is True


def test_synthesize_asks_for_normalized_speech(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("XAI_TTS_VOICE", "eve")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(200, content=b"ID3fake")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert synthesize("It shipped.", client=client) == b"ID3fake"
    assert "\"text_normalization\": true" in seen["body"] or "\"text_normalization\":true" in seen["body"]
    assert "\"voice_id\":\"eve\"" in seen["body"].replace(" ", "") or "\"voice_id\": \"eve\"" in seen["body"]


def test_speech_spells_an_order_id_the_transcript_keeps_whole(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(200, content=b"ID3fake")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert synthesize("Order BK-10245 is in Chicago.", client=client) == b"ID3fake"
    assert "BK-10245" not in seen["body"]
    assert "B K 1 0 2 4 5" in seen["body"]


def test_empty_audio_is_rejected(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    try:
        transcribe(b"", client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))))
        raised = False
    except SpeechError:
        raised = True
    assert raised
