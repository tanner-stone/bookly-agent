"""Chat / Voice switch, autoplay, and hands-free recording.

The browser records WAV after the agent finishes speaking. This process still
sends that audio through speech-to-text, the graph, and text-to-speech.
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_root = Path(__file__).resolve().parent / "components"
_voice = components.declare_component("bookly_voice", path=str(_root / "voice"))
_status = components.declare_component("bookly_listen_status", path=str(_root / "listen_status"))


def voice_bar(
    *,
    mode: str = "chat",
    play_b64: str = "",
    play_token: str = "",
    listen_token: str = "",
    listen_after: bool = True,
):
    """Return a mode change or a new recording, or None.

    ``play_token`` changes only when there is a new clip to speak.
    ``listen_token`` changes when voice mode should open the microphone,
    either immediately or after that clip finishes.
    ``listen_after`` is false once the conversation has ended, so the
    microphone stays closed after the last sentence.
    """

    return _voice(
        mode=mode,
        play_b64=play_b64,
        play_token=play_token,
        listen_token=listen_token,
        listen_after=listen_after,
        default=None,
        key="bookly-voice",
    )


def listen_status():
    """Show Listening or Speaking under the latest message.

    The recorder posts the word. This frame only displays it, so a status
    change does not rerun the app or close the microphone.
    """

    return _status(default=None, key="bookly-listen-status")
