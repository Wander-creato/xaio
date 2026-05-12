# -*- coding: utf-8 -*-
"""Detect a physical tap through the microphone and play claque.mp3.

Dependencies:
    pip install sounddevice numpy pygame
"""

from __future__ import annotations

import os
import queue
import signal
import sys
import time
from pathlib import Path

PIP_INSTALL_COMMAND = "pip install sounddevice numpy pygame"

try:
    import numpy as np
    import sounddevice as sd

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
except ModuleNotFoundError as exc:
    print(f"Dependance manquante : {exc.name}", file=sys.stderr)
    print(f"Installez les dependances avec : {PIP_INSTALL_COMMAND}", file=sys.stderr)
    raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Sensibilite de detection : baissez la valeur pour detecter des chocs plus
# faibles, augmentez-la si le script se declenche trop souvent.
THRESHOLD = 0.25
# ---------------------------------------------------------------------------

MP3_FILE = "claque.mp3"
COOLDOWN_SECONDS = 1.0
SAMPLE_RATE = 44_100
BLOCK_SIZE = 256  # ~5.8 ms a 44.1 kHz pour une bonne reactivite.
CHANNELS = 1
DTYPE = "float32"


def find_input_device() -> int | None:
    """Return the default input device index, or another usable microphone."""
    try:
        default_input = sd.default.device[0]
        if default_input is not None and default_input >= 0:
            device_info = sd.query_devices(default_input, "input")
            if int(device_info.get("max_input_channels", 0)) > 0:
                return int(default_input)
    except (sd.PortAudioError, TypeError, ValueError):
        pass

    try:
        for index, device in enumerate(sd.query_devices()):
            if int(device.get("max_input_channels", 0)) > 0:
                return index
    except sd.PortAudioError:
        return None

    return None


def get_input_sample_rate(device: int) -> int:
    """Prefer the microphone's native sample rate to avoid PortAudio errors."""
    try:
        device_info = sd.query_devices(device, "input")
        sample_rate = int(device_info.get("default_samplerate") or SAMPLE_RATE)
    except (sd.PortAudioError, TypeError, ValueError):
        sample_rate = SAMPLE_RATE

    return sample_rate if sample_rate > 0 else SAMPLE_RATE


def load_sound() -> pygame.mixer.Sound:
    """Initialize pygame's mixer and load the MP3 from this script's folder."""
    sound_path = Path(__file__).with_name(MP3_FILE)
    if not sound_path.is_file():
        raise FileNotFoundError(f"Fichier audio introuvable : {sound_path}")

    pygame.mixer.init(frequency=SAMPLE_RATE)
    return pygame.mixer.Sound(str(sound_path))


def main() -> int:
    trigger_events: queue.SimpleQueue[float] = queue.SimpleQueue()
    stop_requested = False
    last_trigger_at = 0.0

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    def audio_callback(indata: np.ndarray, _frames: int, _time_info: object, status: sd.CallbackFlags) -> None:
        nonlocal last_trigger_at

        if status.input_overflow:
            print("Attention : surcharge du flux micro.", file=sys.stderr)

        now = time.monotonic()
        if now - last_trigger_at < COOLDOWN_SECONDS:
            return

        samples = indata[:, 0] if indata.ndim > 1 else indata
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(np.square(samples))))

        if peak >= THRESHOLD or rms >= THRESHOLD * 0.6:
            last_trigger_at = now
            trigger_events.put(now)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        sound = load_sound()
    except (FileNotFoundError, pygame.error) as exc:
        print(f"Erreur audio : {exc}", file=sys.stderr)
        return 1

    device = find_input_device()
    if device is None:
        print("Erreur : aucun microphone d'entree n'a ete trouve.", file=sys.stderr)
        pygame.mixer.quit()
        return 1

    input_sample_rate = get_input_sample_rate(device)

    try:
        with sd.InputStream(
            device=device,
            channels=CHANNELS,
            samplerate=input_sample_rate,
            blocksize=BLOCK_SIZE,
            dtype=DTYPE,
            callback=audio_callback,
        ):
            print("PRÊT : Tape sur l'ordi", flush=True)

            while not stop_requested:
                try:
                    trigger_events.get(timeout=0.01)
                except queue.Empty:
                    continue

                print("Ah xaio", flush=True)
                sound.play()

    except sd.PortAudioError as exc:
        print(f"Erreur microphone : {exc}", file=sys.stderr)
        return 1
    finally:
        pygame.mixer.quit()

    print("\nArret propre.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
