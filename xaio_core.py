#!/usr/bin/env python3
"""Detect a physical tap through the microphone and play claque.mp3."""

from __future__ import annotations

import os
import queue
import sys
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import numpy as np
    import pygame
    import sounddevice as sd
except ModuleNotFoundError as exc:
    print(
        f"Dependance manquante: {exc.name}\n"
        "Installez les dependances avec:\n"
        "  python -m pip install sounddevice numpy pygame",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Sensibilite principale: baissez ce seuil pour detecter des chocs plus faibles
# ou augmentez-le si le script se declenche trop facilement.
# Valeur attendue: amplitude audio normalisee entre 0.0 et 1.0.
THRESHOLD = 0.18
# ---------------------------------------------------------------------------

COOLDOWN_SECONDS = 1.0
MP3_FILENAME = "claque.mp3"

# Buffers courts = meilleure reactivite. 256 samples ~= 5.8 ms a 44.1 kHz.
SAMPLE_RATE = 44_100
BLOCK_SIZE = 256


def load_sound(mp3_path: Path) -> pygame.mixer.Sound:
    """Initialize pygame's mixer and load the MP3 effect."""
    if not mp3_path.exists():
        raise FileNotFoundError(
            f"Fichier audio introuvable: {mp3_path}\n"
            f"Placez {MP3_FILENAME} dans le meme dossier que ce script."
        )

    pygame.mixer.init(frequency=SAMPLE_RATE)
    return pygame.mixer.Sound(str(mp3_path))


def build_audio_callback(sound: pygame.mixer.Sound, events: queue.Queue[str]):
    """Create the PortAudio callback used by sounddevice.InputStream."""
    last_triggered_at = 0.0

    def audio_callback(indata, frames, callback_time, status) -> None:
        nonlocal last_triggered_at

        if status:
            events.put(f"Avertissement audio: {status}")

        # Convert stereo/multichannel input to a single signal for analysis.
        samples = indata
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)

        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(np.square(samples))))

        # Peak catches very short transients; RMS rejects isolated numeric noise.
        level = max(peak, rms * 2.0)
        now = time.monotonic()

        if level >= THRESHOLD and now - last_triggered_at >= COOLDOWN_SECONDS:
            last_triggered_at = now
            sound.play()  # Non-blocking: pygame mixes the sound in the background.
            events.put("Ah xaio")

    return audio_callback


def open_input_stream(callback):
    """Open the default microphone with a clear error if none is available."""
    try:
        device = sd.query_devices(kind="input")
    except Exception as exc:
        raise RuntimeError(
            "Microphone introuvable. Verifiez qu'un peripherique d'entree audio "
            "est connecte et accessible."
        ) from exc

    if device is None or int(device.get("max_input_channels", 0)) <= 0:
        raise RuntimeError(
            "Microphone introuvable. Aucun peripherique d'entree audio valide."
        )

    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=callback,
    )


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    sound_path = script_dir / MP3_FILENAME
    events: queue.Queue[str] = queue.Queue()

    try:
        sound = load_sound(sound_path)
        callback = build_audio_callback(sound, events)

        with open_input_stream(callback):
            print("PRÊT : Tape sur l'ordi", flush=True)

            while True:
                try:
                    print(events.get(timeout=0.1), flush=True)
                except queue.Empty:
                    pass

    except KeyboardInterrupt:
        print("\nFermeture propre.", flush=True)
        return 0
    except Exception as exc:
        print(f"Erreur: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        pygame.mixer.quit()
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
