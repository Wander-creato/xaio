#!/usr/bin/env python3
"""Detecte un choc via micro et joue claque.mp3 instantanement."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import sounddevice as sd

# Reglage principal de sensibilite (0.0 a 1.0).
THRESHOLD = 0.20

# Buffers courts pour maximiser la reactivite.
SAMPLERATE = 44_100
BLOCKSIZE = 256
CHANNELS = 1
COOLDOWN_SECONDS = 1.0
MP3_FILENAME = "claque.mp3"


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    mp3_path = base_dir / MP3_FILENAME

    if not mp3_path.exists():
        print(f"Erreur: fichier audio introuvable: {mp3_path}")
        return 1

    try:
        pygame.mixer.pre_init(frequency=44_100, channels=2, buffer=256)
        pygame.mixer.init()
        sound = pygame.mixer.Sound(str(mp3_path))
    except pygame.error as exc:
        print(f"Erreur d'initialisation audio (pygame): {exc}")
        return 1

    try:
        default_input = sd.default.device[0]
        if default_input is None or default_input < 0:
            print("Erreur: aucun microphone d'entree par defaut n'est configure.")
            return 1
        sd.check_input_settings(
            device=default_input,
            channels=CHANNELS,
            samplerate=SAMPLERATE,
            dtype="float32",
        )
    except Exception as exc:
        print(f"Erreur micro: {exc}")
        return 1

    stop_flag = {"value": False}
    last_trigger_time = {"value": 0.0}

    def request_stop(_signum: int, _frame: object) -> None:
        stop_flag["value"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def audio_callback(indata: np.ndarray, frames: int, _time_info: object, status: sd.CallbackFlags) -> None:
        del frames
        if status:
            # Les warnings audio n'empechent pas necessairement la detection.
            print(f"[audio-status] {status}", file=sys.stderr)

        if stop_flag["value"]:
            raise sd.CallbackStop

        # Mesure de pic absolu, tres reactive pour capturer les transients.
        peak = float(np.max(np.abs(indata)))
        now = time.monotonic()

        if peak >= THRESHOLD and (now - last_trigger_time["value"]) >= COOLDOWN_SECONDS:
            last_trigger_time["value"] = now
            print("Ah xaio")
            sound.play()

    print("PRÊT : Tape sur l'ordi")
    try:
        with sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            dtype="float32",
            callback=audio_callback,
        ):
            while not stop_flag["value"]:
                time.sleep(0.05)
    except KeyboardInterrupt:
        # Normalement intercepte par le signal handler, garde-fou.
        pass
    except Exception as exc:
        print(f"Erreur durant l'ecoute micro: {exc}")
        return 1
    finally:
        try:
            pygame.mixer.quit()
        except pygame.error:
            pass

    print("Arrêt propre.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
