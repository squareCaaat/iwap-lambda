import os
from pathlib import Path as _PathForNumbaConfig

NUMBA_CACHE_DIR = _PathForNumbaConfig("/tmp") / ".numba_cache"
NUMBA_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

import numba  # noqa: E402
numba.config.DISABLE_CACHE = True
import numpy as np
import librosa
import pretty_midi
from pathlib import Path
import io
from typing import Optional
import soundfile as sf
from pydub import AudioSegment

N_FFT = 2048
HOP_LENGTH = 512
VELOCITY = 100
THRESHOLD_RATIO = 0.1
NOTE_DURATION = 0.5
MAX_MELODY_FREQUENCY = 3000
MAX_NOTE_JUMP = 24
  
INPUT_MP3_PATH = Path(os.getenv("PIANO_MP3_PATH", "/tmp/input.mp3"))
OUTPUT_MIDI_PATH = Path(os.getenv("PIANO_OUTPUT_MIDI_PATH", "/tmp/output.mid"))
DEFAULT_SF2_PATH = Path(os.getenv("PIANO_SF2_PATH", "/opt/soundfont.sf2"))

def freq_to_midi(freq: int) -> int:
    if freq <= 0:
        return -1
    return int(np.round(69 + 12 * np.log2(freq / 440.0)))


def talking_piano():
    audio_samples, sample_rate = librosa.load(INPUT_MP3_PATH, sr=44100, mono=True)

    spectrogram = np.abs(librosa.stft(audio_samples, n_fft=N_FFT, hop_length=HOP_LENGTH))
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=HOP_LENGTH)

    max_freq_index = np.searchsorted(frequencies, MAX_MELODY_FREQUENCY)
    if max_freq_index == 0:
        max_freq_index = len(frequencies)

    midi_file = pretty_midi.PrettyMIDI()
    piano_instrument = pretty_midi.Instrument(program=0)
    threshold_value = THRESHOLD_RATIO * np.max(spectrogram)
    
    current_note = None
    last_note_pitch = None
    
    for time_index, time_value in enumerate(times):
        spectrum_slice = spectrogram[:max_freq_index, time_index]
        peak_index = np.argmax(spectrum_slice)
        peak_magnitude = spectrum_slice[peak_index]
        
        note_number = -1
        
        if peak_magnitude > threshold_value:
            frequency_value = frequencies[peak_index]
            note_number = freq_to_midi(frequency_value)
            if not (0 <= note_number <= 127):
                note_number = -1
        
        if note_number != -1:
            if last_note_pitch is not None and abs(note_number - last_note_pitch) > MAX_NOTE_JUMP:
                note_number = -1
            else:
                last_note_pitch = note_number
        else:
            last_note_pitch = None

        if note_number != -1:
            if current_note is None:
                current_note = pretty_midi.Note(
                    velocity=VELOCITY, pitch=note_number, start=time_value, end=time_value + NOTE_DURATION
                )
                piano_instrument.notes.append(current_note)
            elif current_note.pitch == note_number:
                current_note.end = time_value + NOTE_DURATION
            else:
                current_note = pretty_midi.Note(
                    velocity=VELOCITY, pitch=note_number, start=time_value, end=time_value + NOTE_DURATION
                )
                piano_instrument.notes.append(current_note)
        else:
            current_note = None

    midi_file.instruments.append(piano_instrument)
    
    OUTPUT_MIDI_PATH.parent.mkdir(parents=True, exist_ok=True)
    midi_file.write(str(OUTPUT_MIDI_PATH))
    print(f"MIDI 생성 완료 (파일 저장): {OUTPUT_MIDI_PATH}")
    
    return midi_file, sample_rate


def midi_to_mp3_bytes(
    midi_data: pretty_midi.PrettyMIDI,
    sf2_path: Optional[Path] = None,
    sample_rate: int = 44100,
) -> bytes:
    resolved_sf2_path = Path(sf2_path) if sf2_path is not None else DEFAULT_SF2_PATH
    if not resolved_sf2_path.exists():
        raise ValueError(f"SoundFont 파일을 찾을 수 없습니다: {resolved_sf2_path}")

    try:
        audio_data = midi_data.fluidsynth(fs=sample_rate, sf2_path=str(resolved_sf2_path))
    except Exception as e:
        print(f"MIDI 합성(fluidsynth) 오류: {e}")
        raise
        
    if audio_data.size == 0:
        raise ValueError("MIDI 파일에 음표가 없거나 합성에 실패하여 빈 오디오가 생성되었습니다.")

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio_data, sample_rate, format="WAV", subtype='PCM_16')
    wav_buffer.seek(0)

    mp3_buffer = io.BytesIO()
    try:
        AudioSegment.from_file(wav_buffer, format="wav").export(mp3_buffer, format="mp3")
    except FileNotFoundError:
        raise
    except Exception as e:
        print(f"pydub MP3 export 오류: {e}")
        raise

    return mp3_buffer.getvalue()