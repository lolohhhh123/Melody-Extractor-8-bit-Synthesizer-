#!/usr/bin/env python3
"""
旋律提取器 - 支持记忆显著性分析 / 全曲提取
动态音符时长阈值（基于BPM）+ 延迟确认（消除颤音碎片）

Melody Extractor - Supports memorability saliency analysis or full‑song extraction.
Dynamic note duration threshold (BPM‑based) + delayed confirmation (removes vibrato fragments).
"""

import os
import sys
import numpy as np
import librosa
from scipy.signal import medfilt, savgol_filter
from midiutil import MIDIFile
import warnings
warnings.filterwarnings('ignore')
from rmvpe import RMVPE
import torch
import soundfile as sf
import json

# ==================== Configuration / 配置 ====================
CONFIG = {
    'window_duration': 4.0,
    'hop_duration': 2.0,
    'top_k': 3,
    'n_mels': 128,
    'n_chroma': 12,
    'similarity_threshold': 0.6,
    'pitch_hop_ms': 10,
    'median_filter_duration': 0.1,
    # 新增：稳定倍数，默认 2 即十六分音符
    # stability factor: minimum switch‑stable duration = factor * min_note_duration (default 2 → 16th note)
    'stability_factor': 2.0,   # 最小切换稳定时长 = factor * min_note_duration
    'melody_volume': 0.6,
    'bass_volume': 0.7,
    'arpeggio_volume': 0.3,
    'drum_volume': 0.5,
    'mute_intervals': [],   # 静音区间列表，每个元素为 (start, end) 秒  / mute intervals (start, end) in seconds
}

# Drum patterns for different styles
DRUM_PATTERNS = {
    'rock': {  # 4/4拍，典型摇滚 / 4/4 typical rock
        0: 'kick',
        1: 'snare',
        2: 'kick',
        3: 'snare',
        'offbeat': 'hihat'  # 八分音符踩镲 / eighth‑note hi‑hat
    },
    'pop': {
        0: 'kick',
        1: 'snare',
        2: 'kick',
        3: 'snare',
        'offbeat': 'hihat'
    },
    'funk': {
        0: 'kick',
        1: 'snare',
        2: 'kick',
        3: 'snare',
        '16th': 'hihat'  # 十六分音符踩镲 / sixteenth‑note hi‑hat
    },
    'simple': {
        0: 'kick',
        2: 'snare',
        # 没有 offbeat / no offbeat
    },
}


def mix_melody_with_original(original_audio_path, melody_wav_path, output_mix_path):
    """
    Mix the generated melody WAV with the original song, aligning them in time.
    Preserves the original stereo channels and sample rate.
    将生成的旋律 WAV 与原曲按时间对齐叠加，保留原曲的立体声和原始采样率。

    Parameters
    ----------
    original_audio_path : str
        Path to the original audio file.
    melody_wav_path : str
        Path to the generated melody WAV (typically mono).
    output_mix_path : str
        Path for the mixed output file.
    """
    # 1. Load original audio (preserve stereo, keep original sample rate)
    # 1. 读取原曲（保留立体声，保持原始采样率）
    import librosa
    orig_audio, orig_sr = librosa.load(original_audio_path, sr=None, mono=False)
    # orig_audio shape: (channels, samples) or (samples,) if mono
    if orig_audio.ndim == 1:
        orig_audio = np.expand_dims(orig_audio, axis=0)  # unify to (channels, samples)
    orig_channels = orig_audio.shape[0]
    orig_samples = orig_audio.shape[1]

    # 2. Load melody WAV
    melody, mel_sr = sf.read(melody_wav_path)
    # If melody is stereo, also convert to (channels, samples)
    if melody.ndim == 1:
        melody = np.expand_dims(melody, axis=0)
    mel_channels = melody.shape[0]
    mel_samples = melody.shape[1]

    # 3. Resample melody to original sample rate if necessary
    if mel_sr != orig_sr:
        import librosa
        melody_resampled = []
        for ch in range(mel_channels):
            resampled = librosa.resample(melody[ch], orig_sr=mel_sr, target_sr=orig_sr)
            melody_resampled.append(resampled)
        melody = np.array(melody_resampled)
        mel_sr = orig_sr
        mel_samples = melody.shape[1]

    # 4. Align lengths – zero‑pad the shorter one
    if mel_samples < orig_samples:
        # melody shorter, pad to original length
        pad_width = ((0, 0), (0, orig_samples - mel_samples))
        melody_padded = np.pad(melody, pad_width, mode='constant')
    else:
        # melody longer, truncate to original length
        melody_padded = melody[:, :orig_samples]

    # 5. If melody is mono and original is stereo, duplicate to stereo
    if mel_channels == 1 and orig_channels == 2:
        melody_stereo = np.vstack([melody_padded, melody_padded])  # (2, samples)
    elif mel_channels == orig_channels:
        melody_stereo = melody_padded
    else:
        # other cases: replicate first channel to all original channels
        melody_stereo = np.tile(melody_padded, (orig_channels, 1))

    # 6. Mix: original volume unchanged, melody volume adjustable (default 0.7)
    melody_volume = 0.7  # adjustable
    mix = orig_audio + melody_volume * melody_stereo

    # 7. Prevent clipping: if peak > 0.95, scale down proportionally, preserving dynamics
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix * (0.95 / peak)

    # 8. Save – output expects (samples, channels) layout
    mix_to_write = mix.T  # soundfile expects (samples, channels)
    sf.write(output_mix_path, mix_to_write, orig_sr)
    print(f"✅ 高质量混音 WAV 已保存: {output_mix_path} (立体声, {orig_sr}Hz) / High‑quality mix WAV saved: {output_mix_path} (stereo, {orig_sr}Hz)")


def simple_synth_midi_to_wav(notes, wav_path, sample_rate=22050):
    """
    Synthesize a simple sine‑wave WAV from note list with short fade in/out (10 ms)
    and dynamic peak normalization to avoid clipping.
    用正弦波 + 略长淡入淡出（10ms），并对最终音频做动态归一化（避免削波）生成 WAV。

    Parameters
    ----------
    notes : list of dict
        Each dict contains 'pitch' (MIDI), 'start' (seconds), 'duration' (seconds).
    wav_path : str
        Output WAV path.
    sample_rate : int
        Sample rate for the generated audio.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    if not notes:
        print("❌ 没有音符，无法合成 / No notes to synthesize")
        return False

    total_duration = notes[-1]['start'] + notes[-1]['duration']
    num_samples = int(total_duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    fade_ms = 0.01  # 10 ms fade in/out for smoother transitions
    fade_len = int(fade_ms * sample_rate)

    print(f"🎵 正弦波合成中... 总时长 {total_duration:.2f}s, 音符数 {len(notes)} / Synthesizing sine wave... total duration {total_duration:.2f}s, {len(notes)} notes")

    for note in notes:
        freq = 440.0 * (2.0 ** ((note['pitch'] - 69) / 12.0))
        if freq < 50.0:
            continue

        start_idx = int(note['start'] * sample_rate)
        duration_samples = int(note['duration'] * sample_rate)
        end_idx = min(start_idx + duration_samples, num_samples)
        if end_idx - start_idx < fade_len * 2:
            continue  # ignore extremely short notes

        t = np.arange(end_idx - start_idx) / sample_rate
        wave = 0.4 * np.sin(2 * np.pi * freq * t)  # moderate volume

        # Apply fade in/out
        wave[:fade_len] *= np.linspace(0, 1, fade_len)
        wave[-fade_len:] *= np.linspace(1, 0, fade_len)

        audio[start_idx:end_idx] += wave

    # Dynamic peak normalization: avoid hard clipping while keeping overall loudness
    peak = np.max(np.abs(audio))
    if peak > 0.01:
        audio = audio / peak * 0.95  # leave a little headroom
    else:
        audio[:] = 0.0

    sf.write(wav_path, audio, sample_rate)
    print(f"✅ WAV 已保存 (正弦波): {wav_path} / WAV saved (sine wave): {wav_path}")
    return True


# ==================== 8-bit Multi‑track Synthesizer / 8-bit 多轨合成器 ====================

def synth_note_wave(freq, duration, sample_rate, waveform='square', duty=0.5, volume=0.4, decay=5.0):
    """
    Generate the waveform of a single note with selectable shape and exponential decay envelope.
    生成单个音符的波形

    Parameters
    ----------
    freq : float
        Frequency in Hz.
    duration : float
        Duration in seconds.
    sample_rate : int
    waveform : str
        One of 'square', 'triangle', 'noise', 'sine'.
    duty : float
        Duty cycle for square wave (0.5 = symmetric, <0.5 sharper).
    volume : float
        Peak amplitude before envelope.
    decay : float
        Exponential decay rate (higher = faster decay, recommended 3~8).

    Returns
    -------
    np.ndarray (float32)
        The generated waveform.
    """
    num_samples = int(duration * sample_rate)
    if num_samples <= 0:
        return np.array([])
    t = np.arange(num_samples) / sample_rate

    if waveform == 'square':
        # Use sign function with adjustable duty cycle
        phase = (freq * t) % 1.0
        wave = np.where(phase < duty, 1.0, -1.0)
    elif waveform == 'triangle':
        wave = 2 * np.abs(2 * (freq * t % 1) - 1) - 1
    elif waveform == 'noise':
        wave = np.random.uniform(-1, 1, num_samples)
    else:  # sine
        wave = np.sin(2 * np.pi * freq * t)

    # Exponential decay envelope (simulates chip‑music fast decay)
    envelope = np.exp(-decay * t)
    wave = wave * envelope * volume
    return wave.astype(np.float32)


def synth_drum_note(drum_type, duration, sample_rate, volume=0.5, decay=10.0):
    """
    Generate a short drum hit waveform for the given drum type.
    生成特定鼓音色的短脉冲

    drum_type : str
        'kick', 'snare', 'hihat', 'tom'
    duration : float
        Duration in seconds.
    sample_rate : int
    volume : float
    decay : float

    Returns
    -------
    np.ndarray (float32)
    """
    num_samples = int(duration * sample_rate)
    if num_samples <= 0:
        return np.array([])
    t = np.arange(num_samples) / sample_rate

    if drum_type == 'kick':
        # Kick drum: low‑freq sine + short noise, fast decay
        freq = 60
        wave = np.sin(2 * np.pi * freq * t) * 0.6
        wave += np.random.uniform(-0.2, 0.2, num_samples)  # small amount of noise
        envelope = np.exp(-decay * t)  # fast decay
        wave = wave * envelope

    elif drum_type == 'snare':
        # Snare: mid‑freq noise + short square wave
        noise = np.random.uniform(-1, 1, num_samples)
        # Simple high‑pass via differentiation (approximate)
        carrier = np.sin(2 * np.pi * 200 * t)  # 200 Hz carrier
        wave = 0.5 * carrier + 0.5 * noise
        envelope = np.exp(-decay * t)
        wave = wave * envelope

    elif drum_type == 'hihat':
        # Hi‑hat: high‑freq noise, very short decay
        wave = np.random.uniform(-1, 1, num_samples)
        # High‑pass approximation via first difference
        wave = np.diff(wave, prepend=0)
        envelope = np.exp(-decay * t * 2)  # faster decay
        wave = wave * envelope

    elif drum_type == 'tom':
        # Tom: mid‑low sine, slightly longer decay
        freq = 120
        wave = np.sin(2 * np.pi * freq * t)
        envelope = np.exp(-decay * t * 0.7)
        wave = wave * envelope

    else:
        wave = np.random.uniform(-1, 1, num_samples)
        envelope = np.exp(-decay * t)
        wave = wave * envelope

    # Normalize to target volume
    peak = np.max(np.abs(wave))
    if peak > 1e-6:
        wave = wave / peak * volume
    return wave.astype(np.float32)


def generate_bass_track(melody_notes, bpm, octave_down=1):
    """
    Create a bass track by lowering the melody notes by an octave and shortening to 80% duration.
    生成低音轨道，将主旋律降低八度，时长缩短为80%
    """
    bass_notes = []
    for n in melody_notes:
        if n['pitch'] - 12 * octave_down >= 0:
            bass_notes.append({
                'pitch': n['pitch'] - 12 * octave_down,
                'start': n['start'],
                'duration': n['duration'] * 0.8,  # shorter
                'waveform': 'triangle',
                'volume': 0.3,
                'decay': 6.0
            })
    return bass_notes


def generate_arpeggio_track(melody_notes, bpm, arp_interval=0.1):
    """
    Generate an arpeggio (counter‑melody) track by splitting long notes into triplets
    (root, fifth, octave). Only notes longer than 0.3 s are processed.
    生成副旋律（琶音），将主旋律的每个长音符拆成三连音（根音、五度、高八度）
    """
    arp_notes = []
    for n in melody_notes:
        if n['duration'] < 0.3:
            continue
        pitch = n['pitch']
        # Triad: root, fifth (+7 semitones), octave root (+12)
        offsets = [0, 7, 12]
        duration_per_note = n['duration'] / len(offsets) * 0.6  # leave gaps
        for i, off in enumerate(offsets):
            new_pitch = pitch + off
            if new_pitch > 127:
                continue
            start_time = n['start'] + i * (n['duration'] / len(offsets))
            arp_notes.append({
                'pitch': new_pitch,
                'start': start_time,
                'duration': duration_per_note,
                'waveform': 'square',
                'duty': 0.25,   # sharper
                'volume': 0.25,
                'decay': 4.0
            })
    return arp_notes


def generate_drum_track(duration, bpm, sample_rate, pattern='simple', beat_times=None):
    """
    Generate a drum track with multiple timbres and selectable rhythm patterns.
    If beat_times (detected beat positions in seconds) is provided, drums are aligned to those.
    生成打击乐轨道，支持多种音色和节奏型。
    beat_times: 检测到的节拍时间点列表（秒），若提供则鼓点对齐这些时间

    Parameters
    ----------
    duration : float
        Total length of audio in seconds.
    bpm : float
        Tempo.
    sample_rate : int
    pattern : str
        One of 'simple', 'rock', 'pop', 'funk'.
    beat_times : list or None
        Optional list of beat onset times (seconds) for alignment.

    Returns
    -------
    list of dict
        Each dict has 'start', 'duration', 'drum_type', 'volume', 'decay'.
    """
    beat_duration = 60.0 / bpm
    drum_notes = []

    if beat_times is not None and len(beat_times) > 0:
        # Use detected beat positions
        for i, t in enumerate(beat_times):
            if t > duration:
                break
            beat_pos = i % 4  # assume 4/4 time
            pattern_dict = DRUM_PATTERNS.get(pattern, DRUM_PATTERNS['simple'])
            drum_type = pattern_dict.get(beat_pos, None)
            if drum_type:
                drum_notes.append({
                    'start': t,
                    'duration': min(0.08, beat_duration * 0.15),
                    'drum_type': drum_type,
                    'volume': 0.6 if drum_type in ('kick', 'snare') else 0.4,
                    'decay': 10.0 if drum_type in ('kick', 'snare') else 20.0,
                })
            if 'offbeat' in pattern_dict:
                offbeat_time = t + beat_duration * 0.5
                if offbeat_time < duration:
                    drum_notes.append({
                        'start': offbeat_time,
                        'duration': min(0.04, beat_duration * 0.1),
                        'drum_type': 'hihat',
                        'volume': 0.3,
                        'decay': 20.0,
                    })
            if '16th' in pattern_dict:
                for j in [1, 2, 3]:
                    sixteenth = t + beat_duration * j / 4
                    if sixteenth < duration:
                        drum_notes.append({
                            'start': sixteenth,
                            'duration': min(0.03, beat_duration * 0.08),
                            'drum_type': 'hihat',
                            'volume': 0.25,
                            'decay': 25.0,
                        })
    else:
        # Fallback to even spacing based on BPM
        total_beats = int(duration / beat_duration) + 1
        for beat in range(total_beats):
            t = beat * beat_duration
            if t > duration:
                break
            beat_pos = beat % 4
            pattern_dict = DRUM_PATTERNS.get(pattern, DRUM_PATTERNS['simple'])
            drum_type = pattern_dict.get(beat_pos, None)
            if drum_type:
                drum_notes.append({
                    'start': t,
                    'duration': min(0.08, beat_duration * 0.15),
                    'drum_type': drum_type,
                    'volume': 0.6 if drum_type in ('kick', 'snare') else 0.4,
                    'decay': 10.0 if drum_type in ('kick', 'snare') else 20.0,
                })
            if 'offbeat' in pattern_dict:
                offbeat_time = t + beat_duration * 0.5
                if offbeat_time < duration:
                    drum_notes.append({
                        'start': offbeat_time,
                        'duration': min(0.04, beat_duration * 0.1),
                        'drum_type': 'hihat',
                        'volume': 0.3,
                        'decay': 20.0,
                    })
            if '16th' in pattern_dict:
                for j in [1, 2, 3]:
                    sixteenth = t + beat_duration * j / 4
                    if sixteenth < duration:
                        drum_notes.append({
                            'start': sixteenth,
                            'duration': min(0.03, beat_duration * 0.08),
                            'drum_type': 'hihat',
                            'volume': 0.25,
                            'decay': 25.0,
                        })
    return drum_notes


def synth_multi_track(notes, bpm, sample_rate=22050,
                      enable_bass=True, enable_arpeggio=True, enable_drums=True,
                      time_offset=0.0,
                      melody_volume=0.4, bass_volume=0.3, arpeggio_volume=0.25, drum_volume=0.5,
                      mute_intervals=None, drum_pattern='simple', beat_times=None):
    """
    Main synthesis function: generate 8‑bit‑style multi‑track audio (melody, bass, arpeggio, drums).
    主合成函数：生成包含多轨道的8-bit风格音频

    mute_intervals: list of (start, end) mute intervals in seconds.
    drum_pattern: 'simple', 'rock', 'pop', 'funk'.
    beat_times: optional beat times for drum alignment.

    Returns
    -------
    np.ndarray
        Synthesized audio array.
    """
    if mute_intervals is None:
        mute_intervals = []
    if not notes:
        return np.zeros(0)

    # Helper to check if a time range overlaps any mute interval
    def is_muted(start, duration):
        end = start + duration
        for s, e in mute_intervals:
            if start <= e and end >= s:
                return True
        return False

    total_duration = notes[-1]['start'] + notes[-1]['duration'] + 1.0
    audio = np.zeros(int(total_duration * sample_rate))

    # 1. Main melody (square wave)
    print("   🎵 合成主旋律 (方波) / Synthesizing melody (square)...")
    for n in notes:
        start_abs = n['start'] + time_offset
        if is_muted(start_abs, n['duration']):
            continue
        freq = 440.0 * (2.0 ** ((n['pitch'] - 69) / 12.0))
        if freq < 50 or freq > 20000:
            continue
        wave = synth_note_wave(freq, n['duration'], sample_rate,
                               waveform='square', duty=0.5, volume=melody_volume, decay=5.0)
        start_idx = int(start_abs * sample_rate)
        if start_idx < 0:
            start_idx = 0
        end_idx = start_idx + len(wave)
        if end_idx <= len(audio):
            audio[start_idx:end_idx] += wave

    # 2. Bass (triangle wave)
    if enable_bass:
        print("   🎵 合成低音 (三角波) / Synthesizing bass (triangle)...")
        bass_notes = generate_bass_track(notes, bpm)
        for n in bass_notes:
            start_abs = n['start'] + time_offset
            if is_muted(start_abs, n['duration']):
                continue
            freq = 440.0 * (2.0 ** ((n['pitch'] - 69) / 12.0))
            if freq < 50:
                continue
            wave = synth_note_wave(freq, n['duration'], sample_rate,
                                   waveform='triangle', volume=bass_volume, decay=6.0)
            start_idx = int(start_abs * sample_rate)
            if start_idx < 0:
                start_idx = 0
            end_idx = start_idx + len(wave)
            if end_idx <= len(audio):
                audio[start_idx:end_idx] += wave

    # 3. Arpeggio (25% pulse wave)
    if enable_arpeggio:
        print("   🎵 合成副旋律 (脉冲波 25%) / Synthesizing arpeggio (pulse 25%)...")
        arp_notes = generate_arpeggio_track(notes, bpm)
        for n in arp_notes:
            start_abs = n['start'] + time_offset
            if is_muted(start_abs, n['duration']):
                continue
            freq = 440.0 * (2.0 ** ((n['pitch'] - 69) / 12.0))
            if freq < 50:
                continue
            wave = synth_note_wave(freq, n['duration'], sample_rate,
                                   waveform='square', duty=0.25, volume=arpeggio_volume, decay=4.0)
            start_idx = int(start_abs * sample_rate)
            if start_idx < 0:
                start_idx = 0
            end_idx = start_idx + len(wave)
            if end_idx <= len(audio):
                audio[start_idx:end_idx] += wave

    # 4. Drums
    if enable_drums:
        print("   🥁 合成鼓点 (多音色) / Synthesizing drums (multi‑timbre)...")
        drum_notes = generate_drum_track(total_duration, bpm, sample_rate,
                                         pattern=drum_pattern, beat_times=beat_times)
        for n in drum_notes:
            start_abs = n['start'] + time_offset
            dur = n['duration']
            if is_muted(start_abs, dur):
                continue
            wave = synth_drum_note(n['drum_type'], dur, sample_rate,
                                   volume=n['volume'], decay=n.get('decay', 10.0))
            start_idx = int(start_abs * sample_rate)
            if start_idx < 0:
                start_idx = 0
            end_idx = start_idx + len(wave)
            if end_idx <= len(audio):
                audio[start_idx:end_idx] += wave

    # Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 0.01:
        audio = audio / peak * 0.95
    return audio


class MelodyExtractor:
    """
    Main melody extraction class with memorability analysis and RMVPE/pYIN pitch detection.
    """
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.audio = None
        self.sr = None
        self.duration = 0
        self.detected_bpm = 120
        self.rmvpe_model = None

    def load_rmvpe_model(self, model_path="rmvpe.pt"):
        """
        Load the RMVPE model for pitch detection.
        Returns the model instance.
        """
        if self.rmvpe_model is not None:
            return self.rmvpe_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   加载 RMVPE 模型 (设备: {device})... / Loading RMVPE model (device: {device})...")

        # Initialize model architecture (params: 4, 1, (2, 2) are fixed)
        model = RMVPE(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()
        model = model.to(device)

        self.rmvpe_model = model
        print("   ✅ RMVPE 模型加载成功 / RMVPE model loaded successfully")
        return model

    def load_audio(self, audio_path):
        """
        Load audio file, detect tempo and beat positions.
        Returns self for chaining.
        """
        self.audio, self.sr = librosa.load(audio_path, sr=None, mono=True)
        self.duration = len(self.audio) / self.sr
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=self.audio, sr=self.sr, units='time')
            if len(tempo) > 0 and tempo[0] > 0:
                self.detected_bpm = round(tempo[0])
            if len(beat_frames) > 0:
                self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
                onset_env = librosa.onset.onset_strength(y=self.audio, sr=self.sr)
                self.beat_strengths = onset_env[beat_frames] if len(beat_frames) <= len(onset_env) else None
            else:
                self.beat_times = None
                self.beat_strengths = None
        except:
            self.beat_times = None
            self.beat_strengths = None
            pass
        print(f"✅ 加载音频: {os.path.basename(audio_path)} / Audio loaded: {os.path.basename(audio_path)}")
        print(f"   时长: {self.duration:.2f}秒, 采样率: {self.sr}Hz / Duration: {self.duration:.2f}s, Sample rate: {self.sr}Hz")
        print(f"   🥁 自动检测 BPM: {self.detected_bpm} / Auto‑detected BPM: {self.detected_bpm}")
        return self

    def _extract_features(self, y, sr):
        """
        Compute mel‑spectrogram (dB) and chroma features.
        Returns mel_db, chroma.
        """
        hop_len = int(sr * 0.01)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.config['n_mels'], hop_length=hop_len)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=self.config['n_chroma'], hop_length=hop_len)
        return mel_db, chroma

    def _compute_repetition_score(self, segment, full_features, hop_length):
        """
        Compute how repetitive a segment is within the full track.
        Higher score → more repetition.
        """
        seg_len = segment.shape[1]
        full_len = full_features.shape[1]
        if seg_len > full_len:
            return 0.0
        similarities = []
        step = max(1, seg_len // 2)
        for start in range(0, full_len - seg_len + 1, step):
            window = full_features[:, start:start + seg_len]
            dot = np.sum(segment * window)
            norm = np.linalg.norm(segment) * np.linalg.norm(window)
            if norm > 0:
                sim = dot / norm
                if sim < 0.99:  # exclude exact self‑match
                    similarities.append(sim)
        if not similarities:
            return 0.0
        avg_sim = np.mean(similarities)
        threshold = self.config['similarity_threshold']
        return max(0, (avg_sim - threshold) / (1 - threshold))

    def _compute_surprise_score(self, segment_db, full_db):
        """
        Compute a "surprise" score based on energy and spectral centroid deviation.
        """
        seg_energy = np.mean(segment_db)
        full_energy = np.mean(full_db)
        energy_ratio = max(0, (seg_energy - full_energy) / (np.std(full_db) + 1e-6))
        seg_centroid = np.mean(np.sum(segment_db * np.arange(segment_db.shape[0])[:, None], axis=0) /
                              (np.sum(segment_db, axis=0) + 1e-6))
        full_centroid = np.mean(np.sum(full_db * np.arange(full_db.shape[0])[:, None], axis=0) /
                               (np.sum(full_db, axis=0) + 1e-6))
        centroid_ratio = max(0, (seg_centroid - full_centroid) / (full_centroid + 1e-6))
        return min(1.0, 0.6 * energy_ratio + 0.4 * centroid_ratio)

    def _compute_cleanliness_score(self, chroma_segment):
        """
        Measure how "clean" (narrow pitch distribution) a segment is.
        """
        chroma_mean = np.mean(chroma_segment, axis=1)
        chroma_std = np.std(chroma_mean)
        max_std = 0.288
        return min(1.0, chroma_std / max_std)

    def find_memorable_segments(self):
        """
        Analyze audio structure and return the top‑K most memorable segments.
        Memorability score = 0.5*repetition + 0.3*surprise + 0.2*cleanliness.
        """
        print("\n🔍 正在分析音频结构（寻找高记忆片段）... / Analyzing audio structure (finding memorable segments)...")
        mel_db, chroma = self._extract_features(self.audio, self.sr)
        frame_duration = 0.01
        window_frames = int(self.config['window_duration'] / frame_duration)
        hop_frames = int(self.config['hop_duration'] / frame_duration)
        total_frames = mel_db.shape[1]
        segments = []

        for start in range(0, total_frames - window_frames + 1, hop_frames):
            end = start + window_frames
            seg_mel = mel_db[:, start:end]
            seg_chroma = chroma[:, start:end]

            repetition = self._compute_repetition_score(seg_mel, mel_db, int(self.sr * 0.01))
            surprise = self._compute_surprise_score(seg_mel, mel_db)
            cleanliness = self._compute_cleanliness_score(seg_chroma)

            memory_score = 0.5 * repetition + 0.3 * surprise + 0.2 * cleanliness

            segments.append({
                'start': start * frame_duration,
                'end': end * frame_duration,
                'score': memory_score,
                'repetition': repetition,
                'surprise': surprise,
                'cleanliness': cleanliness,
            })

        segments.sort(key=lambda x: x['score'], reverse=True)
        top_segments = segments[:self.config['top_k']]

        print(f"   找到 {len(segments)} 个候选片段 / Found {len(segments)} candidate segments")
        print(f"   🏆 记忆指数 Top {len(top_segments)}: / Memorability Top {len(top_segments)}:")
        for i, seg in enumerate(top_segments, 1):
            print(f"      #{i}: {seg['start']:.2f}s-{seg['end']:.2f}s  score={seg['score']:.3f}")
        return top_segments

    def extract_pitch_from_segment(self, start_time, end_time, thred):
        """
        Extract pitch (time, frequency) from an audio segment using RMVPE, with
        fallback to CREPE and then pYIN.
        Returns (valid_time, valid_freq) or (None, None) on failure.
        """
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)
        segment = self.audio[start_sample:end_sample]

        # ==================== 1. Try RMVPE ====================
        try:
            if not hasattr(self, 'rmvpe_model') or self.rmvpe_model is None:
                from rmvpe import RMVPE
                import torch
                model_path = self.config.get('rmvpe_model_path', 'rmvpe.pt')
                is_half = False  # CPU: False; GPU: can be True if enough VRAM
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"   🔧 加载 RMVPE 模型（{device}）... / Loading RMVPE model ({device})...")
                self.rmvpe_model = RMVPE(model_path, is_half=is_half, device=device)
                print("   ✅ RMVPE 加载完成 / RMVPE loaded")

            # Resample to 16 kHz (required by RMVPE)
            if self.sr != 16000:
                audio_16k = librosa.resample(segment, orig_sr=self.sr, target_sr=16000)
            else:
                audio_16k = segment
            audio_16k = audio_16k.astype(np.float32)

            # Infer pitch, threshold adjustable (default 0.03)
            f0 = self.rmvpe_model.infer_from_audio(audio_16k, thred=thred)
            print(f"   🔍 RMVPE 返回 f0 长度: {len(f0)}，音频帧数应为 {len(audio_16k)//160} / RMVPE f0 length: {len(f0)}, expected audio frames: {len(audio_16k)//160}")
            hop_length = 160  # RMVPE hop length (10 ms @ 16 kHz)
            times = np.arange(len(f0)) * (hop_length / 16000.0) + start_time

            valid_mask = f0 > 0
            if np.any(valid_mask):
                valid_time = times[valid_mask]
                valid_freq = f0[valid_mask]
                print(f"   🔍 有效音高范围: {valid_time[0]:.2f}s ~ {valid_time[-1]:.2f}s (共 {len(valid_time)} 帧) / Valid pitch range: {valid_time[0]:.2f}s ~ {valid_time[-1]:.2f}s ({len(valid_time)} frames)")
                return valid_time, valid_freq
            else:
                print(f"   ⚠️  RMVPE 未检测到有效音高，尝试备选方案... / RMVPE no valid pitch, trying fallback...")

        except Exception as e:
            print(f"   ⚠️  RMVPE 失败 ({e})，尝试备选方案... / RMVPE failed ({e}), trying fallback...")

        # ==================== 2. Fallback: CREPE ====================
        try:
            import crepe
            result = crepe.predict(segment, self.sr, step_size=self.config['pitch_hop_ms'], verbose=0)
            if len(result) >= 3:
                time = np.array(result[0])
                frequency = np.array(result[1])
                confidence = np.array(result[2])
            else:
                raise ValueError("CREPE 返回长度异常 / CREPE returned abnormal length")

            valid_mask = confidence > 0.5
            if np.any(valid_mask):
                valid_time = time[valid_mask] + start_time
                valid_freq = frequency[valid_mask]
                return valid_time, valid_freq
            else:
                print(f"   ⚠️  CREPE 置信度过低，尝试 pYIN... / CREPE confidence too low, trying pYIN...")

        except Exception as e:
            print(f"   ⚠️  CREPE 失败 ({e})，尝试 pYIN... / CREPE failed ({e}), trying pYIN...")

        # ==================== 3. Final fallback: librosa.pyin ====================
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment, fmin=80, fmax=2000, sr=self.sr,
                hop_length=int(self.sr * self.config['pitch_hop_ms'] / 1000)
            )
            time = np.arange(len(f0)) * (self.config['pitch_hop_ms'] / 1000.0)
            frequency = f0.copy()
            confidence = voiced_probs
            frequency[~voiced_flag] = np.nan

            valid_mask = confidence > 0.5
            if np.any(valid_mask):
                valid_time = time[valid_mask] + start_time
                valid_freq = frequency[valid_mask]
                return valid_time, valid_freq
            else:
                print(f"   ❌ 所有后端均未检测到音高 / All backends failed to detect pitch")
                return None, None
        except Exception as e:
            print(f"   ❌ pYIN 也失败 ({e}) / pYIN also failed ({e})")
            return None, None

    def fix_octave_errors(self, notes, window_size=5, threshold=12):
        """
        Correct isolated octave errors: if a note's pitch is more than an octave below the
        local median and the median is in the mid‑high range, raise it by an octave (+12).
        Real bass sections remain unchanged.
        修正孤立的八度错误：如果某个音符的音高比局部中位数低超过一个八度，
        并且局部中位数在中高音区，则提高八度 (+12)。
        不会改动真正的低音段落。
        """
        if len(notes) < window_size:
            return notes

        fixed = notes.copy()
        changed = True
        max_iter = 3
        while changed and max_iter > 0:
            changed = False
            max_iter -= 1
            for i in range(len(fixed)):
                start = max(0, i - window_size // 2)
                end = min(len(fixed), i + window_size // 2 + 1)
                neighbors = [fixed[j]['pitch'] for j in range(start, end) if j != i]
                if not neighbors:
                    continue
                median_pitch = np.median(neighbors)
                if median_pitch > 48 and median_pitch - fixed[i]['pitch'] >= threshold:
                    fixed[i]['pitch'] += 12
                    changed = True
                    print(f"   🔧 八度修正: {fixed[i]['start']:.2f}s 处 {fixed[i]['pitch']-12} → {fixed[i]['pitch']} / Octave fix: at {fixed[i]['start']:.2f}s {fixed[i]['pitch']-12} → {fixed[i]['pitch']}")
        return fixed

    def freq_to_midi(self, freq):
        """Convert frequency in Hz to MIDI note number (float). Returns None for invalid freq."""
        if freq <= 0 or np.isnan(freq):
            return None
        return 69 + 12 * np.log2(freq / 440.0)

    def quantize_pitch_sequence(self, times, freqs, bpm):
        """
        Convert a raw (time, frequency) stream into a list of note dicts with quantized pitches.
        Applies median/savgol smoothing, removes short fragments, and merges tiny notes.
        """
        print(f"   🛠️ 量化输入：times 长度={len(times)}, 范围={times[0]:.2f}s ~ {times[-1]:.2f}s / Quantizing input: times len={len(times)}, range={times[0]:.2f}s ~ {times[-1]:.2f}s")
        if times is None or len(times) == 0:
            return []

        midi_notes = np.array([self.freq_to_midi(f) for f in freqs])
        valid_mask = ~np.isnan(midi_notes)
        if not np.any(valid_mask):
            return []

        times = times[valid_mask]
        midi_notes = midi_notes[valid_mask]

        # Optional median smoothing
        if len(times) > 3:
            filter_size = max(3, int(self.config['median_filter_duration'] / (times[1] - times[0]) if len(times) > 1 else 3))
            if filter_size % 2 == 0:
                filter_size += 1
            if filter_size > 1 and len(midi_notes) > filter_size:
                midi_notes = medfilt(midi_notes, filter_size)
        if len(midi_notes) > 5:
            midi_notes = savgol_filter(midi_notes, window_length=5, polyorder=2)
        midi_notes_rounded = np.round(midi_notes).astype(int)
        min_note_duration = 60.0 / (bpm * 8) * self.config.get('stability_factor', 2.0)

        notes = []
        current_pitch = midi_notes_rounded[0]
        current_start = times[0]

        # Segment based on pitch changes
        for i in range(1, len(times)):
            pitch = midi_notes_rounded[i]
            if pitch != current_pitch:
                duration = times[i] - current_start
                if duration >= min_note_duration:
                    notes.append({'pitch': current_pitch, 'start': current_start, 'duration': duration})
                current_pitch = pitch
                current_start = times[i]

        # Last note
        duration = times[-1] - current_start
        if duration >= min_note_duration:
            notes.append({'pitch': current_pitch, 'start': current_start, 'duration': duration})

        # Merge very short notes with neighbors
        if len(notes) > 1:
            merged = []
            i = 0
            while i < len(notes):
                if notes[i]['duration'] < min_note_duration:
                    if merged and i+1 < len(notes):
                        next_note = notes[i+1]
                        if next_note['duration'] >= notes[i]['duration']:
                            merged[-1]['duration'] += notes[i]['duration']
                            i += 1  # skip current, proceed to next
                        else:
                            merged[-1]['pitch'] = notes[i]['pitch']
                            merged[-1]['duration'] += notes[i]['duration']
                            i += 1
                    elif merged:
                        merged[-1]['duration'] += notes[i]['duration']
                        i += 1
                    else:
                        notes[i]['duration'] = min_note_duration
                        merged.append(notes[i])
                        i += 1
                else:
                    merged.append(notes[i])
                    i += 1
            notes = merged

        print(f"   📊 量化后生成 {len(notes)} 个音符 / Quantized to {len(notes)} notes")
        if len(notes) > 0:
            first_note = notes[0]
            last_note = notes[-1]
            mid_note = notes[len(notes)//2]
            print(f"   📊 音符分布: 首音符 start={first_note['start']:.2f}s, "
                  f"中位音符 start={mid_note['start']:.2f}s, "
                  f"末音符 start={last_note['start']:.2f}s, 末音符 duration={last_note['duration']:.2f}s / "
                  f"Note distribution: first start={first_note['start']:.2f}s, mid start={mid_note['start']:.2f}s, "
                  f"last start={last_note['start']:.2f}s, last duration={last_note['duration']:.2f}s")
            print(f"   📊 总时长 = {last_note['start'] + last_note['duration'] - first_note['start']:.2f}s / "
                  f"Total duration = {last_note['start'] + last_note['duration'] - first_note['start']:.2f}s")
        notes = self.fix_octave_errors(notes)
        return notes

    def save_midi(self, notes, output_path, bpm):
        """
        Save a list of note dicts to a MIDI file with the given tempo.
        Returns True on success.
        """
        if not notes:
            print("❌ 没有音符可保存 / No notes to save")
            return False

        midi = MIDIFile(1)
        track, channel = 0, 0
        midi.addTempo(track, 0, bpm)

        spb = 60.0 / bpm          # seconds per beat

        # Shift so that the first note starts at beat 0
        time_shift_seconds = notes[0]['start'] if notes else 0
        time_shift_beats = time_shift_seconds / spb

        for note in notes:
            pitch = max(0, min(127, int(note['pitch'])))
            start_beats = note['start'] / spb - time_shift_beats
            duration_beats = note['duration'] / spb
            midi.addNote(track, channel, pitch, start_beats, duration_beats, 100)

        with open(output_path, 'wb') as f:
            midi.writeFile(f)

        print(f"✅ MIDI 已保存: {output_path} / MIDI saved: {output_path}")
        print(f"   包含 {len(notes)} 个音符, BPM={bpm} / Contains {len(notes)} notes, BPM={bpm}")
        return True

    def extract_and_save(self, audio_path, output_midi_path, thred, bpm=None, full_mode=False, render_wav=True, mix=False):
        """
        Main pipeline: load audio, extract melody (full or memorable segments),
        quantize, save MIDI, optionally render to WAV (simple sine or 8‑bit multi‑track),
        and optionally mix with original.
        """
        self.load_audio(audio_path)
        if bpm is None:
            bpm = self.detected_bpm

        all_notes = []

        if full_mode:
            print("\n🎵 模式: 提取全曲主旋律 (Full) / Mode: full‑song extraction")
            times, freqs = self.extract_pitch_from_segment(0, self.duration, thred)
            if times is not None:
                notes = self.quantize_pitch_sequence(times, freqs, bpm)
                all_notes.extend(notes)
                print(f"   ✅ 全曲提取完成，共 {len(notes)} 个音符 / Full extraction done, {len(notes)} notes")
        else:
            print("\n🎵 模式: 仅提取高记忆片段 (Top K) / Mode: memorable segments only")
            top_segments = self.find_memorable_segments()
            for i, seg in enumerate(top_segments, 1):
                print(f"\n   提取片段 #{i}: {seg['start']:.2f}s - {seg['end']:.2f}s / Extracting segment #{i}: {seg['start']:.2f}s - {seg['end']:.2f}s")
                times, freqs = self.extract_pitch_from_segment(seg['start'], seg['end'], thred)
                if times is not None:
                    notes = self.quantize_pitch_sequence(times, freqs, bpm)
                    all_notes.extend(notes)
                    print(f"      提取到 {len(notes)} 个音符 / Extracted {len(notes)} notes")

        if not all_notes:
            print("❌ 未提取到任何音符 / No notes extracted")
            return False

        all_notes.sort(key=lambda x: x['start'])
        success = self.save_midi(all_notes, output_midi_path, bpm)
        if success and render_wav:
            wav_path = os.path.splitext(output_midi_path)[0] + ".wav"
            try:
                if self.config.get('enable_8bit', False):
                    time_offset = self.config.get('time_offset', 0.0)
                    melody_vol = self.config.get('melody_volume', 0.4)
                    bass_vol = self.config.get('bass_volume', 0.3)
                    arp_vol = self.config.get('arpeggio_volume', 0.25)
                    drum_vol = self.config.get('drum_volume', 0.5)
                    mute_intervals = self.config.get('mute_intervals', [])
                    drum_pattern = self.config.get('drum_pattern', 'simple')
                    beat_times = getattr(self, 'beat_times', None)  # use detected beats if available
                    audio = synth_multi_track(all_notes, bpm, self.sr,
                                              time_offset=time_offset,
                                              melody_volume=melody_vol,
                                              bass_volume=bass_vol,
                                              arpeggio_volume=arp_vol,
                                              drum_volume=drum_vol,
                                              mute_intervals=mute_intervals,
                                              drum_pattern=drum_pattern,
                                              beat_times=beat_times)
                    sf.write(wav_path, audio, self.sr)
                    print(f"✅ 8-bit 多轨 WAV 已保存: {wav_path} / 8‑bit multi‑track WAV saved: {wav_path}")
                else:
                    simple_synth_midi_to_wav(all_notes, wav_path)
            except Exception as e:
                print(f"⚠️  渲染失败 ({e}) / Render failed ({e})")
                return success
        if mix:
            mix_path = os.path.splitext(output_midi_path)[0] + "_mix.wav"
            try:
                mix_melody_with_original(audio_path, wav_path, mix_path)
            except Exception as e:
                print(f"⚠️ 混音失败: {e} / Mixing failed: {e}")
        return success


def main():
    """
    Command‑line interface: parse arguments, build configuration, run extraction.
    """
    import argparse
    parser = argparse.ArgumentParser(description='旋律提取器 - 动态BPM音符阈值 + 延迟确认 / Melody Extractor with dynamic BPM threshold and delayed confirmation')
    parser.add_argument('input', help='输入音频文件 / Input audio file')
    parser.add_argument('-o', '--output', help='输出 MIDI 路径 / Output MIDI path', default=None)
    parser.add_argument('--thred', type=float, default=0.001, help='RMVPE threshold (default 0.001)')
    parser.add_argument('--bpm', type=int, help='手动指定 BPM（默认自动检测） / Manually specify BPM')
    parser.add_argument('--top-k', type=int, default=3, help='记忆片段数量 (default 3) / Number of top memorable segments')
    parser.add_argument('--window', type=float, default=4, help='分析窗口秒数 (default 4.0) / Analysis window in seconds')
    parser.add_argument('--full', action='store_true', help='提取全曲主旋律（默认仅提取高记忆片段） / Extract full song melody')
    parser.add_argument('--stability', type=float, default=2, help='稳定倍数 (default 2, 即十六分音符) / Stability factor')
    parser.add_argument('--render-wav', action='store_true', help='同时合成 WAV 音频（方波） / Render WAV audio')
    parser.add_argument('--mix', action='store_true', help='生成旋律与原曲叠加的混音 WAV / Mix with original')
    parser.add_argument('--multitrack', action='store_true', help='启用8-bit多轨合成（包含低音、琶音、鼓） / Enable 8‑bit multi‑track synthesis')
    parser.add_argument('--time-offset', type=float, default=0, help='整体时间偏移（秒），正数后移，负数前移 / Time offset in seconds')
    parser.add_argument('--melody-vol', type=float, default=0.6, help='主旋律音量 (default 0.6) / Melody volume')
    parser.add_argument('--bass-vol', type=float, default=1.0, help='低音音量 (default 1.0) / Bass volume')
    parser.add_argument('--arpeggio-vol', type=float, default=0.3, help='琶音音量 (default 0.3) / Arpeggio volume')
    parser.add_argument('--drum-vol', type=float, default=1.0, help='鼓点音量 (default 1.0) / Drum volume')
    parser.add_argument('--mute-intervals', type=str, default='',
                        help='静音时间段，格式如 "0,5;10,15" 表示0-5秒和10-15秒静音 / Mute intervals "start,end;start,end"')
    parser.add_argument('--drum-pattern', type=str, default='simple',
                        choices=['simple', 'rock', 'pop', 'funk'], help='鼓点节奏型 / Drum pattern')
    parser.add_argument('--config', type=str, help='从 JSON 文件加载配置（命令行参数会覆盖 JSON） / Load config from JSON')
    parser.add_argument('--save-config', type=str, help='将当前配置保存到 JSON 文件 / Save config to JSON')
    args = parser.parse_args()

    # Parse mute intervals
    mute_str = args.mute_intervals
    mute_intervals = []
    if mute_str:
        for part in mute_str.split(';'):
            part = part.strip()
            if not part:
                continue
            try:
                start, end = part.split(',')
                start = float(start.strip())
                end = float(end.strip())
                if start < end:
                    mute_intervals.append((start, end))
            except:
                print(f"⚠️  忽略无效的静音区间: {part} / Ignoring invalid mute interval: {part}")

    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input} / File not found: {args.input}")
        sys.exit(1)

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_melody.mid"

    config = CONFIG.copy()

    # Load JSON config if provided, then override with command‑line args
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            json_config = json.load(f)
            config.update(json_config)
        print(f"✅ 从 JSON 加载配置: {args.config} / Config loaded from JSON: {args.config}")

    if args.bpm is not None:
        config['bpm'] = args.bpm  # will be used in extract_and_save

    config['top_k'] = args.top_k
    config['window_duration'] = args.window
    config['stability_factor'] = args.stability
    config['enable_8bit'] = args.multitrack
    config['time_offset'] = args.time_offset
    config['melody_volume'] = args.melody_vol
    config['bass_volume'] = args.bass_vol
    config['arpeggio_volume'] = args.arpeggio_vol
    config['drum_volume'] = args.drum_vol
    config['mute_intervals'] = mute_intervals
    config['drum_pattern'] = args.drum_pattern

    # Save config if requested
    if args.save_config:
        with open(args.save_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ 配置已保存到: {args.save_config} / Config saved to: {args.save_config}")

    extractor = MelodyExtractor(config)

    success = extractor.extract_and_save(
        args.input, args.output,
        thred=args.thred,
        bpm=args.bpm,
        full_mode=args.full,
        render_wav=args.render_wav,
        mix=args.mix
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
