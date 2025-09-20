# -*- coding: utf-8 -*-
"""Complete Audio Post-Processor with Full Features - Streamlit Version"""

import os
import tempfile
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_silence
import pyloudnorm as pyln
import noisereduce as nr
from scipy import signal

from demucs.pretrained import get_model
from demucs.apply import apply_model

# ==========================
# Audio Processor Class
# ==========================
class CompleteAudioProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.separation_model = None

    # ---------- Voice Isolation ----------
    def load_separation_model(self):
        if self.separation_model is None:
            self.separation_model = get_model('htdemucs')
            self.separation_model.to(self.device)
        return self.separation_model

    def isolate_voice_demucs(self, file_path, strength=0.8):
        model = self.load_separation_model()
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        if sr != 44100:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
            sr = 44100
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        with torch.no_grad():
            sources = apply_model(model, audio_tensor.unsqueeze(0), device=self.device)
        vocals = sources[0, 0].cpu().numpy()
        if strength < 1.0:
            min_len = min(len(vocals), len(audio))
            vocals = vocals[:min_len] * strength + audio[:min_len] * (1 - strength)
        return vocals, sr

    # ---------- Noise Reduction ----------
    def reduce_background(self, audio, sr, reduction_amount):
        return nr.reduce_noise(y=audio, sr=sr, prop_decrease=reduction_amount, n_fft=2048,
                               win_length=2048, hop_length=512, n_std_thresh_stationary=1.5, stationary=True)

    def apply_noise_reduction(self, audio, sr, reduction_amount):
        noise_sample = audio[:int(0.5*sr)]
        return nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample, prop_decrease=reduction_amount)

    # ---------- Loudness Normalization ----------
    def apply_loudness_normalization(self, audio, sr, target_lufs):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
        return normalized

    # ---------- EQ ----------
    def apply_eq(self, audio, sr, low_cut, high_cut, bass_boost, treble_boost):
        if low_cut > 20:
            sos_low = signal.butter(4, low_cut, 'highpass', fs=sr, output='sos')
            audio = signal.sosfilt(sos_low, audio)
        if high_cut < 20000:
            sos_high = signal.butter(4, high_cut, 'lowpass', fs=sr, output='sos')
            audio = signal.sosfilt(sos_high, audio)
        if bass_boost > 0:
            sos_bass = signal.butter(2, 250, 'lowpass', fs=sr, output='sos')
            bass = signal.sosfilt(sos_bass, audio)
            audio = audio + bass * bass_boost * 0.3
        if treble_boost > 0:
            sos_treble = signal.butter(2, 4000, 'highpass', fs=sr, output='sos')
            treble = signal.sosfilt(sos_treble, audio)
            audio = audio + treble * treble_boost * 0.2
        return audio

    # ---------- Compression ----------
    def apply_compression(self, audio, sr, ratio, threshold, attack, release):
        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, audio, sr)
        segment = AudioSegment.from_wav(temp_path)
        compressed = segment.compress_dynamic_range(threshold=threshold, ratio=ratio,
                                                    attack=attack, release=release)
        compressed.export(temp_path, format="wav")
        compressed_audio, _ = sf.read(temp_path)
        return compressed_audio

    # ---------- De-essing & Vocal Clarity ----------
    def apply_deessing(self, audio, sr, deess_amount):
        sos_high = signal.butter(4, 4000, 'highpass', fs=sr, output='sos')
        sibilance = signal.sosfilt(sos_high, audio)
        return audio - sibilance * deess_amount * 0.5

    def enhance_vocal_clarity(self, audio, sr, clarity_amount):
        freqs = [250, 500, 1000, 2000, 4000]
        for freq in freqs:
            sos = signal.butter(2, [freq*0.8, freq*1.2], 'bandpass', fs=sr, output='sos')
            band = signal.sosfilt(sos, audio)
            audio = audio + band * clarity_amount * 0.1
        return audio

    # ---------- Silence Trimming ----------
    def trim_silence(self, audio, sr, threshold_db, fade_duration=0):
        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, audio, sr)
        segment = AudioSegment.from_wav(temp_path)
        silence_ranges = detect_silence(segment, min_silence_len=100, silence_thresh=threshold_db)
        if silence_ranges:
            start_silence = silence_ranges[0][1] if silence_ranges[0][0] == 0 else 0
            end_silence = silence_ranges[-1][0] if silence_ranges[-1][1] == len(segment) else len(segment)
            segment = segment[start_silence:end_silence]
        if fade_duration > 0:
            segment = segment.fade_in(fade_duration).fade_out(fade_duration)
        segment.export(temp_path, format="wav")
        trimmed_audio, _ = sf.read(temp_path)
        return trimmed_audio

# ==========================
# Streamlit UI
# ==========================
st.title("Complete Audio Post-Processor")

uploaded_files = st.file_uploader("Upload audio files", type=['wav','mp3','flac'], accept_multiple_files=True)

# Processing settings
processing_mode = st.radio("Processing Mode", ['Full Mix Processing', 'Voice Isolation Only', 'Music Enhancement'])
isolation_enable = st.checkbox("Enable Voice Isolation")
isolation_strength = st.slider("Isolation Strength", 0.1, 1.0, 0.8)
background_reduction = st.slider("Background Reduction", 0.0, 1.0, 0.7)
noise_enable = st.checkbox("Enable Noise Reduction", value=True)
noise_amount = st.slider("Noise Reduction Amount", 0.1, 1.0, 0.6)
loudness_enable = st.checkbox("Enable Loudness Normalization", value=True)
loudness_target = st.slider("Target LUFS", -30.0, -10.0, -16.0)
eq_enable = st.checkbox("Enable EQ", value=True)
low_cut = st.slider("Low-cut Frequency (Hz)", 20, 500, 100)
high_cut = st.slider("High-cut Frequency (Hz)", 5000, 20000, 15000)
bass_boost = st.slider("Bass Boost Amount", 0.0, 1.0, 0.0)
treble_boost = st.slider("Treble Boost Amount", 0.0, 1.0, 0.0)
comp_enable = st.checkbox("Enable Compression")
comp_ratio = st.slider("Compression Ratio", 1.0, 20.0, 4.0)
comp_threshold = st.slider("Threshold (dB)", -60.0, 0.0, -20.0)
comp_attack = st.slider("Attack (ms)", 1.0, 100.0, 5.0)
comp_release = st.slider("Release (ms)", 10.0, 500.0, 50.0)
deess_enable = st.checkbox("Enable De-essing")
deess_amount = st.slider("De-ess Amount", 0.1, 1.0, 0.5)
vocal_clarity = st.slider("Vocal Clarity Boost", 0.0, 1.0, 0.5)
trim_enable = st.checkbox("Enable Silence Trimming", value=True)
trim_threshold = st.slider("Silence Threshold (dB)", -60, -20, -40)
fade_enable = st.checkbox("Enable Fade In/Out", value=True)
fade_duration = st.slider("Fade Duration (ms)", 10, 500, 100)
normalize_peak = st.checkbox("Normalize Peak to -1 dBFS", value=True)
export_format = st.selectbox("Export Format", ['wav','mp3','flac'])
sample_rate = st.selectbox("Sample Rate", [16000, 22050, 44100, 48000, 96000])

processor = CompleteAudioProcessor()

# ---------- Run Processing ----------
if st.button("Run Processing"):
    if not uploaded_files:
        st.warning("Please upload files!")
    else:
        processed_files = []
        report_data = []
        for uploaded_file in uploaded_files:
            st.write(f"Processing {uploaded_file.name} ...")
            temp_input_path = tempfile.mktemp(suffix=os.path.splitext(uploaded_file.name)[1])
            with open(temp_input_path, 'wb') as f:
                f.write(uploaded_file.read())
            audio, sr = librosa.load(temp_input_path, sr=None, mono=False)
            # Stereo to mono
            if audio.ndim == 2:
                audio = np.mean(audio, axis=0)
            # Voice isolation
            if isolation_enable and processing_mode != 'Music Enhancement':
                try:
                    audio, sr = processor.isolate_voice_demucs(temp_input_path, isolation_strength)
                    if background_reduction > 0:
                        audio = processor.reduce_background(audio, sr, background_reduction)
                except Exception as e:
                    st.error(f"Voice isolation failed: {e}")
            # Noise reduction
            if noise_enable:
                audio = processor.apply_noise_reduction(audio, sr, noise_amount)
            # Loudness normalization
            if loudness_enable:
                audio = processor.apply_loudness_normalization(audio, sr, loudness_target)
            # EQ
            if eq_enable:
                audio = processor.apply_eq(audio, sr, low_cut, high_cut, bass_boost, treble_boost)
            # Compression
            if comp_enable:
                audio = processor.apply_compression(audio, sr, comp_ratio, comp_threshold, comp_attack, comp_release)
            # De-essing
            if deess_enable:
                audio = processor.apply_deessing(audio, sr, deess_amount)
            # Vocal clarity
            audio = processor.enhance_vocal_clarity(audio, sr, vocal_clarity)
            # Silence trimming
            if trim_enable:
                fade = fade_duration if fade_enable else 0
                audio = processor.trim_silence(audio, sr, trim_threshold, fade)
            # Peak normalization
            if normalize_peak:
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak * 0.99
            # Save processed file
            output_path = tempfile.mktemp(suffix=f".{export_format}")
            sf.write(output_path, audio, sample_rate)
            processed_files.append((uploaded_file.name, output_path))
            report_data.append({'filename': uploaded_file.name, 'duration_sec': len(audio)/sr})
            st.success(f"{uploaded_file.name} processed successfully!")

        # Download ZIP
        if processed_files:
            zip_path = tempfile.mktemp(suffix=".zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for orig_name, file_path in processed_files:
                    zipf.write(file_path, arcname=f"processed_{orig_name}")
            with open(zip_path, 'rb') as f:
                st.download_button("Download Processed Files (ZIP)", f, file_name="processed_audio.zip")
            # CSV Report
            df = pd.DataFrame(report_data)
            csv_path = tempfile.mktemp(suffix=".csv")
            df.to_csv(csv_path, index=False)
            with open(csv_path, 'rb') as f:
                st.download_button("Download Processing Report (CSV)", f, file_name="audio_processing_report.csv")
