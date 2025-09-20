# -*- coding: utf-8 -*-
"""Complete Audio Post-Processor with Voice Isolation"""

# Install required packages
!pip install pydub librosa soundfile scipy numpy ipywidgets gradio noisereduce pyloudnorm demucs torchaudio
!pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118  # For GPU support

import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from google.colab import files

# Audio processing libraries
import librosa
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment, effects
from pydub.silence import detect_silence
import pyloudnorm as pyln
import noisereduce as nr
from scipy import signal

# Demucs for voice separation
try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
except:
    !pip install demucs
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

class CompleteAudioProcessor:
    def __init__(self):
        self.uploaded_files = []
        self.processed_files = []
        self.report_data = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.separation_model = None

    def create_ui(self):
        """Create the user interface with controls"""

        # Processing Mode
        self.processing_mode = widgets.RadioButtons(
            options=['Full Mix Processing', 'Voice Isolation Only', 'Music Enhancement'],
            value='Full Mix Processing',
            description='Processing Mode:'
        )

        # Voice Isolation Settings
        self.isolation_enable = widgets.Checkbox(value=False, description='Enable Voice Isolation')
        self.isolation_strength = widgets.FloatSlider(value=0.8, min=0.1, max=1.0, step=0.1,
                                                    description='Isolation Strength')
        self.background_reduction = widgets.FloatSlider(value=0.7, min=0.0, max=1.0, step=0.1,
                                                          description='Background Reduction')

        self.isolation_section = widgets.VBox([
            widgets.HTML("<h4>Voice Isolation Settings</h4>"),
            self.isolation_enable,
            self.isolation_strength,
            self.background_reduction
        ])

        # Loudness Normalization
        self.loudness_enable = widgets.Checkbox(value=True, description='Enable Loudness Normalization')
        self.loudness_target = widgets.FloatSlider(value=-16, min=-30, max=-10, step=0.1,
                                                     description='Target LUFS')

        self.loudness_section = widgets.VBox([
            widgets.HTML("<h4>Loudness Settings</h4>"),
            self.loudness_enable,
            self.loudness_target
        ])

        # Noise Reduction
        self.noise_enable = widgets.Checkbox(value=True, description='Enable Noise Reduction')
        self.noise_amount = widgets.FloatSlider(value=0.6, min=0.1, max=1.0, step=0.1,
                                              description='Noise Reduction Amount')

        self.noise_section = widgets.VBox([
            widgets.HTML("<h4>Noise Reduction</h4>"),
            self.noise_enable,
            self.noise_amount
        ])

        # EQ Settings
        self.eq_enable = widgets.Checkbox(value=True, description='Enable EQ')
        self.low_cut = widgets.IntSlider(value=100, min=20, max=500, step=10,
                                           description='Low-cut Frequency (Hz)')
        self.high_cut = widgets.IntSlider(value=15000, min=5000, max=20000, step=100,
                                            description='High-cut Frequency (Hz)')
        self.bass_boost = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.1,
                                                description='Bass Boost Amount')
        self.treble_boost = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.1,
                                                  description='Treble Boost Amount')

        self.eq_section = widgets.VBox([
            widgets.HTML("<h4>Equalization</h4>"),
            self.eq_enable,
            self.low_cut,
            self.high_cut,
            self.bass_boost,
            self.treble_boost
        ])

        # Compression
        self.comp_enable = widgets.Checkbox(value=False, description='Enable Compression')
        self.comp_ratio = widgets.FloatSlider(value=4.0, min=1.0, max=20.0, step=0.5,
                                                description='Compression Ratio')
        self.comp_threshold = widgets.FloatSlider(value=-20.0, min=-60.0, max=0.0, step=1.0,
                                                    description='Threshold (dB)')
        self.comp_attack = widgets.FloatSlider(value=5.0, min=1.0, max=100.0, step=1.0,
                                                 description='Attack (ms)')
        self.comp_release = widgets.FloatSlider(value=50.0, min=10.0, max=500.0, step=10.0,
                                                  description='Release (ms)')

        self.compression_section = widgets.VBox([
            widgets.HTML("<h4>Compression</h4>"),
            self.comp_enable,
            self.comp_ratio,
            self.comp_threshold,
            self.comp_attack,
            self.comp_release
        ])

        # Voice-specific processing
        self.deess_enable = widgets.Checkbox(value=False, description='Enable De-essing')
        self.deess_amount = widgets.FloatSlider(value=0.5, min=0.1, max=1.0, step=0.1,
                                                  description='De-ess Amount')
        self.vocal_clarity = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.1,
                                                   description='Vocal Clarity Boost')

        self.voice_section = widgets.VBox([
            widgets.HTML("<h4>Voice Enhancement</h4>"),
            self.deess_enable,
            self.deess_amount,
            self.vocal_clarity
        ])

        # Silence Trimming
        self.trim_enable = widgets.Checkbox(value=True, description='Enable Silence Trimming')
        self.trim_threshold = widgets.FloatSlider(value=-40.0, min=-60.0, max=-20.0, step=1.0,
                                                    description='Silence Threshold (dB)')
        self.fade_enable = widgets.Checkbox(value=True, description='Enable Fade In/Out')
        self.fade_duration = widgets.IntSlider(value=100, min=10, max=500, step=10,
                                                 description='Fade Duration (ms)')

        self.trimming_section = widgets.VBox([
            widgets.HTML("<h4>Silence Processing</h4>"),
            self.trim_enable,
            self.trim_threshold,
            self.fade_enable,
            self.fade_duration
        ])


        # Export Options
        self.format_dropdown = widgets.Dropdown(
            options=['WAV', 'MP3', 'FLAC'],
            value='WAV',
            description='Export Format:'
        )

        self.sample_rate_dropdown = widgets.Dropdown(
            options=[16000, 22050, 44100, 48000, 96000],
            value=44100,
            description='Sample Rate:'
        )

        self.bit_depth_dropdown = widgets.Dropdown(
            options=['16-bit', '24-bit', '32-bit'],
            value='16-bit',
            description='Bit Depth:'
        )
        self.normalize_peak = widgets.Checkbox(value=True, description='Normalize to -1.0 dBFS')

        self.export_section = widgets.VBox([
            widgets.HTML("<h4>Export Settings</h4>"),
            self.format_dropdown,
            self.sample_rate_dropdown,
            self.bit_depth_dropdown,
            self.normalize_peak
        ])


        # Buttons
        self.upload_btn = widgets.Button(description="Upload Files", button_style='primary')
        self.process_btn = widgets.Button(description="Run Processing", button_style='success')
        self.download_btn = widgets.Button(description="Download Processed Files (ZIP)")
        self.report_btn = widgets.Button(description="Download Report (CSV)")

        # Output area
        self.output = widgets.Output()

        # Set button callbacks
        self.upload_btn.on_click(self.upload_files)
        self.process_btn.on_click(self.run_processing)
        self.download_btn.on_click(self.download_files)
        self.report_btn.on_click(self.download_report)

        # Create tabs for better organization
        tab_contents = [
            self.isolation_section,
            self.loudness_section,
            self.noise_section,
            self.eq_section,
            self.compression_section,
            self.voice_section,
            self.trimming_section,
            self.export_section
        ]

        tab_titles = ['Isolation', 'Loudness', 'Noise', 'EQ', 'Compression', 'Voice', 'Trimming', 'Export']
        self.tabs = widgets.Tab(children=tab_contents)
        for i, title in enumerate(tab_titles):
            self.tabs.set_title(i, title)

        # Layout
        controls_box = widgets.VBox([
            widgets.HTML("<h2>Complete Audio Post-Processor</h2>"),
            self.processing_mode,
            self.tabs,
            widgets.HTML("<br>"),
            widgets.HBox([self.upload_btn, self.process_btn, self.download_btn, self.report_btn])
        ])

        display(controls_box)
        display(self.output)

    def upload_files(self, b):
        """Handle file upload"""
        self.output.clear_output()
        with self.output:
            uploaded = files.upload()
            self.uploaded_files = list(uploaded.keys())
            print(f"Uploaded {len(self.uploaded_files)} files: {self.uploaded_files}")

    def load_separation_model(self):
        """Load the voice separation model"""
        if self.separation_model is None:
            print("Loading voice separation model (this may take a minute)...")
            self.separation_model = get_model('htdemucs')
            self.separation_model.to(self.device)
        return self.separation_model

    def isolate_voice_demucs(self, file_path, strength=0.8):
        """Isolate voice using Demucs model"""
        model = self.load_separation_model()

        # Load audio with original sample rate and channels
        audio, sr = librosa.load(file_path, sr=None, mono=False)

        # Convert to mono if stereo for processing
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)  # Convert to mono by averaging channels
        else:
            audio_mono = audio

        # Resample to 44100 Hz if needed (Demucs expects this sample rate)
        if sr != 44100:
            audio_mono = librosa.resample(audio_mono, orig_sr=sr, target_sr=44100)
            sr = 44100

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_mono).float().to(self.device)

        # Apply separation model
        with torch.no_grad():
            sources = apply_model(model, audio_tensor.unsqueeze(0), device=self.device)

        # Extract vocals (typically the 0th source in htdemucs)
        vocals = sources[0, 0].cpu().numpy()  # First batch, vocals source

        # Mix with original based on strength
        if strength < 1.0:
            # Ensure same length
            min_len = min(len(vocals), len(audio_mono))
            vocals = vocals[:min_len] * strength + audio_mono[:min_len] * (1 - strength)

        return vocals, sr

    def reduce_background(self, audio, sr, reduction_amount):
        """Reduce background noise while preserving voice"""
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=reduction_amount,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_std_thresh_stationary=1.5,
            stationary=True
        )
        return reduced

    def apply_deessing(self, audio, sr, deess_amount):
        """Apply de-essing to reduce sibilance"""
        # Simple de-esser using multiband compression concept
        sos_high = signal.butter(4, 4000, 'highpass', fs=sr, output='sos')
        sibilance = signal.sosfilt(sos_high, audio)

        # Reduce sibilance
        deessed = audio - (sibilance * deess_amount * 0.5)
        return deessed

    def apply_loudness_normalization(self, audio, sr, target_lufs):
        """Normalize audio to target LUFS"""
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
        return normalized, loudness

    def apply_noise_reduction(self, audio, sr, reduction_amount):
        """Apply noise reduction using noisereduce"""
        # Estimate noise from first 0.5 seconds
        noise_sample = audio[:int(0.5 * sr)]
        reduced = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample, prop_decrease=reduction_amount)
        return reduced

    def apply_eq(self, audio, sr, low_cut, high_cut, bass_boost, treble_boost):
        """Apply comprehensive EQ"""
        # Low cut filter to remove rumble
        if low_cut > 20:
            sos_low = signal.butter(4, low_cut, 'highpass', fs=sr, output='sos')
            audio = signal.sosfilt(sos_low, audio)

        # High cut filter to remove hiss
        if high_cut < 20000:
            sos_high = signal.butter(4, high_cut, 'lowpass', fs=sr, output='sos')
            audio = signal.sosfilt(sos_high, audio)

        # Bass boost (low shelf)
        if bass_boost > 0:
            sos_bass = signal.butter(2, 250, 'lowpass', fs=sr, output='sos')
            bass = signal.sosfilt(sos_bass, audio)
            audio = audio + (bass * bass_boost * 0.3)

        # Treble boost (high shelf)
        if treble_boost > 0:
            sos_treble = signal.butter(2, 4000, 'highpass', fs=sr, output='sos')
            treble = signal.sosfilt(sos_treble, audio)
            audio = audio + (treble * treble_boost * 0.2)

        return audio

    def apply_compression(self, audio, sr, ratio, threshold, attack, release):
        """Apply compression using pydub"""
        # Convert to pydub format
        temp_path = "/tmp/temp_audio.wav"
        sf.write(temp_path, audio, sr)
        audio_segment = AudioSegment.from_wav(temp_path)

        # Apply compression
        compressed = audio_segment.compress_dynamic_range(
            threshold=threshold,
            ratio=ratio,
            attack=attack,
            release=release
        )

        # Convert back to numpy
        compressed.export(temp_path, format="wav")
        compressed_audio, _ = sf.read(temp_path)
        os.remove(temp_path)

        return compressed_audio

    def trim_silence(self, audio, sr, threshold_db, fade_duration=0):
        """Trim silence from beginning and end with optional fade"""
        # Convert to pydub format
        temp_path = "/tmp/temp_audio.wav"
        sf.write(temp_path, audio, sr)
        audio_segment = AudioSegment.from_wav(temp_path)

        # Detect silence
        silence_thresh = threshold_db
        silence_ranges = detect_silence(audio_segment, min_silence_len=100, silence_thresh=silence_thresh)

        if silence_ranges:
            start_silence = silence_ranges[0][1] if silence_ranges[0][0] == 0 else 0
            end_silence = silence_ranges[-1][0] if silence_ranges[-1][1] == len(audio_segment) else len(audio_segment)
            trimmed = audio_segment[start_silence:end_silence]
        else:
            trimmed = audio_segment

        # Apply fade in/out if enabled
        if fade_duration > 0:
            trimmed = trimmed.fade_in(fade_duration).fade_out(fade_duration)

        # Convert back to numpy
        trimmed.export(temp_path, format="wav")
        trimmed_audio, _ = sf.read(temp_path)
        os.remove(temp_path)

        return trimmed_audio

    def enhance_vocal_clarity(self, audio, sr, clarity_amount):
        """Enhance vocal clarity using spectral processing"""
        # Use a comb filter to enhance vocal frequencies
        frequencies = [250, 500, 1000, 2000, 4000]  # Common vocal frequencies

        for freq in frequencies:
            # Create a narrow bandpass filter for each vocal frequency
            sos = signal.butter(2, [freq*0.8, freq*1.2], 'bandpass', fs=sr, output='sos')
            band = signal.sosfilt(sos, audio)
            # Add boosted band back to original
            audio = audio + (band * clarity_amount * 0.1)

        return audio


    def analyze_audio_characteristics(self, audio, sr):
        """Analyze audio for various characteristics"""
        # Calculate spectral centroid for brightness
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

        # Calculate RMS for loudness
        rms = np.sqrt(np.mean(audio**2))

        # Calculate zero-crossing rate for noisiness
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        return {
            'brightness': spectral_centroid,
            'loudness': rms,
            'noisiness': zcr
        }


    def process_file(self, file_path, params):
        """Process a single audio file with given parameters"""
        # Load audio with original properties
        audio, sr = librosa.load(file_path, sr=None, mono=False)
        original_duration = len(audio) / sr

        # Convert to mono for processing if stereo
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
            is_stereo = True
        else:
            audio_mono = audio
            is_stereo = False

        # Initialize report data for this file
        file_report = {
            'filename': os.path.basename(file_path),
            'channels': 'stereo' if is_stereo else 'mono',
            'original_duration': original_duration,
            'final_duration': original_duration,
            'original_lufs': 0,
            'final_lufs': 0,
            'applied_effects': [],
            'brightness': 0,
            'loudness': 0,
            'noisiness': 0
        }

        # Apply processing based on mode
        mode = params['processing_mode']
        processed_audio = audio_mono # Start with mono version


        # Voice isolation (if enabled)
        if params['isolation_enable'] and mode != 'Music Enhancement':
            try:
                processed_audio, sr = self.isolate_voice_demucs(file_path, params['isolation_strength'])
                file_report['applied_effects'].append(f"Voice isolation (strength: {params['isolation_strength']})")

                # Background reduction
                if params['background_reduction'] > 0:
                    processed_audio = self.reduce_background(processed_audio, sr, params['background_reduction'])
                    file_report['applied_effects'].append(f"Background reduction ({params['background_reduction']*100}%)")
            except Exception as e:
                print(f"Error in voice isolation: {e}")
                # Fallback: use original mono audio
                processed_audio, sr = librosa.load(file_path, sr=None, mono=True)
                file_report['applied_effects'].append("Voice isolation failed, used original audio")


        # Full Mix Processing or Music Enhancement (if not voice isolation)
        if mode != 'Voice Isolation Only':
             # Noise reduction
            if params['noise_enable']:
                processed_audio = self.apply_noise_reduction(processed_audio, sr, params['noise_amount'])
                file_report['applied_effects'].append(f"Noise reduction ({params['noise_amount']*100}%)")

            # EQ
            if params['eq_enable']:
                processed_audio = self.apply_eq(processed_audio, sr, params['low_cut'], params['high_cut'],
                                     params['bass_boost'], params['treble_boost'])
                file_report['applied_effects'].append(
                    f"EQ (low-cut {params['low_cut']}Hz, high-cut {params['high_cut']}Hz, "
                    f"bass {params['bass_boost']}, treble {params['treble_boost']})"
                )

            # Compression
            if params['comp_enable']:
                processed_audio = self.apply_compression(processed_audio, sr, params['comp_ratio'],
                                              params['comp_threshold'], params['comp_attack'],
                                              params['comp_release'])
                file_report['applied_effects'].append(
                    f"Compression (ratio {params['comp_ratio']}:1, threshold {params['comp_threshold']}dB)"
                )

        # Voice-specific processing (applied after isolation or on full mix if not music enhancement)
        if mode != 'Music Enhancement':
            if params['deess_enable']:
                processed_audio = self.apply_deessing(processed_audio, sr, params['deess_amount'])
                file_report['applied_effects'].append(f"De-essing ({params['deess_amount']*100}%)")

            if params['vocal_clarity'] > 0:
                processed_audio = self.enhance_vocal_clarity(processed_audio, sr, params['vocal_clarity'])
                file_report['applied_effects'].append(f"Vocal clarity ({params['vocal_clarity']*100}%)")


        # Silence trimming (applied after all other processing)
        if params['trim_enable']:
            fade_duration = params['fade_duration'] if params['fade_enable'] else 0
            processed_audio = self.trim_silence(processed_audio, sr, params['trim_threshold'], fade_duration)
            file_report['final_duration'] = len(processed_audio) / sr
            file_report['applied_effects'].append(
                f"Silence trimming (threshold {params['trim_threshold']}dB)"
            )
            if fade_duration > 0:
                file_report['applied_effects'].append(f"Fade in/out ({fade_duration}ms)")


        # Calculate final LUFS
        meter = pyln.Meter(sr)
        try:
            file_report['final_lufs'] = meter.integrated_loudness(processed_audio)
        except Exception:
             file_report['final_lufs'] = 'N/A' # Handle cases where LUFS calculation might fail


        # Analyze audio characteristics
        analysis = self.analyze_audio_characteristics(processed_audio, sr)
        file_report.update(analysis)

        # Peak normalization if enabled
        if params['normalize_peak']:
            peak = np.max(np.abs(processed_audio))
            if peak > 0:
                processed_audio = processed_audio / peak * 0.99  # Normalize to -1 dBFS
            file_report['applied_effects'].append("Peak normalization to -1.0 dBFS")

        # Resample if needed
        target_sr = params['sample_rate']
        if sr != target_sr:
            # Ensure processed_audio is 1D for resample if it ended up 2D somehow
            if processed_audio.ndim > 1:
                 processed_audio = np.mean(processed_audio, axis=1)
            processed_audio = librosa.resample(y=processed_audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr


        return processed_audio, sr, file_report

    def run_processing(self, b):
        """Run batch processing on all uploaded files"""
        self.output.clear_output()

        if not self.uploaded_files:
            with self.output:
                print("Please upload files first!")
            return

        with self.output:
            print("Starting audio processing...")

            # Collect parameters
            params = {
                'processing_mode': self.processing_mode.value,
                'isolation_enable': self.isolation_enable.value,
                'isolation_strength': self.isolation_strength.value,
                'background_reduction': self.background_reduction.value,
                'loudness_enable': self.loudness_enable.value,
                'loudness_target': self.loudness_target.value,
                'noise_enable': self.noise_enable.value,
                'noise_amount': self.noise_amount.value,
                'eq_enable': self.eq_enable.value,
                'low_cut': self.low_cut.value,
                'high_cut': self.high_cut.value,
                'bass_boost': self.bass_boost.value,
                'treble_boost': self.treble_boost.value,
                'comp_enable': self.comp_enable.value,
                'comp_ratio': self.comp_ratio.value,
                'comp_threshold': self.comp_threshold.value,
                'comp_attack': self.comp_attack.value,
                'comp_release': self.comp_release.value,
                'deess_enable': self.deess_enable.value,
                'deess_amount': self.deess_amount.value,
                'vocal_clarity': self.vocal_clarity.value,
                'trim_enable': self.trim_enable.value,
                'trim_threshold': self.trim_threshold.value,
                'fade_enable': self.fade_enable.value,
                'fade_duration': self.fade_duration.value,
                'format': self.format_dropdown.value.lower(),
                'sample_rate': self.sample_rate_dropdown.value,
                'bit_depth_dropdown': self.bit_depth_dropdown.value, # Pass bit depth
                'normalize_peak': self.normalize_peak.value
            }

            self.processed_files = []
            self.report_data = []

            # Process each file
            for i, file_path in enumerate(self.uploaded_files):
                print(f"Processing {i+1}/{len(self.uploaded_files)}: {file_path}")

                try:
                    processed_audio, sr, file_report = self.process_file(file_path, params)

                    # Save processed file
                    output_filename = f"processed_{os.path.splitext(file_path)[0]}.{params['format']}"

                    # Set appropriate bit depth for WAV
                    subtype = 'FLOAT' # Default for non-WAV or if bit depth is not handled by sf
                    if params['format'] == 'wav':
                        if params['bit_depth_dropdown'] == '16-bit':
                            subtype = 'PCM_16'
                        elif params['bit_depth_dropdown'] == '24-bit':
                            subtype = 'PCM_24'
                        elif params['bit_depth_dropdown'] == '32-bit':
                            subtype = 'PCM_32' # Note: sf might save as FLOAT for 32-bit

                    # Ensure processed audio is 1D (mono) for saving
                    if processed_audio.ndim > 1:
                        processed_audio = np.mean(processed_audio, axis=1)


                    sf.write(output_filename, processed_audio, sr, subtype=subtype)


                    self.processed_files.append(output_filename)
                    self.report_data.append(file_report)

                    print(f"✓ Completed: {output_filename}")
                    print(f"  Applied {len(file_report['applied_effects'])} effects")

                except Exception as e:
                    print(f"✗ Error processing {file_path}: {str(e)}")
                    # Try to continue with next file
                    continue

            print("Audio processing completed!")
            print(f"Successfully processed {len(self.processed_files)} of {len(self.uploaded_files)} files")

    def download_files(self, b):
        """Create ZIP file with processed files"""
        if not self.processed_files:
            with self.output:
                print("No processed files available. Run processing first!")
            return

        with self.output:
            print("Creating ZIP archive...")

            zip_filename = f"processed_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for file in self.processed_files:
                    zipf.write(file)
                    # Clean up temporary files
                    if os.path.exists(file):
                        os.remove(file)

            print(f"ZIP file created: {zip_filename}")
            files.download(zip_filename)

    def download_report(self, b):
        """Generate and download CSV report"""
        if not self.report_data:
            with self.output:
                print("No report data available. Run processing first!")
            return

        with self.output:
            print("Generating report...")

            # Create DataFrame
            df = pd.DataFrame(self.report_data)

            # Format effects list
            df['applied_effects'] = df['applied_effects'].apply(lambda x: '; '.join(x))

            # Add processing timestamp
            df['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Save CSV
            report_filename = f"audio_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(report_filename, index=False)

            print(f"Report generated: {report_filename}")
            files.download(report_filename)

# Main execution
if __name__ == "__main__":
    processor = CompleteAudioProcessor()
    processor.create_ui()
