#!/usr/bin/env python3
"""
Hyundai AI Voice Assistant - Audio Quality Control System
Tamil and Bengali Speech Dataset Quality Analysis

Web-based version with file upload and automatic download
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import librosa
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal
from scipy.signal import find_peaks
from datetime import datetime
from pathlib import Path
import logging
import streamlit as st
import base64
import tempfile
import zipfile
import io

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioQualityAnalyzer:
    """
    Professional audio quality analysis for speech datasets.
    Implements ITU-R BS.1770-4 compliant loudness measurement and
    advanced noise/echo detection algorithms.
    """
    
    def __init__(self, target_lufs=-23.0, lufs_tolerance=3.0):
        """
        Initialize analyzer with target specifications.
        
        Args:
            target_lufs: Target ITU loudness in LUFS (default: -23)
            lufs_tolerance: Acceptable deviation from target (default: ±3)
        """
        self.target_lufs = target_lufs
        self.lufs_tolerance = lufs_tolerance
        self.results = []
        
    def analyze_background_noise(self, y, sr):
        """
        Detect background noise using spectral analysis and noise floor estimation.
        
        Background noise detection strategy:
        1. Extract quiet segments between speech
        2. Calculate noise floor in dBFS
        3. Analyze spectral consistency
        4. Threshold: -50 dBFS (audible noise level)
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            tuple: (has_noise: bool, noise_level_db: float)
        """
        if len(y) < sr * 0.5:  # Need at least 0.5 seconds
            return False, -80.0
        
        # Calculate frame-based RMS energy
        frame_length = min(2048, len(y) // 4)
        hop_length = frame_length // 2
        
        # RMS energy calculation
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
        
        # Find quiet segments (bottom 10 percentile)
        noise_threshold = np.percentile(rms_db, 10)
        quiet_frames = rms_db <= noise_threshold
        
        # Calculate noise floor from quiet segments
        if np.any(quiet_frames):
            noise_floor_db = np.mean(rms_db[quiet_frames])
        else:
            noise_floor_db = np.min(rms_db)
        
        # Spectral analysis for noise characterization
        n_fft = min(2048, len(y) // 2)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Check for consistent low-frequency rumble or high-frequency hiss
        freq_bins = magnitude.shape[0]
        low_freq_energy = np.mean(magnitude[:int(freq_bins * 0.1)])  # Below 1kHz
        high_freq_energy = np.mean(magnitude[int(freq_bins * 0.7):])  # Above 7kHz
        
        # Decision logic: noise detected if floor > -50 dBFS or significant HF/LF energy
        has_noise = (noise_floor_db > -50.0) or \
                   (low_freq_energy > 0.01) or \
                   (high_freq_energy > 0.005)
        
        return has_noise, round(noise_floor_db, 2)
    
    def analyze_echo(self, y, sr):
        """
        Detect echo/reverb using RT60 estimation and impulse response analysis.
        
        Echo detection strategy:
        1. Calculate RT60 (reverberation time)
        2. Analyze energy decay curve
        3. Check for delayed reflections using autocorrelation
        4. Threshold: RT60 > 0.3s indicates problematic echo
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            bool: True if echo detected
        """
        if len(y) < sr * 1.0:  # Need at least 1 second
            return False
        
        # Method 1: Energy decay analysis for RT60
        # Calculate energy envelope
        squared_signal = y ** 2
        window_size = max(int(sr * 0.01), 100)  # 10ms window
        energy_envelope = np.convolve(squared_signal, 
                                     np.ones(window_size)/window_size, 
                                     mode='same')
        
        # Convert to dB
        energy_db = 10 * np.log10(np.maximum(energy_envelope, 1e-10))
        
        # Find peak and analyze decay
        peak_idx = np.argmax(energy_db)
        if peak_idx < len(energy_db) - sr//4:  # Need 0.25s after peak
            decay_curve = energy_db[peak_idx:peak_idx + sr//4]
            
            # Calculate decay rate (simplified RT60)
            if len(decay_curve) > 100:
                # Linear fit to decay curve
                time_axis = np.arange(len(decay_curve)) / sr
                try:
                    slope, _ = np.polyfit(time_axis, decay_curve, 1)
                    rt60_estimate = -60 / slope if slope < -1 else 0.1
                    rt60_estimate = min(rt60_estimate, 2.0)  # Cap at 2 seconds
                except:
                    rt60_estimate = 0.1
            else:
                rt60_estimate = 0.1
        else:
            rt60_estimate = 0.1
        
        # Method 2: Autocorrelation for discrete echoes
        # Look for peaks in autocorrelation indicating delayed reflections
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
        autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize
        
        # Look for significant peaks in 20-500ms range (typical echo delays)
        min_delay_samples = int(sr * 0.02)  # 20ms
        max_delay_samples = min(int(sr * 0.5), len(autocorr))  # 500ms
        
        echo_detected = False
        if max_delay_samples > min_delay_samples:
            echo_region = autocorr[min_delay_samples:max_delay_samples]
            peaks, _ = find_peaks(echo_region, height=0.15)  # 15% correlation threshold
            echo_detected = len(peaks) > 0
        
        # Combined decision: echo present if RT60 > 0.3s OR discrete echoes found
        return (rt60_estimate > 0.3) or echo_detected
    
    def measure_itu_loudness(self, y, sr):
        """
        Measure ITU-R BS.1770-4 compliant integrated loudness.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            float: Integrated loudness in LUFS
        """
        try:
            # Create meter with ITU-R BS.1770-4 standard
            meter = pyln.Meter(sr)
            
            # Measure integrated loudness
            loudness_lufs = meter.integrated_loudness(y)
            
            # Handle edge cases
            if np.isnan(loudness_lufs) or np.isinf(loudness_lufs):
                return -70.0  # Very quiet signal fallback
                
            return round(loudness_lufs, 2)
            
        except Exception as e:
            logger.warning(f"LUFS measurement error: {e}")
            # Fallback: estimate from RMS
            rms = np.sqrt(np.mean(y**2))
            loudness_estimate = 20 * np.log10(rms + 1e-10) - 3.0  # Rough LUFS approximation
            return round(loudness_estimate, 2)
    
    def detect_noise_suppression(self, y, sr):
        """
        Detect artifacts from noise suppression processing.
        
        Noise suppression detection strategy:
        1. Check for spectral holes/gaps (aggressive gating)
        2. Detect musical noise (random tonal artifacts)
        3. Analyze high-frequency cutoff (over-smoothing)
        4. Check for unnatural silence patterns
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            bool: True if noise suppression artifacts detected
        """
        if len(y) < sr * 1.0:  # Need at least 1 second
            return False
        
        # Calculate STFT for spectral analysis
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Feature 1: Spectral gaps (holes in frequency content)
        freq_means = np.mean(magnitude_db, axis=1)
        freq_std = np.std(magnitude_db, axis=1)
        
        # Look for frequency bins with unusually low energy (spectral holes)
        spectral_holes = np.sum(freq_means < (np.median(freq_means) - 20))
        has_spectral_gaps = spectral_holes > (len(freq_means) * 0.1)  # >10% bins affected
        
        # Feature 2: High-frequency cutoff (common in noise suppression)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        hf_mask = freqs > (sr * 0.35)  # Above 35% of Nyquist
        
        if np.any(hf_mask):
            hf_energy = np.mean(magnitude[hf_mask, :])
            lf_energy = np.mean(magnitude[~hf_mask, :])
            hf_ratio = hf_energy / (lf_energy + 1e-10)
            unnatural_hf_cutoff = hf_ratio < 0.01  # Severe HF attenuation
        else:
            unnatural_hf_cutoff = False
        
        # Feature 3: Musical noise (isolated random peaks)
        # Calculate temporal variance of spectral peaks
        peak_counts = []
        for frame in range(magnitude.shape[1]):
            spectrum = magnitude[:, frame]
            peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1)
            peak_counts.append(len(peaks))
        
        if len(peak_counts) > 10:
            peak_variance = np.var(peak_counts)
            musical_noise = peak_variance > 50  # High variance indicates musical noise
        else:
            musical_noise = False
        
        # Feature 4: Unnatural silence patterns (aggressive gating)
        rms = librosa.feature.rms(y=y)[0]
        rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
        
        # Count abrupt transitions to silence
        silence_threshold = np.percentile(rms_db, 5)
        silent_frames = rms_db < silence_threshold
        transitions = np.diff(silent_frames.astype(int))
        abrupt_gates = np.sum(np.abs(transitions)) > (len(rms_db) * 0.2)
        
        # Combined decision: suppression detected if 2+ artifacts present
        artifacts_count = sum([has_spectral_gaps, unnatural_hf_cutoff, 
                             musical_noise, abrupt_gates])
        
        return artifacts_count >= 2
    
    def analyze_file(self, filepath):
        """
        Analyze a single WAV file for all quality metrics.
        
        Args:
            filepath: Path to WAV file
            
        Returns:
            dict: Analysis results
        """
        try:
            # Load audio file
            y, sr = librosa.load(filepath, sr=None, mono=True)
            
            # Normalize to prevent clipping in analysis
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y)) * 0.95
            
            # Run all analyses
            has_noise, noise_level = self.analyze_background_noise(y, sr)
            has_echo = self.analyze_echo(y, sr)
            loudness_lufs = self.measure_itu_loudness(y, sr)
            has_suppression = self.detect_noise_suppression(y, sr)
            
            # Check if loudness is within spec (-23 to -26 LUFS)
            loudness_ok = (self.target_lufs - self.lufs_tolerance <= 
                          loudness_lufs <= 
                          self.target_lufs + self.lufs_tolerance)
            
            result = {
                'Filename': os.path.basename(filepath),
                'Background_Noise': 'Yes' if has_noise else 'No',
                'Noise_Level_dB': noise_level,
                'Echo': 'Yes' if has_echo else 'No',
                'ITU_Loudness_LUFS': loudness_lufs,
                'Loudness_Within_Spec': 'Yes' if loudness_ok else 'No',
                'Noise_Suppression': 'Yes' if has_suppression else 'No',
                'Duration_sec': round(len(y) / sr, 2),
                'Sample_Rate_Hz': sr,
                'Overall_Pass': 'Yes' if (not has_noise and not has_echo and 
                                         loudness_ok and not has_suppression) else 'No'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {filepath}: {e}")
            return {
                'Filename': os.path.basename(filepath),
                'Background_Noise': 'Error',
                'Noise_Level_dB': 'Error',
                'Echo': 'Error',
                'ITU_Loudness_LUFS': 'Error',
                'Loudness_Within_Spec': 'Error',
                'Noise_Suppression': 'Error',
                'Duration_sec': 'Error',
                'Sample_Rate_Hz': 'Error',
                'Overall_Pass': 'Error'
            }
    
    def analyze_files(self, file_paths, output_file='audio_quality_report.xlsx'):
        """
        Analyze all provided WAV files and generate report.
        
        Args:
            file_paths: List of paths to WAV files
            output_file: Output Excel file path
            
        Returns:
            pd.DataFrame: Analysis results
        """
        if not file_paths:
            logger.error("No WAV files provided")
            return None
        
        logger.info(f"Found {len(file_paths)} WAV files to analyze")
        
        # Analyze each file
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Analyzing file {i}/{len(file_paths)}: {os.path.basename(file_path)}")
            result = self.analyze_file(file_path)
            self.results.append(result)
            
            # Progress indicator every 10 files
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(file_paths)} files completed")
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save to Excel with formatting
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Analysis_Results', index=False)
            
            # Add summary sheet
            total_files = len(df)
            passed_files = len(df[df['Overall_Pass'] == 'Yes'])
            
            summary_data = {
                'Metric': [
                    'Total Files Analyzed',
                    'Files Passed All Checks',
                    'Files with Background Noise',
                    'Files with Echo',
                    'Files Outside Loudness Spec',
                    'Files with Noise Suppression',
                    'Pass Rate (%)'
                ],
                'Count': [
                    total_files,
                    passed_files,
                    len(df[df['Background_Noise'] == 'Yes']),
                    len(df[df['Echo'] == 'Yes']),
                    len(df[df['Loudness_Within_Spec'] == 'No']),
                    len(df[df['Noise_Suppression'] == 'Yes']),
                    round((passed_files / total_files * 100) if total_files > 0 else 0, 2)
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Auto-adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        logger.info(f"Analysis complete. Report saved to: {output_file}")
        logger.info(f"Pass rate: {passed_files}/{total_files} files ({passed_files/total_files*100:.1f}%)")
        
        return df


def get_binary_file_downloader_html(bin_file, file_label='File'):
    """
    Generate a download link for a binary file.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


def main():
    """
    Main execution function for Streamlit web app.
    """
    st.set_page_config(
        page_title="Hyundai Audio Quality Control",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Hyundai AI Voice Assistant - Audio Quality Control System")
    st.markdown("""
    This tool analyzes Tamil and Bengali speech datasets for quality assurance.
    Upload WAV files to check for background noise, echo, loudness compliance, and noise suppression artifacts.
    """)
    
    # File upload section
    st.header("Upload Audio Files")
    uploaded_files = st.file_uploader(
        "Choose WAV files", 
        type=['wav', 'WAV'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files")
        
        # Configuration options
        st.sidebar.header("Analysis Settings")
        target_lufs = st.sidebar.slider(
            "Target LUFS", 
            min_value=-30.0, 
            max_value=-10.0, 
            value=-24.5, 
            step=0.5,
            help="Target loudness level in LUFS (ITU-R BS.1770-4)"
        )
        
        lufs_tolerance = st.sidebar.slider(
            "LUFS Tolerance", 
            min_value=0.5, 
            max_value=5.0, 
            value=1.5, 
            step=0.5,
            help="Acceptable deviation from target loudness"
        )
        
        # Analyze button
        if st.button("Analyze Audio Files", type="primary"):
            with st.spinner("Analyzing files..."):
                # Create temporary directory for uploaded files
                with tempfile.TemporaryDirectory() as tmp_dir:
                    file_paths = []
                    
                    # Save uploaded files to temporary directory
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(tmp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                    
                    # Initialize analyzer
                    analyzer = AudioQualityAnalyzer(
                        target_lufs=target_lufs, 
                        lufs_tolerance=lufs_tolerance
                    )
                    
                    # Generate output filename
                    output_filename = f"hyundai_audio_qc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    output_path = os.path.join(tmp_dir, output_filename)
                    
                    # Run analysis
                    results_df = analyzer.analyze_files(file_paths, output_path)
                    
                    if results_df is not None:
                        st.success("Analysis complete!")
                        
                        # Display summary
                        st.header("Analysis Summary")
                        
                        total = len(results_df)
                        passed = len(results_df[results_df['Overall_Pass'] == 'Yes'])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Files", total)
                        col2.metric("Files Passed", passed)
                        col3.metric("Files Failed", total - passed)
                        col4.metric("Pass Rate", f"{passed/total*100:.1f}%" if total > 0 else "0%")
                        
                        # Issue breakdown
                        st.subheader("Issue Breakdown")
                        issues_data = {
                            'Issue Type': [
                                'Background Noise',
                                'Echo',
                                'Loudness Out of Spec',
                                'Noise Suppression Artifacts'
                            ],
                            'Count': [
                                len(results_df[results_df['Background_Noise'] == 'Yes']),
                                len(results_df[results_df['Echo'] == 'Yes']),
                                len(results_df[results_df['Loudness_Within_Spec'] == 'No']),
                                len(results_df[results_df['Noise_Suppression'] == 'Yes'])
                            ]
                        }
                        
                        issues_df = pd.DataFrame(issues_data)
                        st.bar_chart(issues_df.set_index('Issue Type'))
                        
                        # Detailed results table
                        st.subheader("Detailed Results")
                        st.dataframe(results_df)
                        
                        # Download link for the report
                        st.markdown("### Download Report")
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="Download Excel Report",
                                data=f,
                                file_name=output_filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    else:
                        st.error("Analysis failed. Please check the uploaded files.")


if __name__ == "__main__":
    main()