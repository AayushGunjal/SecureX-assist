"""
SecureX-Assist - Advanced Audio Preprocessing
Data augmentation for voice biometric training and anti-spoofing
"""

import numpy as np
import scipy.signal as signal
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class AudioAugmentationEngine:
    """
    Advanced audio data augmentation for voice biometric training
    Generates diverse voice samples from limited enrollment data
    """

    def __init__(self, config: dict):
        self.config = config
        self.augmentation_settings = config.get('augmentation', {})

        # Default augmentation parameters
        self.pitch_shift_range = self.augmentation_settings.get('pitch_shift', [-3, 3])
        self.gain_db_range = self.augmentation_settings.get('gain_db', [-6, 6])
        self.speed_factor_range = self.augmentation_settings.get('speed_factor', [0.95, 1.05])
        self.noise_level = self.augmentation_settings.get('noise_level', 0.01)

        logger.info("AudioAugmentationEngine initialized with advanced augmentation")

    def augment_audio_sample(self, audio_data: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """
        Generate augmented versions of a single audio sample

        Args:
            audio_data: Original audio numpy array (can be 1D or 2D)
            sample_rate: Audio sample rate

        Returns:
            List of augmented audio samples
        """
        # Ensure audio is 1D
        if audio_data.ndim == 2:
            audio_data = audio_data.squeeze()
        elif audio_data.ndim != 1:
            logger.warning(f"Unexpected audio dimensions: {audio_data.shape}, skipping augmentation")
            return [audio_data]

        augmented_samples = [audio_data]  # Include original

        # Generate 4 augmented versions
        for i in range(4):
            augmented = audio_data.copy()

            # Apply random augmentations in sequence
            augmented = self._apply_pitch_shift(augmented, sample_rate)
            augmented = self._apply_gain_variation(augmented)
            augmented = self._apply_speed_variation(augmented, sample_rate)
            augmented = self._add_background_noise(augmented)

            augmented_samples.append(augmented)

        return augmented_samples

    def _apply_pitch_shift(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply random pitch shifting (±3% by default)
        """
        try:
            # Random pitch shift in semitones
            pitch_shift = random.uniform(self.pitch_shift_range[0], self.pitch_shift_range[1])

            if abs(pitch_shift) < 0.1:  # Skip if shift is negligible
                return audio_data

            # Convert semitones to frequency ratio
            ratio = 2 ** (pitch_shift / 12.0)

            # Resample to achieve pitch shift
            new_length = int(len(audio_data) / ratio)
            if new_length > 0:
                shifted_audio = signal.resample(audio_data, new_length)
                # Pad or truncate to original length
                if len(shifted_audio) > len(audio_data):
                    shifted_audio = shifted_audio[:len(audio_data)]
                else:
                    # Pad with zeros
                    padding = np.zeros(len(audio_data) - len(shifted_audio))
                    shifted_audio = np.concatenate([shifted_audio, padding])

                return shifted_audio.astype(audio_data.dtype)
            else:
                return audio_data

        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return audio_data

    def _apply_gain_variation(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply random gain variation (±6 dB by default)
        """
        try:
            # Random gain in dB
            gain_db = random.uniform(self.gain_db_range[0], self.gain_db_range[1])
            gain_factor = 10 ** (gain_db / 20.0)  # Convert dB to linear

            # Apply gain
            augmented = audio_data * gain_factor

            # Prevent clipping
            max_val = np.max(np.abs(augmented))
            if max_val > 1.0:
                augmented = augmented / max_val * 0.95

            return augmented

        except Exception as e:
            logger.warning(f"Gain variation failed: {e}")
            return audio_data

    def _apply_speed_variation(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply random speed variation (±5% by default)
        """
        try:
            # Random speed factor
            speed_factor = random.uniform(self.speed_factor_range[0], self.speed_factor_range[1])

            if abs(speed_factor - 1.0) < 0.01:  # Skip if change is negligible
                return audio_data

            # Resample to achieve speed change
            new_length = int(len(audio_data) / speed_factor)
            if new_length > 0:
                speed_changed = signal.resample(audio_data, new_length)

                # Stretch/compress to original length using interpolation
                original_indices = np.linspace(0, len(speed_changed) - 1, len(audio_data))
                stretched = np.interp(original_indices, np.arange(len(speed_changed)), speed_changed)

                return stretched.astype(audio_data.dtype)
            else:
                return audio_data

        except Exception as e:
            logger.warning(f"Speed variation failed: {e}")
            return audio_data

    def _add_background_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Add subtle background noise to simulate real environments
        """
        try:
            # Generate white noise
            noise = np.random.normal(0, self.noise_level, len(audio_data))

            # Mix with original audio
            noisy_audio = audio_data + noise

            # Normalize to prevent clipping
            max_val = np.max(np.abs(noisy_audio))
            if max_val > 1.0:
                noisy_audio = noisy_audio / max_val * 0.95

            return noisy_audio

        except Exception as e:
            logger.warning(f"Background noise addition failed: {e}")
            return audio_data

    def generate_enrollment_samples(self, original_samples: List[np.ndarray], sample_rate: int) -> List[np.ndarray]:
        """
        Generate comprehensive enrollment dataset from original samples

        Args:
            original_samples: List of original audio samples (typically 3, can be 1D or 2D)
            sample_rate: Audio sample rate

        Returns:
            Extended list including original + augmented samples
        """
        enrollment_dataset = []

        for original in original_samples:
            # Ensure audio is 1D for processing
            if original.ndim == 2:
                processed_original = original.squeeze()
            else:
                processed_original = original
            
            # Add original sample
            enrollment_dataset.append(processed_original)

            # Generate augmented versions
            augmented_versions = self.augment_audio_sample(processed_original, sample_rate)
            enrollment_dataset.extend(augmented_versions[1:])  # Skip original (already added)

        logger.info(f"Generated {len(enrollment_dataset)} enrollment samples from {len(original_samples)} originals")
        return enrollment_dataset


class VoiceQualityAnalyzer:
    """
    Analyze voice quality and detect potential spoofing indicators
    """

    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config.get('audio', {}).get('sample_rate', 16000)

    def analyze_voice_quality(self, audio_data: np.ndarray) -> dict:
        """
        Analyze various voice quality metrics

        Returns:
            Dictionary with quality scores and spoofing indicators
        """
        try:
            # Ensure audio is 1D for analysis
            if audio_data.ndim == 2:
                audio_data = audio_data.squeeze()
            elif audio_data.ndim != 1:
                logger.warning(f"Unexpected audio dimensions: {audio_data.shape}")
                return {'quality_score': 0.0, 'is_live_voice': False}

            # Basic quality metrics
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
            spectral_centroid = self._calculate_spectral_centroid(audio_data)

            # Spoofing indicators
            quality_score = self._calculate_quality_score(audio_data, rms_energy, zero_crossings, spectral_centroid)

            # Debug logging
            logger.info(f"Voice quality metrics - RMS: {rms_energy:.4f}, ZCR: {zero_crossings:.4f}, Centroid: {spectral_centroid:.1f}, Score: {quality_score:.3f}, Live: {quality_score >= 0.25}")

            return {
                'rms_energy': float(rms_energy),
                'zero_crossings': float(zero_crossings),
                'spectral_centroid': float(spectral_centroid),
                'quality_score': float(quality_score),
                'is_live_voice': quality_score >= 0.25  # Match config threshold
            }

        except Exception as e:
            logger.error(f"Voice quality analysis failed: {e}")
            return {'quality_score': 0.0, 'is_live_voice': False}

    def _calculate_spectral_centroid(self, audio_data: np.ndarray) -> float:
        """Calculate spectral centroid (brightness of sound)"""
        try:
            # Compute FFT
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)

            # Magnitude spectrum
            magnitude = np.abs(fft)

            # Spectral centroid
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)

            return abs(centroid)  # Return absolute value

        except Exception:
            return 0.0

    def _calculate_quality_score(self, audio_data: np.ndarray, rms: float, zcr: float, centroid: float) -> float:
        """
        Calculate overall voice quality score (0-1)
        Higher scores indicate better live voice quality
        """
        try:
            score = 0.0

            # RMS energy score (should be reasonable, not too quiet/loud)
            if 0.001 < rms < 0.8:  # MUCH more lenient for processed audio
                score += 0.3
            elif 0.0005 < rms < 1.0:  # Very lenient range
                score += 0.2

            # Zero crossing rate (should be typical for human speech)
            if 0.01 < zcr < 0.5:  # Much more lenient
                score += 0.3
            elif 0.005 < zcr < 0.8:  # Very lenient range
                score += 0.2

            # Spectral centroid (should be in human voice range ~2-5 kHz)
            if 300 < centroid < 10000:  # Much more lenient
                score += 0.4
            elif 100 < centroid < 15000:  # Very lenient range
                score += 0.3

            return min(score, 1.0)  # Cap at 1.0

        except Exception:
            return 0.0