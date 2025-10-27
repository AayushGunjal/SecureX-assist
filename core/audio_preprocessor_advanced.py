"""
Advanced Audio Preprocessing Module
Ultimate Offline Voice Biometric Stack - Component 3

Multi-stage audio processing pipeline:
1. Silence removal (webrtcvad)
2. Volume normalization
3. Noise reduction (noisereduce)
4. Feature extraction (librosa)
5. Quality validation

Features:
- VAD-based silence removal
- Adaptive noise reduction
- Automatic gain control
- Quality metrics
"""

import logging
import numpy as np
import torch
import torchaudio
import warnings
from typing import Tuple, Dict, Optional

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Advanced audio preprocessing for speaker verification.
    Removes silence, cleans noise, normalizes volume.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate: Target sample rate (16kHz for voice biometrics)
        """
        self.sample_rate = sample_rate
        
        # Initialize WebRTC VAD
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(1)  # Aggressiveness 1 (least aggressive)
            logger.info("✅ WebRTC VAD initialized (aggressiveness=1)")
        except ImportError:
            logger.warning("webrtcvad not available, will skip VAD")
            self.vad = None
        
        # Initialize noise reduction
        try:
            import noisereduce
            logger.info("✅ noisereduce library available")
        except ImportError:
            logger.warning("noisereduce not available, will use librosa alternative")
    
    def process(self, audio_array: np.ndarray, 
                remove_silence: bool = True,
                reduce_noise: bool = True,
                normalize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Full audio preprocessing pipeline.
        
        Args:
            audio_array: Input audio (numpy array, float32, 16kHz)
            remove_silence: Whether to remove silence
            reduce_noise: Whether to apply noise reduction
            normalize: Whether to normalize volume
            
        Returns:
            Tuple of (processed_audio, stats_dict)
        """
        stats = {
            "original_length": len(audio_array),
            "original_rms": float(np.sqrt(np.mean(audio_array**2))),
        }
        
        processed = audio_array.copy()
        
        # Step 1: Remove silence
        if remove_silence and self.vad:
            processed = self._remove_silence_vad(processed)
            stats["after_vad_length"] = len(processed)
        
        # Step 2: Reduce noise
        if reduce_noise:
            processed = self._reduce_noise(processed)
            stats["after_noise_reduction_rms"] = float(np.sqrt(np.mean(processed**2)))
        
        # Step 3: Normalize volume
        if normalize:
            processed = self._normalize_volume(processed)
            stats["after_normalization_rms"] = float(np.sqrt(np.mean(processed**2)))
        
        # Step 4: Validate quality
        stats["quality_metrics"] = self._compute_quality_metrics(processed)
        
        logger.info(f"✅ Audio preprocessing complete:")
        logger.info(f"   - Original: {stats['original_length']} samples")
        logger.info(f"   - After VAD: {stats.get('after_vad_length', stats['original_length'])} samples")
        logger.info(f"   - Final RMS: {stats.get('after_normalization_rms', stats['original_rms']):.4f}")
        
        return processed, stats
    
    def _remove_silence_vad(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove silence using WebRTC VAD.
        
        VAD divides audio into 10/20/30ms frames and classifies each as speech/silence.
        """
        if self.vad is None:
            return audio
        
        try:
            # Convert to int16 for WebRTC VAD
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Frame parameters
            frame_duration_ms = 20  # WebRTC VAD typically uses 20ms frames
            frame_size = int(self.sample_rate * frame_duration_ms / 1000)
            
            # Process frames
            voiced_frames = []
            for start in range(0, len(audio_int16), frame_size):
                end = min(start + frame_size, len(audio_int16))
                frame = audio_int16[start:end].tobytes()
                
                # Check if frame contains voice
                if len(frame) == frame_size * 2:  # 2 bytes per 16-bit sample
                    is_voiced = self.vad.is_speech(frame, self.sample_rate)
                    if is_voiced:
                        voiced_frames.append(audio[start:end])
            
            if voiced_frames:
                result = np.concatenate(voiced_frames)
                logger.info(f"VAD: Removed {len(audio) - len(result)} silent samples")
                return result
            else:
                logger.warning("VAD: No voiced frames detected, returning original")
                return audio
                
        except Exception as e:
            logger.warning(f"VAD processing failed: {e}, returning original")
            return audio
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Reduce background noise using spectral subtraction or noisereduce.
        """
        try:
            import noisereduce as nr
            
            # Use noisereduce library (spectral subtraction)
            reduced = nr.reduce_noise(y=audio, sr=self.sample_rate)
            logger.info("✅ Noise reduction applied (noisereduce)")
            return reduced
            
        except ImportError:
            # Fallback: Simple spectral subtraction using librosa
            return self._spectral_subtraction(audio)
    
    def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Simple spectral subtraction for noise reduction.
        
        Assumes first 0.5 seconds are noise.
        """
        try:
            import librosa
            
            # Estimate noise profile (first 0.5s)
            noise_duration_samples = int(0.5 * self.sample_rate)
            noise_profile = audio[:noise_duration_samples]
            
            # Compute spectrograms
            S = librosa.stft(audio)
            S_noise = librosa.stft(noise_profile)
            
            # Spectral subtraction
            S_magnitude = np.abs(S)
            noise_magnitude = np.abs(S_noise).mean(axis=1, keepdims=True)
            
            # Subtract noise
            subtracted = S_magnitude - noise_magnitude
            subtracted = np.maximum(subtracted, 0)  # Prevent negative values
            
            # Reconstruct
            phase = np.angle(S)
            S_reduced = subtracted * np.exp(1j * phase)
            audio_reduced = librosa.istft(S_reduced)
            
            # Match original length
            if len(audio_reduced) > len(audio):
                audio_reduced = audio_reduced[:len(audio)]
            else:
                audio_reduced = np.pad(audio_reduced, (0, len(audio) - len(audio_reduced)))
            
            logger.info("✅ Noise reduction applied (spectral subtraction)")
            return audio_reduced.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Spectral subtraction failed: {e}, returning original")
            return audio
    
    def _normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target RMS level.
        
        Automatic Gain Control (AGC): Adjusts volume to standard level
        to handle different microphone gains.
        """
        target_rms = 0.1  # Target RMS level
        
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms < 1e-6:
            logger.warning("Audio has near-zero RMS, skipping normalization")
            return audio
        
        # Calculate gain
        gain = target_rms / current_rms
        normalized = audio * gain
        
        # Prevent clipping
        if np.max(np.abs(normalized)) > 0.95:
            normalized = normalized / np.max(np.abs(normalized)) * 0.95
        
        logger.info(f"✅ Volume normalized (gain: {gain:.2f}x)")
        return normalized.astype(np.float32)
    
    def _compute_quality_metrics(self, audio: np.ndarray) -> Dict:
        """
        Compute quality metrics for audio validation.
        """
        try:
            import librosa
            
            # RMS energy
            rms = np.sqrt(np.mean(audio**2))
            
            # MFCC statistics
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfcc)
            mfcc_std = np.std(mfcc)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            zcr_mean = np.mean(zcr)
            
            # Signal-to-Noise Ratio estimate
            # Use spectral flatness as proxy for SNR
            spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            snr_estimate = float(spec_centroid.mean())
            
            return {
                "rms_energy": float(rms),
                "mfcc_mean": float(mfcc_mean),
                "mfcc_std": float(mfcc_std),
                "zcr_mean": float(zcr_mean),
                "snr_estimate": float(snr_estimate),
                "duration_sec": len(audio) / self.sample_rate,
            }
            
        except Exception as e:
            logger.warning(f"Quality metrics computation failed: {e}")
            return {"error": str(e)}
    
    def validate_audio_quality(self, audio: np.ndarray, 
                              min_duration: float = 1.0,
                              min_rms: float = 0.01) -> Tuple[bool, str]:
        """
        Validate audio quality for speaker verification.
        
        Args:
            audio: Audio array
            min_duration: Minimum duration in seconds
            min_rms: Minimum RMS energy level
            
        Returns:
            Tuple of (is_valid, reason)
        """
        duration = len(audio) / self.sample_rate
        rms = np.sqrt(np.mean(audio**2))
        
        if duration < min_duration:
            return False, f"Duration too short: {duration:.1f}s < {min_duration}s"
        
        if rms < min_rms:
            return False, f"Signal too quiet: RMS {rms:.4f} < {min_rms}"
        
        return True, "Audio quality OK"
    
    def get_processor_info(self) -> Dict:
        """Get information about the preprocessor."""
        return {
            "sample_rate": self.sample_rate,
            "vad_available": self.vad is not None,
            "pipeline": [
                "Silence removal (WebRTC VAD)",
                "Noise reduction (noisereduce)",
                "Volume normalization (AGC)",
                "Quality validation"
            ]
        }


if __name__ == "__main__":
    logger.info("Testing Audio Preprocessor...")
    
    preprocessor = AudioPreprocessor()
    info = preprocessor.get_processor_info()
    
    print("\n✅ Preprocessor Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
