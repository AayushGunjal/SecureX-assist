"""
SecureX-Assist - Audio Processing Pipeline
Audio capture, Voice Activity Detection, and processing
"""

import sounddevice as sd
import numpy as np
import scipy.signal as signal
from typing import Optional, Tuple, Callable
import logging
from pathlib import Path
import wave
import time

logger = logging.getLogger(__name__)


class AudioRecorder:
    """
    Real-time audio recording with Voice Activity Detection
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config.get('audio', {}).get('sample_rate', 16000)
        self.channels = config.get('audio', {}).get('channels', 1)
        self.chunk_size = config.get('audio', {}).get('chunk_size', 1024)
        
        # Recording state
        self.is_recording = False
        self.audio_buffer = []
        self.should_stop = False
        
        logger.info(f"AudioRecorder initialized: {self.sample_rate}Hz, {self.channels}ch")
    
    def list_devices(self) -> list:
        """List available audio input devices"""
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            return input_devices
        except Exception as e:
            logger.error(f"Failed to list devices: {e}")
            return []
    
    def reduce_background_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply gentle noise reduction to improve voice clarity
        Much less aggressive to preserve natural voice characteristics
        
        Args:
            audio_data: Input audio numpy array
            
        Returns:
            Gently noise-reduced audio
        """
        try:
            # Make a copy to avoid modifying original
            audio = audio_data.copy().flatten()
            
            # 1. Very gentle high-pass filter to remove only DC and very low rumble (< 50 Hz)
            nyquist = self.sample_rate / 2
            low_cutoff = 50 / nyquist  # LOWERED from 80Hz to preserve more low frequencies
            b, a = signal.butter(2, low_cutoff, btype='high')  # LOWER order filter
            audio = signal.filtfilt(b, a, audio)
            
            # 2. Much gentler noise gate - only suppress absolute silence
            noise_threshold = np.percentile(np.abs(audio), 5)  # Bottom 5% is noise (was 15%)
            noise_gate_ratio = 0.8  # Reduce noise to 80% (was 50% - much gentler)
            mask = np.abs(audio) < noise_threshold
            audio[mask] *= noise_gate_ratio
            
            # 3. Skip spectral subtraction for enrollment - it's too destructive
            # This preserves the natural spectral characteristics needed for quality analysis
            
            # 4. Very gentle dynamic range compression
            audio_rms = np.sqrt(np.mean(audio ** 2))
            target_rms = 0.15  # HIGHER target RMS (was 0.1) to preserve dynamics
            if audio_rms > 0:
                compression_ratio = min(target_rms / audio_rms, 2.0)  # Max 2x compression (was 3.0)
                audio = audio * compression_ratio
            
            # 5. Gentle normalization with more headroom
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9  # More headroom (was 0.95)
            
            # Reshape back to original
            if len(audio_data.shape) > 1:
                audio = audio.reshape(-1, 1)
            
            logger.debug("Enhanced background noise reduction applied")
            return audio
            
        except Exception as e:
            logger.warning(f"Noise reduction failed, using original: {e}")
            return audio_data
    
    def record_audio(
        self, 
        duration: float = 5.0,
        callback: Optional[Callable] = None,
        reduce_noise: bool = True
    ) -> Optional[np.ndarray]:
        """
        Record audio for specified duration with optional noise reduction
        
        Args:
            duration: Recording duration in seconds (max limit, can be stopped early)
            callback: Optional callback for progress updates
            reduce_noise: Apply real-time noise reduction (default: True)
            
        Returns:
            Audio data as numpy array (noise-reduced if enabled)
        """
        try:
            logger.info(f"Recording audio for up to {duration} seconds...")
            self.should_stop = False
            
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            
            # Wait for recording to complete or stop signal
            if callback:
                steps = int(duration * 10)  # 10 updates per second
                for i in range(steps):
                    if self.should_stop:
                        sd.stop()
                        # Get recorded samples so far
                        current_frame = int((i / steps) * duration * self.sample_rate)
                        audio_data = audio_data[:current_frame]
                        logger.info(f"Recording stopped early at {i/10:.1f}s")
                        break
                    time.sleep(0.1)
                    progress = (i + 1) / steps
                    callback(progress)
            else:
                # Simple wait with early stop check
                elapsed = 0
                while elapsed < duration and not self.should_stop:
                    time.sleep(0.1)
                    elapsed += 0.1
                if self.should_stop:
                    sd.stop()
                    current_frame = int(elapsed * self.sample_rate)
                    audio_data = audio_data[:current_frame]
                    logger.info(f"Recording stopped early at {elapsed:.1f}s")
            
            if not self.should_stop:
                sd.wait()
            
            logger.info(f"Recording complete: shape={audio_data.shape}")
            
            # Apply automatic gain control to boost quiet voices
            logger.info("Applying automatic gain control...")
            audio_data = AudioProcessor.apply_automatic_gain_control(audio_data, target_level=0.15)
            
            # Apply noise reduction if enabled
            if reduce_noise:
                logger.info("Applying background noise reduction...")
                audio_data = self.reduce_background_noise(audio_data)
                logger.info("Noise reduction complete - Audio cleaned")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return None
    
    def stop_recording(self):
        """Stop ongoing recording"""
        self.should_stop = True
        logger.info("Stop recording signal sent")
    
    def save_audio(self, audio_data: np.ndarray, filepath: str) -> bool:
        """
        Save audio data to WAV file
        
        Args:
            audio_data: Audio numpy array or list
            filepath: Output file path
            
        Returns:
            True if successful
        """
        try:
            # Convert to numpy array if it's a list
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # Ensure it's float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Normalize audio to int16 range
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(filepath, 'w') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            logger.info(f"Audio saved: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False


class VoiceActivityDetector:
    """
    Dual Voice Activity Detection system
    - RMS-based VAD: Lightweight, real-time
    - Silero-VAD: Neural network-based, high accuracy
    """
    
    def __init__(self, config: dict = None, aggressiveness: int = 1):
        # Handle both old and new calling conventions
        if config is not None:
            self.config = config
            self.sample_rate = config.get('audio', {}).get('sample_rate', 16000)
            self.aggressiveness = config.get('audio', {}).get('vad_aggressiveness', aggressiveness)
        else:
            self.config = {}
            self.sample_rate = 16000
            self.aggressiveness = aggressiveness
        
        # VAD thresholds - use config values with more sensitive defaults
        self.rms_threshold = self.config.get('audio', {}).get('min_audio_energy', 0.005)  # More sensitive
        self.min_speech_duration = self.config.get('audio', {}).get('min_speech_duration', 1.0)
        
        # Silero VAD model (lazy load)
        self.silero_model = None
        
        logger.info(f"VoiceActivityDetector initialized with RMS threshold: {self.rms_threshold}")
    
    def detect_voice_rms(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Simple RMS-based voice activity detection - ENHANCED SENSITIVITY
        
        Args:
            audio_data: Audio numpy array
            
        Returns:
            (has_voice, rms_energy)
        """
        try:
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Use adaptive threshold based on audio characteristics
            # For very quiet environments, be more sensitive
            base_threshold = self.rms_threshold
            
            # Check if audio has any significant content at all
            percentile_95 = np.percentile(np.abs(audio_data), 95)
            
            # If the 95th percentile is very low, this might be silence
            # But if RMS is above a very low threshold, consider it voice
            adaptive_threshold = min(base_threshold, percentile_95 * 0.5)
            
            # Minimum threshold to prevent false positives from pure noise
            adaptive_threshold = max(adaptive_threshold, 0.0005)  # Very sensitive minimum
            
            # Check if above threshold
            has_voice = rms > adaptive_threshold
            
            logger.debug(f"RMS VAD: rms={rms:.6f}, adaptive_threshold={adaptive_threshold:.6f}, base_threshold={base_threshold:.6f}, voice={has_voice}")
            return has_voice, float(rms)
            
        except Exception as e:
            logger.error(f"RMS VAD failed: {e}")
            return False, 0.0
    
    def load_silero_vad(self) -> bool:
        """Load Silero VAD model (lazy loading)"""
        if self.silero_model is not None:
            return True
        
        try:
            logger.info("Loading Silero VAD model...")
            import torch
            
            # Load pre-trained Silero VAD
            self.silero_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            self.get_speech_timestamps = utils[0]
            
            logger.info("[OK] Silero VAD model loaded")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}")
            return False
    
    def detect_voice_silero(self, audio_data: np.ndarray) -> Tuple[bool, list]:
        """
        Neural network-based voice activity detection using Silero VAD
        
        Args:
            audio_data: Audio numpy array
            
        Returns:
            (has_voice, speech_timestamps)
        """
        try:
            if not self.load_silero_vad():
                # Fallback to RMS VAD
                has_voice, _ = self.detect_voice_rms(audio_data)
                return has_voice, []
            
            import torch
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_data.flatten()).float()
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.silero_model,
                sampling_rate=self.sample_rate
            )
            
            has_voice = len(speech_timestamps) > 0
            
            logger.debug(f"Silero VAD: {len(speech_timestamps)} speech segments detected")
            return has_voice, speech_timestamps
            
        except Exception as e:
            logger.error(f"Silero VAD failed: {e}")
            # Fallback to RMS
            has_voice, _ = self.detect_voice_rms(audio_data)
            return has_voice, []
    
    def check_speech_quality(self, audio_data: np.ndarray) -> dict:
        """
        Analyze audio quality for speech
        
        Returns dict with quality metrics
        """
        try:
            # Duration check
            duration = len(audio_data) / self.sample_rate
            
            # RMS energy
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Zero crossing rate (indicator of voice vs noise)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data))))
            zcr = zero_crossings / len(audio_data)
            
            # Signal-to-noise estimation
            sorted_audio = np.sort(np.abs(audio_data))
            noise_floor = np.mean(sorted_audio[:len(sorted_audio)//4])
            signal_peak = np.max(np.abs(audio_data))
            snr = 20 * np.log10(signal_peak / (noise_floor + 1e-8))
            
            quality = {
                'duration': duration,
                'rms_energy': float(rms),
                'zero_crossing_rate': float(zcr),
                'snr_db': float(snr),
                'is_sufficient': duration >= self.min_speech_duration and rms > self.rms_threshold
            }
            
            logger.info(f"Speech quality: {quality}")
            return quality
            
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return {'is_sufficient': False}


class AudioProcessor:
    """
    Audio enhancement and preprocessing
    """
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    @staticmethod
    def remove_silence(audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove silence from beginning and end"""
        # Find first and last non-silent samples
        mask = np.abs(audio_data) > threshold
        if not np.any(mask):
            return audio_data
        
        indices = np.where(mask)[0]
        start_idx = indices[0]
        end_idx = indices[-1] + 1
        
        return audio_data[start_idx:end_idx]
    
    @staticmethod
    def apply_bandpass_filter(
        audio_data: np.ndarray, 
        sample_rate: int,
        lowcut: float = 300.0,
        highcut: float = 3400.0
    ) -> np.ndarray:
        """
        Apply bandpass filter to focus on speech frequencies
        Typical speech range: 300-3400 Hz
        """
        try:
            nyquist = sample_rate / 2
            low = lowcut / nyquist
            high = highcut / nyquist
            
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, audio_data.flatten())
            
            return filtered.reshape(audio_data.shape)
            
        except Exception as e:
            logger.error(f"Bandpass filter failed: {e}")
            return audio_data
    
    @staticmethod
    def apply_automatic_gain_control(audio_data: np.ndarray, target_level: float = 0.3) -> np.ndarray:
        """
        Apply Automatic Gain Control to boost quiet audio
        
        Args:
            audio_data: Input audio array
            target_level: Target RMS level (0.0-1.0)
            
        Returns:
            Gain-adjusted audio
        """
        try:
            # Calculate current RMS level
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Calculate gain needed
            if rms > 0:
                gain = target_level / rms
                
                # Limit maximum gain to prevent distortion (max 10x gain)
                gain = min(gain, 10.0)
                
                # Apply gain
                adjusted_audio = audio_data * gain
                
                # Prevent clipping by soft limiting
                max_val = np.max(np.abs(adjusted_audio))
                if max_val > 0.95:
                    # Soft compression for values above 0.95
                    over_limit = adjusted_audio > 0.95
                    adjusted_audio[over_limit] = 0.95 + (adjusted_audio[over_limit] - 0.95) * 0.1
                    
                    under_limit = adjusted_audio < -0.95
                    adjusted_audio[under_limit] = -0.95 + (adjusted_audio[under_limit] + 0.95) * 0.1
                
                logger.debug(f"AGC applied: rms {rms:.4f} -> {np.sqrt(np.mean(adjusted_audio ** 2)):.4f}, gain {gain:.2f}x")
                return adjusted_audio
            else:
                return audio_data
                
        except Exception as e:
            logger.warning(f"AGC failed: {e}")
            return audio_data
