"""
SecureX-Assist - Text-to-Speech System
Audio feedback and confirmation using Piper TTS
"""

import logging
from typing import Optional
import threading
import sounddevice as sd
import numpy as np
import pyttsx3 # Ensure this is installed as a fallback

logger = logging.getLogger("utils.tts") # Use conventional logger name
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - INFO - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class TextToSpeech:
    """
    Text-to-speech engine using Piper for natural voice synthesis
    Provides voice confirmations and status updates
    """

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('tts', {}).get('enabled', True)

        # Piper TTS settings
        self.model_path = config.get('tts', {}).get('model_path', 'en_US-lessac-medium.onnx')
        self.voice = None

        # Fallback TTS engine (pyttsx3)
        self._fallback_engine = None
        self._engine_lock = threading.Lock() # Lock for pyttsx3

        if self.enabled:
            self._initialize_voice()

    def _initialize_voice(self):
        """Initialize Piper voice"""
        try:
            # Try to load the model
            from piper import PiperVoice
            self.voice = PiperVoice.load(self.model_path)
            logger.info(f"Piper TTS initialized with model: {self.model_path}")
        except ImportError:
            logger.warning("piper-tts library not found. pip install piper-tts")
            logger.warning("Falling back to system TTS (pyttsx3).")
            self.voice = None
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS model ({self.model_path}): {e}")
            logger.info("Falling back to system TTS (pyttsx3)...")
            self.voice = None

    def speak(self, text: str, blocking: bool = False):
        """
        Speak text using the best available TTS engine.
        This function is thread-safe.

        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        """
        logger.info(f"DEBUG: Entered TTS.speak with text: {text}, blocking={blocking}")
        if not self.enabled or not text:
            logger.info("DEBUG: TTS.speak early exit (disabled or empty text)")
            return

        if self.voice:
            # --- Use Piper TTS ---
            try:
                # 1. Synthesize audio (this is CPU-bound and blocking)
                audio_data = self.voice.synthesize(text)

                # 2. Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # 3. Play audio
                if blocking:
                    # Play synchronously
                    sd.play(audio_array, samplerate=self.voice.config.sample_rate)
                    sd.wait()
                else:
                    # Play asynchronously
                    sd.play(audio_array, samplerate=self.voice.config.sample_rate)

                logger.info(f"Piper spoke: {text}")
                logger.info("DEBUG: Exiting TTS.speak after Piper TTS")

            except Exception as e:
                logger.error(f"Piper TTS failed: {e}")
                logger.info("DEBUG: Calling _fallback_tts from Piper TTS failure")
                self._fallback_tts(text, blocking)
        else:
            # --- Fallback to system TTS (pyttsx3) ---
            logger.info("DEBUG: Calling _fallback_tts from no Piper voice")
            self._fallback_tts(text, blocking)

    def _fallback_tts(self, text: str, blocking: bool = False):
        """Fallback TTS using pyttsx3 with proper engine management"""
        logger.info(f"DEBUG: Entered _fallback_tts with text: {text}, blocking={blocking}")

        def speak_task():
            # This function runs in a separate thread
            try:
                # pyttsx3 is not thread-safe, so we use a lock
                with self._engine_lock:
                    if self._fallback_engine is None:
                        logger.info("Initializing pyttsx3 fallback engine...")
                        self._fallback_engine = pyttsx3.init()
                        self._fallback_engine.setProperty('rate', 150)
                        self._fallback_engine.setProperty('volume', 0.9)
                        logger.info("pyttsx3 engine initialized.")
                    
                    if self._fallback_engine:
                        self._fallback_engine.say(text)
                        self._fallback_engine.runAndWait()
                        logger.info(f"Fallback TTS spoke: {text}")
                    else:
                        logger.error("Fallback TTS engine is None, cannot speak.")
            except Exception as e:
                logger.error(f"Fallback TTS thread failed: {e}")
            logger.info("DEBUG: Fallback TTS thread finished.")

        try:
            # Start the speak task in a new thread
            thread = threading.Thread(target=speak_task, daemon=True)
            thread.start()

            if blocking:
                # If blocking, wait for this new thread to finish
                thread.join()
                logger.info("DEBUG: Exiting _fallback_tts after blocking.")
            else:
                logger.info("DEBUG: Exiting _fallback_tts immediately (non-blocking).")

        except Exception as e:
            logger.error(f"Failed to start fallback TTS thread: {e}")
            logger.info("DEBUG: Exiting _fallback_tts after exception.")

    def speak_async(self, text: str):
        """Helper function: Speak text asynchronously"""
        self.speak(text, blocking=False)

    def speak_sync(self, text: str):
        """Helper function: Speak text synchronously"""
        self.speak(text, blocking=True)

    def shutdown(self):
        """Shutdown TTS system"""
        logger.info("Shutting down TTS engine...")
        self.enabled = False # Stop any new requests
        
        # Clean up fallback engine
        with self._engine_lock:
            if self._fallback_engine:
                try:
                    self._fallback_engine.stop()
                except Exception as e:
                    logger.warning(f"Error stopping pyttsx3 engine: {e}")
                self._fallback_engine = None
        
        # Stop any sounddevice playback
        try:
            sd.stop()
        except Exception as e:
            logger.warning(f"Error stopping sounddevice: {e}")
            
        logger.info("TTS shutdown complete.")

    # --- Start: Standard Assistant Phrases ---
    # These are just convenient wrappers for self.speak()

    def welcome(self, username: str = "user"):
        """Welcome message"""
        self.speak(f"Welcome to SecureX Assist, {username}")
    
    def authenticate_success(self):
        """Authentication success message"""
        self.speak("Authentication successful. Access granted.")
    
    def authenticate_failed(self):
        """Authentication failed message"""
        self.speak("Authentication failed. Access denied.")
    
    def voice_verification_started(self):
        """Voice verification started"""
        self.speak("Analyzing voice biometric signature")
    
    def voice_verification_passed(self):
        """Voice verification passed"""
        self.speak("Voice verified successfully")
    
    def voice_verification_failed(self):
        """Voice verification failed"""
        self.speak("Voice verification failed. Please try again.")
    
    def liveness_challenge(self, phrase: str):
        """Announce liveness challenge phrase"""
        self.speak(f"Please say the following phrase: {phrase}")
    
    def liveness_passed(self):
        """Liveness check passed"""
        self.speak("Liveness verification passed")
    
    def liveness_failed(self):
        """Liveness check failed"""
        self.speak("Liveness verification failed")
    
    def recording_started(self):
        """Recording started"""
        self.speak("Recording started. Please speak now.")
    
    def recording_complete(self):
        """Recording complete"""
        self.speak("Recording complete")
    
    def enrollment_started(self):
        """Enrollment started"""
        self.speak("Starting voice enrollment. Please speak clearly.")
    
    def enrollment_complete(self):
        """Enrollment complete"""
        self.speak("Voice enrollment complete")
    
    def error(self, message: str = "An error occurred"):
        """Error message"""
        self.speak(message)
    
    # --- End: Standard Assistant Phrases ---
