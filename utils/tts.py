"""
SecureX-Assist - Text-to-Speech System
Audio feedback and confirmation using Piper TTS
"""

import logging
from typing import Optional
import threading
import sounddevice as sd
import numpy as np

logger = logging.getLogger(__name__)

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
        self._engine_lock = threading.Lock()

        # TTS queue for handling concurrent requests
        self._tts_queue = []
        self._tts_thread = None
        self._shutdown = False
        self._lock = threading.Lock()

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
            logger.warning("Piper TTS not available, using fallback")
            self.voice = None
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}")
            logger.info("Falling back to system TTS...")
            self.voice = None

    def speak(self, text: str, blocking: bool = False):
        logger.info(f"DEBUG: Entered TTS.speak with text: {text}, blocking={blocking}")
        """
        Speak text using Piper TTS

        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        """
        if not self.enabled or not text:
            logger.info("DEBUG: TTS.speak early exit (disabled or empty text)")
            return

        if self.voice:
            # Use Piper TTS
            try:
                # Generate audio data
                audio_data = self.voice.synthesize(text)

                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                if blocking:
                    # Play synchronously
                    sd.play(audio_array, samplerate=22050)
                    sd.wait()
                else:
                    # Play asynchronously
                    sd.play(audio_array, samplerate=22050)

                logger.info(f"Spoke: {text}")
                logger.info("DEBUG: Exiting TTS.speak after Piper TTS")

            except Exception as e:
                logger.error(f"Piper TTS failed: {e}")
                logger.info("DEBUG: Calling _fallback_tts from Piper TTS failure")
                self._fallback_tts(text, blocking)
        else:
            logger.info("DEBUG: Calling _fallback_tts from no Piper voice")
            # Fallback to system TTS
            self._fallback_tts(text, blocking)

    def _fallback_tts(self, text: str, blocking: bool = False):
        logger.info(f"DEBUG: Entered _fallback_tts with text: {text}, blocking={blocking}")
        """Fallback TTS using pyttsx3 with proper engine management"""
        try:
            import pyttsx3

            with self._engine_lock:
                if self._fallback_engine is None:
                    self._fallback_engine = pyttsx3.init()
                    self._fallback_engine.setProperty('rate', 150)
                    self._fallback_engine.setProperty('volume', 0.9)

                if blocking:
                    self._fallback_engine.say(text)
                    self._fallback_engine.runAndWait()
                else:
                    # For async, we need to handle threading carefully
                    def speak_async():
                        try:
                            with self._engine_lock:
                                if self._fallback_engine:
                                    self._fallback_engine.say(text)
                                    self._fallback_engine.runAndWait()
                        except Exception as e:
                            logger.error(f"Async fallback TTS failed: {e}")

                    thread = threading.Thread(target=speak_async, daemon=True)
                    thread.start()

            logger.info(f"Fallback TTS spoke: {text}")
            logger.info("DEBUG: Exiting _fallback_tts after speaking")

        except Exception as e:
            logger.error(f"Fallback TTS failed: {e}")
        logger.info("DEBUG: Exiting _fallback_tts after exception")

    def speak_async(self, text: str):
        """Speak text asynchronously"""
        self.speak(text, blocking=False)

    def speak_sync(self, text: str):
        """Speak text synchronously"""
        self.speak(text, blocking=True)

    def shutdown(self):
        """Shutdown TTS system"""
        self._shutdown = True
        if self.voice:
            # Piper voice cleanup if needed
            pass
        
        # Clean up fallback engine
        with self._engine_lock:
            if self._fallback_engine:
                try:
                    self._fallback_engine.stop()
                except:
                    pass
                self._fallback_engine = None
    
    def _speak_async(self, text: str):
        """Legacy async method - now uses queue"""
        self._speak_blocking(text)
    
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
    
    def shutdown(self):
        """Cleanup TTS engine and stop queue processing"""
        self._shutdown = True
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
        
        # Clear queue and wait for thread to finish
        self._tts_queue.clear()
        if self._tts_thread and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=2.0)
