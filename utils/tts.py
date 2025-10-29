"""
SecureX-Assist - Text-to-Speech System
Audio feedback and confirmation
"""

import pyttsx3
import logging
from typing import Optional
import threading

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Text-to-speech engine for audio feedback
    Provides voice confirmations and status updates
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('tts', {}).get('enabled', True)
        
        # TTS settings
        self.rate = config.get('tts', {}).get('rate', 150)
        self.volume = config.get('tts', {}).get('volume', 0.9)
        
        # Initialize engine
        self.engine: Optional[pyttsx3.Engine] = None
        self._lock = threading.Lock()
        
        # TTS queue for handling concurrent requests
        self._tts_queue = []
        self._tts_thread = None
        self._shutdown = False
        
        if self.enabled:
            self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize pyttsx3 engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voice for better clarity
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            logger.info("Text-to-Speech engine initialized")
            
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            self.enabled = False
    
    def speak(self, text: str, blocking: bool = False):
        """
        Speak text using TTS
        
        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        """
        if not self.enabled or not self.engine or not text or not text.strip():
            logger.debug(f"TTS disabled or no text, would say: {text}")
            return
        
        try:
            with self._lock:
                if blocking:
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    # Use a queue-based approach to avoid run loop conflicts
                    if not hasattr(self, '_tts_queue'):
                        self._tts_queue = []
                        self._tts_thread = None
                    
                    self._tts_queue.append(text)
                    self._process_queue_async()
                    
        except RuntimeError as e:
            if "run loop already started" in str(e):
                logger.warning("TTS run loop already started - queuing speech")
                # Queue the speech for later
                if not hasattr(self, '_tts_queue'):
                    self._tts_queue = []
                    self._tts_thread = None
                self._tts_queue.append(text)
                self._process_queue_async()
            else:
                logger.error(f"TTS speech failed: {e}")
        except Exception as e:
            logger.error(f"TTS speech failed: {e}")
    
    def _process_queue_async(self):
        """Process TTS queue asynchronously"""
        if self._tts_thread and self._tts_thread.is_alive():
            return  # Already processing
        
        self._tts_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._tts_thread.start()
    
    def _process_queue(self):
        """Process all queued TTS messages"""
        while self._tts_queue and not self._shutdown:
            try:
                text = self._tts_queue.pop(0)
                self._speak_blocking(text)
            except Exception as e:
                logger.error(f"Queued TTS failed: {e}")
                if self._shutdown:
                    break
    
    def _speak_blocking(self, text: str):
        """Speak text in a blocking manner with proper cleanup"""
        try:
            # Create a fresh engine instance for each speech
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            
            # Set voice if available
            try:
                voices = engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
            except:
                pass
            
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            
        except Exception as e:
            logger.error(f"Blocking TTS failed: {e}")
    
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
