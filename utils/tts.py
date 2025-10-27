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
                    # For non-blocking, use a separate thread with its own engine
                    thread = threading.Thread(target=self._speak_async, args=(text,))
                    thread.daemon = True
                    thread.start()
                    
        except RuntimeError as e:
            if "run loop already started" in str(e):
                logger.warning("TTS run loop already started - using async method")
                # Fall back to async method
                thread = threading.Thread(target=self._speak_async, args=(text,))
                thread.daemon = True
                thread.start()
            else:
                logger.error(f"TTS speech failed: {e}")
        except Exception as e:
            logger.error(f"TTS speech failed: {e}")
    
    def _speak_async(self, text: str):
        """Async speech using a separate engine instance"""
        try:
            # Create a new engine instance for this thread
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
            logger.error(f"Async TTS failed: {e}")
            if "run loop already started" in str(e):
                logger.warning("TTS run loop already started in background thread - skipping speech")
            else:
                logger.debug(f"Background TTS failed: {e}")
        except Exception as e:
            logger.debug(f"Background TTS skipped: {e}")
    
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
        """Cleanup TTS engine"""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
