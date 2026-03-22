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
import time
import sys

logger = logging.getLogger("utils.tts") # Use conventional logger name
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - INFO - %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Add console output for debugging
print("[TTS] Module loaded")


class TextToSpeech:
    """
    Text-to-speech engine using Piper for natural voice synthesis
    Provides voice confirmations and status updates
    """

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('tts', {}).get('enabled', True)
        
        print(f"\n{'='*60}")
        print(f"[TTS] Initialization Started")
        print(f"[TTS] Enabled: {self.enabled}")
        logger.info(f"🔊 TTS Initialization - Enabled: {self.enabled}")

        # Piper TTS settings
        self.model_path = config.get('tts', {}).get('model_path', 'en_US-lessac-medium.onnx')
        print(f"[TTS] Model Path: {self.model_path}")
        logger.info(f"🔊 TTS Model Path: {self.model_path}")
        self.voice = None

        # Fallback TTS engine (pyttsx3)
        self._fallback_engine = None
        self._engine_lock = threading.Lock() # Lock for pyttsx3
        self._engine_ready = False

        if self.enabled:
            self._initialize_voice()
            # Initialize fallback immediately for reliability and speed
            print("[TTS] Pre-initializing fallback engine for faster response...")
            self._init_fallback_engine()
        else:
            print("[TTS] WARNING: TTS is DISABLED in configuration!")
            logger.warning("⚠️ TTS is DISABLED in configuration!")
        
        print(f"{'='*60}\n")

    def _initialize_voice(self):
        """Initialize Piper voice"""
        try:
            # Try to load the model
            from piper import PiperVoice
            import os
            if os.path.exists(self.model_path):
                self.voice = PiperVoice.load(self.model_path)
                print(f"[TTS] ✅ Piper TTS initialized with model: {self.model_path}")
                logger.info(f"Piper TTS initialized with model: {self.model_path}")
            else:
                print(f"[TTS] ❌ Piper model not found: {self.model_path}")
                logger.error(f"Piper model file not found: {self.model_path}")
                self.voice = None
        except ImportError:
            print("[TTS] ⚠️ piper-tts library not found. Using fallback.")
            logger.warning("piper-tts library not found. pip install piper-tts")
            logger.warning("Falling back to system TTS (pyttsx3).")
            self.voice = None
        except Exception as e:
            print(f"[TTS] ❌ Failed to initialize Piper: {e}")
            logger.error(f"Failed to initialize Piper TTS model ({self.model_path}): {e}")
            logger.info("Falling back to system TTS (pyttsx3)...")
            self.voice = None
    
    def _init_fallback_engine(self):
        """Pre-initialize fallback engine for faster response"""
        if self._engine_ready:
            return
        
        try:
            with self._engine_lock:
                if sys.platform == 'win32':
                    print("[TTS] Initializing Windows SAPI engine...")
                    try:
                        self._fallback_engine = pyttsx3.init('sapi5')
                    except:
                        print("[TTS] SAPI5 failed, trying default engine...")
                        self._fallback_engine = pyttsx3.init()
                else:
                    self._fallback_engine = pyttsx3.init()
                
                # Configure engine
                self._fallback_engine.setProperty('rate', 150)
                self._fallback_engine.setProperty('volume', 0.9)
                self._engine_ready = True
                print("[TTS] ✅ Fallback engine ready")
                logger.info("Fallback TTS engine initialized successfully")
        except Exception as e:
            print(f"[TTS] ❌ Failed to initialize fallback: {e}")
            logger.error(f"Failed to initialize fallback engine: {e}")
            self._fallback_engine = None
            self._engine_ready = False

    def speak(self, text: str, blocking: bool = False):
        """
        Speak text using the best available TTS engine.
        This function is thread-safe.

        Args:
            text: Text to speak
            blocking: If True, wait for speech to complete
        """
        print(f"\n[TTS] 🔊 SPEAK: '{text}'")
        logger.info(f"DEBUG: Entered TTS.speak with text: {text}, blocking={blocking}")
        
        if not self.enabled:
            print("[TTS] ⚠️ TTS is disabled")
            logger.info("DEBUG: TTS.speak early exit (disabled)")
            return
        
        if not text or not text.strip():
            print("[TTS] ⚠️ Empty text")
            logger.info("DEBUG: TTS.speak early exit (empty text)")
            return

        if self.voice:
            # --- Use Piper TTS ---
            try:
                print("[TTS] Using Piper TTS...")
                # 1. Synthesize audio - Piper returns a generator that yields AudioChunk objects
                audio_chunks = []
                for chunk in self.voice.synthesize(text):
                    # Each chunk is an AudioChunk object with 'audio' attribute containing bytes
                    if hasattr(chunk, 'audio'):
                        audio_chunks.append(chunk.audio)
                    elif isinstance(chunk, bytes):
                        audio_chunks.append(chunk)
                    elif hasattr(chunk, 'data'):
                        audio_chunks.append(chunk.data)
                    elif hasattr(chunk, '__bytes__'):
                        audio_chunks.append(chunk.__bytes__())
                    else:
                        # Skip unknown chunk types silently
                        continue
                
                # Combine all chunks into a single bytes object
                audio_data = b''.join(audio_chunks)
                
                if not audio_data:
                    raise ValueError("No audio data generated")

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

                print(f"[TTS] ✅ Piper spoke: {text}")
                logger.info(f"Piper spoke: {text}")
                logger.info("DEBUG: Exiting TTS.speak after Piper TTS")
                return  # Success

            except Exception as e:
                print(f"[TTS] ❌ Piper failed: {e}")
                logger.error(f"Piper TTS failed: {e}")
                logger.info("DEBUG: Calling _fallback_tts from Piper TTS failure")
                self._fallback_tts(text, blocking)
        else:
            # --- Fallback to system TTS (pyttsx3) ---
            logger.info("DEBUG: Calling _fallback_tts from no Piper voice")
            self._fallback_tts(text, blocking)

    def _fallback_tts(self, text: str, blocking: bool = False):
        """Fallback TTS using Windows SAPI directly (most reliable)"""
        print(f"[TTS] 🔄 Using fallback TTS for: '{text}'")
        logger.info(f"DEBUG: Entered _fallback_tts with text: {text}, blocking={blocking}")

        def speak_task():
            # This function runs in a separate thread
            success = False
            
            # Method 1: Try Windows SAPI direct (most reliable on Windows)
            if sys.platform == 'win32' and not success:
                try:
                    print("[TTS] Trying direct Windows SAPI...")
                    import win32com.client
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    speaker.Speak(text)
                    print(f"[TTS] ✅ Direct SAPI spoke: {text}")
                    logger.info(f"Windows SAPI spoke: {text}")
                    success = True
                except Exception as sapi_e:
                    print(f"[TTS] ❌ Direct SAPI failed: {sapi_e}")
                    logger.error(f"Windows SAPI failed: {sapi_e}")
            
            # Method 2: Try pyttsx3 if SAPI failed
            if not success:
                try:
                    with self._engine_lock:
                        if self._fallback_engine is None and not self._engine_ready:
                            print("[TTS] Initializing pyttsx3 fallback engine...")
                            logger.info("Initializing pyttsx3 fallback engine...")
                            
                            if sys.platform == 'win32':
                                try:
                                    self._fallback_engine = pyttsx3.init('sapi5')
                                    print("[TTS] Using pyttsx3 with SAPI5")
                                except:
                                    self._fallback_engine = pyttsx3.init()
                            else:
                                self._fallback_engine = pyttsx3.init()
                            
                            # Optimized settings for faster speech
                            self._fallback_engine.setProperty('rate', 180)  # Increased from 150 for faster speech
                            self._fallback_engine.setProperty('volume', 0.9)
                            self._engine_ready = True
                            print("[TTS] ✅ pyttsx3 engine initialized (optimized)")
                        
                        if self._fallback_engine:
                            print(f"[TTS] 🔊 Speaking with pyttsx3: '{text}'")
                            self._fallback_engine.say(text)
                            self._fallback_engine.runAndWait()
                            print(f"[TTS] ✅ pyttsx3 spoke: {text}")
                            logger.info(f"pyttsx3 spoke: {text}")
                            success = True
                        else:
                            print("[TTS] ❌ pyttsx3 engine is None")
                except Exception as e:
                    print(f"[TTS] ❌ pyttsx3 error: {e}")
                    logger.error(f"pyttsx3 failed: {e}")
            
            if not success:
                print(f"[TTS] ❌ All TTS methods failed for: '{text}'")
                logger.error(f"All TTS methods failed for: '{text}'")
            
            logger.info("DEBUG: Fallback TTS thread finished.")

        try:
            # Start the speak task in a new thread
            thread = threading.Thread(target=speak_task, daemon=False)
            thread.start()

            if blocking:
                # If blocking, wait for this new thread to finish
                thread.join(timeout=10)
                logger.info("DEBUG: Exiting _fallback_tts after blocking.")
            else:
                # Give it a moment to start
                time.sleep(0.1)
                logger.info("DEBUG: Exiting _fallback_tts immediately (non-blocking).")

        except Exception as e:
            print(f"[TTS] ❌ Failed to start fallback thread: {e}")
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
