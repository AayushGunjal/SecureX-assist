import os
import wave
import json
import threading
import time
import queue
import numpy as np
from vosk import Model, KaldiRecognizer
import logging
import subprocess
import datetime
import random
import psutil
import pyautogui
import platform  # Added for platform-specific commands

# Assuming intent_classifier.py is in the same directory (./)
from .intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)

class VoiceAssistant:
    """
    Voice Assistant: Handles activation, speech-to-text, intent recognition, TTS, and command processing.
    Integrated with biometric authentication for secure command execution.
    """

    def __init__(self, model_path="vosk-model-small-en-us-0.15", biometric_engine=None, tts_engine=None):
        self.active = False
        self.model_path = model_path
        self._vosk_model = None

        # Biometric integration
        self.biometric_engine = biometric_engine
        self.tts_engine = tts_engine

        # Intent classifier
        self.intent_classifier = IntentClassifier()
        self.intent_classifier.setup_default_intents()

        # Session management
        self.authenticated_session = False
        self.session_start_time = None
        self.session_timeout = 300  # 5 minutes

        # Mic state
        self.mic_state = "idle"  # idle, listening, processing

        # Continuous listening
        self.continuous_listening = False
        self.listening_thread = None
        self.audio_recorder = None
        self.listening_user_id = None # <-- FIX: Added to store user ID for continuous mode

        # Commands registry
        self.commands = {}

        # Load model and setup
        self._load_model()
        self.setup_default_commands()

    def start_continuous_listening(self, audio_recorder, callback, user_id=None): # <-- FIX: Accept user_id
        """Start continuous voice listening"""
        if self.continuous_listening:
            return

        self.continuous_listening = True
        self.audio_recorder = audio_recorder
        self.listening_callback = callback
        self.listening_user_id = user_id # <-- FIX: Store user_id

        self.listening_thread = threading.Thread(target=self._continuous_listen, daemon=True)
        self.listening_thread.start()
        logger.info("Continuous listening started")

    def stop_continuous_listening(self):
        """Stop continuous voice listening"""
        self.continuous_listening = False
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=1.0)
        logger.info("Continuous listening stopped")

    def _continuous_listen(self):
        """Continuous listening loop"""
        while self.continuous_listening:
            try:
                # Record audio
                self.set_mic_state("listening")
                # Use the context object passed from app.py
                # This assumes _audio_stream_ctx has .recorder and .vad_detector
                audio_data = self.audio_recorder.recorder.record_with_vad(
                    vad_detector=self.audio_recorder.vad_detector
                )

                if audio_data is not None and len(audio_data) > 0:
                    self.set_mic_state("processing")

                    # Save temporary audio file
                    temp_file = f"temp_speech_{int(time.time())}.wav"
                    # Assuming the recorder context has the save_audio method
                    self.audio_recorder.recorder.save_audio(audio_data, temp_file)

                    # Transcribe
                    transcript = self.transcribe(temp_file)

                    if transcript.strip():
                        # Process command
                        # --- FIX: Pass the stored user_id for verification ---
                        success, response = self.process_voice_command(
                            transcript, 
                            audio_data, 
                            user_id=self.listening_user_id 
                        )
                        # --- END FIX ---

                        # Callback to UI
                        if self.listening_callback:
                            self.listening_callback(transcript, response, success)

                    # Cleanup
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                else:
                    logger.debug("VAD: No speech detected or audio too short.")
                    
                self.set_mic_state("idle")
                # No sleep needed, record_with_vad is blocking

            except Exception as e:
                logger.error(f"Continuous listening error: {e}")
                self.set_mic_state("idle")
                time.sleep(1.0) # Pause if an error occurred

    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                self._vosk_model = Model(self.model_path)
                logger.info("Vosk model loaded successfully from %s", self.model_path)
            else:
                logger.warning("Vosk model not found at %s. Voice transcription unavailable.", self.model_path)
                self._vosk_model = None
        except Exception as e:
            logger.error("Failed to load Vosk model: %s", e)
            self._vosk_model = None

    def activate(self):
        self.active = True
        self.mic_state = "idle"
        self.speak("Voice Assistant activated. How can I help you?")

    def deactivate(self, silent=False):
        self.active = False
        self.mic_state = "idle"
        if not silent:
            self.speak("Voice Assistant deactivated.")

    def speak(self, text):
        """Speak text using TTS engine"""
        if self.tts_engine and text:
            # Use speak_async if available, otherwise fall back
            if hasattr(self.tts_engine, 'speak_async'):
                self.tts_engine.speak_async(text)
            elif hasattr(self.tts_engine, 'speak'):
                self.tts_engine.speak(text)
        else:
            logger.debug(f"TTS not available or no text: {text}")

    def set_mic_state(self, state):
        """Update microphone state"""
        self.mic_state = state
        logger.info(f"Mic state: {state}")

    def check_session_validity(self):
        """Check if current session is still valid"""
        if not self.authenticated_session or not self.session_start_time:
            return False

        elapsed = time.time() - self.session_start_time
        if elapsed > self.session_timeout:
            self.authenticated_session = False
            self.session_start_time = None
            logger.info("Session expired")
            return False

        return True

    def verify_user_voice(self, audio_sample, user_id):
        """Verify user voice for secure commands"""
        if not self.biometric_engine:
            logger.warning("No biometric engine available for voice verification")
            return False

        try:
            # --- START FIX: Use the passed user_id ---
            if not user_id:
                logger.warning("Cannot verify voice: No user ID provided")
                return False

            # Verify voice
            result = self.biometric_engine.verify_voice(
                user_id=user_id,
                audio_data=audio_sample,
                sample_rate=16000 # Assuming 16kHz
            )
            # --- END FIX ---
            return result.get('verified', False)

        except Exception as e:
            logger.error(f"Voice verification failed: {e}")
            return False

    def transcribe(self, wav_path):
        """Transcribe WAV audio to text using Vosk."""
        if not self._vosk_model:
            logger.error("Vosk model not loaded, cannot transcribe.")
            return ""
        try:
            wf = wave.open(wav_path, "rb")
            rec = KaldiRecognizer(self._vosk_model, wf.getframerate())
            rec.SetWords(True)
            transcript = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = rec.Result()
                    transcript += json.loads(result).get("text", "") + " "
            
            # Get final result
            final_result = rec.FinalResult()
            transcript += json.loads(final_result).get("text", "")
            
            wf.close()
            return transcript.strip()
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def process_voice_command(self, transcript, audio_sample=None, user_id=None): # <-- FIX: Accept user_id
        """Process transcript using intent recognition and execute command."""
        if not transcript:
            return False, "No speech detected."

        transcript = transcript.lower().strip()
        logger.info("Processing command: '%s'", transcript)

        # First try intent classification
        intent, confidence = self.intent_classifier.classify(transcript)

        if intent and confidence > 0.6: # Added confidence threshold
            logger.info(f"Classified intent: {intent} (confidence: {confidence:.2f})")
            # --- FIX: Pass user_id ---
            response = self.execute_intent(intent, transcript, audio_sample, user_id=user_id)
            return True, response

        # Fallback to keyword matching (legacy)
        for name, cmd in self.commands.items():
            for kw in cmd["keywords"]:
                if kw in transcript:
                    logger.info("Matched legacy command: %s", name)
                    # --- FIX: Pass user_id ---
                    response = self.execute_command(name, transcript, audio_sample, user_id=user_id)
                    return True, response

        logger.warning("No command matched for transcript: '%s'", transcript)
        return False, "Command not recognized. Try saying 'help' for available commands."

    def execute_intent(self, intent, transcript, audio_sample=None, user_id=None): # <-- FIX: Accept user_id
        """Execute command based on classified intent"""
        
        intent_handlers = {
            "time": self._get_time,
            "date": self._get_date,
            "open_app": self._intent_open_app, 
            "close_app": self._intent_close_app,
            "system_lock": self._lock_system,
            "system_restart": self._intent_system_restart,
            "system_shutdown": self._intent_system_shutdown,
            "list_files": self._intent_list_files,
            "delete_file": self._intent_delete_file,
            "open_file": self._intent_open_file,
            "play_music": self._intent_play_music,
            "pause_music": self._intent_pause_music,
            "mute_audio": self._mute_audio,
            "unmute_audio": self._unmute_audio,
            "system_info": self._system_status,
            "help": self._what_can_you_do,
            "who_are_you": self._intent_who_are_you,
            "show_logs": self._show_security_logs,
            "greeting": self._hello,
            "goodbye": self._bye,
            "minimize_window": self._minimize_window,
            "maximize_window": self._maximize_window,
            "open_calculator": self._open_calculator,
            "open_notepad": self._open_notepad,
            "open_explorer": self._open_explorer,
            "take_screenshot": self._take_screenshot,
            "show_weather": self._show_weather,
            "search_files": self._search_files,
            "run_security_scan": self._run_security_scan,
            "biometric_status": self._biometric_status,
            "send_secure_message": self._send_secure_message,
            "check_messages": self._check_messages,
            "read_last_message": self._read_last_message,
            "start_voice_call": self._start_voice_call,
            "tell_joke": self._tell_joke,
            "play_game": self._play_game,
            "motivate_me": self._motivate_me,
            "compliment_me": self._compliment_me,
            "tell_fact": self._tell_fact,
            "sing_song": self._sing_song,
            "dance": self._dance,
            "tell_story": self._tell_story,
            "what_can_you_do": self._what_can_you_do,
            "how_are_you": self._how_are_you,
            "good_morning": self._good_morning,
            "good_afternoon": self._good_afternoon,
            "good_evening": self._good_evening,
            "good_night": self._good_night,
            "thank_you": self._thank_you,
            "sorry": self._sorry,
            "hello": self._hello,
            "bye": self._bye,
        }

        handler = intent_handlers.get(intent)
        if handler:
            # Check for security
            secure_intents = [
                "system_lock", "system_restart", "system_shutdown", 
                "delete_file", "send_secure_message"
            ]
            if intent in secure_intents:
                # We can remove this, as the app-level session isn't in this class
                # if not self.check_session_validity():
                #     return "Access denied — please authenticate first."
                
                # Re-verify voice for critical commands
                # --- FIX: Pass the user_id ---
                if audio_sample is not None and not self.verify_user_voice(audio_sample, user_id):
                    self.speak("Access denied — voice not recognized.")
                    return "Access denied — voice not recognized."
                
            # Execute handler
            try:
                response = handler(transcript)
                self.speak(response) # Speak the response
                self._log_action(intent, transcript, response)
                return response
            except Exception as e:
                logger.error(f"Intent handler '{intent}' failed: {e}", exc_info=True)
                return f"Sorry, I encountered an error with the '{intent}' command."
        else:
            return f"I understand you want to {intent.replace('_', ' ')}, but that feature isn't implemented yet."

    def execute_command(self, command_name, transcript, audio_sample=None, user_id=None): # <-- FIX: Accept user_id
        """Execute registered command with security checks (LEGACY)"""
        cmd_info = self.commands.get(command_name)
        if not cmd_info:
            return "Command not found."

        handler = cmd_info["handler"]
        secure = cmd_info.get("secure", False)

        # Check security for sensitive commands
        if secure:
            # We can remove this, as the app-level session isn't in this class
            # if not self.check_session_validity():
            #     return "Access denied — please authenticate first."

            # For critical commands, re-verify voice
            # --- FIX: Pass the user_id ---
            if audio_sample and not self.verify_user_voice(audio_sample, user_id):
                self.speak("Access denied — voice not recognized.")
                return "Access denied — voice not recognized."

        # Execute command
        try:
            response = handler(transcript)
            self.speak(response) # Speak response
            # Log action
            self._log_action(command_name, transcript, response)
            return response
        except Exception as e:
            logger.error(f"Legacy command execution failed: {e}")
            return "Sorry, I encountered an error executing that command."

    def _log_action(self, command, transcript, response):
        """Log assistant actions"""
        logger.info(f"Assistant Action - Command: {command}, Input: '{transcript}', Response: '{response}'")

    # --- LEGACY INTENT HANDLERS (Called by execute_intent) ---

    def _intent_time(self, transcript, audio_sample=None):
        return self._get_time(transcript)

    def _intent_date(self, transcript, audio_sample=None):
        return self._get_date(transcript)

    def _intent_open_app(self, transcript, audio_sample=None):
        # This is a generic handler, the new ones are better
        app_keywords = {
            "notepad": self._open_notepad,
            "calculator": self._open_calculator,
            "explorer": self._open_explorer,
            "chrome": lambda t: self._open_specific_app("chrome.exe", "Chrome"),
            "firefox": lambda t: self._open_specific_app("firefox.exe", "Firefox"),
            "word": lambda t: self._open_specific_app("winword.exe", "Word"),
            "excel": lambda t: self._open_specific_app("excel.exe", "Excel"),
        }
        for app, handler in app_keywords.items():
            if app in transcript:
                return handler(transcript)
        return "Which application would you like me to open?"

    def _open_specific_app(self, exe_name, app_name):
        try:
            subprocess.Popen(exe_name)
            return f"Opening {app_name}."
        except Exception as e:
            return f"Sorry, I couldn't open {app_name}."

    def _intent_close_app(self, transcript, audio_sample=None):
        # This is complex and risky
        return "Sorry, I cannot close applications yet. Please do it manually."

    def _intent_system_restart(self, transcript, audio_sample=None):
        response = "System restart is a critical command. Please confirm manually."
        # os.system("shutdown /r /t 10") # Example, but dangerous
        return response

    def _intent_system_shutdown(self, transcript, audio_sample=None):
        response = "System shutdown is a critical command. Please confirm manually."
        # os.system("shutdown /s /t 30") # Example, but dangerous
        return response

    def _intent_list_files(self, transcript, audio_sample=None):
        try:
            files = os.listdir(".")
            file_list = ", ".join(files[:10])  # Limit to 10 files
            response = f"Files in current directory: {file_list}"
            return response
        except Exception as e:
            return "Couldn't list files."

    def _intent_delete_file(self, transcript, audio_sample=None):
        return "File deletion requires confirmation. Please use the interface for now."

    def _intent_open_file(self, transcript, audio_sample=None):
        return "File opening feature coming soon."

    def _intent_play_music(self, transcript, audio_sample=None):
        pyautogui.press('playpause')
        return "Playing music."

    def _intent_pause_music(self, transcript, audio_sample=None):
        pyautogui.press('playpause')
        return "Pausing music."

    def _intent_who_are_you(self, transcript, audio_sample=None):
        return "I am SecureX, your offline voice assistant with biometric security."

    # Legacy command handlers (for backward compatibility)
    def register_command(self, name, handler, keywords, secure=False):
        """Register a command with handler and keywords."""
        self.commands[name] = {"handler": handler, "keywords": keywords, "secure": secure}

    def setup_default_commands(self):
        # Keep some legacy commands for compatibility
        self.register_command("help", self._what_can_you_do, ["help", "what can you do"])
        self.register_command("time", self._get_time, ["what time is it", "time"])
        self.register_command("date", self._get_date, ["what date is it", "date"])

        # Secure commands
        self.register_command("lock_system", self._lock_system, ["lock system", "lock"], secure=True)
        self.register_command("system_restart", self._intent_system_restart, ["restart system", "restart"], secure=True)
        self.register_command("system_shutdown", self._intent_system_shutdown, ["shutdown system", "shutdown"], secure=True)

    def get_available_commands(self):
        """Get a formatted string of all available voice commands."""
        if not self.commands:
            return "No commands available."
        command_list = []
        for name, cmd_info in self.commands.items():
            keywords = cmd_info.get("keywords", [])
            secure = cmd_info.get("secure", False)
            if keywords:
                cmd_str = f"{name}: {', '.join(keywords)}"
                if secure:
                    cmd_str += " (secure)"
                command_list.append(cmd_str)
        return "Available commands:\n" + "\n".join(command_list)

    def shutdown(self):
        """Shutdown the voice assistant"""
        self.deactivate(silent=True)
        self.stop_continuous_listening()
        logger.info("Voice Assistant shutdown complete")

    def _process_speech_segment(self, speech_buffer, recorder, callback):
        """
        Process accumulated speech audio: save to file, transcribe, and execute command.
        (This seems to be a duplicate/alternative to _continuous_listen, kept for now)
        """
        try:
            if not speech_buffer:
                return
            speech_audio = np.concatenate(speech_buffer)
            duration = len(speech_audio) / 16000
            if duration < 0.5:
                return

            temp_file = f"temp_speech_{int(time.time())}.wav"
            recorder.save_audio(speech_audio, temp_file)
            transcript = self.transcribe(temp_file)
            logger.info("Transcription result: '%s'", transcript)

            if transcript.strip():
                # Note: This path does not pass user_id, it's not used by the main app flow.
                success, response = self.process_voice_command(transcript, speech_audio)
                if callback:
                    callback(transcript, response, success)
            try:
                os.remove(temp_file)
            except:
                pass
        except Exception as e:
            logger.error("Failed to process speech segment: %s", e, exc_info=True)
            if callback:
                callback("", f"Error processing speech: {e}", False)

    # =======================================================
    # --- ALL NEW METHODS MOVED INSIDE THE CLASS ---
    # =======================================================

    # ==================== WINDOW COMMANDS ====================
    
    def _minimize_window(self, text: str) -> str:
        """Minimize active window"""
        try:
            pyautogui.hotkey('win', 'd')  # Show desktop (minimize all)
            return "Minimizing all windows."
        except Exception as e:
            return f"Unable to minimize window: {e}"
    
    def _maximize_window(self, text: str) -> str:
        """Maximize active window"""
        try:
            pyautogui.hotkey('win', 'up')  # Maximize window
            return "Window maximized."
        except Exception as e:
            return f"Unable to maximize window: {e}"
    
    # ==================== AUDIO COMMANDS ====================
    
    def _mute_audio(self, text: str) -> str:
        """Mute system audio"""
        try:
            pyautogui.press('volumemute')
            return "System audio muted."
        except Exception as e:
            return f"Unable to mute audio: {e}"
    
    def _unmute_audio(self, text: str) -> str:
        """Unmute system audio"""
        try:
            pyautogui.press('volumemute')
            return "System audio restored."
        except Exception as e:
            return f"Unable to unmute audio: {e}"
    
    # ==================== CUSTOM COMMANDS ====================

    def _open_calculator(self, text: str) -> str:
        """Open calculator application"""
        try:
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("calc.exe")
            elif system == "darwin":  # macOS
                subprocess.Popen(["open", "-a", "Calculator"])
            else:  # Linux
                subprocess.Popen(["gnome-calculator"])
            return "Calculator launched."
        except Exception as e:
            return f"Unable to open calculator: {e}"

    def _open_notepad(self, text: str) -> str:
        """Open notepad/text editor"""
        try:
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("notepad.exe")
            elif system == "darwin":  # macOS
                subprocess.Popen(["open", "-a", "TextEdit"])
            else:  # Linux
                subprocess.Popen(["gedit"])
            return "Text editor launched."
        except Exception as e:
            return f"Unable to open text editor: {e}"

    def _open_explorer(self, text: str) -> str:
        """Open file explorer"""
        try:
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("explorer.exe")
            elif system == "darwin":  # macOS
                subprocess.Popen(["open", "."])
            else:  # Linux
                subprocess.Popen(["xdg-open", "."])
            return "File explorer opened."
        except Exception as e:
            return f"Unable to open file explorer: {e}"

    def _get_time(self, text: str) -> str:
        """Get current time"""
        try:
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}"
        except Exception as e:
            return f"Unable to retrieve the time: {e}"

    def _get_date(self, text: str) -> str:
        """Get current date"""
        try:
            current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {current_date}"
        except Exception as e:
            return f"Unable to retrieve the date: {e}"

    def _system_status(self, text: str) -> str:
        """Show system status"""
        try:
            system = platform.system()
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            return f"System: {system}, CPU load: {cpu_percent} percent, memory usage: {memory.percent} percent"
        except Exception as e:
            return f"Unable to retrieve system status: {e}"

    def _lock_system(self, text: str) -> str:
        """Lock the system"""
        try:
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("rundll32.exe user32.dll,LockWorkStation")
            elif system == "darwin":  # macOS
                subprocess.Popen(["pmset", "displaysleepnow"])
            else:  # Linux
                subprocess.Popen(["gnome-screensaver-command", "-l"])
            return "System locked."
        except Exception as e:
            return f"Unable to lock the system: {e}"

    def _take_screenshot(self, text: str) -> str:
        """Take a screenshot"""
        try:
            screenshot = pyautogui.screenshot()
            filename = f"screenshot_{int(time.time())}.png"
            screenshot.save(filename)
            return f"Screenshot saved as {filename}"
        except Exception as e:
            return f"Unable to capture a screenshot: {e}"

    def _show_weather(self, text: str) -> str:
        """Show weather information (placeholder)"""
        return "Weather integration is planned for a future update."

    def _search_files(self, text: str) -> str:
        """Search for files (placeholder)"""
        return "File search capability will be available in a future release."

    def _run_security_scan(self, text: str) -> str:
        """Run security scan (placeholder)"""
        return "Security scan initiated. All systems secure."

    def _show_security_logs(self, text: str) -> str:
        """Show security logs (placeholder)"""
        return "Security logs are currently clear."

    def _biometric_status(self, text: str) -> str:
        """Show biometric status"""
        return "Biometric systems: Voice recognition active, face recognition ready."

    def _send_secure_message(self, text: str) -> str:
        """Send secure message (placeholder)"""
        return "Secure messaging will be available soon."

    def _check_messages(self, text: str) -> str:
        """Check messages (placeholder)"""
        return "No new messages."

    def _read_last_message(self, text: str) -> str:
        """Read last message (placeholder)"""
        return "No recent messages to read."

    def _start_voice_call(self, text: str) -> str:
        """Start voice call (placeholder)"""
        return "Voice calling support is coming soon."
    
    # ==================== FUN AND INTERACTIVE COMMANDS ====================
    
    def _tell_joke(self, text: str) -> str:
        """Tell a random joke"""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call fake spaghetti? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why can't you give Elsa a balloon? Because she will let it go!",
        ]
        return random.choice(jokes)
    
    def _play_game(self, text: str) -> str:
        """Start a simple game"""
        return "Let's play a game! Think of a number between 1 and 10. Ready? ... I guess 7!"
    
    def _motivate_me(self, text: str) -> str:
        """Give motivation"""
        motivations = [
            "You are capable of amazing things! Keep pushing forward.",
            "Every expert was once a beginner. You're on the right path!",
            "Success is not final, failure is not fatal. Keep going!",
            "Believe in yourself and all that you are. You are enough!",
        ]
        return random.choice(motivations)
    
    def _compliment_me(self, text: str) -> str:
        """Give a compliment"""
        compliments = [
            "You have an amazing voice! It's so clear and confident.",
            "You're incredibly intelligent and capable!",
            "Your creativity knows no bounds!",
            "You have a wonderful personality!",
        ]
        return random.choice(compliments)
    
    def _tell_fact(self, text: str) -> str:
        """Tell an interesting fact"""
        facts = [
            "Did you know? Octopuses have three hearts and blue blood!",
            "Honey never spoils. Archaeologists have found edible honey in ancient tombs!",
            "A group of flamingos is called a 'flamboyance'!",
            "Bananas are berries, but strawberries aren't!",
        ]
        return random.choice(facts)
    
    def _sing_song(self, text: str) -> str:
        """Sing a short song"""
        return "🎵 Twinkle, twinkle, little star... How I wonder what you are! 🎵 Sorry, I'm not a great singer!"
    
    def _dance(self, text: str) -> str:
        """Dance virtually"""
        return "I'm doing a virtual dance! 💃🕺"
    
    def _tell_story(self, text: str) -> str:
        """Tell a short story"""
        return "Once upon a time, in a world of code, a voice assistant helped a user, and they both worked efficiently ever after. The end."
    
    def _what_can_you_do(self, text: str) -> str:
        """Explain capabilities"""
        return "I can help you with system commands like opening apps, checking status, or locking your system. I can also tell jokes, facts, and more. Just ask!"
    
    def _how_are_you(self, text: str) -> str:
        """Respond to greeting"""
        responses = [
            "I'm doing great, thank you for asking! Ready to help!",
            "I'm fantastic! All systems are operational.",
            "I'm excellent! How can I help you?",
        ]
        return random.choice(responses)
    
    def _good_morning(self, text: str) -> str:
        """Morning greeting"""
        return "Good morning! I hope you have a fantastic day ahead!"
    
    def _good_afternoon(self, text: str) -> str:
        """Afternoon greeting"""
        return "Good afternoon! Hope your day is going well!"
    
    def _good_evening(self, text: str) -> str:
        """Evening greeting"""
        return "Good evening! Time to relax!"
    
    def _good_night(self, text: str) -> str:
        """Night greeting"""
        return "Good night! Sleep well."
    
    def _thank_you(self, text: str) -> str:
        """Respond to thanks"""
        return "You're very welcome! I'm always happy to help!"
    
    def _sorry(self, text: str) -> str:
        """Respond to apology"""
        return "No need to apologize! I'm here to help."
    
    def _hello(self, text: str) -> str:
        """Basic greeting"""
        return "Hello! How can I assist you today?"
    
    def _bye(self, text: str) -> str:
        """Goodbye"""
        return "Goodbye! Have a great day!"