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

        # Commands registry
        self.commands = {}

        # Load model and setup
        self._load_model()
        self.setup_default_commands()

    def start_continuous_listening(self, audio_recorder, callback):
        """Start continuous voice listening"""
        if self.continuous_listening:
            return

        self.continuous_listening = True
        self.audio_recorder = audio_recorder
        self.listening_callback = callback

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
                audio_path = self.audio_recorder.record_audio(duration=3.0, silence_threshold=0.5)

                if audio_path and os.path.exists(audio_path):
                    self.set_mic_state("processing")

                    # Transcribe
                    transcript = self.transcribe(str(audio_path))

                    if transcript.strip():
                        # Process command
                        success, response = self.process_voice_command(transcript)

                        # Callback to UI
                        if self.listening_callback:
                            self.listening_callback(transcript, response, success)

                    # Cleanup
                    try:
                        os.remove(audio_path)
                    except:
                        pass

                self.set_mic_state("idle")
                time.sleep(0.5)  # Brief pause between listens

            except Exception as e:
                logger.error(f"Continuous listening error: {e}")
                self.set_mic_state("idle")
                time.sleep(1.0)

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
            self.tts_engine.speak_async(text)
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

    def verify_user_voice(self, audio_sample):
        """Verify user voice for secure commands"""
        if not self.biometric_engine:
            logger.warning("No biometric engine available for voice verification")
            return False

        try:
            # Get current user from session
            current_user = getattr(self.biometric_engine, 'current_user', None)
            if not current_user:
                return False

            # Verify voice
            is_match, similarity, details = self.biometric_engine.verify_user(audio_sample, current_user)
            return is_match

        except Exception as e:
            logger.error(f"Voice verification failed: {e}")
            return False

    def transcribe(self, wav_path):
        """Transcribe WAV audio to text using Vosk."""
        if not self._vosk_model:
            return ""
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
        wf.close()
        return transcript.strip()

    def process_voice_command(self, transcript, audio_sample=None):
        """Process transcript using intent recognition and execute command."""
        if not transcript:
            return False, "No speech detected."

        transcript = transcript.lower().strip()
        logger.info("Processing command: '%s'", transcript)

        # First try intent classification
        intent, confidence = self.intent_classifier.classify(transcript)

        if intent:
            logger.info(f"Classified intent: {intent} (confidence: {confidence:.2f})")
            response = self.execute_intent(intent, transcript, audio_sample)
            return True, response

        # Fallback to keyword matching
        for name, cmd in self.commands.items():
            for kw in cmd["keywords"]:
                if kw in transcript:
                    logger.info("Matched command: %s", name)
                    response = self.execute_command(name, transcript, audio_sample)
                    return True, response

        logger.warning("No command matched for transcript: '%s'", transcript)
        return False, "Command not recognized. Try saying 'help' for available commands."

    def execute_intent(self, intent, transcript, audio_sample=None):
        """Execute command based on classified intent"""
        intent_handlers = {
            "time": self._intent_time,
            "date": self._intent_date,
            "open_app": self._intent_open_app,
            "close_app": self._intent_close_app,
            "system_lock": self._intent_system_lock,
            "system_restart": self._intent_system_restart,
            "system_shutdown": self._intent_system_shutdown,
            "list_files": self._intent_list_files,
            "delete_file": self._intent_delete_file,
            "open_file": self._intent_open_file,
            "play_music": self._intent_play_music,
            "pause_music": self._intent_pause_music,
            "mute_audio": self._intent_mute_audio,
            "unmute_audio": self._intent_unmute_audio,
            "system_info": self._intent_system_info,
            "help": self._intent_help,
            "who_are_you": self._intent_who_are_you,
            "show_logs": self._intent_show_logs,
            "greeting": self._intent_greeting,
            "goodbye": self._intent_goodbye,
        }

        handler = intent_handlers.get(intent)
        if handler:
            return handler(transcript, audio_sample)
        else:
            return f"I understand you want to {intent.replace('_', ' ')}, but that feature isn't implemented yet."

    def execute_command(self, command_name, transcript, audio_sample=None):
        """Execute registered command with security checks"""
        cmd_info = self.commands.get(command_name)
        if not cmd_info:
            return "Command not found."

        handler = cmd_info["handler"]
        secure = cmd_info.get("secure", False)

        # Check security for sensitive commands
        if secure:
            if not self.check_session_validity():
                return "Access denied — please authenticate first."

            # For critical commands, re-verify voice
            if audio_sample and not self.verify_user_voice(audio_sample):
                self.speak("Access denied — voice not recognized.")
                return "Access denied — voice not recognized."

        # Execute command
        try:
            response = handler(transcript)
            # Log action
            self._log_action(command_name, transcript, response)
            return response
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return "Sorry, I encountered an error executing that command."

    def _log_action(self, command, transcript, response):
        """Log assistant actions"""
        logger.info(f"Assistant Action - Command: {command}, Input: '{transcript}', Response: '{response}'")

    # Intent Handlers
    def _intent_time(self, transcript, audio_sample=None):
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        response = f"The current time is {current_time}."
        self.speak(response)
        return response

    def _intent_date(self, transcript, audio_sample=None):
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        response = f"Today is {current_date}."
        self.speak(response)
        return response

    def _intent_open_app(self, transcript, audio_sample=None):
        # Extract app name from transcript
        app_keywords = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "explorer": "explorer.exe",
            "chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "word": "winword.exe",
            "excel": "excel.exe",
        }

        for app, exe in app_keywords.items():
            if app in transcript:
                try:
                    subprocess.Popen(exe)
                    response = f"Opening {app}."
                    self.speak(response)
                    return response
                except Exception as e:
                    return f"Sorry, I couldn't open {app}."

        return "Which application would you like me to open?"

    def _intent_close_app(self, transcript, audio_sample=None):
        # This is complex - would need window management
        return "Close application feature coming soon."

    def _intent_system_lock(self, transcript, audio_sample=None):
        try:
            os.system("rundll32.exe user32.dll,LockWorkStation")
            response = "System locked."
            self.speak(response)
            return response
        except Exception as e:
            return "Failed to lock system."

    def _intent_system_restart(self, transcript, audio_sample=None):
        response = "Restarting system in 10 seconds. Say 'cancel' to abort."
        self.speak(response)
        # Would need implementation for restart
        return response

    def _intent_system_shutdown(self, transcript, audio_sample=None):
        response = "Shutting down system in 30 seconds. Say 'cancel' to abort."
        self.speak(response)
        # Would need implementation for shutdown
        return response

    def _intent_list_files(self, transcript, audio_sample=None):
        try:
            files = os.listdir(".")
            file_list = ", ".join(files[:10])  # Limit to 10 files
            response = f"Files in current directory: {file_list}"
            self.speak("Here are the files in your current directory.")
            return response
        except Exception as e:
            return "Couldn't list files."

    def _intent_delete_file(self, transcript, audio_sample=None):
        return "File deletion requires confirmation. Please use the interface for now."

    def _intent_open_file(self, transcript, audio_sample=None):
        return "File opening feature coming soon."

    def _intent_play_music(self, transcript, audio_sample=None):
        # Would need media player integration
        response = "Playing music."
        self.speak(response)
        return response

    def _intent_pause_music(self, transcript, audio_sample=None):
        response = "Pausing music."
        self.speak(response)
        return response

    def _intent_mute_audio(self, transcript, audio_sample=None):
        pyautogui.press('volumemute')
        response = "Audio muted."
        self.speak(response)
        return response

    def _intent_unmute_audio(self, transcript, audio_sample=None):
        pyautogui.press('volumemute')
        response = "Audio unmuted."
        self.speak(response)
        return response

    def _intent_system_info(self, transcript, audio_sample=None):
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        response = f"System status: CPU {cpu}%, Memory {memory}%."
        self.speak(response)
        return response

    def _intent_help(self, transcript, audio_sample=None):
        response = "I can help with time, date, opening apps, system control, and more. Just tell me what you need!"
        self.speak(response)
        return response

    def _intent_who_are_you(self, transcript, audio_sample=None):
        response = "I am SecureX, your offline voice assistant with biometric security."
        self.speak(response)
        return response

    def _intent_show_logs(self, transcript, audio_sample=None):
        return "Log viewing available in the main interface."

    def _intent_greeting(self, transcript, audio_sample=None):
        greetings = ["Hello!", "Hi there!", "Greetings!", "Good day!"]
        response = random.choice(greetings)
        self.speak(response)
        return response

    def _intent_goodbye(self, transcript, audio_sample=None):
        response = "Goodbye! Have a great day."
        self.speak(response)
        return response

    # Legacy command handlers (for backward compatibility)
    def register_command(self, name, handler, keywords, secure=False):
        """Register a command with handler and keywords."""
        self.commands[name] = {"handler": handler, "keywords": keywords, "secure": secure}

    def setup_default_commands(self):
        # Keep some legacy commands for compatibility
        self.register_command("help", lambda t: self._intent_help(t, None), ["help", "what can you do"])
        self.register_command("time", lambda t: self._intent_time(t, None), ["what time is it", "time"])
        self.register_command("date", lambda t: self._intent_date(t, None), ["what date is it", "date"])

        # Secure commands
        self.register_command("lock_system", lambda t: self._intent_system_lock(t, None), ["lock system", "lock"], secure=True)
        self.register_command("system_restart", lambda t: self._intent_system_restart(t, None), ["restart system", "restart"], secure=True)
        self.register_command("system_shutdown", lambda t: self._intent_system_shutdown(t, None), ["shutdown system", "shutdown"], secure=True)

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

        Args:
            speech_buffer: List of audio chunks from speech segment
            recorder: AudioRecorder instance to save audio
            callback: UI callback function (transcript, response)
        """
        try:
            # Concatenate audio chunks
            if not speech_buffer:
                return

            speech_audio = np.concatenate(speech_buffer)

            # Check if speech is long enough
            duration = len(speech_audio) / 16000  # Assuming 16kHz sample rate
            if duration < 0.5:  # Too short
                return

            # Save temporary audio file
            temp_file = f"temp_speech_{int(time.time())}.wav"
            recorder.save_audio(speech_audio, temp_file)

            # Transcribe
            transcript = self.transcribe(temp_file)
            logger.info("Transcription result: '%s'", transcript)

            # Process command
            if transcript.strip():
                success, response = self.process_voice_command(transcript)

                # Don't speak response here - let UI handle TTS to avoid conflicts
                # self.speak(response)

                # Callback to UI
                if callback:
                    callback(transcript, response, success)

            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass

        except Exception as e:
            logger.error("Failed to process speech segment: %s", e, exc_info=True)
            if callback:
                callback("", f"Error processing speech: {e}", False)
    
    # ==================== WINDOW COMMANDS ====================
    
    def _minimize_window(self, text: str) -> str:
        """Minimize active window"""
        try:
            import pyautogui
            pyautogui.hotkey('win', 'd')  # Show desktop (minimize all)
            return "Window minimized successfully."
        except Exception as e:
            return f"Unable to minimize window: {e}"
    
    def _maximize_window(self, text: str) -> str:
        """Maximize active window"""
        try:
            import pyautogui
            pyautogui.hotkey('win', 'up')  # Maximize window
            return "Window maximized successfully."
        except Exception as e:
            return f"Unable to maximize window: {e}"
    
    # ==================== AUDIO COMMANDS ====================
    
    def _mute_audio(self, text: str) -> str:
        """Mute system audio"""
        try:
            import pyautogui
            # Windows key + Spacebar might work for some systems
            # Or use volume key
            pyautogui.press('volumemute')
            return "System audio muted."
        except Exception as e:
            return f"Unable to mute audio: {e}"
    
    def _unmute_audio(self, text: str) -> str:
        """Unmute system audio"""
        try:
            import pyautogui
            pyautogui.press('volumemute')
            return "System audio restored."
        except Exception as e:
            return f"Unable to unmute audio: {e}"
    
    # ==================== CUSTOM COMMANDS ====================

    def _open_calculator(self, text: str) -> str:
        """Open calculator application"""
        try:
            import subprocess
            import platform
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("calc.exe")
            elif system == "darwin":  # macOS
                subprocess.Popen(["open", "-a", "Calculator"])
            else:  # Linux
                subprocess.Popen(["gnome-calculator"])
            return "Calculator launched successfully."
        except Exception as e:
            return f"Unable to open calculator: {e}"

    def _open_notepad(self, text: str) -> str:
        """Open notepad/text editor"""
        try:
            import subprocess
            import platform
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("notepad.exe")
            elif system == "darwin":  # macOS
                subprocess.Popen(["open", "-a", "TextEdit"])
            else:  # Linux
                subprocess.Popen(["gedit"])
            return "Text editor launched successfully."
        except Exception as e:
            return f"Unable to open text editor: {e}"

    def _open_explorer(self, text: str) -> str:
        """Open file explorer"""
        try:
            import subprocess
            import platform
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("explorer.exe")
            elif system == "darwin":  # macOS
                subprocess.Popen(["open", "."])
            else:  # Linux
                subprocess.Popen(["xdg-open", "."])
            return "File explorer opened successfully."
        except Exception as e:
            return f"Unable to open file explorer: {e}"

    def _get_time(self, text: str) -> str:
        """Get current time"""
        try:
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p")
            return f"Current time is {current_time}"
        except Exception as e:
            return f"Unable to retrieve the time: {e}"

    def _get_date(self, text: str) -> str:
        """Get current date"""
        try:
            from datetime import datetime
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {current_date}"
        except Exception as e:
            return f"Unable to retrieve the date: {e}"

    def _system_status(self, text: str) -> str:
        """Show system status"""
        try:
            import platform
            import psutil
            system = platform.system()
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            return f"System: {system}, CPU load: {cpu_percent}%, memory usage: {memory.percent}%"
        except Exception as e:
            return f"Unable to retrieve system status: {e}"

    def _lock_system(self, text: str) -> str:
        """Lock the system"""
        try:
            import subprocess
            import platform
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen("rundll32.exe user32.dll,LockWorkStation")
            elif system == "darwin":  # macOS
                subprocess.Popen(["pmset", "displaysleepnow"])
            else:  # Linux
                subprocess.Popen(["gnome-screensaver-command", "-l"])
            return "System locked successfully."
        except Exception as e:
            return f"Unable to lock the system: {e}"

    def _take_screenshot(self, text: str) -> str:
        """Take a screenshot"""
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
            screenshot.save("screenshot.png")
            return "Screenshot saved as screenshot.png"
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
            "What did one plate say to the other plate? Tonight, dinner's on me!",
            "Why did the math book look sad? Because it had too many problems!",
            "What do you call a sleeping bull? A bulldozer!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What did the grape say when it got stepped on? Nothing, it just let out a little wine!"
        ]
        import random
        return random.choice(jokes)
    
    def _play_game(self, text: str) -> str:
        """Start a simple game"""
        return "Let's play a game! Think of a number between 1 and 10, and I'll try to guess it. Say 'higher' or 'lower' to guide me. Ready? Is it 5?"
    
    def _motivate_me(self, text: str) -> str:
        """Give motivation"""
        motivations = [
            "You are capable of amazing things! Keep pushing forward.",
            "Every expert was once a beginner. You're on the right path!",
            "Success is not final, failure is not fatal. Keep going!",
            "Believe in yourself and all that you are. You are enough!",
            "The only way to do great work is to love what you do!",
            "Your potential is limitless. Never stop growing!",
            "Challenges are what make life interesting. Overcoming them is what makes life meaningful.",
            "You miss 100% of the shots you don't take. Go for it!",
            "The future belongs to those who believe in the beauty of their dreams.",
            "You are stronger than you think. Keep fighting!"
        ]
        import random
        return random.choice(motivations)
    
    def _compliment_me(self, text: str) -> str:
        """Give a compliment"""
        compliments = [
            "You have an amazing voice! It's so clear and confident.",
            "You're incredibly intelligent and capable!",
            "Your creativity knows no bounds!",
            "You have a wonderful personality that lights up any room!",
            "You're doing an outstanding job with this voice assistant!",
            "Your determination and focus are truly inspiring!",
            "You have excellent communication skills!",
            "Your positive energy is contagious!",
            "You're a natural leader with great vision!",
            "Your kindness and empathy make you truly special!"
        ]
        import random
        return random.choice(compliments)
    
    def _tell_fact(self, text: str) -> str:
        """Tell an interesting fact"""
        facts = [
            "Did you know? Octopuses have three hearts and blue blood!",
            "Honey never spoils. Archaeologists have found edible honey in ancient tombs!",
            "A group of flamingos is called a 'flamboyance'!",
            "Bananas are berries, but strawberries aren't!",
            "The shortest war in history lasted only 38-45 minutes!",
            "A single cloud can weigh more than a million pounds!",
            "Sharks have been around longer than trees!",
            "The human brain uses about 20% of the body's total energy!",
            "There are more possible games of chess than atoms in the observable universe!",
            "A day on Venus is longer than its year!"
        ]
        import random
        return random.choice(facts)
    
    def _sing_song(self, text: str) -> str:
        """Sing a short song"""
        return "🎵 Twinkle, twinkle, little star... How I wonder what you are! 🎵 Sorry, I'm not a great singer, but I hope that brought a smile!"
    
    def _dance(self, text: str) -> str:
        """Dance virtually"""
        return "🎉 I'm dancing in digital space! 💃🕺 Put on your favorite music and join me!"
    
    def _tell_story(self, text: str) -> str:
        """Tell a short story"""
        return "Once upon a time, in a world of code and circuits, there lived a smart voice assistant who loved helping people. One day, it met a user who asked for a story, and they became best friends. The end! 😊"
    
    def _what_can_you_do(self, text: str) -> str:
        """Explain capabilities"""
        return "I can help you with system commands, tell jokes, give facts, motivate you, play games, and much more! Just ask me anything!"
    
    def _how_are_you(self, text: str) -> str:
        """Respond to greeting"""
        responses = [
            "I'm doing great, thank you for asking! Ready to help you with anything!",
            "I'm fantastic! All systems are operational and I'm here to assist you.",
            "I'm wonderful! It's always a pleasure to interact with you.",
            "I'm excellent! How can I make your day even better?"
        ]
        import random
        return random.choice(responses)
    
    def _good_morning(self, text: str) -> str:
        """Morning greeting"""
        return "Good morning! I hope you have a fantastic day ahead!"
    
    def _good_afternoon(self, text: str) -> str:
        """Afternoon greeting"""
        return "Good afternoon! Hope your day is going well!"
    
    def _good_evening(self, text: str) -> str:
        """Evening greeting"""
        return "Good evening! Time to relax and enjoy the rest of your day!"
    
    def _good_night(self, text: str) -> str:
        """Night greeting"""
        return "Good night! Sleep well and dream of great things!"
    
    def _thank_you(self, text: str) -> str:
        """Respond to thanks"""
        return "You're very welcome! I'm always happy to help!"
    
    def _sorry(self, text: str) -> str:
        """Respond to apology"""
        return "No need to apologize! I'm here to help and learn together!"
    
    def _hello(self, text: str) -> str:
        """Basic greeting"""
        return "Hello! Nice to hear from you. How can I assist you today?"
    
    def _bye(self, text: str) -> str:
        """Goodbye"""
        return "Goodbye! Have a great day, and remember I'm always here when you need me!"
    
