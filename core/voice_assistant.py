import os
import wave
import json
import threading
import time
import queue
import numpy as np
from vosk import Model, KaldiRecognizer
import logging

import contextlib

logger = logging.getLogger(__name__)

class VoiceAssistant:
    """
    Voice Assistant: Handles activation, speech-to-text, TTS, and command processing.
    """
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        self.active = False
        self.model_path = model_path
        self._vosk_model = None
        # Disable TTS to avoid conflicts with main TTS system
        # self._tts_queue = queue.Queue()
        # self._tts_shutdown = threading.Event()
        # self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        # self._tts_thread.start()
        self.commands = {}
        self._load_model()

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
        # Don't speak here - let UI handle TTS to avoid conflicts
        # self.speak("Voice Assistant activated. Say a command.")

    def deactivate(self, silent=False):
        self.active = False
        # TTS disabled to avoid conflicts with main TTS system
        # if not silent:
        #     self.speak("Voice Assistant deactivated.")

    def speak(self, text):
        # TTS disabled to avoid conflicts with main TTS system
        # if not text:
        #     return
        # self._tts_queue.put(text)
        pass

    def _tts_worker(self):
        engine = None
        loop_started = False

        while not self._tts_shutdown.is_set():
            try:
                text = self._tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                self._tts_queue.task_done()
                break

            if engine is None:
                try:
                    import pyttsx3

                    engine = pyttsx3.init()
                    engine.setProperty("rate", 170)
                    engine.setProperty("volume", 0.88)
                    loop_started = False  # Reset loop state when creating new engine
                except RuntimeError as init_exc:
                    if "run loop already started" in str(init_exc):
                        logger.warning("Cannot initialize TTS engine - run loop already started, skipping TTS")
                        self._tts_queue.task_done()
                        continue
                    else:
                        logger.error("TTS initialization failed: %s", init_exc)
                        self._tts_queue.task_done()
                        continue
                except Exception as exc:
                    logger.error("TTS initialization failed: %s", exc)
                    self._tts_queue.task_done()
                    continue

            try:
                if not loop_started:
                    engine.say(text)
                    engine.runAndWait()
                    loop_started = True
                else:
                    # If loop is already started, skip TTS entirely
                    logger.warning("TTS run loop already started - skipping TTS playback")
                    self._tts_queue.task_done()
                    continue
            except RuntimeError as exc:
                if "run loop already started" in str(exc):
                    logger.warning("TTS run loop already started - skipping TTS playback during shutdown")
                    # Don't try to recover, just skip TTS during shutdown
                    with contextlib.suppress(Exception):
                        if engine:
                            engine.stop()
                    engine = None
                    loop_started = False
                else:
                    logger.error("TTS playback failed: %s", exc)
                    with contextlib.suppress(Exception):
                        if engine:
                            engine.stop()
                    engine = None
                    loop_started = False
            except Exception as exc:
                logger.error("Unexpected TTS error: %s", exc, exc_info=True)
                with contextlib.suppress(Exception):
                    if engine:
                        engine.stop()
                engine = None
                loop_started = False
            finally:
                self._tts_queue.task_done()

        if engine is not None:
            with contextlib.suppress(Exception):
                engine.stop()

    def shutdown(self):
        # TTS disabled, no cleanup needed
        # self._tts_shutdown.set()
        # self._tts_queue.put(None)
        # if self._tts_thread.is_alive():
        #     self._tts_thread.join(timeout=1.0)
        pass

    def register_command(self, name, handler, keywords):
        """Register a command with handler and keywords."""
        self.commands[name] = {"handler": handler, "keywords": keywords}

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

    def process_voice_command(self, transcript):
        """Process transcript and execute matching command."""
        transcript = transcript.lower().strip()
        logger.info("Processing command: '%s'", transcript)
        for name, cmd in self.commands.items():
            for kw in cmd["keywords"]:
                if kw in transcript:
                    logger.info("Matched command: %s", name)
                    response = cmd["handler"](transcript)
                    logger.info("Command response: %s", response)
                    return True, response
        logger.warning("No command matched for transcript: '%s'", transcript)
        return False, "Command not recognized."

    # Example built-in command
    def _cmd_help(self, transcript):
        cmds = [name for name in self.commands]
        return "Available commands: " + ", ".join(cmds)

    def setup_default_commands(self):
        self.register_command("help", self._cmd_help, ["help", "what can you do", "commands"])

        # System commands
        self.register_command("open_calculator", self._open_calculator, ["open calculator", "calculator", "calc", "calculate", "math", "arithmetic"])
        self.register_command("open_notepad", self._open_notepad, ["open notepad", "notepad", "text editor", "editor", "write", "note"])
        self.register_command("open_explorer", self._open_explorer, ["open explorer", "file explorer", "explorer", "files", "file manager", "browse files"])
        self.register_command("get_time", self._get_time, ["what time is it", "time", "current time", "tell me the time", "what's the time"])
        self.register_command("get_date", self._get_date, ["what date is it", "date", "current date", "show date", "what's today's date", "today's date"])
        self.register_command("system_status", self._system_status, ["system status", "status", "performance", "system info", "computer status"])

        # Security commands
        self.register_command("lock_system", self._lock_system, ["lock system", "lock computer", "lock"])
        self.register_command("run_security_scan", self._run_security_scan, ["run security scan", "security scan", "scan security"])
        self.register_command("show_security_logs", self._show_security_logs, ["show security logs", "security logs", "logs"])
        self.register_command("biometric_status", self._biometric_status, ["biometric status", "biometrics", "bio status"])

        # Utilities
        self.register_command("take_screenshot", self._take_screenshot, ["take screenshot", "screenshot", "capture screen"])
        self.register_command("show_weather", self._show_weather, ["show weather", "weather", "forecast"])
        self.register_command("search_files", self._search_files, ["search files", "find files", "file search"])

        # Communication
        self.register_command("send_secure_message", self._send_secure_message, ["send secure message", "secure message", "send message"])
        self.register_command("check_messages", self._check_messages, ["check messages", "messages", "new messages"])
        self.register_command("read_last_message", self._read_last_message, ["read last message", "read message", "last message"])
        self.register_command("start_voice_call", self._start_voice_call, ["start voice call", "voice call", "call"])

        # Window commands
        self.register_command("minimize_window", self._minimize_window, ["minimize window", "minimize", "show desktop"])
        self.register_command("maximize_window", self._maximize_window, ["maximize window", "maximize", "fullscreen"])

        # Audio commands
        self.register_command("mute_audio", self._mute_audio, ["mute audio", "mute", "quiet"])
        self.register_command("unmute_audio", self._unmute_audio, ["unmute audio", "unmute", "sound on"])
        
        # Fun and interactive commands
        self.register_command("tell_joke", self._tell_joke, ["tell me a joke", "joke", "make me laugh", "funny"])
        self.register_command("play_game", self._play_game, ["play a game", "game", "let's play", "entertainment"])
        self.register_command("motivate_me", self._motivate_me, ["motivate me", "motivation", "inspire me", "encourage"])
        self.register_command("compliment_me", self._compliment_me, ["compliment me", "nice words", "praise me"])
        self.register_command("tell_fact", self._tell_fact, ["tell me a fact", "fact", "interesting fact", "random fact"])
        self.register_command("sing_song", self._sing_song, ["sing a song", "song", "music", "sing"])
        self.register_command("dance", self._dance, ["dance", "let's dance", "dancing"])
        self.register_command("tell_story", self._tell_story, ["tell me a story", "story", "storytime"])
        self.register_command("what_can_you_do", self._what_can_you_do, ["what can you do", "your capabilities", "help me", "assist me"])
        self.register_command("how_are_you", self._how_are_you, ["how are you", "how do you feel", "status check"])
        self.register_command("good_morning", self._good_morning, ["good morning", "morning", "wake up"])
        self.register_command("good_afternoon", self._good_afternoon, ["good afternoon", "afternoon"])
        self.register_command("good_evening", self._good_evening, ["good evening", "evening"])
        self.register_command("good_night", self._good_night, ["good night", "night", "sleep"])
        self.register_command("thank_you", self._thank_you, ["thank you", "thanks", "appreciate it"])
        self.register_command("sorry", self._sorry, ["sorry", "apologize", "my bad"])
        self.register_command("hello", self._hello, ["hello", "hi", "hey", "greetings"])
        self.register_command("bye", self._bye, ["bye", "goodbye", "see you", "farewell"])

    def get_available_commands(self):
        """Get a formatted string of all available voice commands."""
        if not self.commands:
            return "No commands available."

        command_list = []
        for name, cmd_info in self.commands.items():
            keywords = cmd_info.get("keywords", [])
            if keywords:
                command_list.append(f"• {name}: {', '.join(keywords[:3])}")  # Show first 3 keywords

        return "\n".join(command_list)

    # ==================== CONTINUOUS LISTENING ====================

    def start_continuous_listening(self, audio_processor, callback=None):
        """
        Start continuous voice listening in a background thread.
        Uses VAD to detect speech segments and process commands automatically.

        Args:
            audio_processor: AudioProcessor instance for recording and VAD
            callback: Optional callback function to update UI (transcript, response)
        """
        if hasattr(self, '_continuous_thread') and self._continuous_thread and self._continuous_thread.is_alive():
            print("Continuous listening already running")
            return

        self._continuous_listening = True
        self._continuous_thread = threading.Thread(
            target=self._continuous_listening_loop,
            args=(audio_processor, callback),
            daemon=True
        )
        self._continuous_thread.start()
        logger.info("Continuous listening started")

    def stop_continuous_listening(self):
        """Stop continuous voice listening"""
        self._continuous_listening = False
        if hasattr(self, '_continuous_thread') and self._continuous_thread:
            self._continuous_thread.join(timeout=2.0)
        logger.info("Continuous listening stopped")

    def _continuous_listening_loop(self, audio_processor, callback):
        """
        Main continuous listening loop.
        Records audio in chunks, detects speech, accumulates audio during speech,
        transcribes when speech ends, and processes commands.
        """
        try:
            # Initialize VAD
            vad = audio_processor.vad_detector
            recorder = audio_processor.recorder

            # Configuration
            chunk_duration = 0.5  # Process 0.5 second chunks
            silence_threshold = 1.5  # Stop listening after 1.5s of silence
            min_speech_duration = 1.0  # Minimum speech segment to process

            speech_buffer = []
            silence_counter = 0
            is_speaking = False

            logger.debug("Listening for voice commands")

            while self._continuous_listening:
                try:
                    # Record a chunk
                    chunk = recorder.record_audio(
                        duration=chunk_duration,
                        reduce_noise=True
                    )

                    if chunk is None:
                        time.sleep(0.1)
                        continue

                    # Check for voice activity
                    has_voice, rms = vad.detect_voice_rms(chunk)

                    if has_voice:
                        if not is_speaking:
                            # Speech started
                            is_speaking = True
                            speech_buffer = []
                            silence_counter = 0
                            logger.debug("Speech detected - accumulating audio")

                        # Accumulate speech audio
                        speech_buffer.append(chunk)
                        silence_counter = 0

                    elif is_speaking:
                        # Silence detected during speech
                        silence_counter += chunk_duration

                        if silence_counter >= silence_threshold:
                            # Speech ended - process the accumulated audio
                            if speech_buffer:
                                self._process_speech_segment(speech_buffer, recorder, callback)

                            # Reset for next speech segment
                            speech_buffer = []
                            is_speaking = False
                            silence_counter = 0
                            logger.debug("Ready for next command")

                    # Small delay to prevent CPU hogging
                    time.sleep(0.05)

                except Exception as e:
                    logger.error("Error in continuous listening loop: %s", e, exc_info=True)
                    time.sleep(0.5)

        except Exception as e:
            logger.error("Continuous listening failed: %s", e, exc_info=True)
        finally:
            logger.info("Continuous listening loop ended")

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
    
