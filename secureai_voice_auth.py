"""
SecureAI Voice Assistant - Enhanced Voice Authentication System
Complete voice enrollment, authentication, and password fallback
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog
import threading
import time
import os
import sys
import subprocess
import json
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle

# Core imports
import pyttsx3
import speech_recognition as sr
import requests
from bs4 import BeautifulSoup
import pyautogui
import webbrowser
import psutil
from pynput.keyboard import Key, Controller

# Voice authentication imports
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available. Using basic voice authentication.")

class VoiceAuthenticator:
    def __init__(self):
        self.voice_profiles = {}
        self.voice_profiles_file = "voice_profiles.json"
        self.voice_features_file = "voice_features.pkl"
        self.custom_phrase = None
        self.load_voice_profiles()
        self.load_custom_phrase()
        
    def load_voice_profiles(self):
        """Load existing voice profiles"""
        try:
            if os.path.exists(self.voice_profiles_file):
                with open(self.voice_profiles_file, 'r') as f:
                    self.voice_profiles = json.load(f)
        except:
            self.voice_profiles = {}
    
    def load_custom_phrase(self):
        """Load custom phrase from profiles"""
        try:
            if "primary_user" in self.voice_profiles and "custom_phrase" in self.voice_profiles["primary_user"]:
                self.custom_phrase = self.voice_profiles["primary_user"]["custom_phrase"]
        except:
            self.custom_phrase = None
    
    def save_voice_profiles(self):
        """Save voice profiles"""
        try:
            with open(self.voice_profiles_file, 'w') as f:
                json.dump(self.voice_profiles, f)
        except Exception as e:
            print(f"Error saving voice profiles: {e}")
    
    def extract_voice_features(self, audio_data, sr_rate=22050):
        """Extract voice features from audio"""
        if not LIBROSA_AVAILABLE:
            # Simple fallback - use basic audio properties
            return {
                'energy': float(np.mean(audio_data ** 2)),
                'zero_crossing_rate': float(np.mean(np.abs(np.diff(np.sign(audio_data))))),
                'length': len(audio_data)
            }
        
        try:
            # Advanced features using librosa
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr_rate, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            features = {
                'mfcc_mean': np.mean(mfccs, axis=1).tolist(),
                'mfcc_std': np.std(mfccs, axis=1).tolist(),
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                'energy': float(np.mean(audio_data ** 2))
            }
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def compare_voice_features(self, features1, features2):
        """Compare two voice feature sets"""
        if not features1 or not features2:
            return 0.0
        
        if not LIBROSA_AVAILABLE:
            # Simple comparison
            energy_diff = abs(features1['energy'] - features2['energy'])
            zcr_diff = abs(features1['zero_crossing_rate'] - features2['zero_crossing_rate'])
            similarity = max(0, 1 - (energy_diff + zcr_diff) / 2)
            return similarity
        
        try:
            # Advanced comparison
            mfcc_similarity = np.corrcoef(features1['mfcc_mean'], features2['mfcc_mean'])[0, 1]
            if np.isnan(mfcc_similarity):
                mfcc_similarity = 0
            
            energy_similarity = 1 - abs(features1['energy'] - features2['energy']) / max(features1['energy'], features2['energy'])
            spectral_similarity = 1 - abs(features1['spectral_centroid_mean'] - features2['spectral_centroid_mean']) / max(features1['spectral_centroid_mean'], features2['spectral_centroid_mean'])
            
            overall_similarity = (abs(mfcc_similarity) * 0.6 + energy_similarity * 0.3 + spectral_similarity * 0.1)
            return max(0, min(1, overall_similarity))
        except:
            return 0.0

class SecureAI_Assistant:
    def __init__(self, root):
        self.root = root
        self.root.title("SecureAI Voice Assistant - Enhanced Voice Authentication")
        self.root.geometry("1200x900")
        self.root.configure(bg='#1a1a1a')
        
        # Initialize voice authenticator
        self.voice_auth = VoiceAuthenticator()
        
        # Initialize core components
        self.init_tts()
        self.init_recognition()
        self.init_system_controls()
        
        # Authentication state
        self.is_listening = False
        self.is_authenticated = False
        self.is_activated = False
        self.current_status = "Offline"
        self.auth_attempts = 0
        self.max_auth_attempts = 3
        
        # Voice enrollment state
        self.enrollment_mode = False
        self.enrollment_samples = []
        self.required_samples = 5  # Increased for better accuracy
        self.phrase_setup_mode = False
        
        # Password settings
        self.password_file = "secure_password.hash"
        
        # Thread pool for performance
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Create GUI
        self.setup_gui()
        
        # Check existing authentication setup
        self.check_auth_setup()
        
        # Application dictionary
        self.setup_app_dictionary()
        
    def init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.engine = pyttsx3.init("sapi5")
            voices = self.engine.getProperty("voices")
            if voices and len(voices) > 1:
                self.engine.setProperty("voice", voices[1].id)
            self.engine.setProperty("rate", 200)
        except:
            self.engine = None
            print("TTS engine not available")
    
    def init_recognition(self):
        """Initialize speech recognition"""
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.energy_threshold = 300
    
    def init_system_controls(self):
        """Initialize system controls"""
        try:
            self.keyboard = Controller()
            from ctypes import windll
            self.user32 = windll.user32
            self.APPCOMMAND_VOLUME_UP = 0x0A
            self.APPCOMMAND_VOLUME_DOWN = 0x09
            self.APPCOMMAND_VOLUME_MUTE = 0x08
            self.WM_APPCOMMAND = 0x319
            self.windows_api_available = True
        except:
            self.windows_api_available = False
    
    def setup_app_dictionary(self):
        """Setup application and website dictionaries"""
        self.apps = {
            "notepad": {"cmd": "notepad", "exe": "notepad.exe"},
            "calculator": {"cmd": "calc", "exe": "calc.exe"},
            "paint": {"cmd": "paint", "exe": "mspaint.exe"},
            "cmd": {"cmd": "cmd", "exe": "cmd.exe"},
            "powershell": {"cmd": "powershell", "exe": "powershell.exe"},
            "word": {"cmd": "winword", "exe": "WINWORD.EXE"},
            "excel": {"cmd": "excel", "exe": "EXCEL.EXE"},
            "chrome": {"cmd": "chrome", "exe": "chrome.exe"},
            "firefox": {"cmd": "firefox", "exe": "firefox.exe"},
            "edge": {"cmd": "msedge", "exe": "msedge.exe"},
            "vs code": {"cmd": "code", "exe": "Code.exe"},
            "explorer": {"cmd": "explorer", "exe": "explorer.exe"},
        }
        
        self.websites = {
            "google": "https://www.google.com",
            "youtube": "https://www.youtube.com",
            "facebook": "https://www.facebook.com",
            "twitter": "https://www.twitter.com",
            "github": "https://www.github.com",
            "gmail": "https://gmail.com",
        }
    
    def setup_gui(self):
        """Create the enhanced GUI with voice authentication"""
        # Main container
        main_container = tk.Frame(self.root, bg='#1a1a1a')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title section
        self.create_title_section(main_container)
        
        # Status section
        self.create_status_section(main_container)
        
        # Main content with four columns
        content_frame = tk.Frame(main_container, bg='#1a1a1a')
        content_frame.pack(fill='both', expand=True, pady=20)
        
        # Left column - Voice Authentication
        left_column = tk.Frame(content_frame, bg='#1a1a1a')
        left_column.pack(side='left', fill='y', padx=(0, 10))
        
        # Left-middle column - System Controls
        left_mid_column = tk.Frame(content_frame, bg='#1a1a1a')
        left_mid_column.pack(side='left', fill='y', padx=10)
        
        # Right-middle column - Quick Actions
        right_mid_column = tk.Frame(content_frame, bg='#1a1a1a')
        right_mid_column.pack(side='left', fill='y', padx=10)
        
        # Right column - Response Log
        right_column = tk.Frame(content_frame, bg='#1a1a1a')
        right_column.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Create sections
        self.create_voice_auth_section(left_column)
        self.create_control_section(left_mid_column)
        self.create_quick_actions_section(right_mid_column)
        self.create_response_section(right_column)
        
        # Welcome messages
        self.add_response("🚀 SecureAI Voice Assistant - Custom Phrase Authentication")
        self.add_response("✏️ Set your own authentication phrase - speak it 5 times for training")
        self.add_response("🎤 Advanced voice biometric analysis with environmental adaptation")
        self.add_response("🔐 Multi-layer security: Custom Voice Phrase → Password Fallback → Access")
    
    def create_title_section(self, parent):
        """Create title section"""
        title_frame = tk.Frame(parent, bg='#1a1a1a')
        title_frame.pack(fill='x', pady=(0, 20))
        
        title_label = tk.Label(title_frame,
                              text="🛡️ SecureAI Voice Assistant",
                              font=('Segoe UI', 32, 'bold'),
                              fg='#0078d4',
                              bg='#1a1a1a')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame,
                                 text="Enhanced Voice Authentication • Multi-Sample Enrollment • Password Fallback",
                                 font=('Segoe UI', 12),
                                 fg='#8a8886',
                                 bg='#1a1a1a')
        subtitle_label.pack(pady=(5, 0))
    
    def create_status_section(self, parent):
        """Create status section"""
        status_frame = tk.Frame(parent, bg='#252423', relief='flat', bd=1)
        status_frame.pack(fill='x', pady=(0, 20))
        
        status_inner = tk.Frame(status_frame, bg='#252423')
        status_inner.pack(fill='x', padx=20, pady=15)
        
        # Status indicator
        status_left = tk.Frame(status_inner, bg='#252423')
        status_left.pack(side='left')
        
        self.status_indicator = tk.Canvas(status_left, width=20, height=20, bg='#252423', highlightthickness=0)
        self.status_indicator.pack(side='left', padx=(0, 15))
        self.status_circle = self.status_indicator.create_oval(2, 2, 18, 18, fill='#d13438', outline='')
        
        self.status_label = tk.Label(status_left,
                                    text="Status: Offline",
                                    font=('Segoe UI', 14, 'bold'),
                                    fg='#ffffff',
                                    bg='#252423')
        self.status_label.pack(side='left')
        
        # Authentication status
        self.auth_status_label = tk.Label(status_inner,
                                         text="🔒 Authentication Required",
                                         font=('Segoe UI', 11),
                                         fg='#ff6b6b',
                                         bg='#252423')
        self.auth_status_label.pack(side='right')
    
    def create_voice_auth_section(self, parent):
        """Create voice authentication section"""
        # Voice Authentication Frame
        voice_auth_frame = tk.LabelFrame(parent,
                                        text="🎤 Voice Authentication",
                                        font=('Segoe UI', 12, 'bold'),
                                        fg='#ffffff',
                                        bg='#1a1a1a',
                                        bd=1,
                                        relief='solid')
        voice_auth_frame.pack(fill='x', pady=(0, 15))
        
        voice_inner = tk.Frame(voice_auth_frame, bg='#1a1a1a')
        voice_inner.pack(fill='x', padx=15, pady=15)
        
        # Voice setup section
        tk.Label(voice_inner, text="📝 Voice Setup", font=('Segoe UI', 10, 'bold'), fg='#ffffff', bg='#1a1a1a').pack(anchor='w', pady=(0, 5))
        
        self.phrase_btn = ttk.Button(voice_inner,
                                    text="✏️ Set Custom Phrase",
                                    command=self.set_custom_phrase,
                                    width=25)
        self.phrase_btn.pack(pady=2, fill='x')
        
        # Current phrase display
        self.current_phrase_label = tk.Label(voice_inner,
                                           text="No custom phrase set",
                                           font=('Segoe UI', 9),
                                           fg='#8a8886',
                                           bg='#1a1a1a',
                                           wraplength=200)
        self.current_phrase_label.pack(pady=(2, 8), fill='x')
        
        # Voice enrollment section
        tk.Label(voice_inner, text="📋 Voice Training", font=('Segoe UI', 10, 'bold'), fg='#ffffff', bg='#1a1a1a').pack(anchor='w', pady=(0, 5))
        
        self.enroll_btn = ttk.Button(voice_inner,
                                    text="🎙️ Train Voice (5 samples)",
                                    command=self.start_voice_enrollment,
                                    width=25,
                                    state='disabled')
        self.enroll_btn.pack(pady=2, fill='x')
        
        self.reset_voice_btn = ttk.Button(voice_inner,
                                         text="🗑️ Reset Voice Data",
                                         command=self.reset_voice_data,
                                         width=25)
        self.reset_voice_btn.pack(pady=2, fill='x')
        
        # Separator
        separator1 = ttk.Separator(voice_inner, orient='horizontal')
        separator1.pack(fill='x', pady=10)
        
        # Voice authentication section
        tk.Label(voice_inner, text="🔐 Voice Login", font=('Segoe UI', 10, 'bold'), fg='#ffffff', bg='#1a1a1a').pack(anchor='w', pady=(0, 5))
        
        self.voice_auth_btn = ttk.Button(voice_inner,
                                        text="🎤 Voice Authentication",
                                        command=self.start_voice_authentication,
                                        width=25)
        self.voice_auth_btn.pack(pady=2, fill='x')
        
        # Separator
        separator2 = ttk.Separator(voice_inner, orient='horizontal')
        separator2.pack(fill='x', pady=10)
        
        # Password fallback section
        tk.Label(voice_inner, text="🔑 Password Fallback", font=('Segoe UI', 10, 'bold'), fg='#ffffff', bg='#1a1a1a').pack(anchor='w', pady=(0, 5))
        
        self.password_auth_btn = ttk.Button(voice_inner,
                                           text="🔐 Password Login",
                                           command=self.password_authentication,
                                           width=25)
        self.password_auth_btn.pack(pady=2, fill='x')
        
        self.set_password_btn = ttk.Button(voice_inner,
                                          text="⚙️ Set Password",
                                          command=self.set_password,
                                          width=25)
        self.set_password_btn.pack(pady=2, fill='x')
        
        # Authentication progress
        self.auth_progress = ttk.Progressbar(voice_inner, 
                                           mode='indeterminate',
                                           style='Horizontal.TProgressbar')
        self.auth_progress.pack(fill='x', pady=(10, 0))
    
    def set_custom_phrase(self):
        """Set custom phrase for voice authentication"""
        if self.phrase_setup_mode:
            messagebox.showinfo("Setup Active", "Phrase setup is already in progress!")
            return
            
        # Create custom dialog for phrase input
        phrase_dialog = tk.Toplevel(self.root)
        phrase_dialog.title("Set Custom Voice Phrase")
        phrase_dialog.geometry("500x400")
        phrase_dialog.configure(bg='#1a1a1a')
        phrase_dialog.transient(self.root)
        phrase_dialog.grab_set()
        
        # Center the dialog
        phrase_dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Title
        title_label = tk.Label(phrase_dialog,
                              text="🎤 Set Your Voice Authentication Phrase",
                              font=('Segoe UI', 16, 'bold'),
                              fg='#0078d4',
                              bg='#1a1a1a')
        title_label.pack(pady=20)
        
        # Instructions
        instructions = tk.Label(phrase_dialog,
                               text="Choose a phrase that is:\n• Easy for you to remember and say\n• 5-15 words long\n• Contains your natural speaking style\n• Comfortable to repeat multiple times",
                               font=('Segoe UI', 11),
                               fg='#ffffff',
                               bg='#1a1a1a',
                               justify='left')
        instructions.pack(pady=10)
        
        # Example phrases
        examples_label = tk.Label(phrase_dialog,
                                 text="💡 Example phrases:",
                                 font=('Segoe UI', 10, 'bold'),
                                 fg='#00ff41',
                                 bg='#1a1a1a')
        examples_label.pack(pady=(20, 5))
        
        examples = tk.Label(phrase_dialog,
                           text="• \"Hello SecureAI, this is my voice authenticating access\"\n• \"My name is [Your Name] and I authorize this login\"\n• \"Voice authentication for my personal assistant\"\n• \"Access granted to my secure AI system\"",
                           font=('Segoe UI', 9),
                           fg='#8a8886',
                           bg='#1a1a1a',
                           justify='left')
        examples.pack(pady=5)
        
        # Input frame
        input_frame = tk.Frame(phrase_dialog, bg='#1a1a1a')
        input_frame.pack(pady=20, padx=20, fill='x')
        
        tk.Label(input_frame,
                text="Enter your custom phrase:",
                font=('Segoe UI', 10, 'bold'),
                fg='#ffffff',
                bg='#1a1a1a').pack(anchor='w')
        
        phrase_entry = tk.Text(input_frame,
                              height=3,
                              font=('Segoe UI', 11),
                              wrap=tk.WORD,
                              bg='#2d2d2d',
                              fg='#ffffff',
                              insertbackground='#ffffff',
                              selectbackground='#0078d4')
        phrase_entry.pack(fill='x', pady=5)
        
        # Buttons
        button_frame = tk.Frame(phrase_dialog, bg='#1a1a1a')
        button_frame.pack(pady=20)
        
        def save_phrase():
            phrase = phrase_entry.get(1.0, tk.END).strip()
            if len(phrase) < 10:
                messagebox.showwarning("Phrase Too Short", "Please enter a phrase with at least 10 characters.")
                return
            if len(phrase) > 200:
                messagebox.showwarning("Phrase Too Long", "Please enter a shorter phrase (max 200 characters).")
                return
                
            # Save the phrase
            self.voice_auth.custom_phrase = phrase
            
            # Update current phrase display
            self.current_phrase_label.config(
                text=f"Phrase: \"{phrase}\"",
                fg='#00ff41'
            )
            self.enroll_btn.config(state='normal')
            
            self.add_response(f"✅ Custom phrase set: \"{phrase}\"")
            self.add_response("🎯 Now click 'Train Voice' to record 5 samples")
            self.speak_async("Custom phrase set successfully. Now train your voice with 5 samples.")
            
            phrase_dialog.destroy()
        
        def cancel_phrase():
            phrase_dialog.destroy()
        
        def test_phrase():
            phrase = phrase_entry.get(1.0, tk.END).strip()
            if phrase:
                self.speak_async(phrase)
                self.add_response(f"🔊 Testing phrase: \"{phrase}\"")
        
        ttk.Button(button_frame, text="💾 Save Phrase", command=save_phrase).pack(side='left', padx=5)
        ttk.Button(button_frame, text="🔊 Test Phrase", command=test_phrase).pack(side='left', padx=5)
        ttk.Button(button_frame, text="❌ Cancel", command=cancel_phrase).pack(side='left', padx=5)
    
    def create_control_section(self, parent):
        """Create system control section"""
        # System Controls Frame
        system_frame = tk.LabelFrame(parent,
                                    text="💻 System Controls",
                                    font=('Segoe UI', 12, 'bold'),
                                    fg='#ffffff',
                                    bg='#1a1a1a',
                                    bd=1,
                                    relief='solid')
        system_frame.pack(fill='x', pady=(0, 15))
        
        system_inner = tk.Frame(system_frame, bg='#1a1a1a')
        system_inner.pack(fill='x', padx=15, pady=15)
        
        self.activate_btn = ttk.Button(system_inner,
                                      text="🚀 Activate Assistant",
                                      command=self.activate_assistant,
                                      state='disabled',
                                      width=25)
        self.activate_btn.pack(pady=2, fill='x')
        
        self.listen_btn = ttk.Button(system_inner,
                                    text="👂 Start Listening",
                                    command=self.toggle_listening,
                                    state='disabled',
                                    width=25)
        self.listen_btn.pack(pady=2, fill='x')
        
        ttk.Button(system_inner, text="📸 Screenshot", command=self.take_screenshot, width=25).pack(pady=2, fill='x')
        ttk.Button(system_inner, text="🔒 Lock Screen", command=self.lock_screen, width=25).pack(pady=2, fill='x')
        ttk.Button(system_inner, text="📊 System Status", command=self.show_system_status, width=25).pack(pady=2, fill='x')
    
    def create_quick_actions_section(self, parent):
        """Create quick actions section"""
        quick_frame = tk.LabelFrame(parent,
                                   text="⚡ Quick Actions",
                                   font=('Segoe UI', 12, 'bold'),
                                   fg='#ffffff',
                                   bg='#1a1a1a',
                                   bd=1,
                                   relief='solid')
        quick_frame.pack(fill='both', expand=True)
        
        quick_inner = tk.Frame(quick_frame, bg='#1a1a1a')
        quick_inner.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Information section
        tk.Label(quick_inner, text="📅 Information", font=('Segoe UI', 10, 'bold'), fg='#ffffff', bg='#1a1a1a').pack(anchor='w', pady=(0, 5))
        ttk.Button(quick_inner, text="🕐 Current Time", command=self.get_time, width=20).pack(pady=2, fill='x')
        ttk.Button(quick_inner, text="🌤️ Weather Info", command=self.get_weather, width=20).pack(pady=2, fill='x')
        
        # Search section
        tk.Label(quick_inner, text="🔍 Quick Search", font=('Segoe UI', 10, 'bold'), fg='#ffffff', bg='#1a1a1a').pack(anchor='w', pady=(15, 5))
        ttk.Button(quick_inner, text="🔍 Google", command=lambda: self.open_website("google"), width=20).pack(pady=2, fill='x')
        ttk.Button(quick_inner, text="📺 YouTube", command=lambda: self.open_website("youtube"), width=20).pack(pady=2, fill='x')
        
        # Apps section
        tk.Label(quick_inner, text="🚀 Quick Launch", font=('Segoe UI', 10, 'bold'), fg='#ffffff', bg='#1a1a1a').pack(anchor='w', pady=(15, 5))
        ttk.Button(quick_inner, text="🌐 Chrome", command=lambda: self.open_app("chrome"), width=20).pack(pady=2, fill='x')
        ttk.Button(quick_inner, text="📝 Notepad", command=lambda: self.open_app("notepad"), width=20).pack(pady=2, fill='x')
        ttk.Button(quick_inner, text="🧮 Calculator", command=lambda: self.open_app("calculator"), width=20).pack(pady=2, fill='x')
        
        # Media section
        tk.Label(quick_inner, text="🎵 Media Control", font=('Segoe UI', 10, 'bold'), fg='#ffffff', bg='#1a1a1a').pack(anchor='w', pady=(15, 5))
        ttk.Button(quick_inner, text="▶️ Play/Pause", command=self.media_toggle, width=20).pack(pady=2, fill='x')
        ttk.Button(quick_inner, text="🔊 Volume Up", command=self.volume_up, width=20).pack(pady=2, fill='x')
        ttk.Button(quick_inner, text="🔉 Volume Down", command=self.volume_down, width=20).pack(pady=2, fill='x')
        ttk.Button(quick_inner, text="🔇 Mute Toggle", command=self.volume_mute, width=20).pack(pady=2, fill='x')
    
    def create_response_section(self, parent):
        """Create response section"""
        response_frame = tk.LabelFrame(parent,
                                      text="💬 Assistant Response & Activity Log",
                                      font=('Segoe UI', 12, 'bold'),
                                      fg='#ffffff',
                                      bg='#1a1a1a',
                                      bd=1,
                                      relief='solid')
        response_frame.pack(fill='both', expand=True)
        
        response_inner = tk.Frame(response_frame, bg='#1a1a1a')
        response_inner.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.response_text = scrolledtext.ScrolledText(response_inner,
                                                      height=35,
                                                      font=('Consolas', 10),
                                                      fg='#00ff41',
                                                      bg='#0c0c0c',
                                                      insertbackground='#00ff41',
                                                      selectbackground='#333333',
                                                      selectforeground='#ffffff',
                                                      wrap=tk.WORD,
                                                      relief='flat',
                                                      bd=1)
        self.response_text.pack(fill='both', expand=True)
        
        # Control buttons
        log_controls = tk.Frame(response_inner, bg='#1a1a1a')
        log_controls.pack(fill='x', pady=(10, 0))
        
        ttk.Button(log_controls, text="🗑️ Clear Log", command=self.clear_log).pack(side='left', padx=(0, 10))
        ttk.Button(log_controls, text="💾 Save Log", command=self.save_log).pack(side='left')
    
    # Authentication Methods
    def check_auth_setup(self):
        """Check existing authentication setup"""
        voice_profiles_exist = os.path.exists(self.voice_auth.voice_profiles_file) and self.voice_auth.voice_profiles
        password_exists = os.path.exists(self.password_file)
        
        # Update custom phrase display
        if self.voice_auth.custom_phrase:
            self.current_phrase_label.config(
                text=f"Phrase: \"{self.voice_auth.custom_phrase}\"",
                fg='#00ff41'
            )
            self.enroll_btn.config(state='normal')
        else:
            self.current_phrase_label.config(
                text="❌ No custom phrase set - Click 'Set Custom Phrase' first",
                fg='#ff6b6b'
            )
            self.enroll_btn.config(state='disabled')
        
        if voice_profiles_exist:
            self.add_response(f"✅ Voice profiles found: {len(self.voice_auth.voice_profiles)} profile(s)")
            if self.voice_auth.custom_phrase:
                self.voice_auth_btn.config(state='normal')
                self.add_response(f"🎤 Voice phrase: \"{self.voice_auth.custom_phrase}\"")
        else:
            self.add_response("⚠️ No voice profiles found. Set custom phrase and train your voice.")
            
        if password_exists:
            self.add_response("✅ Password authentication available")
        else:
            self.add_response("⚠️ No password set. Click 'Set Password' for fallback authentication.")
            
        if not self.voice_auth.custom_phrase:
            self.add_response("🎯 Step 1: Set your custom phrase")
            self.add_response("🎯 Step 2: Train your voice with 5 samples")
        elif not voice_profiles_exist:
            self.add_response("🎯 Step 2: Train your voice with 5 samples")
    
    def start_voice_enrollment(self):
        """Start voice enrollment process"""
        if not self.voice_auth.custom_phrase:
            messagebox.showwarning("No Custom Phrase", "Please set your custom phrase first!")
            return
            
        if self.enrollment_mode:
            messagebox.showinfo("Enrollment Active", "Voice enrollment is already in progress!")
            return
            
        self.enrollment_mode = True
        self.enrollment_samples = []
        self.enroll_btn.config(text="⏹️ Stop Training", command=self.stop_voice_enrollment)
        self.auth_progress.start(10)
        
        self.add_response("🎙️ Starting voice training process...")
        self.add_response(f"📋 Recording {self.required_samples} samples of your custom phrase")
        self.add_response(f"🗣️ You will repeat: \"{self.voice_auth.custom_phrase}\"")
        self.add_response("� Speak naturally - different tones, speeds, and positions are good!")
        
        self.speak_async(f"Starting voice training. You will repeat your phrase {self.required_samples} times. Speak naturally and clearly.")
        
        # Start enrollment in thread
        threading.Thread(target=self.voice_enrollment_worker, daemon=True).start()
    
    def voice_enrollment_worker(self):
        """Voice enrollment worker thread"""
        try:
            custom_phrase = self.voice_auth.custom_phrase
            
            for i in range(self.required_samples):
                if not self.enrollment_mode:
                    break
                    
                self.add_response(f"🎤 Sample {i+1}/{self.required_samples}")
                self.add_response(f"🗣️ Say: \"{custom_phrase}\"")
                
                # Give different instructions for variety
                instructions = [
                    "Speak normally and clearly",
                    "Try speaking a bit slower",
                    "Speak at normal speed",
                    "Try from a slightly different position",
                    "Final sample - speak naturally"
                ]
                
                instruction = instructions[i] if i < len(instructions) else "Speak naturally"
                self.add_response(f"💡 {instruction}")
                self.speak_async(f"Sample {i+1}. {instruction}. Please say your phrase.")
                
                # Wait a moment for user to prepare
                time.sleep(2)
                
                # Record voice sample
                sample = self.record_voice_sample(custom_phrase, 10)  # 10 second timeout for custom phrases
                
                if sample:
                    self.enrollment_samples.append({
                        'phrase': custom_phrase,
                        'features': sample,
                        'timestamp': datetime.now().isoformat(),
                        'sample_number': i + 1
                    })
                    self.add_response(f"✅ Sample {i+1} recorded successfully")
                    
                    # Provide feedback
                    if i < self.required_samples - 1:
                        self.add_response(f"📊 {self.required_samples - (i+1)} more samples needed")
                else:
                    self.add_response(f"❌ Sample {i+1} failed - retrying...")
                    i -= 1  # Retry this sample
                    continue
                
                # Short pause between samples
                if i < self.required_samples - 1:
                    time.sleep(1.5)
            
            if len(self.enrollment_samples) >= self.required_samples:
                self.save_voice_enrollment()
            else:
                self.add_response("❌ Training incomplete - not enough samples")
                
        except Exception as e:
            self.add_response(f"❌ Training error: {str(e)}")
        finally:
            self.stop_voice_enrollment()
    
    def record_voice_sample(self, expected_phrase, timeout=10):
        """Record a single voice sample"""
        try:
            self.add_response("🎤 Recording... Speak now!")
            
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                # Listen for the phrase
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
            
            self.add_response("🔄 Processing audio...")
            
            # Convert to features
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize
            
            features = self.voice_auth.extract_voice_features(audio_data)
            
            # Try to recognize speech for verification (optional)
            try:
                recognized_text = self.recognizer.recognize_google(audio, language="en-in")
                self.add_response(f"🗣️ Heard: \"{recognized_text}\"")
                
                # More flexible matching for custom phrases
                expected_words = set(expected_phrase.lower().split())
                recognized_words = set(recognized_text.lower().split())
                
                # Calculate word overlap percentage
                if expected_words:
                    overlap = len(expected_words & recognized_words) / len(expected_words)
                    
                    if overlap >= 0.5:  # At least 50% word match
                        self.add_response(f"✅ Good match ({overlap:.1%} word overlap)")
                        return features
                    else:
                        self.add_response(f"⚠️ Low match ({overlap:.1%}) - but voice features captured")
                        # Still return features as voice patterns are more important than exact words
                        return features
                else:
                    return features
                    
            except sr.UnknownValueError:
                self.add_response("🎤 Speech unclear, but voice patterns captured")
                # Voice features are still valid even if speech recognition fails
                return features
            except Exception as recognition_error:
                self.add_response("🎤 Speech recognition failed, but voice patterns captured")
                # Voice biometric features are still usable
                return features
                
        except sr.WaitTimeoutError:
            self.add_response("⏰ Recording timeout - please speak sooner")
            return None
        except Exception as e:
            self.add_response(f"❌ Recording error: {str(e)}")
            return None
    
    def save_voice_enrollment(self):
        """Save voice enrollment data"""
        try:
            user_id = "primary_user"  # In a multi-user system, this would be dynamic
            
            # Calculate average features
            all_features = [sample['features'] for sample in self.enrollment_samples]
            
            if not LIBROSA_AVAILABLE:
                # Simple averaging for basic features
                avg_features = {
                    'energy': np.mean([f['energy'] for f in all_features]),
                    'zero_crossing_rate': np.mean([f['zero_crossing_rate'] for f in all_features]),
                    'length': np.mean([f['length'] for f in all_features])
                }
            else:
                # Advanced averaging for MFCC features
                avg_features = {
                    'mfcc_mean': np.mean([f['mfcc_mean'] for f in all_features], axis=0).tolist(),
                    'mfcc_std': np.mean([f['mfcc_std'] for f in all_features], axis=0).tolist(),
                    'spectral_centroid_mean': np.mean([f['spectral_centroid_mean'] for f in all_features]),
                    'zero_crossing_rate_mean': np.mean([f['zero_crossing_rate_mean'] for f in all_features]),
                    'energy': np.mean([f['energy'] for f in all_features])
                }
            
            self.voice_auth.voice_profiles[user_id] = {
                'features': avg_features,
                'samples': self.enrollment_samples,
                'custom_phrase': self.voice_auth.custom_phrase,
                'created': datetime.now().isoformat(),
                'sample_count': len(self.enrollment_samples),
                'version': '2.0'  # Version with custom phrases
            }
            
            self.voice_auth.save_voice_profiles()
            
            self.add_response("✅ Voice training completed successfully!")
            self.add_response(f"🎯 {len(self.enrollment_samples)} voice samples analyzed and saved")
            self.add_response(f"🔐 Your phrase: \"{self.voice_auth.custom_phrase}\"")
            self.add_response("🚀 Voice authentication is now ready!")
            
            self.voice_auth_btn.config(state='normal')
            self.enroll_btn.config(text="🔄 Re-train Voice", command=self.start_voice_enrollment)
            
            self.speak_async("Voice training completed successfully. Your voice authentication is now ready to use.")
            
        except Exception as e:
            self.add_response(f"❌ Failed to save voice enrollment: {str(e)}")
    
    def stop_voice_enrollment(self):
        """Stop voice enrollment"""
        self.enrollment_mode = False
        if self.voice_auth.voice_profiles and "primary_user" in self.voice_auth.voice_profiles:
            self.enroll_btn.config(text="🔄 Re-train Voice", command=self.start_voice_enrollment)
        else:
            self.enroll_btn.config(text="🎙️ Train Voice (5 samples)", command=self.start_voice_enrollment)
        self.auth_progress.stop()
        self.add_response("⏹️ Voice training stopped")
    
    def start_voice_authentication(self):
        """Start voice authentication process"""
        if not self.voice_auth.voice_profiles:
            messagebox.showwarning("No Voice Profiles", "Please train your voice first!")
            return
            
        if not self.voice_auth.custom_phrase:
            messagebox.showwarning("No Custom Phrase", "Please set your custom phrase first!")
            return
            
        self.auth_attempts += 1
        self.add_response(f"🎤 Voice authentication attempt {self.auth_attempts}/{self.max_auth_attempts}")
        self.auth_progress.start(10)
        
        # Use the custom phrase
        phrase = self.voice_auth.custom_phrase
        self.add_response(f"🗣️ Please say your phrase:")
        self.add_response(f"💬 \"{phrase}\"")
        self.speak_async("Please say your custom authentication phrase.")
        
        # Start authentication in thread
        threading.Thread(target=self.voice_authentication_worker, 
                        args=(phrase,), daemon=True).start()
    
    def voice_authentication_worker(self, expected_phrase):
        """Voice authentication worker"""
        try:
            self.add_response("🎙️ Starting voice authentication...")
            
            # Record authentication sample
            sample_features = self.record_voice_sample(expected_phrase, timeout=12)
            
            if not sample_features:
                self.add_response("❌ Voice recording failed")
                self.auth_progress.stop()
                return
            
            # Compare with enrolled profiles
            user_id = "primary_user"
            if user_id not in self.voice_auth.voice_profiles:
                self.add_response("❌ No voice profile found")
                self.auth_progress.stop()
                return
            
            enrolled_features = self.voice_auth.voice_profiles[user_id]['features']
            similarity = self.voice_auth.compare_voice_features(sample_features, enrolled_features)
            
            self.add_response(f"🔍 Analyzing voice patterns...")
            self.add_response(f"📊 Voice similarity score: {similarity:.1%}")
            
            # Dynamic threshold based on number of samples
            sample_count = self.voice_auth.voice_profiles[user_id].get('sample_count', 3)
            if sample_count >= 5:
                threshold = 0.55  # Lower threshold for more samples (better training)
            else:
                threshold = 0.60  # Higher threshold for fewer samples
            
            self.add_response(f"🎯 Required threshold: {threshold:.1%}")
            
            if similarity >= threshold:
                self.add_response(f"✅ Voice match confirmed! ({similarity:.1%})")
                self.authentication_success("voice")
            else:
                self.add_response(f"❌ Voice authentication failed")
                self.add_response(f"⚠️ Similarity {similarity:.1%} below required {threshold:.1%}")
                
                if self.auth_attempts >= self.max_auth_attempts:
                    self.add_response("🚫 Maximum voice authentication attempts reached")
                    self.add_response("🔑 Please use password authentication for access")
                    self.speak_async("Voice authentication failed multiple times. Please use password login.")
                else:
                    remaining = self.max_auth_attempts - self.auth_attempts
                    self.add_response(f"🔄 {remaining} attempts remaining. Try speaking more clearly.")
                    self.speak_async("Authentication failed. Try speaking more clearly.")
                    
        except Exception as e:
            self.add_response(f"❌ Voice authentication error: {str(e)}")
        finally:
            self.auth_progress.stop()
    
    def password_authentication(self):
        """Password authentication fallback"""
        if not os.path.exists(self.password_file):
            messagebox.showwarning("No Password Set", "Please set a password first!")
            return
        
        password = simpledialog.askstring("Password Authentication", 
                                         "Enter your password:", 
                                         show='*')
        
        if password:
            if self.verify_password(password):
                self.authentication_success("password")
            else:
                self.add_response("❌ Incorrect password")
                messagebox.showerror("Authentication Failed", "Incorrect password!")
        else:
            self.add_response("🚫 Password authentication cancelled")
    
    def set_password(self):
        """Set password for fallback authentication"""
        password = simpledialog.askstring("Set Password", 
                                         "Enter a new password:", 
                                         show='*')
        
        if password and len(password) >= 6:
            confirm_password = simpledialog.askstring("Confirm Password", 
                                                     "Confirm your password:", 
                                                     show='*')
            
            if password == confirm_password:
                self.save_password(password)
                self.add_response("✅ Password set successfully")
                messagebox.showinfo("Success", "Password set successfully!")
            else:
                self.add_response("❌ Passwords don't match")
                messagebox.showerror("Error", "Passwords don't match!")
        elif password:
            self.add_response("❌ Password must be at least 6 characters")
            messagebox.showerror("Error", "Password must be at least 6 characters!")
    
    def save_password(self, password):
        """Save password hash"""
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            with open(self.password_file, 'w') as f:
                f.write(password_hash)
        except Exception as e:
            self.add_response(f"❌ Failed to save password: {str(e)}")
    
    def verify_password(self, password):
        """Verify password"""
        try:
            with open(self.password_file, 'r') as f:
                stored_hash = f.read().strip()
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            return password_hash == stored_hash
        except:
            return False
    
    def authentication_success(self, method):
        """Handle successful authentication"""
        self.is_authenticated = True
        self.auth_attempts = 0
        
        self.add_response(f"✅ Authentication successful via {method}!")
        self.update_status("Authenticated", '#107c10')
        self.auth_status_label.config(text="🔓 Authenticated", fg='#00ff41')
        
        # Enable controls
        self.activate_btn.config(state='normal')
        
        self.speak_async(f"Authentication successful via {method}. Ready to activate.")
    
    def reset_voice_data(self):
        """Reset all voice data"""
        if messagebox.askyesno("Reset Voice Data", 
                              "Are you sure you want to delete all voice profiles and custom phrases? This cannot be undone!"):
            try:
                if os.path.exists(self.voice_auth.voice_profiles_file):
                    os.remove(self.voice_auth.voice_profiles_file)
                if os.path.exists(self.voice_auth.voice_features_file):
                    os.remove(self.voice_auth.voice_features_file)
                
                self.voice_auth.voice_profiles = {}
                self.voice_auth.custom_phrase = None
                self.voice_auth_btn.config(state='disabled')
                self.enroll_btn.config(state='disabled', text="🎙️ Train Voice (5 samples)")
                
                # Reset phrase display
                self.current_phrase_label.config(
                    text="❌ No custom phrase set - Click 'Set Custom Phrase' first",
                    fg='#ff6b6b'
                )
                
                self.add_response("🗑️ All voice data has been reset")
                self.add_response("� Step 1: Set your custom phrase")
                self.add_response("🎯 Step 2: Train your voice with 5 samples")
                messagebox.showinfo("Reset Complete", "All voice data has been reset! Please set up your custom phrase and retrain your voice.")
                
            except Exception as e:
                self.add_response(f"❌ Failed to reset voice data: {str(e)}")
    
    # Core Functions (similar to previous version)
    def speak_async(self, text):
        """Non-blocking speech"""
        def speak_worker():
            try:
                if self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except:
                pass
        self.executor.submit(speak_worker)
    
    def add_response(self, message):
        """Add message to response log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\\n"
        
        self.response_text.insert(tk.END, formatted_message)
        
        # Color coding
        if "✅" in message:
            start_idx = self.response_text.index(f"end-{len(formatted_message)}c")
            end_idx = self.response_text.index("end-1c")
            self.response_text.tag_add("success", start_idx, end_idx)
            self.response_text.tag_config("success", foreground="#00ff41")
        elif "❌" in message:
            start_idx = self.response_text.index(f"end-{len(formatted_message)}c")
            end_idx = self.response_text.index("end-1c")
            self.response_text.tag_add("error", start_idx, end_idx)
            self.response_text.tag_config("error", foreground="#ff4444")
        
        self.response_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_status(self, status, color='#d13438'):
        """Update status indicator"""
        self.current_status = status
        self.status_label.config(text=f"Status: {status}")
        self.status_indicator.itemconfig(self.status_circle, fill=color)
        self.root.update_idletasks()
    
    def activate_assistant(self):
        """Activate assistant after authentication"""
        if not self.is_authenticated:
            messagebox.showwarning("Authentication Required", "Please authenticate first!")
            return
        
        self.is_activated = True
        self.update_status("Activated", '#0078d4')
        self.add_response("🚀 SecureAI Assistant activated!")
        
        self.listen_btn.config(state='normal')
        self.activate_btn.config(text="✅ Active", state='disabled')
        
        self.greet_user()
    
    def greet_user(self):
        """Greet user"""
        hour = int(datetime.now().hour)
        if 0 <= hour < 12:
            greeting = "Good morning! SecureAI is ready to assist you."
        elif 12 <= hour < 18:
            greeting = "Good afternoon! SecureAI is ready to assist you."
        else:
            greeting = "Good evening! SecureAI is ready to assist you."
        
        self.add_response(f"👋 {greeting}")
        self.speak_async(greeting)
    
    # Voice Command Processing
    def toggle_listening(self):
        """Toggle voice listening"""
        if not self.is_activated:
            messagebox.showwarning("Not Activated", "Please activate assistant first!")
            return
        
        if not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()
    
    def start_listening(self):
        """Start voice recognition"""
        self.is_listening = True
        self.listen_btn.config(text="🛑 Stop Listening")
        self.update_status("Listening...", '#0078d4')
        self.add_response("👂 Voice recognition activated")
        
        def listen_worker():
            while self.is_listening:
                try:
                    with sr.Microphone() as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=5)
                    
                    self.update_status("Processing...", '#ffaa00')
                    query = self.recognizer.recognize_google(audio, language="en-in")
                    self.add_response(f"🗣️ You said: '{query}'")
                    
                    self.process_voice_command(query.lower())
                    
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.add_response("❓ Could not understand")
                except Exception as e:
                    self.add_response(f"❌ Recognition error: {str(e)}")
                
                if self.is_listening:
                    self.update_status("Listening...", '#0078d4')
                    time.sleep(0.5)
        
        threading.Thread(target=listen_worker, daemon=True).start()
    
    def stop_listening(self):
        """Stop voice recognition"""
        self.is_listening = False
        self.listen_btn.config(text="👂 Start Listening")
        self.update_status("Activated", '#107c10')
        self.add_response("🛑 Voice recognition stopped")
    
    def process_voice_command(self, query):
        """Process voice commands"""
        try:
            if "hello" in query or "hi" in query:
                response = "Hello! How can I help you?"
                self.add_response(f"🤖 {response}")
                self.speak_async(response)
            elif "time" in query:
                self.get_time()
            elif "weather" in query:
                self.get_weather()
            elif "open" in query:
                for app in self.apps.keys():
                    if app in query:
                        self.open_app(app)
                        return
                self.add_response("❓ Which app would you like to open?")
            elif "volume up" in query:
                self.volume_up()
            elif "volume down" in query:
                self.volume_down()
            elif "mute" in query:
                self.volume_mute()
            elif "screenshot" in query:
                self.take_screenshot()
            elif "lock screen" in query:
                self.lock_screen()
            elif "go to sleep" in query or "stop" in query:
                self.stop_listening()
                self.speak_async("Going to sleep mode")
            else:
                self.add_response("🤖 I'm not sure how to help with that")
        except Exception as e:
            self.add_response(f"❌ Command error: {str(e)}")
    
    # Quick Action Functions (same as previous version)
    def get_time(self):
        current_time = datetime.now().strftime("%I:%M %p")
        response = f"Current time is {current_time}"
        self.add_response(f"🕐 {response}")
        self.speak_async(response)
    
    def get_weather(self):
        def weather_worker():
            try:
                self.add_response("🌤️ Fetching weather...")
                url = "https://www.google.com/search?q=weather"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                response = requests.get(url, headers=headers, timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                temp = soup.find("div", class_="BNeawe iBp4i AP7Wnd")
                desc = soup.find("div", class_="BNeawe tAd8D AP7Wnd")
                
                if temp and desc:
                    weather_info = f"Weather: {desc.text}, Temperature: {temp.text}"
                    self.add_response(f"🌤️ {weather_info}")
                    self.speak_async(weather_info)
                else:
                    self.add_response("⚠️ Weather not available")
            except:
                self.add_response("❌ Weather fetch failed")
        
        self.executor.submit(weather_worker)
    
    def open_website(self, site):
        if site in self.websites:
            webbrowser.open(self.websites[site])
            self.add_response(f"🌐 {site.title()} opened")
            self.speak_async(f"{site} opened")
    
    def open_app(self, app_name):
        def app_worker():
            try:
                if app_name in self.apps:
                    subprocess.Popen(self.apps[app_name]["cmd"], shell=True)
                    self.add_response(f"🚀 {app_name.title()} launched")
                    self.speak_async(f"{app_name} opened")
            except Exception as e:
                self.add_response(f"❌ Error opening {app_name}")
        
        self.executor.submit(app_worker)
    
    # Media Controls
    def volume_up(self):
        def volume_worker():
            try:
                if self.windows_api_available:
                    hwnd = self.user32.GetForegroundWindow()
                    for _ in range(3):
                        self.user32.SendMessageW(hwnd, self.WM_APPCOMMAND, hwnd, self.APPCOMMAND_VOLUME_UP << 16)
                        time.sleep(0.05)
                else:
                    for _ in range(3):
                        self.keyboard.press(Key.media_volume_up)
                        self.keyboard.release(Key.media_volume_up)
                        time.sleep(0.05)
                
                self.add_response("🔊 Volume increased")
                self.speak_async("Volume up")
            except:
                self.add_response("❌ Volume control error")
        
        self.executor.submit(volume_worker)
    
    def volume_down(self):
        def volume_worker():
            try:
                if self.windows_api_available:
                    hwnd = self.user32.GetForegroundWindow()
                    for _ in range(3):
                        self.user32.SendMessageW(hwnd, self.WM_APPCOMMAND, hwnd, self.APPCOMMAND_VOLUME_DOWN << 16)
                        time.sleep(0.05)
                else:
                    for _ in range(3):
                        self.keyboard.press(Key.media_volume_down)
                        self.keyboard.release(Key.media_volume_down)
                        time.sleep(0.05)
                
                self.add_response("🔉 Volume decreased")
                self.speak_async("Volume down")
            except:
                self.add_response("❌ Volume control error")
        
        self.executor.submit(volume_worker)
    
    def volume_mute(self):
        def mute_worker():
            try:
                if self.windows_api_available:
                    hwnd = self.user32.GetForegroundWindow()
                    self.user32.SendMessageW(hwnd, self.WM_APPCOMMAND, hwnd, self.APPCOMMAND_VOLUME_MUTE << 16)
                else:
                    self.keyboard.press(Key.media_volume_mute)
                    self.keyboard.release(Key.media_volume_mute)
                
                self.add_response("🔇 Volume mute toggled")
                self.speak_async("Mute toggled")
            except:
                self.add_response("❌ Mute control error")
        
        self.executor.submit(mute_worker)
    
    def media_toggle(self):
        def media_worker():
            try:
                self.keyboard.press(Key.media_play_pause)
                self.keyboard.release(Key.media_play_pause)
                self.add_response("▶️ Media toggled")
                self.speak_async("Media toggled")
            except:
                self.add_response("❌ Media control error")
        
        self.executor.submit(media_worker)
    
    # System Functions
    def take_screenshot(self):
        def screenshot_worker():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot = pyautogui.screenshot()
                filename = f"screenshot_{timestamp}.png"
                screenshot.save(filename)
                self.add_response(f"📸 Screenshot saved as {filename}")
                self.speak_async("Screenshot taken")
            except:
                self.add_response("❌ Screenshot failed")
        
        self.executor.submit(screenshot_worker)
    
    def lock_screen(self):
        try:
            pyautogui.hotkey("win", "l")
            self.add_response("🔒 Screen locked")
            self.speak_async("Screen locked")
        except:
            self.add_response("❌ Screen lock failed")
    
    def show_system_status(self):
        def status_worker():
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                status_info = f"💻 CPU: {cpu_percent}% | Memory: {memory.percent}%"
                self.add_response(status_info)
                self.speak_async(f"CPU {cpu_percent} percent, Memory {memory.percent} percent")
            except:
                self.add_response("❌ System status error")
        
        self.executor.submit(status_worker)
    
    # Utility Functions
    def clear_log(self):
        self.response_text.delete(1.0, tk.END)
        self.add_response("🗑️ Log cleared")
    
    def save_log(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"secureai_log_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.response_text.get(1.0, tk.END))
            self.add_response(f"💾 Log saved as {filename}")
        except:
            self.add_response("❌ Save failed")

def main():
    """Main function"""
    print("🚀 Starting SecureAI Voice Assistant - Enhanced Authentication...")
    
    root = tk.Tk()
    
    # Center window
    width, height = 1200, 900
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    app = SecureAI_Assistant(root)
    
    def on_closing():
        if messagebox.askokcancel("Quit SecureAI", 
                                "Are you sure you want to quit?"):
            try:
                app.executor.shutdown(wait=False)
                if app.engine:
                    app.engine.stop()
            except:
                pass
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    print("✅ SecureAI Enhanced Authentication Ready!")
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Failed to start SecureAI: {e}")
        input("Press Enter to exit...")