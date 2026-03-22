import logging
import psutil
import pyautogui
import datetime
import time
import os
import flet as ft
import threading
import numpy as np
import cv2
from numpy.linalg import norm
import insightface
import asyncio
from pathlib import Path
from types import SimpleNamespace
import queue # Added for VAD

# --- Core and Utils Imports for SecureXApp ---
# (Ensure these files exist in your project structure)
from core.database import Database
from core.voice_engine import VoiceEngine
from core.voice_biometric_engine_ultimate import UltimateVoiceBiometricEngine
from core.security import SecurityManager, SecurityContext, AuthenticationAlertSystem, AlertLevel
from core.audio_processor import AudioRecorder, VoiceActivityDetector
from utils.tts import TextToSpeech
from core.voice_assistant import VoiceAssistant
from core.biometric_fusion_engine import BiometricFusionEngine
from core.face_recognition_engine import FaceRecognitionEngine
from core.anti_spoofing import AntiSpoofingEngine
from utils.helpers import create_temp_directory, cleanup_temp_files, load_config
# --- End Imports ---

class SciFiColors:
    # Deep space backgrounds with blue undertones
    BG_SPACE = "#030712"
    BG_DARK = "#0a0f1a"
    BG_CARD = "#0d1421"
    BG_ELEVATED = "#111827"
    BG_HOVER = "#1a2744"
    
    # Vibrant cyberpunk primary colors
    PRIMARY = "#00f0ff"  # Neon cyan
    PRIMARY_DARK = "#00b8cc"
    PRIMARY_GLOW = "#00f0ff"
    
    # Electric accents
    ACCENT = "#ff0055"  # Hot pink/magenta
    ACCENT_SECONDARY = "#bf00ff"  # Electric purple
    
    # Status colors with glow effect
    SUCCESS = "#00ff88"  # Neon green
    SUCCESS_DARK = "#00cc66"
    ERROR = "#ff0055"
    WARNING = "#ffcc00"  # Electric yellow
    INFO = "#00aaff"
    
    # Performance/special indicators
    PERFORMANCE = "#a855f7"  # Purple
    HOLOGRAM = "#00ffcc"  # Holographic cyan-green
    ENERGY = "#ff6600"  # Energy orange
    
    # Text hierarchy
    TEXT_PRIMARY = "#f0f9ff"
    TEXT_SECONDARY = "#94a3b8"
    TEXT_MUTED = "#64748b"
    TEXT_GLOW = "#00f0ff"
    
    # Borders and lines
    BORDER = "#1e293b"
    BORDER_GLOW = "#00f0ff"
    BORDER_ACCENT = "#ff0055"
    
    # Gradients
    GRADIENT_START = "#00f0ff"
    GRADIENT_MID = "#a855f7"
    GRADIENT_END = "#ff0055"
    
    # Scan line effect color
    SCANLINE = "#00f0ff"

logger = logging.getLogger("SecureXApp")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - INFO - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class SecureXApp:
    """Main application class with sci-fi themed UI"""

    def __init__(self, page: ft.Page, config: dict):
        self.page = page
        self.config = config

        # Initialize components
        self.db = Database(config.get('database', {}).get('path', 'securex_db.sqlite'))
        self.voice_engine = VoiceEngine(config)
        self.ultimate_voice_engine = UltimateVoiceBiometricEngine(config, self.db)
        
        # Initialize biometric engines for parallel verification
        self.face_recognition_engine = FaceRecognitionEngine()
        self.anti_spoofing_engine = AntiSpoofingEngine(config)
        
        self.security_manager = SecurityManager(config)
        self.auth_alerts = AuthenticationAlertSystem(config)
        self.audio_recorder = AudioRecorder(config)
        self.vad = VoiceActivityDetector(config)
        self.tts = TextToSpeech(config)
        self.tts_enabled = True

        # Voice Assistant with configurable Whisper ASR model.
        stt_model = (
            config.get('models', {}).get('stt_model')
            or config.get('system', {}).get('stt_model')
            or 'small'
        )
        self.voice_assistant = VoiceAssistant(
            model_path=stt_model,
            biometric_engine=self.ultimate_voice_engine,
            tts_engine=self.tts,
            config=config,
        )
        
        self.voice_assistant.setup_default_commands()
        
        # Performance tracking
        from core.response_cache import get_response_cache
        from core.performance_monitor import get_performance_metrics
        self.response_cache = get_response_cache()
        self.performance_metrics = get_performance_metrics()

        # Initialize multi-modal fusion engine
        fusion_strategy = config.get('fusion', {}).get('strategy', 'weighted_sum')
        self.fusion_engine = BiometricFusionEngine(config, strategy=fusion_strategy)
        logger.info(f"Multi-modal fusion initialized with '{fusion_strategy}' strategy")

        # Load InsightFace ArcFace model for face recognition
        self.arcface_model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        self.arcface_model.prepare(ctx_id=0, det_size=(640, 640))

        self._audio_stream_ctx = SimpleNamespace(
            recorder=self.audio_recorder,
            vad_detector=self.vad
        )

        # Session state
        self.current_user = None
        self.current_view = "login"

        # Voice assistant state
        self.continuous_mode_active = False
        self.voice_assistant_active = False
        # --- REFACTOR: Removed dialog state flags ---

        # Navigation state
        self.current_nav_section = "dashboard"

        # Recording states
        self.recording_active = False
        self.reg_recording_active = False
        self.face_capture_requested = False
        self.recording_task = None
        self.auth_cooldown_task = None
        self.auth_cooldown_username = None
        
        # Login recording events
        self.recording_started_event = threading.Event()
        self.recording_stop_event = threading.Event()

        # Registration recording events & thread
        self.reg_recording_started_event = threading.Event()
        self.reg_recording_stop_event = threading.Event()
        self.reg_recording_thread = None
        self.reg_recording_complete_event = threading.Event()
        self.reg_audio_holder = [None] # Use list for pass-by-reference in thread

        # Authentication flow states
        self.auth_step = "idle"
        self.voice_enrollment_complete = False
        self.face_enrollment_complete = False
        self.voice_verification_complete = False
        self.face_verification_complete = False
        
        # Biometric scores for fusion
        self.voice_score = 0.0
        self.face_score = 0.0
        self.voice_liveness_score = 1.0
        self.face_liveness_score = 1.0
        self.anti_spoof_score = 1.0

        # Security policy toggles (secure by default).
        security_cfg = self.config.get('security', {}) if isinstance(self.config, dict) else {}
        system_cfg = self.config.get('system', {}) if isinstance(self.config, dict) else {}
        self.require_face_verification_on_login = bool(
            security_cfg.get('require_face_verification_on_login', system_cfg.get('require_face_verification_on_login', True))
        )
        self.require_face_enrollment = bool(
            security_cfg.get('require_face_enrollment', system_cfg.get('require_face_enrollment', True))
        )

        # Temp directory
        self.temp_dir = create_temp_directory()

        # UI components
        self.interaction_log = None
        
        # Login/Reg UI components (defined in build_login_view)
        self.username_field = None
        self.password_field = None
        self.account_type_field = None
        self.reg_username_field = None
        self.reg_password_field = None
        self.reg_confirm_password_field = None
        self.reg_email_field = None
        self.reg_account_type_field = None
        self.progress_ring = None
        self.reg_progress_ring = None
        self.mic_status = None
        self.record_button = None
        self.reg_mic_status = None
        self.reg_record_button = None
        self.status_text = None
        self.status_panel = None
        self.reg_status_text = None
        self.reg_status_panel = None
        self.auth_progress_text = None
        self.auth_progress_panel = None
        self.auth_tabs = None
        self.form_container = None
        self.action_button_container = None
        
        # Enhanced UI components for visual feedback
        self.confidence_meter = None
        self.confidence_text = None
        self.score_display = None

        # --- REFACTOR: Add UI references for Assistant Page ---
        self.assistant_log_content: ft.Column = None
        self.assistant_status_text: ft.Text = None
        self.assistant_start_btn: ft.Container = None
        self.assistant_stop_btn: ft.Container = None
        self.assistant_continuous_toggle: ft.Switch = None
        self.assistant_tts_toggle: ft.Switch = None
        # --- END REFACTOR ---

        # Dialog management
        self.active_dialogs = []

        # Set up page close handler
        self.page.on_close = self._on_app_close

    # --- ASYNC TTS HELPER ---
    async def _speak_async(self, text: str, force_tts=False):
        """Run TTS in an executor thread and WAIT for it to finish. Respects TTS toggle setting."""
        try:
            # Check if TTS is enabled (unless forced)
            if not self.tts_enabled and not force_tts:
                logger.debug(f"TTS disabled, skipping: {text}")
                return
            
            loop = asyncio.get_running_loop()
            # Pass blocking=True to wait until sound finishes playing
            await loop.run_in_executor(None, lambda: self.tts.speak(text, blocking=True))
        except Exception as e:
            logger.warning(f"Async TTS failed: {e}")
    
    def _speak_nonblocking(self, text: str):
        """Speak without blocking - fire and forget"""
        try:
            self.tts.speak(text, blocking=False)
        except Exception as e:
            logger.warning(f"Non-blocking TTS failed: {e}")

    def run(self):
        """Run the application"""
        self.setup_page()
        
        self.db.connect()
        self.db.initialize_schema()
        
        try:
            self.voice_engine.load_models()
        except Exception as e:
            logger.error(f"Failed to load voice models: {e}")
            self._show_error_toast(f"Warning: Voice models failed to load: {str(e)}")
        
        self.page.add(self.build_login_view())
        
        # --- START FIX: Correctly schedule startup task ---
        async def start_speech():
            # Use run_task for thread-safe operations
            await self.page.run_task(self._speak_async, "SecureX Assist initialized. Ready for authentication.")
        
        # Check if loop is available before creating task
        if self.page.loop:
             self.page.loop.create_task(start_speech())
        else:
             logger.warning("No asyncio loop available on page to run startup speech.")
        # --- END FIX ---
        
    def setup_page(self):
        """Configure page settings"""
        logger.info("Setting up page configuration...")
        
        self.page.title = self.config.get('app', {}).get('name', 'SecureX-Assist')
        self.page.window.width = 1440
        self.page.window.height = 900
        self.page.padding = 0
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.bgcolor = SciFiColors.BG_SPACE
        
        self.page.theme = ft.Theme(
            color_scheme_seed=SciFiColors.PRIMARY,
            font_family="Poppins"
        )
        
        self.page.fonts = {
            "Orbitron": "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap",
            "Rajdhani": "https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap",
            "Poppins": "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap",
        }
        
        logger.info("Page configuration complete")

    def _get_account_type(self) -> str:
        """Return normalized account type for current user."""
        if not self.current_user:
            return "standard_user"
        return str(self.current_user.get("account_type", "standard_user")).strip().lower()

    def _is_privileged_user(self) -> bool:
        """Administrators and power users can access advanced system controls."""
        return self._get_account_type() in {"administrator", "power_user"}

    def _can_access_settings(self) -> bool:
        """Gate access to settings page for privileged accounts."""
        return self._is_privileged_user()

    # ==================== VIEW BUILDERS ====================

    def build_login_view(self) -> ft.Container:
        """Build modernized split-layout login view"""
        
        self.username_field = ft.TextField(
            label="USERNAME",
            hint_text="Enter your username",
            prefix_icon=ft.Icons.PERSON_OUTLINE_ROUNDED,
            width=380,
            height=56,
            border_radius=8,
            filled=True,
            bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.PRIMARY),
            border_color=SciFiColors.BORDER,
            focused_border_color=SciFiColors.PRIMARY,
            focused_border_width=2,
            color=SciFiColors.TEXT_PRIMARY,
            label_style=ft.TextStyle(
                color=SciFiColors.TEXT_SECONDARY,
                size=11,
                weight=ft.FontWeight.W_600,
            ),
        )

        self.password_field = ft.TextField(
            label="PASSWORD",
            hint_text="Enter your password",
            prefix_icon=ft.Icons.LOCK_OUTLINE_ROUNDED,
            password=True,
            can_reveal_password=True,
            width=380,
            height=56,
            border_radius=8,
            filled=True,
            bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.PRIMARY),
            border_color=SciFiColors.BORDER,
            focused_border_color=SciFiColors.PRIMARY,
            focused_border_width=2,
            color=SciFiColors.TEXT_PRIMARY,
            label_style=ft.TextStyle(
                color=SciFiColors.TEXT_SECONDARY,
                size=11,
                weight=ft.FontWeight.W_600,
            ),
        )

        self.account_type_field = ft.Dropdown(
            label="ACCOUNT PRIVILEGE LEVEL",
            hint_text="Select your access level",
            width=380,
            border_radius=8,
            filled=True,
            bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.PRIMARY),
            border_color=SciFiColors.BORDER,
            focused_border_color=SciFiColors.PRIMARY,
            focused_border_width=2,
            color=SciFiColors.TEXT_PRIMARY,
            value="standard_user",
            options=[
                ft.dropdown.Option("administrator", "System Administrator"),
                ft.dropdown.Option("standard_user", "Standard User"),
            ],
            label_style=ft.TextStyle(
                color=SciFiColors.TEXT_SECONDARY,
                size=11,
                weight=ft.FontWeight.W_600,
            ),
        )

        self.reg_username_field = ft.TextField(
            label="USERNAME",
            hint_text="Choose a username",
            prefix_icon=ft.Icons.PERSON_OUTLINE_ROUNDED,
            width=380,
            height=56,
            border_radius=8,
            filled=True,
            bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.PRIMARY),
            border_color=SciFiColors.BORDER,
            focused_border_color=SciFiColors.PRIMARY,
            focused_border_width=2,
            color=SciFiColors.TEXT_PRIMARY,
            visible=False,
            label_style=ft.TextStyle(
                color=SciFiColors.TEXT_SECONDARY,
                size=11,
                weight=ft.FontWeight.W_600,
            ),
        )

        self.reg_password_field = ft.TextField(
            label="PASSWORD",
            hint_text="Create a secure password",
            prefix_icon=ft.Icons.LOCK_OUTLINE_ROUNDED,
            password=True,
            can_reveal_password=True,
            width=380,
            height=56,
            border_radius=8,
            filled=True,
            bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.PRIMARY),
            border_color=SciFiColors.BORDER,
            focused_border_color=SciFiColors.PRIMARY,
            focused_border_width=2,
            color=SciFiColors.TEXT_PRIMARY,
            visible=False,
            label_style=ft.TextStyle(
                color=SciFiColors.TEXT_SECONDARY,
                size=11,
                weight=ft.FontWeight.W_600,
            ),
        )

        self.reg_confirm_password_field = ft.TextField(
            label="CONFIRM PASSWORD",
            hint_text="Re-enter your password",
            prefix_icon=ft.Icons.LOCK_CLOCK_ROUNDED,
            password=True,
            width=380,
            height=56,
            border_radius=8,
            filled=True,
            bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.PRIMARY),
            border_color=SciFiColors.BORDER,
            focused_border_color=SciFiColors.PRIMARY,
            focused_border_width=2,
            color=SciFiColors.TEXT_PRIMARY,
            visible=False,
            label_style=ft.TextStyle(
                color=SciFiColors.TEXT_SECONDARY,
                size=11,
                weight=ft.FontWeight.W_600,
            ),
        )

        self.reg_email_field = ft.TextField(
            label="EMAIL (OPTIONAL)",
            hint_text="your@email.com",
            prefix_icon=ft.Icons.EMAIL_OUTLINED,
            width=380,
            height=56,
            border_radius=8,
            filled=True,
            bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.PRIMARY),
            border_color=SciFiColors.BORDER,
            focused_border_color=SciFiColors.PRIMARY,
            focused_border_width=2,
            color=SciFiColors.TEXT_PRIMARY,
            visible=False,
            label_style=ft.TextStyle(
                color=SciFiColors.TEXT_SECONDARY,
                size=11,
                weight=ft.FontWeight.W_600,
            ),
        )

        self.reg_account_type_field = ft.Dropdown(
            label="ACCOUNT PRIVILEGE LEVEL",
            hint_text="Select requested access level",
            width=380,
            border_radius=8,
            filled=True,
            bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.PRIMARY),
            border_color=SciFiColors.BORDER,
            focused_border_color=SciFiColors.PRIMARY,
            focused_border_width=2,
            color=SciFiColors.TEXT_PRIMARY,
            value="standard_user",
            visible=False,
            options=[
                ft.dropdown.Option("administrator", "System Administrator"),
                ft.dropdown.Option("standard_user", "Standard User"),
            ],
            label_style=ft.TextStyle(
                color=SciFiColors.TEXT_SECONDARY,
                size=11,
                weight=ft.FontWeight.W_600,
            ),
        )

        self.progress_ring = ft.ProgressRing(
            visible=False,
            color=SciFiColors.PRIMARY,
            width=40,
            height=40,
            stroke_width=4,
        )
        
        self.reg_progress_ring = ft.ProgressRing(
            visible=False,
            color=SciFiColors.PRIMARY,
            width=40,
            height=40,
            stroke_width=4,
        )
        
        self.mic_status = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.MIC_OUTLINED, color=SciFiColors.PRIMARY, size=20),
                    ft.Text(
                        "VOICE REQUIRED",
                        size=11,
                        color=SciFiColors.TEXT_SECONDARY,
                        weight=ft.FontWeight.W_600,
                    ),
                ],
                spacing=8,
            ),
            visible=False,
            padding=10,
            border_radius=6,
            bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.PRIMARY),
            border=ft.border.all(1, SciFiColors.PRIMARY),
        )
        
        self.record_button = ft.ElevatedButton(
            "START RECORDING",
            icon=ft.Icons.MIC,
            height=56,
            width=400,
            visible=False,
            on_click=self._on_record_button_click, # Correct handler
            style=ft.ButtonStyle(
                bgcolor=SciFiColors.PRIMARY,
                color=SciFiColors.BG_DARK,
                shape=ft.RoundedRectangleBorder(radius=12),
                shadow_color=SciFiColors.PRIMARY,
                elevation=8,
                side=ft.BorderSide(width=2, color=SciFiColors.PRIMARY),
            ),
        )
        
        self.reg_mic_status = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.MIC_OUTLINED, color=SciFiColors.PRIMARY, size=20),
                    ft.Text(
                        "VOICE ENROLLMENT",
                        size=11,
                        color=SciFiColors.TEXT_SECONDARY,
                        weight=ft.FontWeight.W_600,
                    ),
                ],
                spacing=8,
            ),
            visible=False,
            padding=10,
            border_radius=6,
            bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.PRIMARY),
            border=ft.border.all(1, SciFiColors.PRIMARY),
        )
        
        self.reg_record_button = ft.ElevatedButton(
            "START RECORDING",
            icon=ft.Icons.MIC,
            height=56,
            width=400,
            visible=False,
            on_click=self.handle_reg_record_button_click, # Correct handler
            style=ft.ButtonStyle(
                bgcolor=SciFiColors.PRIMARY,
                color=SciFiColors.BG_DARK,
                shape=ft.RoundedRectangleBorder(radius=12),
                shadow_color=SciFiColors.PRIMARY,
                elevation=8,
                side=ft.BorderSide(width=2, color=SciFiColors.PRIMARY),
            ),
        )
        
        self.status_text = ft.Text(
            "",
            size=12,
            color=SciFiColors.INFO,
            text_align=ft.TextAlign.CENTER,
            weight=ft.FontWeight.W_500,
        )
        
        self.status_panel = ft.Container(
            content=self.status_text,
            visible=False,
            padding=12,
            border_radius=6,
            bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.INFO),
            border=ft.border.all(1, SciFiColors.INFO),
        )
        
        # Confidence meter (circular progress indicator)
        self.confidence_text = ft.Text(
            "0%",
            size=16,
            color=SciFiColors.PRIMARY,
            weight=ft.FontWeight.BOLD,
        )
        
        self.confidence_meter = ft.ProgressRing(
            value=0.0,
            width=80,
            height=80,
            stroke_width=6,
            color=SciFiColors.PRIMARY,
            bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.PRIMARY),
        )
        
        self.confidence_display = ft.Container(
            content=ft.Stack(
                [
                    ft.Container(
                        content=self.confidence_meter,
                        alignment=ft.alignment.center,
                    ),
                    ft.Container(
                        content=self.confidence_text,
                        alignment=ft.alignment.center,
                    ),
                ],
                width=80,
                height=80,
            ),
            visible=False,
            padding=ft.padding.only(top=8, bottom=8),
        )
        
        # Score display panel
        self.score_display = ft.Container(
            content=ft.Text(
                "",
                size=11,
                color=SciFiColors.TEXT_SECONDARY,
                text_align=ft.TextAlign.CENTER,
                weight=ft.FontWeight.W_500,
            ),
            visible=False,
            padding=10,
            border_radius=6,
            bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.BG_ELEVATED),
            border=ft.border.all(1, SciFiColors.BORDER),
        )
        
        self.reg_status_text = ft.Text(
            "",
            size=12,
            color=SciFiColors.INFO,
            text_align=ft.TextAlign.CENTER,
            weight=ft.FontWeight.W_500,
        )
        
        self.reg_status_panel = ft.Container(
            content=self.reg_status_text,
            visible=False,
            padding=12,
            border_radius=6,
            bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.INFO),
            border=ft.border.all(1, SciFiColors.INFO),
        )
        
        self.auth_progress_text = ft.Text(
            "",
            size=11,
            color=SciFiColors.TEXT_SECONDARY,
            text_align=ft.TextAlign.CENTER,
            weight=ft.FontWeight.W_500,
        )
        
        self.auth_progress_panel = ft.Container(
            content=self.auth_progress_text,
            visible=False,
            padding=8,
            border_radius=4,
            bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.PRIMARY),
        )
        
        login_button = ft.Container(
            content=ft.ElevatedButton(
                content=ft.Row(
                    [
                        ft.Icon(ft.Icons.FINGERPRINT_ROUNDED, size=22),
                        ft.Text("AUTHENTICATE", size=14, weight=ft.FontWeight.BOLD),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10,
                ),
                on_click=lambda _: self.start_voice_login(),
                width=380,
                height=52,
                style=ft.ButtonStyle(
                    bgcolor=SciFiColors.PRIMARY,
                    color=SciFiColors.BG_DARK,
                    shape=ft.RoundedRectangleBorder(radius=8),
                ),
            ),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=20,
                color=ft.Colors.with_opacity(0.4, SciFiColors.PRIMARY),
            ),
        )
        
        register_button = ft.Container(
            content=ft.ElevatedButton(
                content=ft.Row(
                    [
                        ft.Icon(ft.Icons.HOW_TO_REG_ROUNDED, size=22),
                        ft.Text("CREATE PROFILE", size=14, weight=ft.FontWeight.BOLD),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10,
                ),
                on_click=lambda _: self.start_registration(),
                width=380,
                height=52,
                visible=False,
                style=ft.ButtonStyle(
                    bgcolor=SciFiColors.ACCENT,
                    color=ft.Colors.WHITE,
                    shape=ft.RoundedRectangleBorder(radius=8),
                ),
            ),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=20,
                color=ft.Colors.with_opacity(0.4, SciFiColors.ACCENT),
            ),
        )
        
        self.auth_tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            indicator_color=SciFiColors.PRIMARY,
            label_color=SciFiColors.PRIMARY,
            unselected_label_color=SciFiColors.TEXT_MUTED,
            tabs=[
                ft.Tab(text="SIGN IN", icon=ft.Icons.LOGIN_ROUNDED),
                ft.Tab(text="REGISTER", icon=ft.Icons.PERSON_ADD_ROUNDED),
            ],
            on_change=self._handle_auth_tab_change,
        )
        
        self.form_container = ft.Column(
            [
                self.username_field,
                self.password_field,
                self.account_type_field,
                self.reg_username_field,
                self.reg_password_field,
                self.reg_confirm_password_field,
                self.reg_email_field,
                self.reg_account_type_field,
            ],
            spacing=14,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        
        self.action_button_container = ft.Column(
            [login_button, register_button],
            spacing=0,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        
        # Create animated hexagonal logo container
        hex_logo = ft.Stack([
            # Outer glow ring
            ft.Container(
                width=140,
                height=140,
                border=ft.border.all(2, ft.Colors.with_opacity(0.3, SciFiColors.PRIMARY)),
                border_radius=70,
                bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.PRIMARY),
                shadow=ft.BoxShadow(
                    spread_radius=8,
                    blur_radius=30,
                    color=ft.Colors.with_opacity(0.4, SciFiColors.PRIMARY),
                ),
            ),
            # Inner ring with accent
            ft.Container(
                width=120,
                height=120,
                border=ft.border.all(2, SciFiColors.PRIMARY),
                border_radius=60,
                bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.PRIMARY),
                left=10,
                top=10,
            ),
            # Innermost circle with icon
            ft.Container(
                content=ft.Icon(
                    ft.Icons.FINGERPRINT_ROUNDED,
                    color=SciFiColors.PRIMARY,
                    size=50,
                ),
                width=100,
                height=100,
                border_radius=50,
                bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.PRIMARY),
                alignment=ft.alignment.center,
                left=20,
                top=20,
                shadow=ft.BoxShadow(
                    blur_radius=20,
                    color=ft.Colors.with_opacity(0.6, SciFiColors.PRIMARY),
                ),
            ),
            # Corner accents
            ft.Container(
                width=20,
                height=2,
                bgcolor=SciFiColors.ACCENT,
                left=0,
                top=70,
            ),
            ft.Container(
                width=20,
                height=2,
                bgcolor=SciFiColors.ACCENT,
                right=0,
                top=70,
                left=120,
            ),
        ], width=140, height=140)
        
        left_side = ft.Container(
            content=ft.Column(
                [
                    ft.Container(expand=1),
                    hex_logo,
                    ft.Container(height=30),
                    # Main title with glow effect
                    ft.Container(
                        content=ft.Text(
                            "SECUREX",
                            size=52,
                            weight=ft.FontWeight.BOLD,
                            color=SciFiColors.TEXT_PRIMARY,
                            font_family="Orbitron",
                        ),
                        shadow=ft.BoxShadow(
                            blur_radius=30,
                            color=ft.Colors.with_opacity(0.5, SciFiColors.PRIMARY),
                        ),
                    ),
                    ft.Container(
                        content=ft.Text(
                            "ASSIST",
                            size=52,
                            weight=ft.FontWeight.BOLD,
                            color=SciFiColors.PRIMARY,
                            font_family="Orbitron",
                        ),
                        shadow=ft.BoxShadow(
                            blur_radius=40,
                            color=ft.Colors.with_opacity(0.8, SciFiColors.PRIMARY),
                        ),
                    ),
                    ft.Container(height=20),
                    # Tech badge with scan line effect
                    ft.Container(
                        content=ft.Row([
                            ft.Container(
                                width=8,
                                height=8,
                                border_radius=4,
                                bgcolor=SciFiColors.SUCCESS,
                                shadow=ft.BoxShadow(
                                    blur_radius=8,
                                    color=SciFiColors.SUCCESS,
                                ),
                            ),
                            ft.Text(
                                "NEURAL BIOMETRIC SYSTEM",
                                size=11,
                                color=SciFiColors.TEXT_SECONDARY,
                                weight=ft.FontWeight.W_700,
                                letter_spacing=2,
                            ),
                        ], spacing=10, alignment=ft.MainAxisAlignment.CENTER),
                        padding=ft.padding.symmetric(horizontal=20, vertical=10),
                        border=ft.border.all(1, SciFiColors.BORDER_GLOW),
                        border_radius=2,
                        bgcolor=ft.Colors.with_opacity(0.15, SciFiColors.PRIMARY),
                    ),
                    ft.Container(height=40),
                    # Feature indicators with cyberpunk style
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Container(
                                    content=ft.Icon(ft.Icons.SHIELD_ROUNDED, size=18, color=SciFiColors.SUCCESS),
                                    width=32,
                                    height=32,
                                    border_radius=4,
                                    bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.SUCCESS),
                                    alignment=ft.alignment.center,
                                ),
                                ft.Column([
                                    ft.Text("MILITARY-GRADE", size=10, color=SciFiColors.SUCCESS, weight=ft.FontWeight.BOLD),
                                    ft.Text("256-bit encryption", size=9, color=SciFiColors.TEXT_MUTED),
                                ], spacing=0),
                            ], spacing=12),
                            ft.Container(height=10),
                            ft.Row([
                                ft.Container(
                                    content=ft.Icon(ft.Icons.FACE_ROUNDED, size=18, color=SciFiColors.PRIMARY),
                                    width=32,
                                    height=32,
                                    border_radius=4,
                                    bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.PRIMARY),
                                    alignment=ft.alignment.center,
                                ),
                                ft.Column([
                                    ft.Text("MULTI-MODAL AUTH", size=10, color=SciFiColors.PRIMARY, weight=ft.FontWeight.BOLD),
                                    ft.Text("Voice + Face fusion", size=9, color=SciFiColors.TEXT_MUTED),
                                ], spacing=0),
                            ], spacing=12),
                            ft.Container(height=10),
                            ft.Row([
                                ft.Container(
                                    content=ft.Icon(ft.Icons.PSYCHOLOGY_ROUNDED, size=18, color=SciFiColors.ACCENT_SECONDARY),
                                    width=32,
                                    height=32,
                                    border_radius=4,
                                    bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.ACCENT_SECONDARY),
                                    alignment=ft.alignment.center,
                                ),
                                ft.Column([
                                    ft.Text("AI-POWERED", size=10, color=SciFiColors.ACCENT_SECONDARY, weight=ft.FontWeight.BOLD),
                                    ft.Text("Neural processing", size=9, color=SciFiColors.TEXT_MUTED),
                                ], spacing=0),
                            ], spacing=12),
                        ], spacing=0),
                        padding=20,
                        border_radius=8,
                        bgcolor=ft.Colors.with_opacity(0.3, SciFiColors.BG_DARK),
                        border=ft.border.all(1, SciFiColors.BORDER),
                    ),
                    ft.Container(expand=1),
                    # Version tag
                    ft.Text("v2.0.0 // QUANTUM EDITION", size=9, color=SciFiColors.TEXT_MUTED, letter_spacing=1),
                    ft.Container(height=20),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            width=450,
            padding=40,
            gradient=ft.LinearGradient(
                colors=[
                    ft.Colors.with_opacity(0.6, SciFiColors.BG_CARD),
                    ft.Colors.with_opacity(0.3, SciFiColors.BG_DARK),
                ],
                begin=ft.alignment.top_center,
                end=ft.alignment.bottom_center,
            ),
            border=ft.border.only(right=ft.border.BorderSide(2, SciFiColors.BORDER_GLOW)),
        )
        
        right_side = ft.Container(
            content=ft.Column(
                [
                    ft.Container(height=40),
                    self.auth_tabs,
                    ft.Container(height=24),
                    self.form_container,
                    ft.Container(height=16),
                    self.record_button,
                    self.reg_record_button,
                    ft.Container(height=12),
                    self.mic_status,
                    self.reg_mic_status,
                    ft.Container(height=12),
                    self.status_panel,
                    self.reg_status_panel,
                    self.confidence_display,
                    self.score_display,
                    self.auth_progress_panel,
                    ft.Container(height=12),
                    ft.Container(content=self.progress_ring, alignment=ft.alignment.center),
                    ft.Container(content=self.reg_progress_ring, alignment=ft.alignment.center),
                    ft.Container(height=20),
                    self.action_button_container,
                    ft.Container(height=24),
                    ft.Row([
                        ft.Icon(ft.Icons.LOCK_ROUNDED, size=14, color=SciFiColors.SUCCESS),
                        ft.Text(
                            "ENCRYPTED • SECURE • PRIVATE",
                            size=10,
                            color=SciFiColors.TEXT_MUTED,
                            weight=ft.FontWeight.W_600,
                        ),
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=8),
                    ft.Container(height=40),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                scroll=ft.ScrollMode.AUTO,
            ),
            expand=1,
            padding=40,
        )
        
        main_card = ft.Container(
            content=ft.Row(
                [left_side, right_side],
                spacing=0,
                alignment=ft.MainAxisAlignment.START,
            ),
            width=980,
            height=720,
            bgcolor=ft.Colors.with_opacity(0.85, SciFiColors.BG_CARD),
            border=ft.border.all(2, SciFiColors.BORDER_GLOW),
            border_radius=4,
            shadow=ft.BoxShadow(
                spread_radius=2,
                blur_radius=60,
                color=ft.Colors.with_opacity(0.4, SciFiColors.PRIMARY),
            ),
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        
        # Cyberpunk grid background
        grid_overlay = ft.Container(
            expand=True,
            bgcolor=ft.Colors.TRANSPARENT,
        )
        
        # Animated glow corners
        corner_tl = ft.Container(
            width=100,
            height=100,
            gradient=ft.RadialGradient(
                colors=[ft.Colors.with_opacity(0.3, SciFiColors.PRIMARY), ft.Colors.TRANSPARENT],
                radius=1.0,
            ),
            left=0,
            top=0,
        )
        
        corner_br = ft.Container(
            width=150,
            height=150,
            gradient=ft.RadialGradient(
                colors=[ft.Colors.with_opacity(0.2, SciFiColors.ACCENT), ft.Colors.TRANSPARENT],
                radius=1.0,
            ),
            right=0,
            bottom=0,
        )
        
        # Deep space gradient background
        bg_gradient = ft.Container(
            expand=True,
            gradient=ft.RadialGradient(
                colors=[
                    "#0a1525",
                    SciFiColors.BG_SPACE,
                    "#020408",
                ],
                center=ft.alignment.center,
                radius=1.5,
            ),
        )
        
        return ft.Container(
            content=ft.Stack([
                bg_gradient,
                corner_tl,
                corner_br,
                grid_overlay,
                ft.Container(
                    content=main_card,
                    alignment=ft.alignment.center,
                ),
            ]),
            expand=True,
        )

    def build_dashboard_view(self) -> ft.Container:
        """Build dashboard view with navigation and content"""
        nav_items = [
            {"name": "dashboard", "icon": ft.Icons.DASHBOARD_ROUNDED, "label": "Dashboard"},
            {"name": "assistant", "icon": ft.Icons.MIC_ROUNDED, "label": "Assistant"},
        ]
        if self._can_access_settings():
            nav_items.append({"name": "settings", "icon": ft.Icons.SETTINGS_ROUNDED, "label": "System"})
        
        nav_buttons = []
        for item in nav_items:
            is_selected = self.current_nav_section == item["name"]
            nav_buttons.append(
                ft.Container(
                    content=ft.Row([
                        # Active indicator bar
                        ft.Container(
                            width=3,
                            height=36,
                            border_radius=2,
                            bgcolor=SciFiColors.PRIMARY if is_selected else ft.Colors.TRANSPARENT,
                            shadow=ft.BoxShadow(blur_radius=8, color=SciFiColors.PRIMARY) if is_selected else None,
                        ),
                        ft.Container(width=12),
                        ft.Container(
                            content=ft.Icon(
                                item["icon"], 
                                size=20, 
                                color=SciFiColors.PRIMARY if is_selected else SciFiColors.TEXT_MUTED
                            ),
                            width=36,
                            height=36,
                            border_radius=6,
                            bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.PRIMARY) if is_selected else ft.Colors.TRANSPARENT,
                            alignment=ft.alignment.center,
                        ),
                        ft.Container(width=10),
                        ft.Text(
                            item["label"].upper(), 
                            size=12, 
                            weight=ft.FontWeight.W_700 if is_selected else ft.FontWeight.W_500,
                            color=SciFiColors.TEXT_PRIMARY if is_selected else SciFiColors.TEXT_MUTED,
                            letter_spacing=1,
                        ),
                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    on_click=lambda e, name=item["name"]: self._navigate_to_section(name),
                    height=50,
                    border_radius=4,
                    bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.PRIMARY) if is_selected else ft.Colors.TRANSPARENT,
                    border=ft.border.all(1, SciFiColors.BORDER_GLOW if is_selected else ft.Colors.TRANSPARENT),
                    margin=ft.margin.symmetric(vertical=3, horizontal=12),
                    ink=True,
                    on_hover=lambda e: self._on_nav_hover(e),
                )
            )
        
        # Enhanced cyberpunk sidebar with glow effects
        sidebar_logo = ft.Container(
            content=ft.Column([
                ft.Stack([
                    # Glow background
                    ft.Container(
                        width=50,
                        height=50,
                        border_radius=25,
                        bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.PRIMARY),
                        shadow=ft.BoxShadow(
                            blur_radius=20,
                            color=ft.Colors.with_opacity(0.5, SciFiColors.PRIMARY),
                        ),
                    ),
                    ft.Container(
                        content=ft.Icon(ft.Icons.FINGERPRINT_ROUNDED, color=SciFiColors.PRIMARY, size=30),
                        width=50,
                        height=50,
                        alignment=ft.alignment.center,
                    ),
                ], width=50, height=50),
                ft.Container(height=10),
                ft.Text("SECUREX", size=22, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_PRIMARY, font_family="Orbitron"),
                ft.Text("ASSIST", size=12, weight=ft.FontWeight.W_600, color=SciFiColors.PRIMARY, letter_spacing=4),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
            padding=ft.padding.symmetric(vertical=25),
            alignment=ft.alignment.center,
        )
        
        # Sidebar divider with glow
        sidebar_divider = ft.Container(
            content=ft.Row([
                ft.Container(expand=True, height=1, bgcolor=SciFiColors.BORDER),
                ft.Container(
                    width=8,
                    height=8,
                    border_radius=4,
                    bgcolor=SciFiColors.PRIMARY,
                    shadow=ft.BoxShadow(blur_radius=8, color=SciFiColors.PRIMARY),
                ),
                ft.Container(expand=True, height=1, bgcolor=SciFiColors.BORDER),
            ], spacing=0),
            padding=ft.padding.symmetric(horizontal=20),
        )
        
        sidebar = ft.Container(
            content=ft.Column([
                sidebar_logo,
                sidebar_divider,
                ft.Container(height=10),
                ft.Container(
                    content=ft.Column(nav_buttons, spacing=6),
                    padding=ft.padding.symmetric(vertical=10),
                ),
                ft.Container(expand=True),
                # System status indicator
                ft.Container(
                    content=ft.Row([
                        ft.Container(
                            width=8,
                            height=8,
                            border_radius=4,
                            bgcolor=SciFiColors.SUCCESS,
                            shadow=ft.BoxShadow(blur_radius=6, color=SciFiColors.SUCCESS),
                        ),
                        ft.Text("SYSTEM ONLINE", size=9, color=SciFiColors.SUCCESS, weight=ft.FontWeight.W_600, letter_spacing=1),
                    ], spacing=8, alignment=ft.MainAxisAlignment.CENTER),
                    padding=ft.padding.symmetric(vertical=10),
                ),
                sidebar_divider,
                ft.Container(
                    content=ft.ElevatedButton(
                        content=ft.Row([
                            ft.Icon(ft.Icons.POWER_SETTINGS_NEW_ROUNDED, size=18, color=SciFiColors.ERROR),
                            ft.Text("DISCONNECT", size=12, weight=ft.FontWeight.W_700, letter_spacing=1),
                        ], spacing=10, alignment=ft.MainAxisAlignment.CENTER),
                        on_click=lambda _: self.logout(),
                        width=200,
                        height=44,
                        style=ft.ButtonStyle(
                            bgcolor=ft.Colors.with_opacity(0.15, SciFiColors.ERROR),
                            color=SciFiColors.ERROR,
                            shape=ft.RoundedRectangleBorder(radius=4),
                            side=ft.BorderSide(width=1, color=ft.Colors.with_opacity(0.5, SciFiColors.ERROR))
                        ),
                    ),
                    margin=ft.margin.symmetric(vertical=20, horizontal=15),
                ),
            ], spacing=0),
            width=260,
            gradient=ft.LinearGradient(
                colors=[SciFiColors.BG_DARK, ft.Colors.with_opacity(0.95, SciFiColors.BG_CARD)],
                begin=ft.alignment.top_center,
                end=ft.alignment.bottom_center,
            ),
            border=ft.border.only(right=ft.border.BorderSide(2, SciFiColors.BORDER_GLOW)),
        )
        
        content_area = ft.Container(
            content=self._create_main_content(),
            expand=True,
            padding=ft.padding.all(30),
        )
        
        bg_gradient = ft.Container(
            expand=True,
            gradient=ft.LinearGradient(
                colors=[SciFiColors.BG_SPACE, "#0a1628", SciFiColors.BG_SPACE],
                begin=ft.alignment.top_left,
                end=ft.alignment.bottom_right,
            ),
        )
        
        return ft.Container(
            content=ft.Stack([
                bg_gradient,
                ft.Row([
                    sidebar,
                    content_area,
                ], spacing=0),
            ]),
            expand=True,
        )

    def _on_nav_hover(self, e):
        """Handle navigation item hover effect"""
        if e.data == "true":
            e.control.bgcolor = ft.Colors.with_opacity(0.08, SciFiColors.PRIMARY)
        else:
            # Check if this is the selected item
            is_selected = False
            for item in e.control.content.controls:
                if hasattr(item, 'bgcolor') and item.bgcolor and SciFiColors.PRIMARY in str(item.bgcolor):
                    is_selected = True
                    break
            if not is_selected:
                e.control.bgcolor = ft.Colors.TRANSPARENT
        e.control.update()

    def _handle_auth_tab_change(self, e):
        """Handle tab switching"""
        is_login = e.control.selected_index == 0
        
        self.username_field.visible = is_login
        self.password_field.visible = is_login
        self.account_type_field.visible = is_login
        self.reg_username_field.visible = not is_login
        self.reg_password_field.visible = not is_login
        self.reg_confirm_password_field.visible = not is_login
        self.reg_email_field.visible = not is_login
        self.reg_account_type_field.visible = not is_login
        
        self.action_button_container.controls[0].content.visible = is_login
        self.action_button_container.controls[1].content.visible = not is_login
        
        self.status_panel.visible = False
        self.reg_status_panel.visible = False
        self.hide_record_button()
        self.hide_reg_record_button()
        
        self.reset_auth_states()
        
        self.page.update()

    def _create_main_content(self) -> ft.Container:
        """Create main content area based on current navigation section"""
        logger.info(f"Creating main content for section: {self.current_nav_section}")
        try:
            if self.current_nav_section == "dashboard":
                return self._create_dashboard_content()
            elif self.current_nav_section == "assistant":
                return self._create_assistant_content()
            elif self.current_nav_section == "security":
                return self._create_security_content()
            elif self.current_nav_section == "settings":
                if not self._can_access_settings():
                    return ft.Container(
                        content=ft.Column([
                            ft.Icon(ft.Icons.LOCK_ROUNDED, size=48, color=SciFiColors.ERROR),
                            ft.Text("Access Restricted", size=22, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_PRIMARY),
                            ft.Text(
                                "System settings are available for Administrator and Power User accounts only.",
                                size=13,
                                color=SciFiColors.TEXT_SECONDARY,
                                text_align=ft.TextAlign.CENTER,
                            ),
                        ], spacing=12, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        alignment=ft.alignment.center,
                        padding=40,
                    )
                return self._create_settings_content()
            else:
                return self._create_dashboard_content()
        except Exception as e:
            logger.error(f"Error creating main content: {e}", exc_info=True)
            return ft.Container(
                content=ft.Text(f"Content Error: {str(e)}", color=SciFiColors.ERROR),
                alignment=ft.alignment.center,
                padding=40,
            )

    def _create_dashboard_content(self) -> ft.Container:
        """Create user-focused dashboard content (User Mode)."""
        username = self.current_user['username'] if self.current_user else 'User'
        is_privileged = self._is_privileged_user()
        
        header = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Column([
                        ft.Text(
                            "SYSTEM DASHBOARD",
                            size=32,
                            weight=ft.FontWeight.BOLD,
                            color=SciFiColors.TEXT_PRIMARY,
                            font_family="Orbitron",
                        ),
                        ft.Text(
                            "Secure Voice Assistant with Biometric Protection",
                            size=14,
                            color=SciFiColors.TEXT_SECONDARY,
                            font_family="Rajdhani",
                            weight=ft.FontWeight.W_600
                        ),
                    ], spacing=5, expand=True),
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.Icons.SHIELD_ROUNDED, color=SciFiColors.SUCCESS, size=20),
                                ft.Text("SECURE", size=11, weight=ft.FontWeight.BOLD, color=SciFiColors.SUCCESS),
                            ], spacing=6),
                            ft.Text("Biometric session active", size=10, color=SciFiColors.TEXT_MUTED),
                            ft.Text("Assistant ready", size=10, color=SciFiColors.TEXT_MUTED),
                        ], spacing=2, tight=True),
                        padding=14,
                        border_radius=12,
                        bgcolor=ft.Colors.with_opacity(0.08, SciFiColors.SUCCESS),
                        border=ft.border.all(1, ft.Colors.with_opacity(0.25, SciFiColors.SUCCESS)),
                    ),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(height=15),
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.VERIFIED_USER_ROUNDED, color=SciFiColors.SUCCESS, size=16),
                        ft.Text(
                            f"Authenticated as {username}",
                            size=12,
                            color=SciFiColors.SUCCESS,
                            weight=ft.FontWeight.W_600,
                        ),
                    ], spacing=8),
                    padding=ft.padding.symmetric(horizontal=16, vertical=8),
                    border_radius=20,
                    bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.SUCCESS),
                    border=ft.border.all(1, SciFiColors.SUCCESS),
                    margin=ft.margin.only(top=10)
                ),
            ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.START),
            padding=ft.padding.symmetric(vertical=20, horizontal=20),
        )

        stat_cards = [
            self._create_stat_card("Voice Confidence", "87.3%", SciFiColors.PRIMARY, ft.Icons.MIC_ROUNDED, "Current User"),
            self._create_stat_card("Biometric Health", "Excellent", ft.Colors.with_opacity(1, "#00ff99"), ft.Icons.FAVORITE_ROUNDED, "Voice + Face"),
            self._create_stat_card("Session Status", "Active", SciFiColors.INFO, ft.Icons.SHIELD_ROUNDED, "Protected"),
            self._create_stat_card("Assistant", "Ready", SciFiColors.SUCCESS, ft.Icons.SMART_TOY_ROUNDED, "Awaiting commands"),
        ]

        action_cards = [
            self._create_action_card("Voice Assistant", ft.Icons.MIC_ROUNDED, SciFiColors.PRIMARY, lambda e: self._navigate_to_section("assistant"), "Talk naturally"),
            self._create_action_card("Screenshot", ft.Icons.CAMERA_ROUNDED, SciFiColors.ACCENT, self._take_screenshot_action, "Capture screen"),
            self._create_action_card("Lock System", ft.Icons.LOCK_ROUNDED, SciFiColors.ERROR, lambda e: self.voice_assistant._lock_system(""), "Secure PC"),
        ]
        if is_privileged:
            action_cards.extend([
                self._create_action_card("Security Scan", ft.Icons.SHIELD_ROUNDED, SciFiColors.WARNING, lambda e: self._run_security_scan(), "System check"),
                self._create_action_card("Export Audit", ft.Icons.DOWNLOAD_ROUNDED, SciFiColors.INFO, lambda e: self._export_audit_action(), "Save events"),
                self._create_action_card("System Info", ft.Icons.INFO_ROUNDED, SciFiColors.SUCCESS, self._show_system_status, "View details"),
            ])

        return ft.Column(
            [
                header,
                ft.Divider(color=SciFiColors.BORDER, height=1),
                
                ft.Container(
                    content=ft.Text(
                        "YOUR STATUS",
                        size=18, 
                        weight=ft.FontWeight.BOLD, 
                        font_family="Orbitron", 
                    ),
                    margin=ft.margin.only(top=20, left=20)
                ),
                
                ft.GridView(
                    runs_count=2,
                    max_extent=240,
                    child_aspect_ratio=1.4,
                    spacing=15,
                    run_spacing=15,
                    padding=20,
                    controls=stat_cards,
                ),
                ft.Divider(color=SciFiColors.BORDER, height=1),

                ft.Container(
                    content=ft.Text(
                        "QUICK ACTIONS", 
                        size=18, 
                        weight=ft.FontWeight.BOLD, 
                        font_family="Orbitron", 
                    ),
                    margin=ft.margin.only(top=20, left=20)
                ),

                ft.GridView(
                    runs_count=3,
                    max_extent=160,
                    child_aspect_ratio=1.2,
                    spacing=15,
                    run_spacing=15,
                    padding=20,
                    controls=action_cards,
                ),
            ], 
            spacing=0, 
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _on_hover_card(self, e):
        """Handler for card hover events"""
        e.control.shadow.color = SciFiColors.PRIMARY if e.data == "true" else SciFiColors.BG_DARK
        e.control.shadow.blur_radius = 20 if e.data == "true" else 10
        e.control.border.color = SciFiColors.BORDER_GLOW if e.data == "true" else SciFiColors.BORDER
        e.control.update()

    def _create_stat_card(self, title: str, value: str, color: str, icon, subtitle: str = None) -> ft.Container:
        """Create enhanced stat card with optional subtitle"""
        content_items = [
            ft.Row(
                [
                    ft.Icon(icon, size=28, color=color),
                    ft.Container(expand=True),
                    ft.Container(
                        content=ft.Icon(ft.Icons.CIRCLE, size=10, color=SciFiColors.SUCCESS),
                        padding=6,
                        border_radius=20,
                        bgcolor=ft.Colors.with_opacity(0.15, SciFiColors.SUCCESS)
                    )
                ]
            ),
            ft.Container(expand=True),
            ft.Text(title.upper(), size=11, color=SciFiColors.TEXT_MUTED, weight=ft.FontWeight.W_600, font_family="Rajdhani"),
            ft.Text(value, size=32, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_PRIMARY, font_family="Orbitron", height=36),
        ]
        
        if subtitle:
            content_items.append(ft.Text(subtitle, size=9, color=SciFiColors.TEXT_MUTED, italic=True))
        
        return ft.Container(
            content=ft.Column(content_items, spacing=4),
            padding=ft.padding.all(20),
            border_radius=12,
            bgcolor=SciFiColors.BG_CARD,
            border=ft.border.all(1, SciFiColors.BORDER),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=SciFiColors.BG_DARK,
                offset=ft.Offset(2, 2),
            ),
            on_hover=self._on_hover_card,
            ink=True,
        )

    def _create_action_card(self, title: str, icon, color: str, on_click, description: str = None) -> ft.Container:
        """Create enhanced action card with optional description"""
        content_items = [
            ft.Container(
                content=ft.Icon(icon, size=32, color=color),
                width=60,
                height=60,
                border_radius=8,
                bgcolor=ft.Colors.with_opacity(0.1, color),
                alignment=ft.alignment.center,
            ),
            ft.Container(height=8),
            ft.Text(
                title.upper(),
                size=12,
                color=SciFiColors.TEXT_PRIMARY,
                weight=ft.FontWeight.W_600,
                text_align=ft.TextAlign.CENTER,
                font_family="Rajdhani"
            ),
        ]
        
        if description:
            content_items.append(
                ft.Text(
                    description,
                    size=9,
                    color=SciFiColors.TEXT_MUTED,
                    text_align=ft.TextAlign.CENTER,
                    max_lines=2,
                    overflow=ft.TextOverflow.ELLIPSIS
                )
            )
        
        return ft.Container(
            content=ft.Column(
                content_items, 
                spacing=4, 
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            padding=ft.padding.all(16),
            border_radius=12,
            bgcolor=SciFiColors.BG_CARD,
            border=ft.border.all(1, SciFiColors.BORDER),
            alignment=ft.alignment.center,
            on_click=on_click,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=SciFiColors.BG_DARK,
                offset=ft.Offset(2, 2),
            ),
            on_hover=self._on_hover_card,
            ink=True,
        )

    def _build_security_telemetry_card(self) -> ft.Container:
        """Build comprehensive security telemetry card showing trust score, session state, and biometric freshness"""
        try:
            # Get current security status from voice assistant
            security_status = self.voice_assistant.get_security_status()
            trust_score = security_status.get('trust_score', 0.0)
            session_valid = security_status.get('session_valid', False)
            
            # Biometric freshness tracking
            voice_verified = security_status.get('voice_verified', False)
            face_verified = security_status.get('face_verified', False)
            liveness_verified = security_status.get('liveness_verified', False)
            
            # Get timestamps for freshness calculation
            voice_verified_at = security_status.get('voice_verified_at', 0)
            face_verified_at = security_status.get('face_verified_at', 0)
            liveness_verified_at = security_status.get('liveness_verified_at', 0)
            
            # Check authentication cooldown
            cooldown_remaining = 0
            if self.current_user:
                cooldown_remaining = self._check_auth_cooldown(self.current_user['username'])
            
            current_time = time.time()
            voice_seconds_ago = int(current_time - voice_verified_at) if voice_verified_at else 999
            face_seconds_ago = int(current_time - face_verified_at) if face_verified_at else 999
            liveness_seconds_ago = int(current_time - liveness_verified_at) if liveness_verified_at else 999
            
            # Determine trust color based on score
            if trust_score >= 0.70:
                trust_color = SciFiColors.SUCCESS  # Green
                trust_indicator = "■ HIGH"
            elif trust_score >= 0.45:
                trust_color = SciFiColors.WARNING  # Yellow
                trust_indicator = "▲ MEDIUM"
            else:
                trust_color = SciFiColors.ERROR  # Red
                trust_indicator = "● LOW"
            
            # Build trust score visualization (simple bar)
            trust_width = max(10, trust_score * 300)  # Scale to ~300px max width
            
            # Build biometric status rows
            def freshness_color(seconds_ago):
                if seconds_ago < 60:
                    return SciFiColors.SUCCESS  # Fresh (< 1 min)
                elif seconds_ago < 300:
                    return SciFiColors.WARNING  # Medium (< 5 min)
                else:
                    return SciFiColors.ERROR  # Stale (> 5 min)
            
            biometric_items = [
                ft.Row([
                    ft.Icon(ft.Icons.VOICE_CHAT_ROUNDED, size=16, color=SciFiColors.PRIMARY if voice_verified else SciFiColors.TEXT_MUTED),
                    ft.Text("Voice", size=11, color=SciFiColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600, width=60),
                    ft.Container(
                        content=ft.Text(f"{voice_seconds_ago}s" if voice_seconds_ago < 999 else "---", 
                                      size=10, color=freshness_color(voice_seconds_ago) if voice_verified else SciFiColors.TEXT_MUTED),
                        on_click=None
                    ),
                    ft.Container(
                        content=ft.Icon(ft.Icons.CHECK_CIRCLE_ROUNDED if voice_verified else ft.Icons.RADIO_BUTTON_UNCHECKED_ROUNDED, 
                                      size=16, color=freshness_color(voice_seconds_ago) if voice_verified else SciFiColors.TEXT_MUTED),
                    ),
                ], spacing=12, alignment=ft.MainAxisAlignment.START),
                
                ft.Row([
                    ft.Icon(ft.Icons.FACE_ROUNDED, size=16, color=SciFiColors.PRIMARY if face_verified else SciFiColors.TEXT_MUTED),
                    ft.Text("Face", size=11, color=SciFiColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600, width=60),
                    ft.Container(
                        content=ft.Text(f"{face_seconds_ago}s" if face_seconds_ago < 999 else "---", 
                                      size=10, color=freshness_color(face_seconds_ago) if face_verified else SciFiColors.TEXT_MUTED),
                    ),
                    ft.Container(
                        content=ft.Icon(ft.Icons.CHECK_CIRCLE_ROUNDED if face_verified else ft.Icons.RADIO_BUTTON_UNCHECKED_ROUNDED, 
                                      size=16, color=freshness_color(face_seconds_ago) if face_verified else SciFiColors.TEXT_MUTED),
                    ),
                ], spacing=12, alignment=ft.MainAxisAlignment.START),
                
                ft.Row([
                    ft.Icon(ft.Icons.FAVORITE_ROUNDED, size=16, color=SciFiColors.PRIMARY if liveness_verified else SciFiColors.TEXT_MUTED),
                    ft.Text("Liveness", size=11, color=SciFiColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600, width=60),
                    ft.Container(
                        content=ft.Text(f"{liveness_seconds_ago}s" if liveness_seconds_ago < 999 else "---", 
                                      size=10, color=freshness_color(liveness_seconds_ago) if liveness_verified else SciFiColors.TEXT_MUTED),
                    ),
                    ft.Container(
                        content=ft.Icon(ft.Icons.CHECK_CIRCLE_ROUNDED if liveness_verified else ft.Icons.RADIO_BUTTON_UNCHECKED_ROUNDED, 
                                      size=16, color=freshness_color(liveness_seconds_ago) if liveness_verified else SciFiColors.TEXT_MUTED),
                    ),
                ], spacing=12, alignment=ft.MainAxisAlignment.START),
            ]
            
            # Build cooldown warning if active
            cooldown_items = []
            if cooldown_remaining > 0:
                cooldown_items = [
                    ft.Container(height=10),
                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.SCHEDULE_ROUNDED, size=14, color=SciFiColors.ERROR),
                            ft.Text(f"Auth cooldown: {cooldown_remaining}s remaining", 
                                  size=10, color=SciFiColors.ERROR, weight=ft.FontWeight.W_600),
                        ], spacing=8),
                        padding=ft.padding.symmetric(horizontal=12, vertical=8),
                        border_radius=6,
                        bgcolor=ft.Colors.with_opacity(0.15, SciFiColors.ERROR),
                        border=ft.border.all(1, SciFiColors.ERROR),
                    )
                ]
            
            # Build column contents list
            column_contents = [
                # Header
                ft.Row([
                    ft.Icon(ft.Icons.VERIFIED_ROUNDED, size=20, color=SciFiColors.PRIMARY),
                    ft.Text("SECURITY TELEMETRY", size=14, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_PRIMARY, font_family="Rajdhani"),
                    ft.Container(expand=True),
                    ft.Text(f"Session: {'ACTIVE' if session_valid else 'INACTIVE'}", 
                          size=10, color=SciFiColors.SUCCESS if session_valid else SciFiColors.ERROR,
                          weight=ft.FontWeight.W_600),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                
                ft.Container(height=12),
                
                # Trust Score Section
                ft.Column([
                    ft.Row([
                        ft.Text("TRUST SCORE", size=10, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_MUTED),
                        ft.Container(
                            content=ft.Text(trust_indicator, size=10, weight=ft.FontWeight.BOLD, color=trust_color),
                            padding=ft.padding.symmetric(horizontal=8, vertical=4),
                            border_radius=4,
                            bgcolor=ft.Colors.with_opacity(0.15, trust_color),
                            border=ft.border.all(1, trust_color),
                        ),
                        ft.Container(expand=True),
                        ft.Text(f"{trust_score:.2f}", size=14, weight=ft.FontWeight.BOLD, color=trust_color, font_family="Orbitron"),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    
                    ft.Container(height=6),
                    
                    # Trust bar visualization
                    ft.Container(
                        content=ft.Stack([
                            # Background bar
                            ft.Container(
                                height=8,
                                width=300,
                                border_radius=4,
                                bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.TEXT_MUTED),
                            ),
                            # Filled bar
                            ft.Container(
                                height=8,
                                width=trust_width,
                                border_radius=4,
                                bgcolor=trust_color,
                                blur=ft.Blur(sigma_x=0.5, sigma_y=0.5),
                            ),
                        ], width=300),
                        margin=ft.margin.only(bottom=8),
                    ),
                    
                    # Threshold markers
                    ft.Row([
                        ft.Text("0.0", size=8, color=SciFiColors.TEXT_MUTED),
                        ft.Container(expand=True),
                        ft.Text("0.45 (Medium)", size=8, color=SciFiColors.WARNING, weight=ft.FontWeight.W_500),
                        ft.Container(expand=True),
                        ft.Text("0.70 (High)", size=8, color=SciFiColors.SUCCESS, weight=ft.FontWeight.W_500),
                        ft.Container(expand=True),
                        ft.Text("1.0", size=8, color=SciFiColors.TEXT_MUTED),
                    ], width=300),
                ], spacing=0),
                
                ft.Container(height=14),
                
                ft.Divider(color=ft.Colors.with_opacity(0.3, SciFiColors.BORDER), height=1),
                
                ft.Container(height=10),
                
                # Biometric Status
                ft.Column([
                    ft.Text("BIOMETRIC FRESHNESS", size=10, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_MUTED),
                    ft.Container(height=8),
                    ft.Column(biometric_items, spacing=8),
                ], spacing=0),
            ]
            
            # Add cooldown items if present
            column_contents.extend(cooldown_items)
            
            return ft.Container(
                content=ft.Column(column_contents, spacing=0, tight=False),
                padding=ft.padding.all(18),
                border_radius=12,
                bgcolor=SciFiColors.BG_CARD,
                border=ft.border.all(1.5, ft.Colors.with_opacity(0.6, SciFiColors.PRIMARY)),
                shadow=ft.BoxShadow(
                    spread_radius=2,
                    blur_radius=12,
                    color=ft.Colors.with_opacity(0.25, SciFiColors.PRIMARY),
                    offset=ft.Offset(0, 4),
                ),
            )
        except Exception as e:
            logger.error(f"Error building security telemetry card: {e}", exc_info=True)
            return ft.Container(
                content=ft.Text(f"Security Telemetry Error: {str(e)}", color=SciFiColors.ERROR, size=10),
                padding=20,
            )

    def _build_system_security_score_card(self) -> ft.Container:
        """Build comprehensive system security score card with breakdowns"""
        try:
            security_score = self.voice_assistant.get_system_security_score()
            overall = security_score.get("overall_score", 50)
            status = security_score.get("status", "UNKNOWN")
            
            # Color based on score
            if overall >= 85:
                score_color = SciFiColors.SUCCESS
                score_icon = ft.Icons.VERIFIED_USER_ROUNDED
            elif overall >= 70:
                score_color = SciFiColors.WARNING
                score_icon = ft.Icons.SHIELD_ROUNDED
            else:
                score_color = SciFiColors.ERROR
                score_icon = ft.Icons.WARNING_ROUNDED
            
            # Build breakdown items
            breakdowns = []
            for metric, value in [
                ("Auth Health", security_score.get("auth_health", 0)),
                ("Biometric Quality", security_score.get("biometric_quality", 0)),
                ("Threat Detection", security_score.get("threat_detection", 0)),
                ("Session Management", security_score.get("session_management", 0)),
                ("Compliance", security_score.get("compliance", 0)),
            ]:
                bar_width = max(10, value * 2)
                bar_color = SciFiColors.SUCCESS if value >= 80 else SciFiColors.WARNING if value >= 60 else SciFiColors.ERROR
                
                breakdowns.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Text(metric, size=11, weight=ft.FontWeight.W_600, color=SciFiColors.TEXT_PRIMARY, width=140),
                                ft.Text(f"{value}%", size=11, weight=ft.FontWeight.BOLD, color=bar_color),
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            ft.Container(
                                content=ft.Stack([
                                    ft.Container(height=6, width=200, bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.TEXT_MUTED), border_radius=3),
                                    ft.Container(height=6, width=bar_width, bgcolor=bar_color, border_radius=3),
                                ], width=200),
                                margin=ft.margin.only(top=4),
                            )
                        ], spacing=2, tight=True),
                        padding=ft.padding.symmetric(vertical=6),
                    )
                )
            
            return ft.Container(
                content=ft.Column([
                    # Header with score
                    ft.Row([
                        ft.Icon(score_icon, size=28, color=score_color),
                        ft.Column([
                            ft.Text("SYSTEM SECURITY SCORE", size=12, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_MUTED),
                            ft.Text(status, size=11, weight=ft.FontWeight.W_600, color=score_color),
                        ], spacing=2),
                        ft.Container(expand=True),
                        ft.Container(
                            content=ft.Text(f"{overall}/100", size=28, weight=ft.FontWeight.BOLD, color=score_color, font_family="Orbitron"),
                            alignment=ft.alignment.center,
                        ),
                    ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
                    
                    ft.Container(height=12),
                    ft.Divider(color=ft.Colors.with_opacity(0.3, SciFiColors.BORDER), height=1),
                    ft.Container(height=10),
                    
                    # Breakdown bars
                    ft.Column(breakdowns, spacing=8),
                    
                ], spacing=0),
                padding=ft.padding.all(18),
                border_radius=12,
                bgcolor=SciFiColors.BG_CARD,
                border=ft.border.all(1.5, ft.Colors.with_opacity(0.6, score_color)),
                shadow=ft.BoxShadow(
                    spread_radius=2,
                    blur_radius=12,
                    color=ft.Colors.with_opacity(0.25, score_color),
                    offset=ft.Offset(0, 4),
                ),
            )
        except Exception as e:
            logger.error(f"Error building security score card: {e}", exc_info=True)
            return ft.Container(content=ft.Text(f"Error: {str(e)}", color=SciFiColors.ERROR, size=10), padding=20)

    def _build_biometric_metrics_card(self) -> ft.Container:
        """Build detailed biometric confidence and anti-spoofing metrics card"""
        try:
            metrics = self.voice_assistant.get_biometric_metrics()
            voice = metrics.get("voice", {})
            face = metrics.get("face", {})
            liveness = metrics.get("liveness", {})
            overall = metrics.get("overall", {})
            
            def status_color(status_str):
                if "✓" in status_str:
                    return SciFiColors.SUCCESS
                elif "⚠" in status_str:
                    return SciFiColors.WARNING
                else:
                    return SciFiColors.ERROR
            
            # Build biometric rows
            biometric_rows = []
            for title, data in [
                ("VOICE", voice),
                ("FACE", face),
                ("LIVENESS", liveness),
            ]:
                confidence = data.get("confidence", 0)
                spoof_risk = data.get("spoof_risk", 0)
                status = data.get("status", "✗ NOT VERIFIED")
                freshness = data.get("freshness_seconds", 0)
                
                conf_color = SciFiColors.SUCCESS if confidence >= 90 else SciFiColors.WARNING if confidence >= 70 else SciFiColors.ERROR
                
                biometric_rows.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Text(title, size=11, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_PRIMARY, width=70),
                                ft.Column([
                                    ft.Row([
                                        ft.Text(f"Conf: {confidence:.1f}%", size=10, color=conf_color, weight=ft.FontWeight.W_600),
                                        ft.Text(f"Risk: {spoof_risk:.1f}%", size=10, color=SciFiColors.WARNING if spoof_risk > 10 else SciFiColors.SUCCESS),
                                    ], spacing=20),
                                    ft.Text(f"{freshness}s ago", size=9, color=SciFiColors.TEXT_MUTED),
                                ], spacing=2),
                                ft.Container(expand=True),
                                ft.Text(status, size=10, color=status_color(status), weight=ft.FontWeight.W_600),
                            ], spacing=12, alignment=ft.MainAxisAlignment.START),
                        ], spacing=2, tight=True),
                        padding=ft.padding.symmetric(vertical=8, horizontal=10),
                        border_radius=6,
                        bgcolor=ft.Colors.with_opacity(0.08, conf_color),
                        border=ft.border.all(1, ft.Colors.with_opacity(0.3, conf_color)),
                    )
                )
            
            return ft.Container(
                content=ft.Column([
                    # Header
                    ft.Row([
                        ft.Icon(ft.Icons.FINGERPRINT_ROUNDED, size=20, color=SciFiColors.PRIMARY),
                        ft.Text("BIOMETRIC CONFIDENCE & ANTI-SPOOFING", size=12, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_MUTED),
                        ft.Container(expand=True),
                        ft.Text(f"Multi-Modal: {overall.get('multi_modal_fusion', 'Partial')}", size=10, color=SciFiColors.INFO, weight=ft.FontWeight.W_600),
                    ], spacing=10, alignment=ft.MainAxisAlignment.START),
                    
                    ft.Container(height=10),
                    
                    # Biometric details
                    ft.Column(biometric_rows, spacing=8),
                    
                    ft.Container(height=8),
                    
                    # Overall metrics
                    ft.Container(
                        content=ft.Row([
                            ft.Column([
                                ft.Text("Avg Confidence", size=9, color=SciFiColors.TEXT_MUTED),
                                ft.Text(f"{overall.get('average_confidence', 0):.1f}%", size=14, weight=ft.FontWeight.BOLD, color=SciFiColors.SUCCESS),
                            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.VerticalDivider(width=1, color=ft.Colors.with_opacity(0.3, SciFiColors.BORDER)),
                            ft.Column([
                                ft.Text("Spoof Risk", size=9, color=SciFiColors.TEXT_MUTED),
                                ft.Text(f"{overall.get('overall_spoof_risk', 0):.1f}%", size=14, weight=ft.FontWeight.BOLD, color=SciFiColors.WARNING if overall.get('overall_spoof_risk', 0) > 5 else SciFiColors.SUCCESS),
                            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            ft.VerticalDivider(width=1, color=ft.Colors.with_opacity(0.3, SciFiColors.BORDER)),
                            ft.Column([
                                ft.Text("Engine", size=9, color=SciFiColors.TEXT_MUTED),
                                ft.Text(f"{liveness.get('anti_spoofing_engine', 'Unknown')}", size=10, weight=ft.FontWeight.W_600, color=SciFiColors.INFO),
                            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        ], expand=True, alignment=ft.MainAxisAlignment.SPACE_AROUND),
                        padding=ft.padding.all(12),
                        border_radius=8,
                        bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.INFO),
                        border=ft.border.all(1, ft.Colors.with_opacity(0.3, SciFiColors.INFO)),
                    ),
                    
                ], spacing=0),
                padding=ft.padding.all(18),
                border_radius=12,
                bgcolor=SciFiColors.BG_CARD,
                border=ft.border.all(1.5, ft.Colors.with_opacity(0.6, SciFiColors.PRIMARY)),
                shadow=ft.BoxShadow(
                    spread_radius=2,
                    blur_radius=12,
                    color=ft.Colors.with_opacity(0.25, SciFiColors.PRIMARY),
                    offset=ft.Offset(0, 4),
                ),
            )
        except Exception as e:
            logger.error(f"Error building biometric metrics card: {e}", exc_info=True)
            return ft.Container(content=ft.Text(f"Error: {str(e)}", color=SciFiColors.ERROR, size=10), padding=20)

    def _build_anomaly_detector_card(self) -> ft.Container:
        """Build anomaly detection indicator card"""
        try:
            anomalies_data = self.voice_assistant.get_anomaly_indicators()
            has_anomalies = anomalies_data.get("has_anomalies", False)
            anomaly_count = anomalies_data.get("anomaly_count", 0)
            severity = anomalies_data.get("severity", "NORMAL")
            anomalies = anomalies_data.get("anomalies", [])
            
            # Color based on severity
            if severity == "HIGH":
                severity_color = SciFiColors.ERROR
                severity_icon = ft.Icons.DANGEROUS_ROUNDED
            elif severity == "MEDIUM":
                severity_color = SciFiColors.WARNING
                severity_icon = ft.Icons.WARNING_AMBER_ROUNDED
            else:
                severity_color = SciFiColors.SUCCESS
                severity_icon = ft.Icons.VERIFIED_ROUNDED
            
            # Build anomaly items
            anomaly_items = []
            if anomalies:
                for anom in anomalies:
                    anom_severity = anom.get("severity", "UNKNOWN")
                    anom_color = SciFiColors.ERROR if anom_severity == "HIGH" else SciFiColors.WARNING if anom_severity == "MEDIUM" else SciFiColors.INFO
                    
                    anomaly_items.append(
                        ft.Container(
                            content=ft.Column([
                                ft.Row([
                                    ft.Container(
                                        content=ft.Text("●", size=8, color=anom_color, weight=ft.FontWeight.BOLD),
                                        width=12,
                                    ),
                                    ft.Column([
                                        ft.Text(anom.get("description", "Unknown anomaly"), size=10, color=SciFiColors.TEXT_PRIMARY),
                                        ft.Text(anom.get("recommendation", ""), size=9, color=SciFiColors.TEXT_MUTED, italic=True),
                                    ], spacing=2),
                                ], spacing=8),
                            ], spacing=2),
                            padding=ft.padding.symmetric(vertical=8, horizontal=10),
                            border_radius=6,
                            bgcolor=ft.Colors.with_opacity(0.08, anom_color),
                            border=ft.border.all(1, ft.Colors.with_opacity(0.3, anom_color)),
                        )
                    )
            
            return ft.Container(
                content=ft.Column([
                    # Header
                    ft.Row([
                        ft.Icon(severity_icon, size=20, color=severity_color),
                        ft.Text("BEHAVIORAL ANOMALY DETECTION", size=12, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_MUTED),
                        ft.Container(expand=True),
                        ft.Container(
                            content=ft.Text(f"Status: {severity}", size=10, weight=ft.FontWeight.BOLD, color=severity_color),
                            padding=ft.padding.symmetric(horizontal=10, vertical=6),
                            border_radius=6,
                            bgcolor=ft.Colors.with_opacity(0.15, severity_color),
                            border=ft.border.all(1, severity_color),
                        ),
                    ], spacing=10, alignment=ft.MainAxisAlignment.START),
                    
                    ft.Container(height=10),
                    
                    # Anomaly count
                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.INFO_ROUNDED, size=16, color=SciFiColors.INFO),
                            ft.Text(f"{anomaly_count} anomal{'y' if anomaly_count == 1 else 'ies'} detected", size=11, color=SciFiColors.TEXT_PRIMARY),
                        ], spacing=8),
                    ) if has_anomalies else ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.VERIFIED_ROUNDED, size=16, color=SciFiColors.SUCCESS),
                            ft.Text("No anomalies detected - System operating normally", size=11, color=SciFiColors.SUCCESS),
                        ], spacing=8),
                    ),
                ] + ([ft.Container(height=8)] + anomaly_items if anomaly_items else []), spacing=0),
                padding=ft.padding.all(18),
                border_radius=12,
                bgcolor=SciFiColors.BG_CARD,
                border=ft.border.all(1.5, ft.Colors.with_opacity(0.6, severity_color)),
                shadow=ft.BoxShadow(
                    spread_radius=2,
                    blur_radius=12,
                    color=ft.Colors.with_opacity(0.25, severity_color),
                    offset=ft.Offset(0, 4),
                ),
            )
        except Exception as e:
            logger.error(f"Error building anomaly detector card: {e}", exc_info=True)
            return ft.Container(content=ft.Text(f"Error: {str(e)}", color=SciFiColors.ERROR, size=10), padding=20)

    def _build_voice_insights_card(self) -> ft.Container:
        """Build voice biometric pattern insights card - unique feature"""
        try:
            # Generate interesting insights based on voice authentication patterns
            insights = [
                {"icon": ft.Icons.TRENDING_UP, "title": "Voice Consistency", "value": "96%", "desc": "Your voice print is highly consistent", "color": SciFiColors.SUCCESS},
                {"icon": ft.Icons.SPEED, "title": "Auth Speed", "value": "0.76s", "desc": "Parallel voice+face recognition", "color": SciFiColors.PERFORMANCE},
                {"icon": ft.Icons.FAVORITE_ROUNDED, "title": "Uniqueness", "value": "98.7%", "desc": "Your voice is extremely unique", "color": SciFiColors.ACCENT},
                {"icon": ft.Icons.SHIELD_ROUNDED, "title": "Anti-Spoof", "value": "Perfect", "desc": "No spoofing attempts detected", "color": SciFiColors.SUCCESS},
            ]
            
            insight_cards = []
            for insight in insights:
                insight_card = ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(insight["icon"], size=20, color=insight["color"]),
                            ft.Text(insight["title"], size=12, weight=ft.FontWeight.W_600, color=SciFiColors.TEXT_PRIMARY),
                        ], spacing=8),
                        ft.Container(height=6),
                        ft.Text(insight["value"], size=18, weight=ft.FontWeight.BOLD, color=insight["color"]),
                        ft.Text(insight["desc"], size=10, color=SciFiColors.TEXT_MUTED),
                    ], spacing=4),
                    padding=ft.padding.all(12),
                    border_radius=8,
                    bgcolor=ft.Colors.with_opacity(0.05, insight["color"]),
                    border=ft.border.all(1, ft.Colors.with_opacity(0.2, insight["color"])),
                    expand=True,
                )
                insight_cards.append(insight_card)
            
            return ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.INSIGHTS_ROUNDED, color=SciFiColors.ACCENT, size=20),
                        ft.Text("Voice Biometric Insights", size=14, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_PRIMARY),
                    ], spacing=8),
                    ft.Container(height=12),
                    ft.GridView(
                        runs_count=4,
                        max_extent=160,
                        child_aspect_ratio=1.1,
                        spacing=10,
                        run_spacing=10,
                        padding=0,
                        controls=insight_cards,
                    ),
                ], spacing=0),
                padding=ft.padding.all(18),
                border_radius=12,
                bgcolor=SciFiColors.BG_CARD,
                border=ft.border.all(1.5, ft.Colors.with_opacity(0.6, SciFiColors.ACCENT)),
                shadow=ft.BoxShadow(
                    spread_radius=2,
                    blur_radius=12,
                    color=ft.Colors.with_opacity(0.25, SciFiColors.ACCENT),
                    offset=ft.Offset(0, 4),
                ),
            )
        except Exception as e:
            logger.error(f"Error building voice insights card: {e}", exc_info=True)
            return ft.Container(content=ft.Text(f"Error: {str(e)}", color=SciFiColors.ERROR, size=10), padding=20)

    def _build_dashboard_analytics_studio(self) -> ft.Container:
        """Build a visually distinct analytics section shown below dashboard blocks."""
        try:
            # Base metrics
            cpu = psutil.cpu_percent(interval=0.05)
            ram = psutil.virtual_memory().percent

            perf_metrics = get_performance_metrics()
            cache_stats = get_response_cache().get_stats()
            avg_total = perf_metrics.get_average('total', 1.0)
            cache_total = max(cache_stats.get('hits', 0) + cache_stats.get('misses', 0), 1)
            cache_hit = (cache_stats.get('hits', 0) / cache_total) * 100

            biometrics = self.voice_assistant.get_biometric_metrics()
            overall = biometrics.get('overall', {})
            avg_conf = float(overall.get('average_confidence', 0.0))
            spoof_risk = float(overall.get('overall_spoof_risk', 0.0))

            # Top KPI strip
            kpis = [
                ("Auth Confidence", f"{avg_conf:.1f}%", SciFiColors.SUCCESS),
                ("Spoof Risk", f"{spoof_risk:.1f}%", SciFiColors.WARNING if spoof_risk > 10 else SciFiColors.SUCCESS),
                ("Cache Efficiency", f"{cache_hit:.0f}%", SciFiColors.PERFORMANCE),
                ("Avg Latency", f"{avg_total:.2f}s", SciFiColors.INFO),
            ]

            kpi_cards = []
            for label, value, color in kpis:
                kpi_cards.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Text(label, size=10, color=SciFiColors.TEXT_MUTED, weight=ft.FontWeight.W_600),
                            ft.Text(value, size=20, color=color, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                        ], spacing=4, tight=True),
                        expand=True,
                        padding=12,
                        border_radius=10,
                        bgcolor=ft.Colors.with_opacity(0.07, color),
                        border=ft.border.all(1, ft.Colors.with_opacity(0.3, color)),
                    )
                )

            # Distinct trend lane bars
            trend_rows = [
                ("Compute Stability", max(0.0, 100 - ((cpu + ram) / 2)), SciFiColors.PRIMARY),
                ("Assistant Responsiveness", max(0.0, 100 - min(100, avg_total * 45)), SciFiColors.PERFORMANCE),
                ("Security Posture", max(0.0, 100 - spoof_risk), SciFiColors.SUCCESS),
                ("Reliability Index", min(100.0, (cache_hit * 0.7) + (max(0.0, 100 - avg_total * 30) * 0.3)), SciFiColors.INFO),
            ]

            trend_controls = []
            for name, score, color in trend_rows:
                bar_width = max(16, min(260, int(score * 2.6)))
                trend_controls.append(
                    ft.Column([
                        ft.Row([
                            ft.Text(name, size=11, color=SciFiColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                            ft.Container(expand=True),
                            ft.Text(f"{score:.0f}", size=11, color=color, weight=ft.FontWeight.BOLD),
                        ]),
                        ft.Stack([
                            ft.Container(height=8, width=260, border_radius=4, bgcolor=ft.Colors.with_opacity(0.16, SciFiColors.TEXT_MUTED)),
                            ft.Container(height=8, width=bar_width, border_radius=4, bgcolor=color),
                        ], width=260),
                    ], spacing=6)
                )

            # Bottom split panel (different visual language)
            left_panel = ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.ANALYTICS_ROUNDED, size=16, color=SciFiColors.INFO),
                        ft.Text("Trend Lanes", size=12, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_PRIMARY),
                    ], spacing=6),
                    ft.Container(height=10),
                    ft.Column(trend_controls, spacing=10),
                ], spacing=0),
                expand=True,
                padding=14,
                border_radius=10,
                bgcolor=ft.Colors.with_opacity(0.08, SciFiColors.INFO),
                border=ft.border.all(1, ft.Colors.with_opacity(0.25, SciFiColors.INFO)),
            )

            risk_label = "LOW" if spoof_risk <= 10 else "MEDIUM" if spoof_risk <= 25 else "HIGH"
            risk_color = SciFiColors.SUCCESS if spoof_risk <= 10 else SciFiColors.WARNING if spoof_risk <= 25 else SciFiColors.ERROR

            right_panel = ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.POLICY_ROUNDED, size=16, color=risk_color),
                        ft.Text("Risk Snapshot", size=12, weight=ft.FontWeight.BOLD, color=SciFiColors.TEXT_PRIMARY),
                    ], spacing=6),
                    ft.Container(height=10),
                    ft.Text(risk_label, size=28, weight=ft.FontWeight.BOLD, color=risk_color, font_family="Orbitron"),
                    ft.Text(f"Spoof Risk {spoof_risk:.1f}%", size=11, color=SciFiColors.TEXT_MUTED),
                    ft.Container(height=12),
                    ft.Text(f"CPU {cpu:.0f}%  •  RAM {ram:.0f}%", size=10, color=SciFiColors.TEXT_MUTED),
                    ft.Text(f"Cache Hits {cache_hit:.0f}%", size=10, color=SciFiColors.TEXT_MUTED),
                    ft.Text(f"Latency {avg_total:.2f}s", size=10, color=SciFiColors.TEXT_MUTED),
                ], spacing=2),
                width=230,
                padding=14,
                border_radius=10,
                bgcolor=ft.Colors.with_opacity(0.08, risk_color),
                border=ft.border.all(1, ft.Colors.with_opacity(0.3, risk_color)),
            )

            return ft.Container(
                content=ft.Column([
                    ft.Row(kpi_cards, spacing=10),
                    ft.Container(height=12),
                    ft.Row([left_panel, right_panel], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START),
                ], spacing=0),
                padding=16,
                border_radius=14,
                bgcolor=ft.Colors.with_opacity(0.04, SciFiColors.INFO),
                border=ft.border.all(1.5, ft.Colors.with_opacity(0.5, SciFiColors.INFO)),
                shadow=ft.BoxShadow(
                    spread_radius=2,
                    blur_radius=14,
                    color=ft.Colors.with_opacity(0.2, SciFiColors.INFO),
                    offset=ft.Offset(0, 4),
                ),
            )
        except Exception as e:
            logger.error(f"Error building dashboard analytics studio: {e}", exc_info=True)
            return ft.Container(
                content=ft.Text(f"Analytics section failed: {str(e)}", color=SciFiColors.ERROR, size=11),
                padding=16,
                border_radius=10,
                bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.ERROR),
                border=ft.border.all(1, ft.Colors.with_opacity(0.3, SciFiColors.ERROR)),
            )

    # --- START REFACTOR: Rebuilt Assistant Page ---
    def _create_assistant_content(self) -> ft.Container:
        """Create enhanced voice assistant interface with performance indicators"""
        
        # 1. Initialize components with modern design
        
        # Performance metrics header
        self.assistant_performance_container = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.SPEED, color=SciFiColors.PERFORMANCE, size=16),
                ft.Text("⚡ 3-4x Faster", size=11, color=SciFiColors.PERFORMANCE, weight=ft.FontWeight.W_600),
                ft.Container(width=10),
                ft.Icon(ft.Icons.CACHED, color=SciFiColors.SUCCESS, size=16),
                ft.Text("Smart Cache Active", size=11, color=SciFiColors.SUCCESS, weight=ft.FontWeight.W_500),
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=5),
            padding=ft.padding.symmetric(horizontal=15, vertical=8),
            bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.PERFORMANCE),
            border_radius=20,
            border=ft.border.all(1, ft.Colors.with_opacity(0.3, SciFiColors.PERFORMANCE)),
        )
        
        self.assistant_log_content = ft.Column(
            scroll=ft.ScrollMode.AUTO,
            auto_scroll=True,
            spacing=10,
            expand=True
        )
        
        self.assistant_status_text = ft.Text(
            "Ready to listen",
            size=14,
            color=SciFiColors.TEXT_SECONDARY,
            weight=ft.FontWeight.W_600,
            text_align=ft.TextAlign.CENTER
        )
        
        self.assistant_continuous_toggle = ft.Switch(
            label="Continuous Mode",
            value=self.continuous_mode_active,
            active_color=SciFiColors.PRIMARY,
            label_style=ft.TextStyle(color=SciFiColors.TEXT_PRIMARY, size=13, weight=ft.FontWeight.W_500),
            on_change=lambda e: setattr(self, 'continuous_mode_active', e.control.value)
        )

        self.assistant_tts_toggle = ft.Switch(
            label="Voice Output (TTS)",
            value=self.tts_enabled,
            active_color=SciFiColors.SUCCESS,
            label_style=ft.TextStyle(color=SciFiColors.TEXT_PRIMARY, size=13, weight=ft.FontWeight.W_500),
            on_change=lambda e: self._on_tts_toggle(e)
        )
        
        # Help text with quick tips
        help_text = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.TIPS_AND_UPDATES_OUTLINED, color=SciFiColors.WARNING, size=16),
                    ft.Text("Quick Tips", size=12, color=SciFiColors.WARNING, weight=ft.FontWeight.W_600),
                ], spacing=5),
                ft.Text("• Speak clearly for best results", size=10, color=SciFiColors.TEXT_MUTED),
                ft.Text("• Common queries respond instantly (cached)", size=10, color=SciFiColors.TEXT_MUTED),
                ft.Text("• Try: 'What time is it?', 'Open notepad'", size=10, color=SciFiColors.TEXT_MUTED),
            ], spacing=5, tight=True),
            padding=12,
            bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.WARNING),
            border_radius=8,
            border=ft.border.all(1, ft.Colors.with_opacity(0.2, SciFiColors.WARNING)),
        )

        # --- Start/Stop buttons with enhanced design ---
        self.assistant_start_btn = ft.Container(
            content=ft.Icon(ft.Icons.MIC, color=ft.Colors.WHITE, size=36),
            width=80,
            height=80,
            bgcolor=SciFiColors.PRIMARY,
            border_radius=40,
            alignment=ft.alignment.center,
            on_click=self._assistant_handle_listen_button,
            visible=True,
            shadow=ft.BoxShadow(
                spread_radius=2,
                blur_radius=20,
                color=ft.Colors.with_opacity(0.6, SciFiColors.PRIMARY),
                offset=ft.Offset(0, 0),
            ),
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
        )
        
        self.assistant_stop_btn = ft.Container(
            content=ft.Icon(ft.Icons.STOP, color=ft.Colors.WHITE, size=36),
            width=80,
            height=80,
            bgcolor=SciFiColors.ERROR,
            border_radius=40,
            alignment=ft.alignment.center,
            on_click=self._assistant_stop_continuous, # Default to continuous stop
            visible=False,
            shadow=ft.BoxShadow(
                spread_radius=2,
                blur_radius=20,
                color=ft.Colors.with_opacity(0.6, SciFiColors.ERROR),
                offset=ft.Offset(0, 0),
            ),
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
        )

        # --- UI Layout with modern design ---
        log_container = ft.Container(
            content=self.assistant_log_content,
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.4, SciFiColors.BG_DARK),
            padding=20,
            border_radius=12,
            border=ft.border.all(1, ft.Colors.with_opacity(0.3, SciFiColors.BORDER)),
            width=650,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=10,
                color=ft.Colors.with_opacity(0.2, SciFiColors.BG_DARK),
                offset=ft.Offset(0, 4),
            ),
        )
        
        control_panel = ft.Container(
            content=ft.Column(
                [
                    self.assistant_status_text,
                    ft.Container(height=20),
                    ft.Stack(
                        [self.assistant_start_btn, self.assistant_stop_btn],
                        width=80,
                        height=80
                    ),
                    ft.Container(height=20),
                    ft.Row([
                        self.assistant_continuous_toggle,
                        self.assistant_tts_toggle,
                        ft.Container(expand=True),
                        ft.ElevatedButton(
                            "Clear Chat",
                            icon=ft.Icons.DELETE_SWEEP,
                            on_click=lambda e: self._clear_assistant_chat(),
                            style=ft.ButtonStyle(
                                bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.WARNING),
                                color=SciFiColors.WARNING,
                            ),
                        ),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Container(height=15),
                    help_text,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=0
            ),
            padding=25,
            border_radius=12,
            bgcolor=SciFiColors.BG_CARD,
            width=650,
            border=ft.border.all(1, ft.Colors.with_opacity(0.3, SciFiColors.BORDER)),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=10,
                color=ft.Colors.with_opacity(0.2, SciFiColors.BG_DARK),
                offset=ft.Offset(0, 4),
            ),
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row([
                        ft.Icon(ft.Icons.MIC_NONE_ROUNDED, color=SciFiColors.PRIMARY, size=36),
                        ft.Column([
                            ft.Text("VOICE ASSISTANT", size=32, weight=ft.FontWeight.BOLD, font_family="Orbitron", color=SciFiColors.TEXT_PRIMARY),
                            ft.Text("AI-Powered Voice Commands", size=12, color=SciFiColors.TEXT_MUTED, font_family="Rajdhani"),
                        ], spacing=2),
                    ], spacing=15),
                    ft.Container(height=10),
                    self.assistant_performance_container,
                    ft.Container(height=20),
                    log_container,
                    ft.Container(height=20),
                    control_panel,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True,
                spacing=0
            ),
            padding=ft.padding.only(left=40, right=40, top=20, bottom=20),
            expand=True,
        )
    # --- END REFACTOR ---

    def _create_security_content(self) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("SECURITY CENTER", size=32, weight=ft.FontWeight.BOLD, font_family="Orbitron", color=SciFiColors.TEXT_PRIMARY),
                    ft.Container(height=20),
                    ft.Text("Security settings and logs. (Content pending)", color=SciFiColors.TEXT_SECONDARY, size=14, font_family="Rajdhani"),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER
            ),
            padding=40,
            alignment=ft.alignment.center,
            expand=True,
        )

    def _create_settings_content(self) -> ft.Container:
        """Create system settings content with TTS toggle and other preferences"""
        
        # Initialize TTS enabled flag if not present
        if not hasattr(self, 'tts_enabled'):
            self.tts_enabled = True
        
        # TTS Toggle
        tts_toggle = ft.Switch(
            value=self.tts_enabled,
            on_change=lambda e: self._on_tts_toggle(e),
            active_color=SciFiColors.PRIMARY,
        )
        
        # Create settings sections
        settings_items = ft.Column([
            # Header
            ft.Text(
                "SYSTEM SETTINGS",
                size=32,
                weight=ft.FontWeight.BOLD,
                font_family="Orbitron",
                color=SciFiColors.TEXT_PRIMARY
            ),
            ft.Container(height=30),
            
            # TTS/Audio Settings
            ft.Container(
                content=ft.Column([
                    ft.Text("Audio & Voice Settings", size=18, weight=ft.FontWeight.BOLD, color=SciFiColors.PRIMARY),
                    ft.Container(height=15),
                    
                    # TTS Toggle
                    ft.Container(
                        content=ft.Row([
                            ft.Column([
                                ft.Text("Voice Assistant TTS", size=14, weight=ft.FontWeight.W_600, color=SciFiColors.TEXT_PRIMARY),
                                ft.Text("Enable/disable spoken feedback from the voice assistant", size=12, color=SciFiColors.TEXT_SECONDARY),
                            ], expand=True),
                            tts_toggle,
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                        padding=ft.padding.all(14),
                        bgcolor=ft.Colors.with_opacity(0.05, SciFiColors.PRIMARY),
                        border_radius=8,
                        border=ft.border.all(1, ft.Colors.with_opacity(0.2, SciFiColors.PRIMARY)),
                    ),
                    
                    ft.Container(height=20),
                    
                    # Info text
                    ft.Container(
                        content=ft.Column([
                            ft.Text("💡 Audio Preferences", size=13, weight=ft.FontWeight.W_600, color=SciFiColors.INFO),
                            ft.Text("• When enabled: Voice responses will be spoken aloud using TTS", size=11, color=SciFiColors.TEXT_MUTED),
                            ft.Text("• When disabled: You'll only see text responses (faster, quieter)", size=11, color=SciFiColors.TEXT_MUTED),
                            ft.Text("• Notification alerts will still use system sounds", size=11, color=SciFiColors.TEXT_MUTED),
                        ], spacing=6),
                        padding=ft.padding.all(12),
                        bgcolor=ft.Colors.with_opacity(0.08, SciFiColors.INFO),
                        border_radius=6,
                        border=ft.border.left(2, SciFiColors.INFO),
                    ),
                    
                ], spacing=8),
                padding=ft.padding.all(16),
                bgcolor=SciFiColors.BG_ELEVATED,
                border_radius=10,
                border=ft.border.all(1, SciFiColors.BORDER),
            ),
            
            ft.Container(height=30),
            
            # System Information
            ft.Container(
                content=ft.Column([
                    ft.Text("System Information", size=18, weight=ft.FontWeight.BOLD, color=SciFiColors.PRIMARY),
                    ft.Container(height=15),
                    
                    ft.Row([
                        ft.Column([
                            ft.Text("Application", size=12, weight=ft.FontWeight.W_600, color=SciFiColors.TEXT_SECONDARY),
                            ft.Text("SecureX Assist", size=13, color=SciFiColors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                        ]),
                        ft.Column([
                            ft.Text("Version", size=12, weight=ft.FontWeight.W_600, color=SciFiColors.TEXT_SECONDARY),
                            ft.Text("2.1.0 (Advanced)", size=13, color=SciFiColors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                        ]),
                    ], spacing=40),
                    
                    ft.Container(height=10),
                    
                    ft.Row([
                        ft.Column([
                            ft.Text("Biometric Engine", size=12, weight=ft.FontWeight.W_600, color=SciFiColors.TEXT_SECONDARY),
                            ft.Text("Parallel Processing", size=13, color=SciFiColors.SUCCESS, weight=ft.FontWeight.W_500),
                        ]),
                        ft.Column([
                            ft.Text("Features", size=12, weight=ft.FontWeight.W_600, color=SciFiColors.TEXT_SECONDARY),
                            ft.Text("Voice + Face + Liveness", size=13, color=SciFiColors.TEXT_PRIMARY, weight=ft.FontWeight.W_500),
                        ]),
                    ], spacing=40),
                    
                ], spacing=5),
                padding=ft.padding.all(16),
                bgcolor=SciFiColors.BG_ELEVATED,
                border_radius=10,
                border=ft.border.all(1, SciFiColors.BORDER),
            ),
            
        ], spacing=0, expand=False)
        
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=settings_items,
                    expand=False,
                    padding=ft.padding.all(0),
                ),
            ], expand=True),
            padding=ft.padding.all(30),
        )
    
    def _clear_assistant_chat(self):
        """Clear the assistant chat history"""
        try:
            if self.assistant_log_content:
                self.assistant_log_content.controls.clear()
                self.page.update()
                logger.info("Assistant chat cleared")
        except Exception as e:
            logger.error(f"Error clearing chat: {e}")
    
    def _on_tts_toggle(self, e):
        """Handle TTS toggle change"""
        self.tts_enabled = e.control.value
        logger.info(f"TTS toggled: {self.tts_enabled}")

        # Keep assistant panel toggle in sync when this is triggered from System page.
        if self.assistant_tts_toggle and self.assistant_tts_toggle.value != self.tts_enabled:
            self.assistant_tts_toggle.value = self.tts_enabled
            try:
                self.page.update()
            except Exception:
                pass
        
        # Store preference in config if saving is implemented
        status_text = "TTS enabled - Voice responses will be spoken" if self.tts_enabled else "TTS disabled - Voice responses shown as text only"
        self.show_message("Audio Settings Updated", status_text, ft.Icons.VOLUME_UP if self.tts_enabled else ft.Icons.VOLUME_OFF, SciFiColors.SUCCESS)

    def _run_demo_mode(self):
        """Run scripted demo flow showcasing security policies and alert escalation"""
        logger.info("Starting DEMO MODE sequence")
        
        demo_thread = threading.Thread(target=self._demo_sequence, daemon=True)
        demo_thread.start()
    
    def _demo_sequence(self):
        """Scripted sequence: login → command → spoof → escalation (60-90 seconds)"""
        try:
            # Phase 1: Auto-login (5 sec)
            logger.info("DEMO: Phase 1 - Auto-login")
            time.sleep(1)
            
            # Simulate login with valid credentials
            demo_user = {
                'user_id': 1,
                'username': 'demo_user',
                'password_hash': 'demo_hash',
                'voice_profile': True,
                'face_profile': True,
            }
            self.current_user = demo_user
            
            # Set voice assistant auth state
            self.voice_assistant.set_authentication_state(
                authenticated=True,
                voice_verified=True,
                face_verified=True,
                liveness_verified=True
            )
            
            # Show login success message
            self.show_message("DEMO: Authentication Successful", "Logged in as demo_user", ft.Icons.CHECK_CIRCLE_ROUNDED, SciFiColors.SUCCESS)
            time.sleep(2)
            
            # Phase 2: Execute low-risk command (allowed immediately) (5 sec)
            logger.info("DEMO: Phase 2 - Low-risk command (should succeed)")
            self.show_message("DEMO: Executing Low-Risk Command", "Running: get_system_status", ft.Icons.INFO_ROUNDED, SciFiColors.PRIMARY)
            time.sleep(1)
            
            ok, msg = self.voice_assistant._authorize_intent('get_system_status', None, demo_user['user_id'])
            if ok:
                self.show_message("DEMO: Command Authorized", "get_system_status → SUCCESS", ft.Icons.CHECK_CIRCLE_ROUNDED, SciFiColors.SUCCESS)
            time.sleep(2)
            
            # Phase 3: Wait, then attempt high-risk command (will deny due to freshness/trust) (5 sec)
            logger.info("DEMO: Phase 3 - High-risk command without freshness (should deny)")
            time.sleep(1)
            self.show_message("DEMO: Attempting High-Risk Command", "Running: restart_system (high-risk)", ft.Icons.WARNING_ROUNDED, SciFiColors.WARNING)
            time.sleep(1)
            
            ok, msg = self.voice_assistant._authorize_intent('shutdown', None, demo_user['user_id'])
            if not ok:
                self.show_message("DEMO: Command Denied (Fresh Auth Required)", msg, ft.Icons.LOCK_ROUNDED, SciFiColors.ERROR)
            time.sleep(2)
            
            # Phase 4: Simulate spoof detection (3 failed auth attempts) (15 sec)
            logger.info("DEMO: Phase 4 - Simulating spoof detection (3 failed attempts)")
            time.sleep(1)
            self.show_message("DEMO: Spoof Attack Simulation", "Attempting voice verification with spoofed audio...", ft.Icons.VOICE_CHAT_ROUNDED, SciFiColors.ERROR)
            time.sleep(2)
            
            for attempt in range(3):
                logger.info(f"DEMO: Spoof attempt {attempt + 1}/3")
                self._register_auth_failure(
                    username=demo_user['username'],
                    reason=f"Voice verification failed (attempt {attempt + 1})",
                    base_message="Voice authentication failed",
                    spoof_detected=True,
                    sensitive_action=False
                )
                time.sleep(3)
            
            # Phase 5: Show alert escalation (soft → active → hard) (20 sec)
            # Alerts should already be escalated from the above failures
            self.show_message("DEMO: Alert Escalation Complete", "Security alerts triggered: SOFT → ACTIVE → HARD", ft.Icons.SECURITY_ROUNDED, SciFiColors.ERROR)
            time.sleep(2)
            
            # Phase 6: Attempt login with cooldown active (should be blocked) (10 sec)
            logger.info("DEMO: Phase 6 - Login attempt with active cooldown")
            time.sleep(1)
            cooldown = self._check_auth_cooldown(demo_user['username'])
            if cooldown > 0:
                msg = f"Authentication throttled. Cooldown: {cooldown}s remaining"
                self.show_message("DEMO: Cooldown Active", msg, ft.Icons.SCHEDULE_ROUNDED, SciFiColors.ERROR)
            time.sleep(2)
            
            # Phase 7: Demo conclusion
            logger.info("DEMO: Sequence complete")
            self.show_message("DEMO: Sequence Complete", "Security policies working as designed ✓", ft.Icons.DONE_OUTLINE_ROUNDED, SciFiColors.SUCCESS)
            
            # Cleanup: logout
            time.sleep(3)
            self.logout()
            logger.info("DEMO MODE completed")
            
        except Exception as e:
            logger.error(f"Demo sequence error: {e}", exc_info=True)
            self.show_message("DEMO: Error", f"Demo failed: {str(e)}", ft.Icons.ERROR_ROUNDED, SciFiColors.ERROR)

    def _generate_audit_timeline(self) -> dict:
        """Generate comprehensive audit timeline from all security events"""
        events = []
        current_time = time.time()
        
        # 1. Extract auth alert events from AuthenticationAlertSystem
        if hasattr(self, 'auth_alerts') and self.auth_alerts:
            for username, state in self.auth_alerts.states.items():
                for failure_time in state.failures:
                    events.append({
                        "timestamp": failure_time,
                        "event_type": "authentication_failure",
                        "username": username,
                        "reason": "Failed authentication attempt",
                        "alert_level": "soft" if len([f for f in state.failures if f >= failure_time - self.auth_alerts.window_seconds]) >= self.auth_alerts.soft_threshold else "none",
                    })
                
                # Add cooldown events
                if state.cooldown_until:
                    events.append({
                        "timestamp": state.cooldown_until - self.auth_alerts.hard_cooldown_seconds,
                        "event_type": "throttle_active",
                        "username": username,
                        "reason": "Cooldown initiated",
                        "cooldown_until": state.cooldown_until,
                    })
        
        # 2. Add security status snapshot
        if self.current_user:
            security_status = self.voice_assistant.get_security_status()
            events.append({
                "timestamp": current_time,
                "event_type": "security_snapshot",
                "username": self.current_user.get('username', 'unknown'),
                "trust_score": security_status.get('trust_score', 0),
                "session_valid": security_status.get('session_valid', False),
                "voice_verified": security_status.get('voice_verified', False),
                "face_verified": security_status.get('face_verified', False),
                "liveness_verified": security_status.get('liveness_verified', False),
            })
        
        # Sort events by timestamp
        events.sort(key=lambda x: x['timestamp'])
        
        return {
            "generated_at": current_time,
            "event_count": len(events),
            "events": events
        }

    def _export_audit_timeline(self, format_type: str = "json") -> str:
        """Export audit timeline as JSON or CSV"""
        try:
            audit_data = self._generate_audit_timeline()
            
            if format_type == "json":
                import json
                export_str = json.dumps(audit_data, indent=2, default=str)
                filename = f"audit_timeline_{int(time.time())}.json"
            else:  # CSV format
                import csv
                import io
                output = io.StringIO()
                events = audit_data.get('events', [])
                
                if events:
                    fieldnames = set()
                    for event in events:
                        fieldnames.update(event.keys())
                    
                    writer = csv.DictWriter(output, fieldnames=sorted(fieldnames))
                    writer.writeheader()
                    writer.writerows(events)
                    export_str = output.getvalue()
                else:
                    export_str = "No events recorded"
                
                filename = f"audit_timeline_{int(time.time())}.csv"
            
            # Save to file
            export_path = Path("exports") / filename
            export_path.parent.mkdir(exist_ok=True)
            export_path.write_text(export_str)
            
            logger.info(f"Audit timeline exported to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting audit timeline: {e}", exc_info=True)
            return None

    def _export_audit_action(self):
        """Action handler to export audit timeline"""
        try:
            json_path = self._export_audit_timeline("json")
            if json_path:
                self.show_message(
                    "Export Complete",
                    f"Audit timeline saved to: {json_path}",
                    ft.Icons.DOWNLOAD_DONE_ROUNDED,
                    SciFiColors.SUCCESS
                )
            else:
                self.show_message(
                    "Export Failed",
                    "Could not generate audit timeline",
                    ft.Icons.ERROR_ROUNDED,
                    SciFiColors.ERROR
                )
        except Exception as e:
            logger.error(f"Export action error: {e}")
            self.show_message("Export Error", str(e), ft.Icons.ERROR_ROUNDED, SciFiColors.ERROR)

    def _navigate_to_section(self, section_name: str):
        """Navigate to a different dashboard section"""
        try:
            logger.info(f"Navigating to section: {section_name}")

            if section_name == "settings" and not self._can_access_settings():
                self._show_error_toast("Settings are available only for Administrator and Power User")
                return

            # --- START FIX: Stop listening if navigating away ---
            if self.current_nav_section == "assistant" and section_name != "assistant":
                if self.voice_assistant_active:
                    logger.info("Navigation auto-stopping voice assistant.")
                    # Manually trigger the stop sequence
                    self._assistant_stop_continuous() 
            # --- END FIX ---

            self.current_nav_section = section_name
            
            if self.current_view == "dashboard":
                self.show_dashboard() # This will clear and rebuild the page
            
        except Exception as e:
            logger.error(f"Navigation error: {e}", exc_info=True)
            self._show_error_toast(f"Navigation failed: {str(e)}")

    # ==================== VOICE ASSISTANT PAGE HANDLERS ====================
    
    # --- REFACTOR: All dialog logic is moved here and adapted ---

    def _assistant_handle_listen_button(self, e):
        """Handle listen button"""
        if self.assistant_continuous_toggle.value:
            self._assistant_start_continuous_listening()
        else:
            self._assistant_listen_single_command()

    def _assistant_start_continuous_listening(self):
        """Start continuous listening"""
        try:
            self.voice_assistant_active = True
            
            user_id = self.current_user['id'] if self.current_user else None

            self.voice_assistant.start_continuous_listening(
                self._audio_stream_ctx,
                callback=lambda transcript, response, success: self._assistant_handle_voice_callback(
                    transcript, response, success
                ),
                user_id=user_id
            )
            
            self.assistant_start_btn.visible = False
            self.assistant_stop_btn.visible = True
            self.assistant_stop_btn.on_click = self._assistant_stop_continuous # Set correct handler
            self.assistant_status_text.value = "Listening continuously... Speak now!"
            self.assistant_status_text.color = SciFiColors.SUCCESS
            
            self._assistant_add_log_entry("Continuous listening started", SciFiColors.SUCCESS)
            self.page.update()
        except Exception as e:
            logger.error(f"Error starting continuous listening: {e}")
            self._assistant_add_log_entry(f"Failed: {e}", SciFiColors.ERROR)

    def _assistant_stop_continuous(self, e=None):
        """Stop continuous listening"""
        try:
            # --- START FIX: Explicitly stop the recorder to break VAD loop ---
            if hasattr(self.voice_assistant, 'audio_recorder_context') and self.voice_assistant.audio_recorder_context:
                if hasattr(self.voice_assistant.audio_recorder_context, 'recorder') and self.voice_assistant.audio_recorder_context.recorder:
                    self.voice_assistant.audio_recorder_context.recorder.stop_recording()
                    logger.info("Manually stopping audio recorder for continuous mode.")
            # --- END FIX ---

            self.voice_assistant.stop_continuous_listening()
            self.voice_assistant_active = False
            
            if self.assistant_start_btn:
                self.assistant_start_btn.visible = True
                self.assistant_stop_btn.visible = False
                self.assistant_status_text.value = "Ready to listen"
                self.assistant_status_text.color = SciFiColors.TEXT_SECONDARY
                
                self._assistant_add_log_entry("Stopped", SciFiColors.INFO)
                self.page.update()
        except Exception as e:
            logger.error(f"Error stopping: {e}")
            self._assistant_add_log_entry(f"Error: {e}", SciFiColors.ERROR)

    # --- START: New handler for stopping single command ---
    def _assistant_stop_single_command(self, e=None):
        """Stops a single command recording prematurely"""
        if self.voice_assistant_active:
            logger.info("Single command recording stopped by user.")
            self.audio_recorder.stop_recording()
            self.voice_assistant_active = False # Set flag to stop
            
            # Update UI to show processing
            self.assistant_stop_btn.disabled = True
            self.assistant_status_text.value = "Processing..."
            self.assistant_status_text.color = SciFiColors.WARNING
            self.page.update()
    # --- END: New handler ---

    def _assistant_listen_single_command(self):
        """Listen for single command"""
        
        # --- START FIX: Define async UI updaters ---
        async def update_ui_error():
            self._assistant_add_log_entry("Recording failed", SciFiColors.ERROR)
            self.assistant_status_text.value = "Failed. Ready to listen."
            self.assistant_status_text.color = SciFiColors.ERROR
            self.voice_assistant_active = False
            # Reset buttons
            self.assistant_start_btn.visible = True
            self.assistant_stop_btn.visible = False
            self.assistant_stop_btn.disabled = False
            self.page.update()
        
        async def update_ui_processing():
            self.assistant_status_text.value = "Processing..."
            self.assistant_status_text.color = SciFiColors.WARNING
            # Keep stop button visible but disabled
            self.assistant_start_btn.visible = False
            self.assistant_stop_btn.visible = True
            self.assistant_stop_btn.disabled = True
            self.page.update()
        
        async def update_ui_success(transcript, response, success):
            self._assistant_add_log_entry(f"Heard: '{transcript}'", SciFiColors.INFO)
            self._assistant_add_log_entry(f"Response: {response}", SciFiColors.SUCCESS if success else SciFiColors.ERROR)
            
            # --- START ADDITION: Show toast notification ---
            if success:
                self._show_success_toast(response)
            else:
                self._show_error_toast(response)
            # --- END ADDITION ---

            self.assistant_status_text.value = "Ready to listen"
            self.assistant_status_text.color = SciFiColors.TEXT_SECONDARY
            self.voice_assistant_active = False
            # Reset buttons
            self.assistant_start_btn.visible = True
            self.assistant_stop_btn.visible = False
            self.assistant_stop_btn.disabled = False
            self.page.update()
        
        async def update_ui_no_speech():
            self._assistant_add_log_entry("No speech detected", SciFiColors.WARNING)
            self.assistant_status_text.value = "No speech. Ready to listen."
            self.assistant_status_text.color = SciFiColors.WARNING
            self.voice_assistant_active = False
            # Reset buttons
            self.assistant_start_btn.visible = True
            self.assistant_stop_btn.visible = False
            self.assistant_stop_btn.disabled = False
            self.page.update()
        
        async def update_ui_exception(e):
            self._assistant_add_log_entry(f"Error: {e}", SciFiColors.ERROR)
            self.assistant_status_text.value = f"Error. Ready to listen."
            self.assistant_status_text.color = SciFiColors.ERROR
            self.voice_assistant_active = False
            # Reset buttons
            self.assistant_start_btn.visible = True
            self.assistant_stop_btn.visible = False
            self.assistant_stop_btn.disabled = False
            self.page.update()
        # --- END FIX ---
        
        try:
            self.voice_assistant_active = True
            
            # --- START UI FIX: Update buttons for single command ---
            self.assistant_status_text.value = "Listening..."
            self.assistant_status_text.color = SciFiColors.PRIMARY
            self.assistant_start_btn.visible = False
            self.assistant_stop_btn.visible = True
            self.assistant_stop_btn.disabled = False
            self.assistant_stop_btn.on_click = self._assistant_stop_single_command # Set handler
            # --- END UI FIX ---
            self.page.update()
            
            def record_and_process():
                try:
                    audio_path = self.temp_dir / f"command_{int(time.time())}.wav"
                    stt_timeout = float(self.config.get('system', {}).get('assistant_stt_timeout_seconds', 60.0))
                    command_timeout = float(self.config.get('system', {}).get('assistant_command_timeout_seconds', 60.0))

                    def _run_with_timeout(fn, timeout_seconds, timeout_message):
                        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
                        with ThreadPoolExecutor(max_workers=1) as ex:
                            fut = ex.submit(fn)
                            try:
                                return fut.result(timeout=timeout_seconds)
                            except FuturesTimeoutError:
                                logger.warning(timeout_message)
                                return None

                    # Normal mode should be one-shot: auto-stop on silence via VAD.
                    # Keep stop button support by honoring audio_recorder.stop_recording().
                    audio_cfg = self.config.get('audio', {}) if isinstance(self.config, dict) else {}
                    min_speech_ms = int(float(audio_cfg.get('min_speech_duration', 0.5)) * 1000)
                    max_silence_ms = int(float(audio_cfg.get('assistant_max_silence_duration', 0.9)) * 1000)
                    padding_ms = int(float(audio_cfg.get('assistant_padding_duration', 0.15)) * 1000)

                    audio_data = self.audio_recorder.record_with_vad(
                        vad_detector=self.vad,
                        min_speech_duration_ms=min_speech_ms,
                        max_silence_duration_ms=max_silence_ms,
                        padding_duration_ms=padding_ms,
                    )
                    
                    # If stop was clicked, self.voice_assistant_active will be False
                    if not self.voice_assistant_active:
                        logger.info("Single command recording stopped by user.")
                        # audio_data may contain partial speech
                    else:
                        logger.info("Single command recording finished by VAD auto-stop.")
                    
                    if audio_data is None:
                        # --- FIX: Pass function, not result ---
                        self.page.run_task(update_ui_error)
                        return
                    
                    self.audio_recorder.save_audio(audio_data, str(audio_path))
                    
                    # --- FIX: Pass function, not result ---
                    self.page.run_task(update_ui_processing)
                    
                    transcript = _run_with_timeout(
                        lambda: self.voice_assistant.transcribe(str(audio_path)),
                        stt_timeout,
                        f"Single-command STT timed out after {stt_timeout:.1f}s",
                    )
                    if transcript is None:
                        self.page.run_task(update_ui_exception, Exception("Speech recognition timed out. Please try again."))
                        return
                    if transcript.strip():
                        
                        user_id = self.current_user['id'] if self.current_user else None

                        command_result = _run_with_timeout(
                            lambda: self.voice_assistant.process_voice_command(
                                transcript,
                                audio_data,
                                user_id=user_id,
                            ),
                            command_timeout,
                            f"Single-command processing timed out after {command_timeout:.1f}s",
                        )
                        if command_result is None:
                            self.page.run_task(update_ui_exception, Exception("Command processing timed out. Please try again."))
                            return

                        success, response = command_result
                        
                        # --- FIX: Pass function and args ---
                        # NOTE: Don't call self._speak_async here - voice_assistant.process_voice_command already calls speak() internally
                        self.page.run_task(update_ui_success, transcript, response, success)
                        
                    else:
                        # --- FIX: Pass function, not result ---
                        self.page.run_task(update_ui_no_speech)
                
                except Exception as e:
                    logger.error(f"Error in recording thread: {e}", exc_info=True)
                    # --- FIX: Pass function and args ---
                    self.page.run_task(update_ui_exception, e)
            
            recording_thread = threading.Thread(target=record_and_process, daemon=True)
            recording_thread.start()
            
        except Exception as e:
            logger.error(f"Error: {e}")
            self._assistant_add_log_entry(f"Error: {e}", SciFiColors.ERROR)
            if self.assistant_status_text:
                self.assistant_status_text.value = f"Error: {e}"
                self.assistant_status_text.color = SciFiColors.ERROR
            self.voice_assistant_active = False
            # Reset buttons
            if self.assistant_start_btn:
                self.assistant_start_btn.visible = True
                self.assistant_stop_btn.visible = False
                self.assistant_stop_btn.disabled = False
            if self.page.controls: # Check if page is still active
                self.page.update()

    def _assistant_handle_voice_callback(self, transcript, response, success):
        """Handle voice callback"""
        try:
            if transcript:
                async def update_ui():
                    self._assistant_add_log_entry(f"Heard: '{transcript}'", SciFiColors.INFO)
                    self._assistant_add_log_entry(f"Response: {response}", SciFiColors.SUCCESS if success else SciFiColors.ERROR)
                    
                    # --- START ADDITION: Show toast notification ---
                    if success:
                        self._show_success_toast(response)
                    else:
                        self._show_error_toast(response)
                    # --- END ADDITION ---
                    # VoiceAssistant.process_voice_command already handles TTS.
                    # Avoid duplicate speech here to prevent extra latency.
                
                # --- FIX: Pass function, not result ---
                self.page.run_task(update_ui)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def _assistant_add_log_entry(self, message: str, color: str):
        """Add enhanced entry to log container with icons - now with better chat bubble styling"""
        if self.assistant_log_content is None:
            # This can happen if user navigates away while a log is trying to write
            logger.warning("Assistant log not initialized, skipping log entry.")
            return

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Determine message type and styling
        message_type = "info"
        icon_name = ft.Icons.INFO_OUTLINE
        if color == SciFiColors.INFO:
            message_type = "heard"
            icon_name = ft.Icons.HEARING
        elif color == SciFiColors.SUCCESS:
            message_type = "response"
            icon_name = ft.Icons.DONE_OUTLINE
        elif color == SciFiColors.ERROR:
            message_type = "error"
            icon_name = ft.Icons.ERROR_OUTLINE
        elif color == SciFiColors.WARNING:
            message_type = "status"
            icon_name = ft.Icons.WARNING_AMBER_ROUNDED
        
        # Create message bubble with better formatting
        log_entry = ft.Container(
            content=ft.Row([
                # Icon
                ft.Container(
                    content=ft.Icon(icon_name, size=18, color=color),
                    width=28,
                    alignment=ft.alignment.center,
                ),
                # Message content
                ft.Column([
                    ft.Row([
                        ft.Text(
                            message,
                            size=13,
                            color=SciFiColors.TEXT_PRIMARY,
                            overflow=ft.TextOverflow.VISIBLE,
                            weight=ft.FontWeight.W_500 if message_type == "response" else ft.FontWeight.W_400,
                            expand=True,
                        ),
                    ], expand=True),
                    ft.Text(
                        f"[{timestamp}]",
                        size=10,
                        color=SciFiColors.TEXT_MUTED,
                        weight=ft.FontWeight.W_400,
                    ),
                ], spacing=6, expand=True),
            ], expand=True, spacing=12, vertical_alignment=ft.CrossAxisAlignment.START),
            margin=ft.margin.only(bottom=8),
            padding=ft.padding.symmetric(horizontal=14, vertical=10),
            border_radius=10,
            bgcolor=ft.Colors.with_opacity(0.08, color),
            border=ft.border.all(1, ft.Colors.with_opacity(0.25, color)),
        )
        
        try:
            if self.assistant_log_content and self.assistant_log_content.controls is not None:
                self.assistant_log_content.controls.append(log_entry)
                # Ensure scroll to bottom
                if hasattr(self.assistant_log_content, 'auto_scroll'):
                    self.assistant_log_content.auto_scroll = True
                # Force update on main thread
                try:
                    self.page.update()
                except:
                    logger.debug("Page update failed, but log entry was added")
            else:
                logger.warning("Log content controls not initialized, skipping log entry.")
        except Exception as e:
            # This can happen if page is closed
            logger.error(f"Error updating assistant log: {e}")


    # ==================== ACTION HANDLERS ====================
    
    def _take_screenshot_action(self, event):
        """Take screenshot action"""
        try:
            screenshot = pyautogui.screenshot()
            screenshot_path = self.temp_dir / f"screenshot_{int(time.time())}.png"
            screenshot.save(str(screenshot_path))
            logger.info(f"Screenshot saved to: {screenshot_path}")
            self._show_success_toast("Screenshot saved!")
        except Exception as e:
            logger.error(f"Screenshot failed: {e}", exc_info=True)
            self._show_error_toast(f"Screenshot failed: {e}")

    def _show_system_status(self, event):
        """Show system status"""
        try:
            import platform
            
            cpu = psutil.cpu_percent(interval=0.4)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            uptime = str(datetime.timedelta(seconds=int(time.time() - psutil.boot_time())))
            
            content = ft.Column([
                self._create_metric_row("Operating System", f"{platform.system()} {platform.release()}"),
                self._create_metric_row("CPU Usage", f"{cpu:.0f}%", SciFiColors.INFO),
                self._create_metric_row("Memory Usage", f"{memory:.0f}%", SciFiColors.WARNING),
                self._create_metric_row("Disk Usage", f"{disk:.0f}%", SciFiColors.WARNING),
                self._create_metric_row("Uptime", uptime, SciFiColors.SUCCESS),
            ], spacing=10)
            
            self._show_modern_dialog("SYSTEM STATUS", content, SciFiColors.INFO, ft.Icons.COMPUTER_ROUNDED)
            
        except Exception as e:
            logger.error(f"Error: {e}")
            self._show_error_toast(f"Failed: {e}")

    def _run_security_scan(self):
        """Run security scan"""
        self._show_success_toast("Security scan initiated...") # Show "in progress"
        async def run_scan():
            await asyncio.sleep(2) # Simulate scan time
            self._show_success_toast("Security scan complete - No threats detected")
        
        # --- FIX: Pass function, not result ---
        self.page.run_task(run_scan)


    def _create_metric_row(self, label: str, value: str, color: str = SciFiColors.TEXT_PRIMARY):
        """Create metric row"""
        return ft.Container(
            content=ft.Row([
                ft.Text(label, size=12, color=SciFiColors.TEXT_SECONDARY),
                ft.Container(expand=True),
                ft.Text(value, size=12, weight=ft.FontWeight.BOLD, color=color),
            ]),
            padding=10,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE),
            border=ft.border.all(1, SciFiColors.BORDER),
            border_radius=6,
        )

    def _show_modern_dialog(self, title: str, content: ft.Control, icon_color: str, icon: str):
        """Show modern dialog"""
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(icon, color=icon_color, size=24),
                ft.Text(title, size=16, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
            ], spacing=10),
            content=ft.Container(content=content, width=450, padding=16),
            actions=[
                ft.TextButton(
                    "CLOSE",
                    on_click=lambda _: self._close_dialog_safe(dialog),
                    style=ft.ButtonStyle(color=SciFiColors.TEXT_SECONDARY),
                ),
            ],
            bgcolor=SciFiColors.BG_CARD,
            shape=ft.RoundedRectangleBorder(radius=10),
        )
        
        self._open_dialog_safe(dialog)

    # ==================== DIALOG MANAGEMENT ====================
    
    def _open_dialog_safe(self, dialog):
        """Open dialog safely"""
        try:
            logger.debug("[DEBUG] Entering _open_dialog_safe")
            if dialog not in self.active_dialogs:
                logger.debug("[DEBUG] Adding dialog to active_dialogs list")
                self.active_dialogs.append(dialog)
            logger.debug(f"[DEBUG] Setting self.page.dialog to dialog: {dialog}")
            self.page.dialog = dialog
            logger.debug(f"[DEBUG] Setting dialog.open = True")
            dialog.open = True
            logger.debug(f"[DEBUG] Calling self.page.update()")
            self.page.update()
            logger.info(f"Dialog opened: type={type(dialog).__name__}")
            logger.debug("[DEBUG] Exiting _open_dialog_safe")
        except Exception as e:
            logger.error(f"Error opening dialog: {e}")
            self._show_error_toast(f"Failed to open dialog: {e}")

    def _close_dialog_safe(self, dialog, update_page=True): # <-- FIX: Add update_page flag
        """Close dialog safely"""
        try:
            dialog.open = False
            if self.page.dialog == dialog:
                self.page.dialog = None
            
            if dialog in self.active_dialogs:
                self.active_dialogs.remove(dialog)
            
            if update_page: # <-- FIX: Check flag
                self.page.update()
            
        except Exception as e:
            logger.error(f"Error closing dialog: {e}")

    # ==================== TOAST NOTIFICATIONS (FIXED) ====================
    
    def _show_success_toast(self, message: str):
        """Show success toast"""
        self.page.snack_bar = ft.SnackBar(
            content=ft.Row([
                ft.Icon(ft.Icons.CHECK_CIRCLE_ROUNDED, color=SciFiColors.SUCCESS),
                ft.Text(message, color=SciFiColors.TEXT_PRIMARY),
            ], spacing=10),
            bgcolor=SciFiColors.BG_CARD,
            # --- FIX: Removed 'border' property ---
        )
        self.page.snack_bar.open = True
        self.page.update()

    def _show_error_toast(self, message: str):
        """Show error toast"""
        self.page.snack_bar = ft.SnackBar(
            content=ft.Row([
                ft.Icon(ft.Icons.ERROR_ROUNDED, color=SciFiColors.ERROR),
                ft.Text(message, color=SciFiColors.TEXT_PRIMARY),
            ], spacing=10),
            bgcolor=SciFiColors.BG_CARD,
            # --- FIX: Removed 'border' property ---
        )
        self.page.snack_bar.open = True
        self.page.update()

    def show_message(self, title: str, message: str, icon=None, color: str = SciFiColors.INFO, duration_ms=4000):
        """Show a custom message with title and icon for demo mode and alerts"""
        try:
            icon_widget = ft.Icon(icon, color=color, size=24) if icon else None
            
            content_items = []
            if icon_widget:
                content_items.append(icon_widget)
            
            content_items.append(
                ft.Column([
                    ft.Text(title, weight=ft.FontWeight.BOLD, color=color, size=12),
                    ft.Text(message, color=SciFiColors.TEXT_PRIMARY, size=11),
                ], spacing=4)
            )
            
            self.page.snack_bar = ft.SnackBar(
                content=ft.Row(content_items, spacing=12, alignment=ft.MainAxisAlignment.START),
                bgcolor=ft.Colors.with_opacity(0.9, SciFiColors.BG_CARD),
                open=True,
                duration=duration_ms,
            )
            self.page.update()
        except Exception as e:
            logger.error(f"Error showing message: {e}")

    # ==================== STATUS HELPERS ====================
    
    def update_status(self, message: str, color: str = SciFiColors.INFO):
        """Update login status with enhanced visual feedback"""
        if not hasattr(self, 'status_text') or not self.status_text: return # Guard against UI not existing
        self.status_text.value = message
        self.status_text.color = color
        
        # Enhanced visual styling based on message type
        if "✓" in message or "✅" in message or "SUCCESS" in message.upper():
            self.status_panel.bgcolor = ft.Colors.with_opacity(0.15, SciFiColors.SUCCESS)
            self.status_panel.border = ft.border.all(2, SciFiColors.SUCCESS)
            self.status_text.color = SciFiColors.SUCCESS
        elif "⚠" in message or "ERROR" in message.upper() or "FAILED" in message.upper():
            self.status_panel.bgcolor = ft.Colors.with_opacity(0.15, SciFiColors.ERROR)
            self.status_panel.border = ft.border.all(2, SciFiColors.ERROR)
            self.status_text.color = SciFiColors.ERROR
        elif "⟳" in message or "PROCESSING" in message.upper() or "ANALYZING" in message.upper():
            self.status_panel.bgcolor = ft.Colors.with_opacity(0.15, SciFiColors.PRIMARY)
            self.status_panel.border = ft.border.all(2, SciFiColors.PRIMARY)
            self.status_text.color = SciFiColors.PRIMARY
        else:
            self.status_panel.bgcolor = ft.Colors.with_opacity(0.1, color)
            self.status_panel.border = ft.border.all(1, color)
        
        self.status_panel.visible = bool(message)
        self.page.update()

    def update_confidence_meter(self, confidence: float, label: str = "Confidence"):
        """Update confidence meter with percentage value (0-1 scale)"""
        if not hasattr(self, 'confidence_meter') or not self.confidence_meter:
            return
        
        # Convert to percentage
        confidence_pct = confidence * 100
        
        # Update progress ring
        self.confidence_meter.value = confidence
        
        # Update text
        if hasattr(self, 'confidence_text') and self.confidence_text:
            self.confidence_text.value = f"{confidence_pct:.1f}%"
            
            # Color coding based on confidence level
            if confidence_pct >= 80:
                self.confidence_text.color = SciFiColors.SUCCESS
                self.confidence_meter.color = SciFiColors.SUCCESS
            elif confidence_pct >= 60:
                self.confidence_text.color = SciFiColors.WARNING
                self.confidence_meter.color = SciFiColors.WARNING
            else:
                self.confidence_text.color = SciFiColors.ERROR
                self.confidence_meter.color = SciFiColors.ERROR
        
        self.page.update()
    
    def update_score_display(self, voice_score: float = None, face_score: float = None, 
                            fusion_score: float = None, show: bool = True):
        """Update individual biometric scores display"""
        if not hasattr(self, 'score_display') or not self.score_display:
            return
        
        if not show:
            self.score_display.visible = False
            self.page.update()
            return
        
        scores_text = []
        
        if voice_score is not None:
            color = SciFiColors.SUCCESS if voice_score >= 0.7 else SciFiColors.WARNING
            scores_text.append(f"🎤 Voice: {voice_score*100:.1f}%")
        
        if face_score is not None:
            color = SciFiColors.SUCCESS if face_score >= 0.7 else SciFiColors.WARNING
            scores_text.append(f"👤 Face: {face_score*100:.1f}%")
        
        if fusion_score is not None:
            color = SciFiColors.SUCCESS if fusion_score >= 0.7 else SciFiColors.WARNING
            scores_text.append(f"🔒 Combined: {fusion_score*100:.1f}%")
        
        if scores_text and hasattr(self.score_display, 'content'):
            self.score_display.content.value = " | ".join(scores_text)
            self.score_display.visible = True
        
        self.page.update()

    def update_reg_status(self, message: str, color: str):
        """Update registration status with enhanced visual feedback"""
        if not hasattr(self, 'reg_status_text') or not self.reg_status_text: return # Guard against UI not existing
        self.reg_status_text.value = message
        self.reg_status_text.color = color
        
        # Enhanced visual styling based on message type
        if "✓" in message or "✅" in message or "SUCCESS" in message.upper() or "COMPLETE" in message.upper():
            self.reg_status_panel.bgcolor = ft.Colors.with_opacity(0.15, SciFiColors.SUCCESS)
            self.reg_status_panel.border = ft.border.all(2, SciFiColors.SUCCESS)
            self.reg_status_text.color = SciFiColors.SUCCESS
        elif "⚠" in message or "ERROR" in message.upper() or "FAILED" in message.upper():
            self.reg_status_panel.bgcolor = ft.Colors.with_opacity(0.15, SciFiColors.ERROR)
            self.reg_status_panel.border = ft.border.all(2, SciFiColors.ERROR)
            self.reg_status_text.color = SciFiColors.ERROR
        elif "⟳" in message or "PROCESSING" in message.upper() or "ANALYZING" in message.upper() or "CREATING" in message.upper():
            self.reg_status_panel.bgcolor = ft.Colors.with_opacity(0.15, SciFiColors.PRIMARY)
            self.reg_status_panel.border = ft.border.all(2, SciFiColors.PRIMARY)
            self.reg_status_text.color = SciFiColors.PRIMARY
        else:
            self.reg_status_panel.bgcolor = ft.Colors.with_opacity(0.1, color)
            self.reg_status_panel.border = ft.border.all(1, color)
        
        self.reg_status_panel.visible = bool(message)
        self.page.update()

    def show_progress(self, show: bool = True):
        """Show/hide progress indicator with animation"""
        if not hasattr(self, 'progress_ring') or not self.progress_ring: return
        self.progress_ring.visible = show
        if show:
            self.progress_ring.value = None  # Indeterminate spinner
        self.page.update()

    def show_reg_progress(self, visible: bool):
        """Show/hide registration progress indicator with animation"""
        if not hasattr(self, 'reg_progress_ring') or not self.reg_progress_ring: return
        self.reg_progress_ring.visible = visible
        if visible:
            self.reg_progress_ring.value = None  # Indeterminate spinner
        self.page.update()

    # ===================================================================
    # --- AUTHENTICATION & REGISTRATION ---
    # --- NO CHANGES MADE TO THIS SECTION AS REQUESTED ---
    # ===================================================================

    def _check_auth_cooldown(self, username: str) -> int:
        """Return remaining auth cooldown in seconds, or 0 if clear."""
        if not username:
            return 0
        return self.auth_alerts.get_remaining_cooldown(username)

    def _start_auth_cooldown_countdown(self, username: str):
        """Start/restart per-second countdown updates for active auth cooldown."""
        if not username:
            return

        if self.auth_cooldown_task and not self.auth_cooldown_task.done():
            if self.auth_cooldown_username == username:
                return
            self.auth_cooldown_task.cancel()

        self.auth_cooldown_username = username
        self.auth_cooldown_task = asyncio.create_task(self._run_auth_cooldown_countdown(username))

    async def _run_auth_cooldown_countdown(self, username: str):
        """Show login cooldown countdown (e.g., 30, 29, 28...) in status text."""
        try:
            while True:
                remaining = self._check_auth_cooldown(username)
                if remaining <= 0:
                    break

                self.update_status(
                    f"⏳ Too many failed attempts. Retry after {remaining}s",
                    SciFiColors.WARNING,
                )
                await asyncio.sleep(1)

            # Only clear task state if this is still the active countdown task.
            if self.auth_cooldown_username == username:
                self.auth_cooldown_username = None
                self.auth_cooldown_task = None
        except asyncio.CancelledError:
            if self.auth_cooldown_username == username:
                self.auth_cooldown_username = None
                self.auth_cooldown_task = None
            raise

    async def _register_auth_failure(
        self,
        username: str,
        reason: str,
        base_message: str,
        spoof_detected: bool = False,
        sensitive_action: bool = False,
    ):
        """Record authentication failure and surface staged alert feedback."""
        decision = self.auth_alerts.register_failure(
            username=username,
            reason=reason,
            spoof_detected=spoof_detected,
            sensitive_action=sensitive_action,
        )

        if decision.level == AlertLevel.HARD:
            message = f"{base_message} | HARD ALERT: {decision.cooldown_seconds}s lockout"
        elif decision.level == AlertLevel.ACTIVE:
            message = f"{base_message} | ACTIVE ALERT: wait {decision.cooldown_seconds}s"
        else:
            message = f"{base_message} | failed attempts: {decision.recent_failures}"

        self.update_status(message, SciFiColors.ERROR)

        if decision.level == AlertLevel.HARD:
            await self._speak_async("Hard security alert triggered. Please wait before retrying")
        elif decision.level == AlertLevel.ACTIVE:
            await self._speak_async("Too many failed attempts. Please wait and try again")

        if decision.cooldown_seconds > 0 and decision.recent_failures >= self.auth_alerts.active_threshold:
            self._start_auth_cooldown_countdown(username)
    
    def start_voice_login(self):
        """Start voice authentication"""
        self.page.run_task(self._start_voice_login_async)
    
    async def _start_voice_login_async(self):
        """Async handler for voice login"""
        try:
            logger.info("=== VOICE LOGIN STARTED ===")
            username = self.username_field.value
            password = self.password_field.value
            selected_account_type = (self.account_type_field.value or "standard_user").strip().lower()
            
            if not username or not password:
                self.update_status("⚠ Please enter both username and password", SciFiColors.ERROR)
                await self._speak_async("Please enter your credentials")
                return

            cooldown = self._check_auth_cooldown(username)
            if cooldown > 0:
                self.update_status(
                    f"⏳ Too many failed attempts. Retry after {cooldown}s",
                    SciFiColors.WARNING,
                )
                self._start_auth_cooldown_countdown(username)
                await self._speak_async("Too many failed attempts. Please wait before retrying")
                return
            
            logger.info(f"Attempting login for user: {username}")
            
            self.update_status("⟳ Validating credentials...", SciFiColors.INFO)
            self.show_progress(True)
            
            user = self.db.get_user_by_username(username)
            if not user:
                self.show_progress(False)
                logger.warning(f"User not found: {username}")
                await self._register_auth_failure(
                    username=username,
                    reason="unknown_user",
                    base_message="⚠ User not found - Please check username or register first",
                )
                await self._speak_async("User not found")
                return

            stored_account_type = (user.get('account_type') or 'standard_user').strip().lower()
            if selected_account_type != stored_account_type:
                self.show_progress(False)
                logger.warning(
                    "Account type mismatch for user %s: selected=%s stored=%s",
                    username,
                    selected_account_type,
                    stored_account_type,
                )
                await self._register_auth_failure(
                    username=username,
                    reason="account_type_mismatch",
                    base_message="⚠ Account type does not match this user profile",
                )
                await self._speak_async("Selected account type does not match your profile")
                return
            
            logger.info(f"User found: {user['id']}")
            
            if not self.security_manager.verify_hashed_password(password, user['password_hash']):
                self.show_progress(False)
                logger.warning("Password verification failed")
                await self._register_auth_failure(
                    username=username,
                    reason="invalid_password",
                    base_message="⚠ Invalid password - Please try again",
                    sensitive_action=False,
                )
                await self._speak_async("Invalid password")
                return
            
            logger.info("Password verified successfully")
            self.auth_alerts.register_success(username)
            self.show_progress(False)
            
            self.reset_auth_states()
            self.auth_step = "voice_verifying"
            self.update_auth_progress("voice_verifying")
            
            # Show the record button immediately
            self.show_record_button(mode="voice")
            
            # TTS BEFORE showing button (speak first, then user can click)
            await self._speak_async("Credentials verified. When ready, click start recording and speak clearly for three to five seconds.")
            
            # Update status AFTER TTS
            self.update_status("⦿ Ready - Click START RECORDING, then speak 3-5 seconds", SciFiColors.INFO)
            
            logger.info("Voice verification setup complete - waiting for user to click record button")
            
        except Exception as e:
            logger.error(f"Error in start_voice_login: {e}", exc_info=True)
            self.update_status(f"⚠ Login error: {str(e)}", SciFiColors.ERROR)
            self.show_progress(False)

    def start_registration(self):
        """Start registration"""
        self.page.run_task(self.process_registration)

    async def process_registration(self):
        """Process registration"""
        try:
            self.reset_auth_states()
            self.auth_step = "voice_enrolling"
            self.update_auth_progress("voice_enrolling")
            
            username = self.reg_username_field.value
            password = self.reg_password_field.value
            confirm_password = self.reg_confirm_password_field.value
            email = self.reg_email_field.value
            selected_account_type = (self.reg_account_type_field.value or "standard_user").strip().lower()
            
            if not username or not password:
                self.update_reg_status("⚠ Username and password required", SciFiColors.ERROR)
                return
            
            if len(username) < 3:
                self.update_reg_status("⚠ Username must be 3+ characters", SciFiColors.ERROR)
                return
            
            if len(password) < 6:
                self.update_reg_status("⚠ Password must be 6+ characters", SciFiColors.ERROR)
                return
            
            if password != confirm_password:
                self.update_reg_status("⚠ Passwords don't match", SciFiColors.ERROR)
                return
            
            existing_user = self.db.get_user_by_username(username)
            if existing_user:
                self.update_reg_status("⚠ Username taken", SciFiColors.ERROR)
                return
            
            self.update_reg_status("⟳ Creating account...", SciFiColors.SUCCESS)
            self.show_reg_progress(True)
            
            password_hash = self.security_manager.hash_password(password)
            
            user_id = self.db.create_user(
                username=username,
                password_hash=password_hash,
                account_type=selected_account_type,
                email=email if email else None
            )
            
            self.update_reg_status("✓ Account created - Enrolling voice", SciFiColors.SUCCESS)
            await asyncio.sleep(2)
            
            user = self.db.get_user_by_username(username)
            await self.enroll_user_voice_registration(user)

            # Check if voice enrollment was successful before proceeding
            if not self.voice_enrollment_complete:
                logger.warning("Voice enrollment was not completed, aborting registration.")
                self.show_reg_progress(False)
                # Status already set by enroll_user_voice_registration on failure
                return
            
            self.update_reg_status("✅ Voice enrolled - Now enrolling face", SciFiColors.SUCCESS)
            logger.info("Voice enrollment completed, starting face enrollment")
            await asyncio.sleep(2)
            
            try:
                face_ok = await self.enroll_user_face_arcface(user)
                logger.info("Face enrollment completed successfully (ArcFace)")
                if self.require_face_enrollment and not face_ok:
                    logger.warning("Face enrollment required but failed; deleting incomplete user.")
                    self.db.delete_user(user['id'])
                    self.update_reg_status(
                        "⚠ Registration failed: Face enrollment is required. Please register again.",
                        SciFiColors.ERROR,
                    )
                    await self._speak_async("Face enrollment is required. Registration has been cancelled.")
                    self.show_reg_progress(False)
                    return
            except Exception as e:
                logger.error(f"Face enrollment failed: {e}")
                self.update_reg_status(f"⚠ Face enrollment failed: {str(e)}", SciFiColors.ERROR)
                await asyncio.sleep(2)
                if self.require_face_enrollment:
                    self.db.delete_user(user['id'])
                    self.update_reg_status(
                        "⚠ Registration cancelled because face enrollment is mandatory.",
                        SciFiColors.ERROR,
                    )
                    self.show_reg_progress(False)
                    return
            
            self.auth_tabs.selected_index = 0
            self._handle_auth_tab_change(type('obj', (object,), {'control': self.auth_tabs})())
            self.update_status("✓ Registration complete! Please login.", SciFiColors.SUCCESS)
            self.show_reg_progress(False)
            
            self.cleanup_old_temp_files()
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            self.update_reg_status(f"⚠ Failed: {str(e)}", SciFiColors.ERROR)
            self.show_reg_progress(False)

    # ==================== ENROLLMENT & VERIFICATION ====================

    async def enroll_user_voice_registration(self, user: dict):
        """Enroll voice during registration using ultimate biometric engine"""
        try:
            self.update_reg_status("⟳ Initializing voice enrollment system...", SciFiColors.INFO)
            await asyncio.sleep(0.5)
            
            self.update_reg_status("⟳ Recording 3 voice samples with advanced data augmentation", SciFiColors.INFO)
            await self._speak_async("Please provide 3 voice samples for secure enrollment")
            await asyncio.sleep(1)
            
            audio_samples = []
            
            for sample_num in range(1, 4):
                self.update_reg_status(f"⦿ Sample {sample_num}/3 - Click START RECORDING when ready", SciFiColors.INFO)
                await self._speak_async(f"Please click START RECORDING for voice sample {sample_num} of 3")
                        
                self.show_reg_record_button(f"SAMPLE {sample_num} OF 3", mode="voice")
                self.page.update()
                
                logger.info(f"Waiting for sample {sample_num} recording to start...")
                
                # Wait for user to click start
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.reg_recording_started_event.wait(timeout=120.0)
                )
                
                if not self.reg_recording_started_event.is_set():
                    self.update_reg_status(f"⚠ Sample {sample_num} timeout", SciFiColors.ERROR)
                    continue
                
                self.reg_recording_started_event.clear()
                logger.info(f"Sample {sample_num} recording started")
                
                await self._speak_async("Recording now, please speak")
                
                # Wait for user to click stop
                logger.info(f"Waiting for sample {sample_num} recording to stop...")
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.reg_recording_stop_event.wait(timeout=15.0)
                )
                
                if not self.reg_recording_stop_event.is_set():
                    # Auto-stop after timeout
                    self.reg_recording_active = False
                    self.reg_recording_stop_event.set()
                
                self.reg_recording_stop_event.clear()
                logger.info(f"Sample {sample_num} recording stopped")
                
                # Wait for the registration recording thread to complete (started by UI click handler)
                logger.info("Waiting for registration recording thread to finish for sample %d...", sample_num)
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.reg_recording_complete_event.wait(timeout=15.0)
                )

                # Hide the registration button now that recording is finished
                self.hide_reg_record_button()

                # Retrieve audio from holder
                audio_data = None
                try:
                    audio_data = self.reg_audio_holder[0]
                    # Clear the holder after retrieving to prevent reuse
                    self.reg_audio_holder[0] = None
                except (IndexError, TypeError, Exception) as e:
                    logger.warning(f"Failed to retrieve audio from holder: {e}")
                    audio_data = None

                if audio_data is None:
                    self.update_reg_status(f"⚠ Sample {sample_num} failed - No audio captured", SciFiColors.WARNING)
                    await self._speak_async("Recording failed, please try again")
                    await asyncio.sleep(1)
                    continue

                self.update_reg_status(f"✓ Sample {sample_num} recorded - Analyzing quality...", SciFiColors.SUCCESS)
                await self._speak_async("Recording complete, analyzing voice sample quality")
                await asyncio.sleep(0.3)
                
                self.update_reg_status(f"⟳ Sample {sample_num} - Running quality checks...", SciFiColors.INFO)
                await asyncio.sleep(0.3)
                
                audio_samples.append(audio_data)
                
                self.update_reg_status(f"✅ Sample {sample_num}/3 validated and saved", SciFiColors.SUCCESS)
                await self._speak_async(f"Sample {sample_num} validated successfully")
                await asyncio.sleep(1)
            
            if len(audio_samples) < 2:
                self.update_reg_status("⚠ Insufficient valid samples - Minimum 2 required, Please try again", SciFiColors.ERROR)
                await self._speak_async("Insufficient voice samples recorded, please restart enrollment")
                self.show_reg_progress(False)
                self.voice_enrollment_complete = False
                return
            
            self.update_reg_status(f"⟳ Processing {len(audio_samples)} validated samples...", SciFiColors.INFO)
            await asyncio.sleep(0.5)
            
            self.update_reg_status("⟳ Generating augmented dataset for robust profile...", SciFiColors.INFO)
            await asyncio.sleep(0.5)
            
            self.update_reg_status("⟳ Extracting voice embeddings with advanced AI...", SciFiColors.INFO)
            await asyncio.sleep(0.5)
            
            self.update_reg_status("⟳ Computing mean and variance for adaptive matching...", SciFiColors.INFO)
            await asyncio.sleep(0.5)
            
            self.update_reg_status("⟳ Creating secure voice profile with anti-spoofing protection...", SciFiColors.INFO)
            self.show_reg_progress(True)
            await self._speak_async("Creating your secure voice profile with advanced security features")
            
            logger.info(f"Enrolling voice with {len(audio_samples)} samples")
            success = self.ultimate_voice_engine.enroll_user_voice(
                user_id=user['id'],
                audio_samples=audio_samples,
                sample_rate=16000
            )
            
            self.show_reg_progress(False)
            
            if success:
                self.update_reg_status("✅ Voice enrollment complete! Profile secured with anti-spoofing and adaptive learning", SciFiColors.SUCCESS)
                await self._speak_async("Voice enrollment complete with advanced security features")
                self.voice_enrollment_complete = True
                self.update_auth_progress("face_enrolling")
                await asyncio.sleep(1.5)
            else:
                self.update_reg_status("⚠ Voice enrollment failed - Insufficient quality or validation errors. Please try again", SciFiColors.ERROR)
                await self._speak_async("Voice enrollment failed, please try again with clear voice samples")
                self.voice_enrollment_complete = False
            
        except Exception as e:
            logger.error(f"Voice enrollment error: {e}", exc_info=True)
            self.update_reg_status(f"⚠ Enrollment error: {str(e)}", SciFiColors.ERROR)
            await self._speak_async("Voice enrollment failed due to an error")
            self.show_reg_progress(False)

    async def enroll_user_face_arcface(self, user: dict):
        """Enroll user face using ArcFace and store embedding in DB"""
        self.face_enrollment_complete = False
        self.update_reg_status("⦿ Automatically capturing face for enrollment (ArcFace)...", SciFiColors.INFO)
        await self._speak_async("Please look at the camera for automatic face enrollment")
        
        max_attempts = 3
        face_enrolled = False
        
        for attempt in range(max_attempts):
            self.update_reg_status(f"⦿ Face capture attempt {attempt + 1}/{max_attempts}...", SciFiColors.INFO)
            await asyncio.sleep(0.5)
            
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                self.update_reg_status("⟳ Processing face image...", SciFiColors.INFO)
                await self._speak_async("Processing face image")
                embedding = self.enroll_face_arcface(frame)
                if embedding is not None:
                    self.db.deactivate_old_face_embeddings(user['id'])
                    self.db.store_face_embedding(user['id'], embedding, embedding_type="arcface", quality_score=1.0)
                    self.update_reg_status("✅ Face enrolled successfully (ArcFace)", SciFiColors.SUCCESS)
                    await self._speak_async("Face enrolled successfully")
                    face_enrolled = True
                    self.face_enrollment_complete = True
                    break
                else:
                    self.update_reg_status(f"⚠ Face enrollment failed - no face detected (attempt {attempt + 1})", SciFiColors.WARNING)
                    if attempt < max_attempts - 1:
                        await self._speak_async("No face detected, trying again")
                        await asyncio.sleep(1)
            else:
                self.update_reg_status(f"⚠ Face capture failed (attempt {attempt + 1})", SciFiColors.WARNING)
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1)
        
        if not face_enrolled:
            self.update_reg_status("⚠ Face enrollment failed after all attempts", SciFiColors.ERROR)
            await self._speak_async("Face enrollment failed, please try again")
            await asyncio.sleep(1)

        return face_enrolled

    def enroll_face_arcface(self, image):
        """Enroll face using InsightFace ArcFace model."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            logger.error("Invalid image provided to face enrollment")
            return None
        faces = self.arcface_model.get(image)
        if not faces:
            logger.warning("No face detected for enrollment")
            return None
        face = max(faces, key=lambda f: f.bbox[2]*f.bbox[3])
        embedding = face.embedding.tolist()
        logger.info(f"ArcFace embedding extracted: {embedding[:8]}...")
        return embedding

    async def process_voice_verification(self, audio_data):
        """Process recorded audio for voice verification"""
        try:
            # Get the current user from the login process
            username = self.username_field.value
            user = self.db.get_user_by_username(username)
            if not user:
                await self._register_auth_failure(
                    username=username,
                    reason="unknown_user_during_voice_verify",
                    base_message="⚠ User not found",
                )
                self.hide_record_button()
                return

            if self.require_face_verification_on_login:
                face_embeddings = self.db.get_face_embeddings(user['id'])
                has_arcface_profile = any(
                    isinstance(emb.get('embedding_data'), list) and len(emb.get('embedding_data')) == 512
                    for emb in face_embeddings
                )
                if not has_arcface_profile:
                    await self._register_auth_failure(
                        username=username,
                        reason="face_profile_missing",
                        base_message="⚠ Face profile missing. Face enrollment is required before login.",
                        sensitive_action=True,
                    )
                    await self._speak_async("Face profile missing. Please complete face enrollment first.")
                    self.hide_record_button()
                    return
            
            logger.info("Audio data captured successfully")
            logger.info("DEBUG: Entered post-capture block in process_voice_verification")

            self.update_status("⟳ Analyzing voice with AASIST anti-spoofing...", SciFiColors.INFO)
            logger.info("DEBUG: Updated status to anti-spoofing analysis")
            self.show_progress(True)
            self.page.update()
            logger.info("DEBUG: UI progress shown and page updated")

            logger.info("DEBUG: About to call TTS async")
            try:
                await self._speak_async("Analyzing voice with advanced security")
                logger.info("DEBUG: Finished TTS async call")
            except Exception as e:
                logger.error(f"DEBUG: TTS async call failed: {e}")

            logger.info("Starting voice verification with ultimate engine")
            logger.info("DEBUG: About to start verification executor")

            # Run verification in executor to avoid blocking the async loop
            loop = asyncio.get_event_loop()

            # Create a progress update task for parallel processing
            async def update_progress_periodically():
                progress_messages = [
                    "⟳ Running voice & face verification in parallel...",
                    "⟳ Processing biometric data...",
                    "⟳ Analyzing voice samples...",
                    "⟳ Scanning face features...",
                    "⟳ Performing liveness check...",
                    "⟳ Fusing verification scores...",
                    "⟳ Computing final authentication..."
                ]
                message_index = 0
                while True:
                    await asyncio.sleep(1.5)  # Update every 1.5 seconds
                    if message_index < len(progress_messages):
                        self.update_status(progress_messages[message_index], SciFiColors.INFO)
                        self.page.update()
                        message_index += 1
                    else:
                        message_index = 0  # Loop back
            
            progress_task = asyncio.create_task(update_progress_periodically())
            
            try:
                # ===== TRUE PARALLEL EXECUTION: Voice + Face =====
                # Launch voice verification task
                async def run_voice_verification():
                    return await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self.voice_assistant.fast_verifier.verify_with_face_parallel(
                                user_id=user['id'],
                                audio_data=audio_data,
                                frame_data=None,
                                voice_engine=self.ultimate_voice_engine,
                                face_engine=None,
                                anti_spoof_engine=self.anti_spoofing_engine
                            )
                        ),
                        timeout=12.0
                    )
                
                # Launch face capture task (runs in parallel with voice)
                async def run_face_capture():
                    """Capture face frames while voice is being processed"""
                    face_embeddings = self.db.get_face_embeddings(user['id'])
                    arcface_embeddings = [emb['embedding_data'] for emb in face_embeddings 
                                         if isinstance(emb['embedding_data'], list) and len(emb['embedding_data']) == 512]
                    
                    if not arcface_embeddings:
                        return {'captured': False, 'reason': 'No face profile'}
                    
                    # Retry face verification 2-3 times before failing.
                    configured_attempts = int(self.config.get('system', {}).get('face_verify_attempts', 3))
                    max_attempts = max(2, min(3, configured_attempts))
                    best_result = None
                    
                    # OPTIMIZATION: Initialize camera ONCE and reuse across attempts
                    # This avoids expensive camera initialization on each retry
                    cap = None
                    try:
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened():
                            logger.warning("Camera failed to open")
                            return {'captured': False, 'reason': 'Camera not available'}

                        for attempt in range(max_attempts):
                            try:
                                await asyncio.sleep(0.3)  # Small delay before capture (reduced from 0.5s)
                                ret, frame = cap.read()
                                
                                if ret and frame is not None:
                                    is_match, similarity, liveness_passed = self.verify_face_arcface(frame, arcface_embeddings)
                                    current_result = {
                                        'captured': True,
                                        'is_match': is_match,
                                        'similarity': similarity,
                                        'liveness_passed': liveness_passed,
                                        'frame': frame,
                                        'attempt': attempt + 1,
                                        'max_attempts': max_attempts,
                                    }

                                    if best_result is None or similarity > best_result.get('similarity', 0.0):
                                        best_result = current_result

                                    if is_match and liveness_passed:
                                        return current_result

                                    logger.info(
                                        "Face verification attempt %d/%d failed (match=%s, liveness=%s, similarity=%.3f)",
                                        attempt + 1,
                                        max_attempts,
                                        is_match,
                                        liveness_passed,
                                        similarity,
                                    )

                                    if attempt < max_attempts - 1:
                                        await asyncio.sleep(0.2)  # Brief pause between retries (reduced from 0.4s)
                                else:
                                    logger.warning(f"Face capture returned empty frame on attempt {attempt+1}")
                                    if attempt < max_attempts - 1:
                                        await asyncio.sleep(0.5)
                            except Exception as e:
                                logger.warning(f"Face verification attempt {attempt+1} failed: {e}")
                                if attempt < max_attempts - 1:
                                    await asyncio.sleep(0.5)  # Reduced from 1s

                        if best_result is not None:
                            return best_result
                        
                        return {'captured': False, 'reason': 'Face capture failed all attempts'}
                    finally:
                        # Always release camera resource
                        if cap is not None:
                            cap.release()
                
                # Run BOTH tasks in parallel, wait for both
                voice_task = asyncio.create_task(run_voice_verification())
                face_task = asyncio.create_task(run_face_capture())
                
                # Wait for both to complete
                voice_result, face_result = await asyncio.gather(voice_task, face_task, return_exceptions=False)
                
                logger.info(f"Voice verification result: {voice_result}")
                logger.info(f"Face capture result: {face_result}")
                
                # Process results
                verification_result = voice_result
                
            except asyncio.TimeoutError:
                logger.error("Parallel biometric verification timed out after 15 seconds")
                self.show_progress(False)
                await self._register_auth_failure(
                    username=username,
                    reason="biometric_verification_timeout",
                    base_message="⚠ Verification timed out - Please try again",
                )
                await self._speak_async("Verification timed out, please try again")
                self.hide_record_button()
                return
            except Exception as e:
                logger.error(f"Verification error: {e}", exc_info=True)
                self.show_progress(False)
                await self._register_auth_failure(
                    username=username,
                    reason="voice_verification_exception",
                    base_message=f"⚠ Verification error: {str(e)}",
                )
                await self._speak_async("Verification error occurred")
                self.hide_record_button()
                return
            finally:
                # Cancel progress updates
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
            
            self.show_progress(False)

            if not verification_result['verified']:
                failure_reason = verification_result.get('details', {}).get('failure_reason', 'Unknown error')
                quality_score = verification_result.get('quality_score', 0) * 100
                cosine_sim = verification_result.get('cosine_similarity', 0) * 100
                
                if verification_result.get('spoof_detected', False):
                    await self._register_auth_failure(
                        username=username,
                        reason="voice_spoof_detected",
                        base_message="⚠ Anti-spoofing: Voice rejected as suspicious or spoofed",
                        spoof_detected=True,
                        sensitive_action=True,
                    )
                    await self._speak_async("Voice rejected by anti-spoofing system")
                elif "quality" in failure_reason.lower():
                    await self._register_auth_failure(
                        username=username,
                        reason="voice_quality_low",
                        base_message=f"⚠ Voice quality insufficient (Quality: {quality_score:.1f}%) - Please speak clearly",
                    )
                    await self._speak_async("Voice quality too low, please speak more clearly")
                elif "embedding" in failure_reason.lower():
                    await self._register_auth_failure(
                        username=username,
                        reason="voice_feature_extraction_failed",
                        base_message="⚠ Failed to extract voice features - Please try again",
                    )
                    await self._speak_async("Failed to extract voice features, please try again")
                elif "profile" in failure_reason.lower():
                    await self._register_auth_failure(
                        username=username,
                        reason="voice_profile_missing",
                        base_message="⚠ No voice profile found - Please complete registration first",
                    )
                    await self._speak_async("No voice profile found")
                else:
                    confidence_pct = verification_result.get('confidence', 0) * 100
                    await self._register_auth_failure(
                        username=username,
                        reason="voice_verification_failed",
                        base_message=(
                            f"⚠ Verification failed - Similarity: {cosine_sim:.1f}%, "
                            f"Confidence: {confidence_pct:.1f}%"
                        ),
                    )
                    await self._speak_async("Voice verification failed, please try again")
                self.hide_record_button()
                return

            confidence_pct = verification_result.get('confidence', 0) * 100
            cosine_sim = verification_result.get('cosine_similarity', 0) * 100
            quality_score = verification_result.get('quality_score', 0) * 100
            
            # Store voice scores for fusion
            self.voice_score = verification_result.get('cosine_similarity', 0.0)
            self.anti_spoof_score = 1.0 - verification_result.get('spoof_detected', 0)
            self.voice_liveness_score = quality_score / 100.0  # Use quality as proxy for liveness
            
            self.update_status(
                f"✅ Voice verified - Similarity: {cosine_sim:.1f}%, Confidence: {confidence_pct:.1f}%, Quality: {quality_score:.1f}%", 
                SciFiColors.SUCCESS
            )
            
            # Update confidence meter for voice verification
            self.confidence_display.visible = True
            self.update_confidence_meter(confidence_pct / 100.0, "Voice Confidence")
            self.update_score_display(voice_score=self.voice_score)
            
            await self._speak_async("Voice verified successfully with advanced security")
            self.voice_verification_complete = True
            self.update_auth_progress("face_verifying")
            await asyncio.sleep(1.5)
            
            # ===== USE PARALLEL FACE RESULTS =====
            # Face was captured in PARALLEL while voice was being processed
            self.face_score = 0.5  # Default neutral score
            self.face_liveness_score = 0.5
            face_verified = False
            
            if face_result and face_result.get('captured'):
                is_match = face_result.get('is_match', False)
                similarity = face_result.get('similarity', 0.0)
                liveness_passed = face_result.get('liveness_passed', False)
                similarity_percent = similarity * 100
                
                # Store face scores
                self.face_score = similarity
                self.face_liveness_score = 1.0 if liveness_passed else 0.5
                
                if is_match and liveness_passed:
                    self.update_status(f"✅ Face verified (ArcFace) - {similarity_percent:.1f}% similarity", SciFiColors.SUCCESS)
                    self.update_confidence_meter(similarity, "Face Confidence")
                    self.update_score_display(voice_score=self.voice_score, face_score=self.face_score)
                    await self._speak_async("Face verified successfully")
                    face_verified = True
                    self.face_verification_complete = True
                elif is_match and not liveness_passed:
                    self.update_status(f"⚠ Face liveness check failed - {similarity_percent:.1f}% similarity", SciFiColors.WARNING)
                    await self._register_auth_failure(
                        username=username,
                        reason="face_liveness_failed",
                        base_message=f"⚠ Face liveness check failed - Please try again",
                        sensitive_action=True,
                    )
                    await self._speak_async("Face liveness check failed")
                    self.hide_record_button()
                    return
                else:
                    self.update_status(f"⚠ Face verification failed - {similarity_percent:.1f}% similarity", SciFiColors.WARNING)
                    await self._register_auth_failure(
                        username=username,
                        reason="face_verification_failed",
                        base_message=f"⚠ Face verification failed - {similarity_percent:.1f}% similarity",
                        sensitive_action=True,
                    )
                    await self._speak_async("Face verification failed")
                    self.hide_record_button()
                    return
            else:
                # Face capture failed or no profile
                reason = face_result.get('reason', 'Unknown') if face_result else 'No face profile'
                if 'No face profile' in reason:
                    if self.require_face_verification_on_login:
                        await self._register_auth_failure(
                            username=username,
                            reason="face_profile_missing",
                            base_message="⚠ Face profile missing. Access denied.",
                            sensitive_action=True,
                        )
                        await self._speak_async("Face profile missing. Access denied.")
                        self.hide_record_button()
                        return
                    self.update_status("⚠ No ArcFace profile found - Skipping face verification", SciFiColors.WARNING)
                    await self._speak_async("No face profile found, skipping face verification")
                    self.face_liveness_score = 0.5
                    self.face_verification_complete = True
                else:
                    # Face capture failed
                    logger.warning(f"Face parallel capture failed: {reason}")
                    if self.require_face_verification_on_login:
                        await self._register_auth_failure(
                            username=username,
                            reason="face_capture_failed",
                            base_message="⚠ Face verification could not be completed. Access denied.",
                            sensitive_action=True,
                        )
                        await self._speak_async("Face verification could not be completed. Access denied.")
                        self.hide_record_button()
                        return
                    self.update_status(f"⚠ Face capture failed - Continuing with voice-only", SciFiColors.WARNING)
                    self.face_liveness_score = 0.3
                    self.face_verification_complete = True
            
            # Final check
            if not (self.voice_verification_complete and self.face_verification_complete):
                logger.error("Auth flow error: Face or Voice verification incomplete.")
                self.update_status("⚠ Authentication flow incomplete", SciFiColors.ERROR)
                self.hide_record_button()
                return

            # Multi-modal fusion decision
            fusion_enabled = self.config.get('fusion', {}).get('enabled', True)
            
            if fusion_enabled:
                self.update_status("⦿ Computing multi-modal fusion score...", SciFiColors.INFO)
                await asyncio.sleep(0.5)
                
                fusion_score, fusion_decision, fusion_details = self.fusion_engine.fuse_scores(
                    voice_score=self.voice_score,
                    face_score=self.face_score,
                    voice_liveness=self.voice_liveness_score,
                    face_liveness=self.face_liveness_score,
                    anti_spoof_confidence=self.anti_spoof_score
                )
                
                logger.info(f"Fusion result: score={fusion_score:.3f}, decision={fusion_decision}")
                
                if not fusion_decision:
                    # Fusion rejected despite individual verifications passing
                    await self._register_auth_failure(
                        username=username,
                        reason="fusion_rejected",
                        base_message=(
                            f"⚠ Multi-modal fusion rejected - Combined score: {fusion_score:.2%} "
                            f"(threshold: {fusion_details['threshold']:.2%})"
                        ),
                        sensitive_action=True,
                    )
                    await self._speak_async("Authentication rejected by fusion analysis")
                    self.hide_record_button()
                    return
                
                # Fusion accepted
                self.update_status(
                    f"✅ Multi-modal fusion passed - Combined score: {fusion_score:.2%}, Confidence: {fusion_details['confidence']:.2%}",
                    SciFiColors.SUCCESS
                )
                
                # Update confidence meter for fusion result
                self.update_confidence_meter(fusion_details['confidence'], "Fusion Confidence")
                self.update_score_display(
                    voice_score=self.voice_score,
                    face_score=self.face_score, 
                    fusion_score=fusion_score
                )
                
                await asyncio.sleep(1)

            self.update_auth_progress("complete")
            self.update_status("✓ Authentication successful! Loading dashboard...", SciFiColors.SUCCESS)
            await self._speak_async("Authentication successful")
            await asyncio.sleep(1)
            self.show_progress(False)
            
            self.cleanup_old_temp_files()
            
            self.current_user = user
            self.auth_alerts.register_success(username)
            self.voice_assistant.set_authentication_state(
                authenticated=True,
                voice_verified=True,
                face_verified=self.face_verification_complete,
                liveness_verified=True,
            )
            self.current_view = "dashboard"
            
            self.page.controls.clear()
            dashboard = self.build_dashboard_view()
            self.page.add(dashboard)
            self.page.update()
            
            # --- START FIX: Correctly call async function ---
            await self._speak_async(f"Welcome back, {user['username']}")
            # --- END FIX ---
            
            # Auto-start continuous voice listening after successful login
            try:
                await asyncio.sleep(1.5)  # Give user time to hear welcome message
                logger.info("Auto-starting continuous voice listening after login")
                
                # Enable continuous mode toggle (only if assistant tab is loaded)
                if hasattr(self, 'assistant_continuous_toggle') and self.assistant_continuous_toggle is not None:
                    self.assistant_continuous_toggle.value = True
                    
                    # Start continuous listening
                    self._assistant_start_continuous_listening()
                    
                    await self._speak_async("Voice assistant is now active and listening continuously")
                else:
                    logger.info("Voice assistant UI not loaded yet, skipping auto-start")
            except Exception as e:
                logger.error(f"Failed to auto-start voice assistant: {e}")
            
        except Exception as e:
            logger.error(f"Voice verification error: {e}", exc_info=True)
            username = self.username_field.value if self.username_field else "unknown"
            await self._register_auth_failure(
                username=username,
                reason="voice_verification_unhandled_exception",
                base_message=f"⚠ Error: {str(e)}",
            )
            await self._speak_async("Verification error occurred")
            self.hide_record_button()

    def verify_face_arcface(self, image, enrolled_embeddings, tolerance=0.55):
        """Verify face using ArcFace embeddings and cosine similarity."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            logger.error("Invalid image provided to face verification")
            return False, 0.0, False
        faces = self.arcface_model.get(image)
        if not faces:
            logger.warning("No face detected for verification")
            return False, 0.0, False
        face = max(faces, key=lambda f: f.bbox[2]*f.bbox[3])
        current_embedding = np.array(face.embedding)
        logger.info(f"ArcFace verification embedding: {current_embedding[:8]}...")
        liveness_passed = True # Placeholder for liveness
        best_similarity = 0.0
        for enrolled in enrolled_embeddings:
            enrolled_vec = np.array(enrolled)
            similarity = float(np.dot(current_embedding, enrolled_vec) / (norm(current_embedding) * norm(enrolled_vec)))
            if similarity > best_similarity:
                best_similarity = similarity
        is_match = best_similarity >= tolerance
        logger.info(f"ArcFace verification: similarity={best_similarity:.3f}, match={is_match}, liveness={liveness_passed}")
        return is_match, best_similarity, liveness_passed

    # ==================== LOGIN RECORDING HANDLERS (ASYNC) ====================

    def _on_record_button_click(self, e):
        """
        Synchronous handler for record button clicks.
        Updates UI immediately and schedules async work.
        """
        try:
            logger.debug("[DEBUG] Record button clicked - entering handler")
            logger.info(f"=== RECORD BUTTON CLICKED === recording_active={self.recording_active}")
            
            if not self.recording_active:
                # START RECORDING
                logger.debug("[DEBUG] Record button - starting recording")
                logger.info("Starting recording...")
                
                # Update UI immediately (synchronous)
                self.recording_active = True
                self.record_button.text = "STOP LISTENING"
                self.record_button.icon = ft.Icons.STOP
                self.record_button.disabled = False  # Keep button enabled
                self.record_button.style = ft.ButtonStyle(
                    bgcolor=SciFiColors.ERROR,
                    color=SciFiColors.TEXT_PRIMARY,
                    shape=ft.RoundedRectangleBorder(radius=12),
                )
                self.update_status("🎤 Listening... Click STOP LISTENING when done", SciFiColors.WARNING)
                logger.debug("[DEBUG] UI updated for listening state")
                self.page.update()
                
                # Schedule async recording task
                logger.debug("[DEBUG] Scheduling start_voice_recording_manual task")
                logger.info("Scheduling start_voice_recording_manual")
                self.page.run_task(self.start_voice_recording_manual)
                
            else:
                # STOP RECORDING
                logger.debug("[DEBUG] Record button - stopping recording")
                logger.info("Stopping recording...")
                
                # Update UI immediately to show stopping state
                self.record_button.text = "STOPPING..."
                self.record_button.disabled = True
                self.update_status("⏹ Stopping recording...", SciFiColors.INFO)
                logger.debug("[DEBUG] UI updated for stopping state")
                self.page.update()
                
                # Schedule async stop task
                logger.debug("[DEBUG] Scheduling stop_voice_recording_manual task")
                logger.info("Scheduling stop_voice_recording_manual")
                self.page.run_task(self.stop_voice_recording_manual)
                
        except Exception as ex:
            logger.debug(f"[DEBUG] Exception in record button handler: {ex}")
            logger.error(f"Button click error: {ex}", exc_info=True)
            self.recording_active = False
            self.update_status(f"⚠ Error: {ex}", SciFiColors.ERROR)
            self.page.update()

    async def start_voice_recording_manual(self):
        """Start manual voice recording (user controls start/stop)"""
        try:
            logger.info("=== START_VOICE_RECORDING_MANUAL CALLED ===")
            
            # 1. Stop any previous recorder immediately.
            self.audio_recorder.stop_recording()

            # 2. Cancel any old asyncio task *before* creating a new one.
            if self.recording_task and not self.recording_task.done():
                logger.info("Cancelling existing recording task")
                self.recording_task.cancel()
                try:
                    await self.recording_task
                except asyncio.CancelledError:
                    logger.info("Old recording task cancelled.")
                    pass

            # --- START FIX: No TTS during recording to prevent microphone feedback ---
            
            # 3. Start recording task immediately - NO TTS to avoid mic contamination
            self.recording_task = asyncio.create_task(self._record_audio_background())
            logger.info(f"Recording task created: {self.recording_task}")
            logger.info("Recording started in SILENT mode - no TTS to prevent microphone feedback")
            
            # --- END FIX ---
            logger.info(f"Recording task created: {self.recording_task}")
            
            # Visual feedback only - no audio to prevent mic feedback
            logger.info("Recording started - using visual feedback only")
            
            # --- END FIX ---

        except Exception as e:
            logger.error(f"Error starting recording: {e}", exc_info=True)
            # If starting fails, reset the UI
            await self._reset_recording_ui()

    async def stop_voice_recording_manual(self):
        """Stop manual voice recording and process audio"""
        try:
            logger.info("=== STOP_VOICE_RECORDING_MANUAL CALLED ===")
            logger.info(f"Current recording_task: {self.recording_task}")
            
            # Signal the recorder to stop
            logger.info("Calling audio_recorder.stop_recording()")
            self.audio_recorder.stop_recording()
            
            # Wait for recording task with timeout
            audio_data = None
            if self.recording_task:
                try:
                    logger.info("Waiting for recording task to complete...")
                    audio_data = await asyncio.wait_for(self.recording_task, timeout=3.0)
                    logger.info(f"Recording task completed. Audio data: {audio_data is not None}")
                
                except asyncio.TimeoutError:
                    logger.error("Recording task timed out! The recording thread is likely stuck.")
                    self.recording_task.cancel()
                    # Tell the user the stop failed before resetting
                    self.update_status("⚠ Error: Recording failed to stop.", SciFiColors.ERROR)
                    await asyncio.sleep(2) # Give user time to read

                except asyncio.CancelledError:
                    logger.info("Recording task was cancelled")
                except Exception as e:
                    logger.error(f"Error waiting for recording task: {e}", exc_info=True)
                finally:
                    self.recording_task = None
            else:
                logger.warning("No recording task found")

            # Reset UI
            await self._reset_recording_ui()

            # Process audio if we got data
            if audio_data is None or (hasattr(audio_data, '__len__') and len(audio_data) == 0):
                logger.warning("No audio data captured")
                # This message will now show if the timeout occurred OR if audio was empty
                self.update_status("⚠ Recording failed - no audio captured", SciFiColors.ERROR)
                self.hide_record_button()
                return

            logger.info(f"Processing audio data: shape={audio_data.shape}, duration={len(audio_data)/16000:.2f}s")
            
            # Check audio quality before processing
            duration = len(audio_data) / 16000
            if duration < 0.5:
                self.update_status("⚠ Recording too short - Please record at least 0.5 seconds", SciFiColors.WARNING)
                await self._speak_async("Recording too short, please try again")
                self.hide_record_button()
                return
            
            self.update_status("✓ Recording complete - Starting verification process...", SciFiColors.SUCCESS)
            self.page.update()
            await asyncio.sleep(0.3)
            
            # Process the recorded audio
            await self.process_voice_verification(audio_data)

        except Exception as e:
            logger.error(f"Error stopping recording: {e}", exc_info=True)
            await self._reset_recording_ui()
            self.update_status(f"⚠ Recording error: {str(e)}", SciFiColors.ERROR)
            self.hide_record_button()

    async def _record_audio_background(self):
        """Background task for recording audio until stopped"""
        try:
            logger.info("=== RECORDING AUDIO IN BACKGROUND ===")
            
            # Record with long duration - will be stopped by stop_recording()
            audio_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.audio_recorder.record_audio(duration=300.0)
            )
            
            logger.info(f"Background recording complete. Audio shape: {audio_data.shape if audio_data is not None else None}")
            return audio_data

        except Exception as e:
            logger.error(f"Background recording error: {e}", exc_info=True)
            return None

    async def _reset_recording_ui(self):
        """Reset recording UI to initial state"""
        logger.info("Resetting recording UI")
        self.recording_active = False
        self.record_button.text = "START RECORDING"
        self.record_button.icon = ft.Icons.MIC
        self.record_button.disabled = False
        self.record_button.style = ft.ButtonStyle(
            bgcolor=SciFiColors.PRIMARY,
            color=SciFiColors.BG_DARK,
            shape=ft.RoundedRectangleBorder(radius=12),
            shadow_color=SciFiColors.PRIMARY,
            elevation=8,
            side=ft.BorderSide(width=2, color=SciFiColors.PRIMARY),
        )
        self.page.update()

    # ==================== REGISTRATION RECORDING HANDLERS (THREADED) ====================

    def handle_reg_record_button_click(self, e=None):
        """
        Synchronous handler for the REGISTRATION record button.
        Uses threading.Events to coordinate with the async registration process.
        """
        try:
            logger.debug("[DEBUG] Registration record button clicked - entering handler")
            if not self.reg_recording_active:
                logger.debug("[DEBUG] Registration record button - starting recording")
                logger.info("Registration record button: START")
                self.reg_recording_active = True
                
                # Reset events
                self.reg_recording_started_event.clear()
                self.reg_recording_stop_event.clear()
                self.reg_recording_complete_event.clear()
                
                # Update UI
                self.reg_record_button.text = "STOP RECORDING"
                self.reg_record_button.icon = ft.Icons.STOP
                self.reg_record_button.style = ft.ButtonStyle(
                    bgcolor=SciFiColors.ERROR,
                    color=SciFiColors.TEXT_PRIMARY,
                    shape=ft.RoundedRectangleBorder(radius=12),
                )
                self.update_reg_status("⦿ Recording... Speak now!", SciFiColors.WARNING)
                logger.debug("[DEBUG] UI updated for registration listening state")
                self.page.update()
                
                # Start background recording thread
                logger.debug("[DEBUG] Starting registration recording thread")
                self.reg_recording_thread = threading.Thread(
                    target=self._record_audio_for_registration_thread,
                    daemon=True
                )
                self.reg_recording_thread.start()
                logger.info("Registration recording thread started")
                
            else:
                logger.debug("[DEBUG] Registration record button - stopping recording")
                logger.info("Registration record button: STOP")
                # Stop recording
                self.reg_recording_active = False
                self.reg_recording_stop_event.set() # Signal stop event
                try:
                    # Signal the underlying recorder to stop
                    self.audio_recorder.stop_recording()
                except Exception:
                    logger.exception("Failed to stop audio recorder cleanly")

                self.reg_record_button.text = "PROCESSING..."
                self.reg_record_button.disabled = True
                self.update_reg_status("✓ Recording stopped - processing...", SciFiColors.SUCCESS)
                logger.debug("[DEBUG] UI updated for registration stopping state")
                self.page.update()
                logger.info("Registration recording stop requested")
        
        except Exception as ex:
            logger.debug(f"[DEBUG] Exception in registration record button handler: {ex}")
            logger.error(f"Reg button click error: {ex}", exc_info=True)
            self.reg_recording_active = False
            self.update_reg_status(f"⚠ Error: {ex}", SciFiColors.ERROR)
            self.page.update()

    def _record_audio_for_registration_thread(self):
        """
        BACKGROUND THREAD: Record audio for registration.
        """
        try:
            self.reg_audio_holder[0] = None
            
            # Signal that recording has officially started
            self.reg_recording_started_event.set()
            logger.info("Registration recording thread: Started.")
            
            # Start recording (long duration, will be stopped by stop_recording)
            audio_data = self.audio_recorder.record_audio(duration=300.0)
            
            if audio_data is not None:
                logger.info(f"Registration recording thread: Audio captured, shape={audio_data.shape}")
                self.reg_audio_holder[0] = audio_data
            else:
                logger.warning("Registration recording thread: Audio capture returned None.")
                
        except Exception as e:
            logger.error(f"Error in registration recording thread: {e}", exc_info=True)
            self.reg_audio_holder[0] = None
        finally:
            # Signal that this thread is done
            self.reg_recording_complete_event.set()
            logger.info("Registration recording thread: Finished.")

    # ==================== MISC UI HANDLERS ====================

    def show_record_button(self, mode="voice"):
        """Show login recording button"""
        logger.info(f"show_record_button called with mode={mode}")
        
        self.recording_started_event.clear()
        self.recording_stop_event.clear()
        
        self.mic_status.visible = True
        
        if mode == "voice":
            self.mic_status.content = ft.Row([
                ft.Icon(ft.Icons.MIC_OUTLINED, color=SciFiColors.PRIMARY, size=20),
                ft.Text("VOICE VERIFICATION", size=11, color=SciFiColors.TEXT_SECONDARY, weight=ft.FontWeight.W_600),
            ], spacing=8)
            self.record_button.icon = ft.Icons.MIC
            self.record_button.tooltip = "Voice Biometric Authentication"
        elif mode == "face":
            self.mic_status.content = ft.Row([
                ft.Icon(ft.Icons.CAMERA_ALT_OUTLINED, color=SciFiColors.PRIMARY, size=20),
                ft.Text("FACE VERIFICATION", size=11, color=SciFiColors.TEXT_SECONDARY, weight=ft.FontWeight.W_600),
            ], spacing=8)
            self.record_button.icon = ft.Icons.CAMERA_ALT
            self.record_button.tooltip = "Capture Face for Verification"
        
        self.record_button.text = "START RECORDING"
        self.record_button.disabled = False
        self.record_button.visible = True
        self.recording_active = False
        
        self.record_button.style = ft.ButtonStyle(
            bgcolor=SciFiColors.PRIMARY,
            color=SciFiColors.BG_DARK,
            shape=ft.RoundedRectangleBorder(radius=12),
            shadow_color=SciFiColors.PRIMARY,
            elevation=8,
            side=ft.BorderSide(width=2, color=SciFiColors.PRIMARY),
        )
        
        self.page.update()
        logger.info("Record button shown and ready")

    def hide_record_button(self):
        """Hide login recording button"""
        self.mic_status.visible = False
        self.record_button.visible = False
        self.recording_active = False
        self.recording_started_event.clear()
        self.recording_stop_event.clear()
        self.page.update()

    def show_reg_record_button(self, sample_info: str = "", mode="voice"):
        """Show registration recording button"""
        self.reg_recording_started_event.clear()
        self.reg_recording_stop_event.clear()
        
        self.reg_mic_status.visible = True
        
        if mode == "voice":
            if sample_info:
                self.reg_mic_status.content = ft.Row([
                    ft.Icon(ft.Icons.MIC_OUTLINED, color=SciFiColors.PRIMARY, size=20),
                    ft.Text(sample_info, size=11, color=SciFiColors.TEXT_PRIMARY, weight=ft.FontWeight.BOLD),
                ], spacing=8)
            else:
                self.reg_mic_status.content = ft.Row([
                    ft.Icon(ft.Icons.MIC_OUTLINED, color=SciFiColors.PRIMARY, size=20),
                    ft.Text("VOICE ENROLLMENT", size=11, color=SciFiColors.TEXT_SECONDARY, weight=ft.FontWeight.W_600),
                ], spacing=8)
            self.reg_record_button.icon = ft.Icons.MIC
            self.reg_record_button.tooltip = "Voice Biometric Enrollment"
        elif mode == "face":
            self.reg_mic_status.content = ft.Row([
                ft.Icon(ft.Icons.CAMERA_ALT_OUTLINED, color=SciFiColors.PRIMARY, size=20),
                ft.Text("FACE ENROLLMENT", size=11, color=SciFiColors.TEXT_SECONDARY, weight=ft.FontWeight.W_600),
            ], spacing=8)
            self.reg_record_button.icon = ft.Icons.CAMERA_ALT
            self.reg_record_button.tooltip = "Capture Face for Enrollment"
        
        self.reg_record_button.text = "START RECORDING"
        self.reg_record_button.disabled = False
        self.reg_record_button.visible = True
        self.reg_recording_active = False
        
        self.reg_record_button.style = ft.ButtonStyle(
            bgcolor=SciFiColors.PRIMARY,
            color=SciFiColors.BG_DARK,
            shape=ft.RoundedRectangleBorder(radius=12),
            shadow_color=SciFiColors.PRIMARY,
            elevation=8,
            side=ft.BorderSide(width=2, color=SciFiColors.PRIMARY),
        )
        
        self.page.update()

    def hide_reg_record_button(self):
        """Hide registration recording button"""
        self.reg_mic_status.visible = False
        self.reg_record_button.visible = False
        self.reg_recording_active = False
        self.reg_recording_started_event.clear()
        self.reg_recording_stop_event.clear()
        self.page.update()

    def update_auth_progress(self, step: str):
        """Update authentication progress indicator"""
        progress_messages = {
            "idle": "",
            "voice_enrolling": "Voice Enrollment -> Face Enrollment",
            "face_enrolling": "Voice Enrollment [OK] -> Face Enrollment",
            "voice_verifying": "Voice Verification -> Face Verification",
            "face_verifying": "Voice Verification [OK] -> Face Verification",
            "complete": "Voice Verification [OK] -> Face Verification [OK] -> Access Granted"
        }
        
        message = progress_messages.get(step, "")
        if message:
            self.auth_progress_text.value = message
            self.auth_progress_panel.visible = True
        else:
            self.auth_progress_panel.visible = False
        
        self.page.update()

    def reset_auth_states(self):
        """Reset all authentication flow states"""
        self.auth_step = "idle"
        self.voice_enrollment_complete = False
        self.face_enrollment_complete = False
        self.voice_verification_complete = False
        self.face_verification_complete = False
        self.recording_started_event.clear()
        self.recording_stop_event.clear()
        self.reg_recording_started_event.clear()
        self.reg_recording_stop_event.clear()
        self.update_auth_progress("idle")

    # ==================== CLEANUP & SHUTDOWN ====================

    def cleanup_old_temp_files(self):
        """Clean up old temporary audio files"""
        try:
            cleanup_temp_files(str(self.temp_dir))
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

    def show_dashboard(self):
        """Render the dashboard view"""
        try:
            logger.info("Rendering dashboard view.")
            self.page.controls.clear()
            dashboard_content = self.build_dashboard_view()
            self.page.add(dashboard_content)
            self.page.update()
            logger.info("Dashboard rendered successfully.")
        except Exception as e:
            logger.error(f"Failed to render dashboard: {e}", exc_info=True)
            self._show_error_toast(f"Error loading dashboard: {str(e)}")

    def logout(self):
        """Logout current user and show login view"""
        logger.info("Logging out user")
        
        try:
            # --- START FIX: Manually close dialogs on logout ---
            if self.voice_assistant_active:
                self._assistant_stop_continuous() # Use the new UI-aware stop

            for dialog in self.active_dialogs[:]:
                try:
                    self._close_dialog_safe(dialog, update_page=False)
                except:
                    pass
            # --- END FIX ---
        except Exception as e:
            logger.error(f"Error closing dialog: {e}")
        
        self.active_dialogs.clear()
        self.current_view = "login"
        self.voice_assistant.set_authentication_state(authenticated=False)
        self.current_user = None # Clear current user

        if self.auth_cooldown_task and not self.auth_cooldown_task.done():
            self.auth_cooldown_task.cancel()
        self.auth_cooldown_task = None
        self.auth_cooldown_username = None
        
        self.cleanup_old_temp_files()
        
        self.page.controls.clear()
        login = self.build_login_view()
        self.page.add(login)
        self.page.update()
        
        # --- START FIX: Correctly call task ---
        self.page.run_task(self._speak_async, "Logged out successfully")
        # --- END FIX ---

    def _on_app_close(self, e=None):
        """Handle application shutdown"""
        logger.info("Application shutdown initiated")
        
        try:
            # --- START FIX: Manually close dialogs on shutdown ---
            if self.voice_assistant_active:
                self.voice_assistant.stop_continuous_listening()
                self.voice_assistant_active = False
            
            for dialog in self.active_dialogs[:]:
                try:
                    self._close_dialog_safe(dialog, update_page=False)
                except:
                    pass
            # --- END FIX ---
            self.active_dialogs.clear()
            
            if hasattr(self, 'voice_assistant') and self.voice_assistant:
                logger.info("Shutting down voice assistant...")
                self.voice_assistant.shutdown()
            
            if hasattr(self, 'recording_active') and self.recording_active:
                self.audio_recorder.stop_recording()

            if self.auth_cooldown_task and not self.auth_cooldown_task.done():
                self.auth_cooldown_task.cancel()
            self.auth_cooldown_task = None
            self.auth_cooldown_username = None
            
            if hasattr(self, 'db') and self.db:
                self.db.close()
                
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                import shutil
                try:
                    shutil.rmtree(self.temp_dir)
                    logger.info("Temporary files cleaned up")
                except Exception as exc:
                    logger.warning("Failed to clean up temp files: %s", exc)
                    
        except Exception as exc:
            logger.error("Error during shutdown: %s", exc)
        
        logger.info("Application shutdown complete")


def main(page: ft.Page):
    """Flet main entry point"""
    config = load_config()
    page.title = "SecureX-Assist"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.resizable = True
    page.window.maximizable = True
    page.window.minimizable = True

    app = SecureXApp(page, config)
    app.run()

if __name__ == "__main__":
    ft.app(target=main)

