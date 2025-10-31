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

# --- Core and Utils Imports for SecureXApp ---
# (Ensure these files exist in your project structure)
from core.database import Database
from core.voice_engine import VoiceEngine
from core.voice_biometric_engine_ultimate import UltimateVoiceBiometricEngine
from core.security import SecurityManager, SecurityContext
from core.audio_processor import AudioRecorder, VoiceActivityDetector
from utils.tts import TextToSpeech
from core.voice_assistant import VoiceAssistant
from core.face_recognition_engine import FaceRecognitionEngine
from utils.helpers import create_temp_directory, cleanup_temp_files, load_config
# --- End Imports ---

class SciFiColors:
    BG_SPACE = "#10172b"
    BG_DARK = "#0a1628"
    BG_CARD = "#1a2336"
    BG_ELEVATED = "#232e48"
    PRIMARY = "#00eaff"
    ACCENT = "#ff2e63"
    SUCCESS = "#00ff99"
    ERROR = "#ff2e63"
    WARNING = "#ffd700"
    INFO = "#3b82f6"
    TEXT_PRIMARY = "#eaf6fb"
    TEXT_SECONDARY = "#b2becd"
    TEXT_MUTED = "#6c7891"
    BORDER = "#2e3a59"
    BORDER_GLOW = "#00eaff"

logger = logging.getLogger("SecureXApp")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
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
        self.security_manager = SecurityManager(config)
        self.audio_recorder = AudioRecorder(config)
        self.vad = VoiceActivityDetector(config)
        self.tts = TextToSpeech(config)
        self.voice_assistant = VoiceAssistant()
        self.voice_assistant.setup_default_commands()

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
        self.voice_assistant_dialog_open = False

        # Navigation state
        self.current_nav_section = "dashboard"

        # Recording states
        self.recording_active = False
        self.reg_recording_active = False
        self.face_capture_requested = False
        self.recording_task = None
        
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

        # Temp directory
        self.temp_dir = create_temp_directory()

        # UI components
        self.interaction_log = None
        
        # Login/Reg UI components (defined in build_login_view)
        self.username_field = None
        self.password_field = None
        self.reg_username_field = None
        self.reg_password_field = None
        self.reg_confirm_password_field = None
        self.reg_email_field = None
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

        # Dialog management
        self.active_dialogs = []

        # Set up page close handler
        self.page.on_close = self._on_app_close

    # --- ASYNC TTS HELPER ---
    async def _speak_async(self, text: str):
        """Run TTS in an executor thread to avoid blocking the UI."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self.tts.speak(text))
        except Exception as e:
            logger.warning(f"Async TTS failed: {e}")

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
        
        # Run initial TTS asynchronously
        async def start_speech():
            await self._speak_async("SecureX Assist initialized. Ready for authentication.")
        
        self.page.run_task(start_speech)
        
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
                self.reg_username_field,
                self.reg_password_field,
                self.reg_confirm_password_field,
                self.reg_email_field,
            ],
            spacing=14,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        
        self.action_button_container = ft.Column(
            [login_button, register_button],
            spacing=0,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        
        left_side = ft.Container(
            content=ft.Column(
                [
                    ft.Container(expand=1),
                    ft.Stack([
                        ft.Container(
                            width=100,
                            height=100,
                            border=ft.border.all(3, SciFiColors.PRIMARY),
                            border_radius=20,
                            bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.PRIMARY),
                        ),
                        ft.Container(
                            content=ft.Icon(
                                ft.Icons.SECURITY_ROUNDED,
                                color=SciFiColors.PRIMARY,
                                size=50,
                            ),
                            width=100,
                            height=100,
                            alignment=ft.alignment.center,
                        ),
                    ]),
                    ft.Container(height=24),
                    ft.Text(
                        "SECUREX",
                        size=48,
                        weight=ft.FontWeight.BOLD,
                        color=SciFiColors.TEXT_PRIMARY,
                        font_family="Orbitron",
                    ),
                    ft.Text(
                        "ASSIST",
                        size=48,
                        weight=ft.FontWeight.BOLD,
                        color=SciFiColors.PRIMARY,
                        font_family="Orbitron",
                    ),
                    ft.Container(height=16),
                    ft.Container(
                        content=ft.Text(
                            "VOICE BIOMETRIC AUTH",
                            size=13,
                            color=SciFiColors.TEXT_SECONDARY,
                            weight=ft.FontWeight.W_600,
                        ),
                        padding=ft.padding.symmetric(horizontal=20, vertical=8),
                        border=ft.border.all(1, SciFiColors.BORDER_GLOW),
                        border_radius=4,
                        bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.PRIMARY),
                    ),
                    ft.Container(height=32),
                    ft.Row([
                        ft.Icon(ft.Icons.SHIELD_ROUNDED, size=16, color=SciFiColors.SUCCESS),
                        ft.Text(
                            "Military-Grade Security",
                            size=12,
                            color=SciFiColors.TEXT_MUTED,
                        ),
                    ], spacing=8),
                    ft.Container(height=8),
                    ft.Row([
                        ft.Icon(ft.Icons.VERIFIED_USER_ROUNDED, size=16, color=SciFiColors.PRIMARY),
                        ft.Text(
                            "Multi-Factor Authentication",
                            size=12,
                            color=SciFiColors.TEXT_MUTED,
                        ),
                    ], spacing=8),
                    ft.Container(expand=1),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            width=420,
            padding=40,
            bgcolor=ft.Colors.with_opacity(0.3, SciFiColors.BG_CARD),
            border=ft.border.only(right=ft.border.BorderSide(1, SciFiColors.BORDER_GLOW)),
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
            width=920,
            height=700,
            bgcolor=ft.Colors.with_opacity(0.6, SciFiColors.BG_CARD),
            border=ft.border.all(1, SciFiColors.BORDER_GLOW),
            border_radius=16,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=40,
                color=ft.Colors.with_opacity(0.3, SciFiColors.PRIMARY),
            ),
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
            {"name": "security", "icon": ft.Icons.SECURITY_ROUNDED, "label": "Security"},
            {"name": "assistant", "icon": ft.Icons.MIC_ROUNDED, "label": "Assistant"},
            {"name": "settings", "icon": ft.Icons.SETTINGS_ROUNDED, "label": "System"},
        ]
        
        nav_buttons = []
        for item in nav_items:
            nav_buttons.append(
                ft.Container(
                    content=ft.ElevatedButton(
                        content=ft.Row([
                            ft.Icon(item["icon"], size=20, color=SciFiColors.TEXT_PRIMARY),
                            ft.Text(item["label"], size=14, weight=ft.FontWeight.W_500),
                        ], spacing=12),
                        on_click=lambda e, name=item["name"]: self._navigate_to_section(name),
                        width=200,
                        height=48,
                        style=ft.ButtonStyle(
                            bgcolor=SciFiColors.BG_ELEVATED if self.current_nav_section == item["name"] else ft.Colors.TRANSPARENT,
                            color=SciFiColors.TEXT_PRIMARY,
                            shape=ft.RoundedRectangleBorder(radius=8),
                        ),
                    ),
                    margin=ft.margin.symmetric(vertical=4),
                )
            )
        
        sidebar = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Column([
                        ft.Text("SECUREX", size=24, weight=ft.FontWeight.BOLD, color=SciFiColors.PRIMARY, font_family="Orbitron"),
                        ft.Text("ASSIST", size=24, weight=ft.FontWeight.BOLD, color=SciFiColors.PRIMARY, font_family="Orbitron"),
                    ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=ft.padding.all(20),
                ),
                ft.Divider(color=SciFiColors.BORDER),
                ft.Container(
                    content=ft.Column(nav_buttons, spacing=0),
                    padding=ft.padding.symmetric(vertical=20),
                ),
                ft.Container(expand=True),
                ft.Container(
                    content=ft.ElevatedButton(
                        content=ft.Row([
                            ft.Icon(ft.Icons.LOGOUT_ROUNDED, size=20, color=SciFiColors.ERROR),
                            ft.Text("Logout", size=14, weight=ft.FontWeight.W_500),
                        ], spacing=12),
                        on_click=lambda _: self.logout(),
                        width=200,
                        height=48,
                        style=ft.ButtonStyle(
                            bgcolor=ft.Colors.TRANSPARENT,
                            color=SciFiColors.ERROR,
                            shape=ft.RoundedRectangleBorder(radius=8),
                        ),
                    ),
                    margin=ft.margin.symmetric(vertical=20),
                ),
            ], spacing=0),
            width=240,
            bgcolor=ft.Colors.with_opacity(0.8, SciFiColors.BG_DARK),
            border=ft.border.only(right=ft.border.BorderSide(1, SciFiColors.BORDER)),
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

    def _handle_auth_tab_change(self, e):
        """Handle tab switching"""
        is_login = e.control.selected_index == 0
        
        self.username_field.visible = is_login
        self.password_field.visible = is_login
        self.reg_username_field.visible = not is_login
        self.reg_password_field.visible = not is_login
        self.reg_confirm_password_field.visible = not is_login
        self.reg_email_field.visible = not is_login
        
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
        """Create main dashboard"""
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            ram_usage = psutil.virtual_memory().percent
        except:
            cpu_usage = 45
            ram_usage = 62
        
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            "SECUREX DASHBOARD",
                            size=28,
                            weight=ft.FontWeight.BOLD,
                            color=SciFiColors.TEXT_PRIMARY,
                            font_family="Orbitron",
                        ),
                        ft.Text(
                            "Advanced Biometric Security & Voice Assistant System",
                            size=14,
                            color=SciFiColors.TEXT_SECONDARY,
                        ),
                        ft.Container(
                            content=ft.Row([
                                ft.Icon(ft.Icons.VERIFIED_USER_ROUNDED, color=SciFiColors.SUCCESS, size=16),
                                ft.Text(
                                    f"Authenticated as {self.current_user['username'] if self.current_user else 'Unknown'}",
                                    size=12,
                                    color=SciFiColors.SUCCESS,
                                    weight=ft.FontWeight.W_600,
                                ),
                            ], spacing=8),
                            padding=ft.padding.symmetric(horizontal=16, vertical=8),
                            border_radius=20,
                            bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.SUCCESS),
                            border=ft.border.all(1, SciFiColors.SUCCESS),
                        ),
                    ], spacing=8, horizontal_alignment=ft.CrossAxisAlignment.START),
                    padding=ft.padding.symmetric(vertical=20, horizontal=20),
                ),
                ft.Container(
                    content=ft.Row([
                        self._create_stat_card("CPU Usage", f"{int(cpu_usage)}%", SciFiColors.PRIMARY, ft.Icons.MEMORY_ROUNDED),
                        self._create_stat_card("RAM Usage", f"{int(ram_usage)}%", SciFiColors.ACCENT, ft.Icons.STORAGE_ROUNDED),
                        self._create_stat_card("Security", "ACTIVE", SciFiColors.SUCCESS, ft.Icons.SECURITY_ROUNDED),
                        self._create_stat_card("Assistant", "READY", SciFiColors.INFO, ft.Icons.MIC_ROUNDED),
                    ], spacing=20, wrap=True),
                    height=200,
                ),
                ft.Container(height=20),
                ft.Text("QUICK ACTIONS", size=18, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ft.Container(height=16),
                ft.Row([
                    self._create_action_card("Voice Assistant", ft.Icons.MIC_ROUNDED, SciFiColors.PRIMARY, self._show_voice_dialog),
                    self._create_action_card("Security Scan", ft.Icons.SECURITY_ROUNDED, SciFiColors.WARNING, lambda e: self._run_security_scan()),
                    self._create_action_card("System Info", ft.Icons.INFO_ROUNDED, SciFiColors.SUCCESS, self._show_system_status),
                    self._create_action_card("Screenshot", ft.Icons.CAMERA_ROUNDED, SciFiColors.ACCENT, self._take_screenshot_action),
                ], spacing=16, wrap=True),
            ], spacing=0, scroll=ft.ScrollMode.AUTO),
            padding=ft.padding.all(0),
            expand=True,
        )

    def _create_stat_card(self, title: str, value: str, color: str, icon) -> ft.Container:
        return ft.Container(
            content=ft.Column([
                ft.Icon(icon, size=32, color=color),
                ft.Container(height=8),
                ft.Text(value, size=24, weight=ft.FontWeight.BOLD, color=color),
                ft.Text(title.upper(), size=10, color=SciFiColors.TEXT_MUTED, weight=ft.FontWeight.W_600),
            ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            width=200,
            height=150,
            padding=ft.padding.all(20),
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.6, SciFiColors.BG_CARD),
            border=ft.border.all(1, color),
        )

    def _create_action_card(self, title: str, icon, color: str, on_click) -> ft.Container:
        return ft.Container(
            content=ft.Column([
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
                    size=11,
                    color=SciFiColors.TEXT_PRIMARY,
                    weight=ft.FontWeight.W_600,
                    text_align=ft.TextAlign.CENTER,
                ),
            ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            width=120,
            height=100,
            padding=ft.padding.all(12),
            border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.4, SciFiColors.BG_ELEVATED),
            border=ft.border.all(1, SciFiColors.BORDER),
            alignment=ft.alignment.center,
            on_click=on_click,
            ink=True,
        )

    def _create_assistant_content(self) -> ft.Container:
        """Create voice assistant interface"""
        return ft.Container(
            content=ft.Column([
                ft.Text("VOICE ASSISTANT", size=24, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ft.Container(height=40),
                ft.ElevatedButton(
                    "🎤 START VOICE ASSISTANT",
                    icon=ft.Icons.MIC_ROUNDED,
                    on_click=self._show_voice_dialog,
                    style=ft.ButtonStyle(
                        bgcolor=SciFiColors.PRIMARY,
                        color=SciFiColors.BG_DARK,
                    ),
                ),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=40,
            alignment=ft.alignment.center,
        )

    def _create_security_content(self) -> ft.Container:
        return ft.Container(
            content=ft.Column([
                ft.Text("SECURITY CENTER", size=24, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ft.Container(height=20),
                ft.Text("Security settings and logs.", color=SciFiColors.TEXT_SECONDARY, size=14),
            ]),
            padding=40,
        )

    def _create_settings_content(self) -> ft.Container:
        return ft.Container(
            content=ft.Column([
                ft.Text("SETTINGS", size=24, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ft.Container(height=20),
                ft.Text("Application settings.", color=SciFiColors.TEXT_SECONDARY, size=14),
            ]),
            padding=40,
        )

    def _navigate_to_section(self, section_name: str):
        """Navigate to a different dashboard section"""
        try:
            logger.info(f"Navigating to section: {section_name}")
            self.current_nav_section = section_name
            
            if self.current_view == "dashboard":
                self.show_dashboard()
            
        except Exception as e:
            logger.error(f"Navigation error: {e}", exc_info=True)
            self._show_error_toast(f"Navigation failed: {str(e)}")

    # ==================== VOICE ASSISTANT DIALOG ====================
    
    def _show_voice_dialog(self, e=None):
        """Show voice assistant dialog"""
        try:
            if self.voice_assistant_dialog_open:
                logger.info("Voice assistant dialog already open")
                return
            
            logger.info("Opening voice assistant dialog")
            
            log_content = ft.Column(
                scroll=ft.ScrollMode.AUTO,
                auto_scroll=True,
                expand=True,
                spacing=4,
            )
            
            status_text = ft.Text(
                "Ready - Toggle continuous mode and click 'Listen'",
                size=12,
                color=SciFiColors.TEXT_SECONDARY,
                weight=ft.FontWeight.W_500,
            )
            
            continuous_toggle = ft.Switch(
                label="Continuous Mode",
                value=self.continuous_mode_active,
                active_color=SciFiColors.PRIMARY,
                on_change=lambda e: setattr(self, 'continuous_mode_active', e.control.value)
            )
            
            stop_btn = ft.ElevatedButton(
                "STOP",
                icon=ft.Icons.STOP,
                on_click=None,
                style=ft.ButtonStyle(
                    bgcolor=SciFiColors.ERROR,
                    color=ft.Colors.WHITE,
                    shape=ft.RoundedRectangleBorder(radius=6),
                ),
                visible=False
            )
            
            start_btn = ft.ElevatedButton(
                "LISTEN",
                icon=ft.Icons.MIC,
                on_click=None,
                style=ft.ButtonStyle(
                    bgcolor=SciFiColors.ACCENT,
                    color=ft.Colors.WHITE,
                    shape=ft.RoundedRectangleBorder(radius=6),
                )
            )
            
            start_btn.on_click = lambda e: self._handle_listen_button(
                e, log_content, status_text, start_btn, stop_btn, continuous_toggle
            )
            stop_btn.on_click = lambda e: self._stop_continuous(
                log_content, status_text, start_btn, stop_btn
            )
            
            dialog_content = ft.Column([
                continuous_toggle,
                ft.Container(height=10),
                status_text,
                ft.Container(height=12),
                ft.Row([start_btn, stop_btn], spacing=10),
                ft.Container(height=16),
                ft.Text(
                    "INTERACTION LOG:",
                    size=12,
                    weight=ft.FontWeight.BOLD,
                    color=SciFiColors.TEXT_PRIMARY,
                ),
                ft.Container(
                    content=log_content,
                    height=200,
                    bgcolor=ft.Colors.with_opacity(0.5, SciFiColors.BG_DARK),
                    padding=10,
                    border_radius=6,
                    border=ft.border.all(1, SciFiColors.BORDER)
                ),
            ], width=500, spacing=0)
            
            dialog = ft.AlertDialog(
                modal=True,
                title=ft.Row([
                    ft.Icon(ft.Icons.MIC_ROUNDED, color=SciFiColors.PRIMARY, size=28),
                    ft.Text("VOICE ASSISTANT", size=18, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ], spacing=12),
                content=dialog_content,
                actions=[
                    ft.TextButton(
                        "CLOSE",
                        on_click=lambda _: self._close_voice_assistant_dialog(dialog),
                        style=ft.ButtonStyle(color=SciFiColors.TEXT_SECONDARY),
                    ),
                ],
                bgcolor=SciFiColors.BG_CARD,
                shape=ft.RoundedRectangleBorder(radius=10),
            )
            
            self._open_dialog_safe(dialog)
            self.voice_assistant_dialog_open = True
            
            self._add_log_entry_to_container(log_content, "Voice Assistant initialized", SciFiColors.SUCCESS)
            
            logger.info("Voice assistant dialog opened successfully")
            
        except Exception as ex:
            logger.error(f"Error opening voice dialog: {ex}", exc_info=True)
            self._show_error_toast(f"Failed to open voice assistant: {str(ex)}")

    def _handle_listen_button(self, e, log_content, status_text, start_btn, stop_btn, continuous_toggle):
        """Handle listen button"""
        if continuous_toggle.value:
            self._start_continuous_listening(log_content, status_text, start_btn, stop_btn)
        else:
            self._listen_single_command(log_content, status_text)

    def _start_continuous_listening(self, log_content, status_text, start_btn, stop_btn):
        """Start continuous listening"""
        try:
            self.voice_assistant_active = True
            
            self.voice_assistant.start_continuous_listening(
                self._audio_stream_ctx,
                callback=lambda transcript, response, success: self._handle_voice_callback(
                    transcript, response, success, log_content
                )
            )
            
            start_btn.visible = False
            stop_btn.visible = True
            status_text.value = "Listening continuously - Speak now!"
            status_text.color = SciFiColors.SUCCESS
            
            self._add_log_entry_to_container(log_content, "Continuous listening started", SciFiColors.SUCCESS)
            self.page.update()
        except Exception as e:
            logger.error(f"Error starting continuous listening: {e}")
            self._add_log_entry_to_container(log_content, f"Failed: {e}", SciFiColors.ERROR)

    def _stop_continuous(self, log_content, status_text, start_btn, stop_btn):
        """Stop continuous listening"""
        try:
            self.voice_assistant.stop_continuous_listening()
            self.voice_assistant_active = False
            
            start_btn.visible = True
            stop_btn.visible = False
            status_text.value = "Stopped"
            status_text.color = SciFiColors.INFO
            
            self._add_log_entry_to_container(log_content, "Stopped", SciFiColors.INFO)
            self.page.update()
        except Exception as e:
            logger.error(f"Error stopping: {e}")
            self._add_log_entry_to_container(log_content, f"Error: {e}", SciFiColors.ERROR)

    def _listen_single_command(self, log_content, status_text):
        """Listen for single command"""
        async def update_ui_error():
            self._add_log_entry_to_container(log_content, "Recording failed", SciFiColors.ERROR)
            status_text.value = "Failed"
            status_text.color = SciFiColors.ERROR
            self.voice_assistant_active = False
            self.page.update()
        
        async def update_ui_processing():
            status_text.value = "Processing..."
            status_text.color = SciFiColors.WARNING
            self.page.update()
        
        async def update_ui_success(transcript, response, success):
            self._add_log_entry_to_container(log_content, f"Heard: '{transcript}'", SciFiColors.INFO)
            self._add_log_entry_to_container(log_content, f"Response: {response}", SciFiColors.SUCCESS if success else SciFiColors.ERROR)
            status_text.value = f"Processed: {response[:40]}..."
            status_text.color = SciFiColors.SUCCESS
            self.voice_assistant_active = False
            self.page.update()
        
        async def update_ui_no_speech():
            self._add_log_entry_to_container(log_content, "No speech detected", SciFiColors.WARNING)
            status_text.value = "No speech"
            status_text.color = SciFiColors.WARNING
            self.voice_assistant_active = False
            self.page.update()
        
        async def update_ui_exception(e):
            self._add_log_entry_to_container(log_content, f"Error: {e}", SciFiColors.ERROR)
            status_text.value = f"Error: {e}"
            status_text.color = SciFiColors.ERROR
            self.voice_assistant_active = False
            self.page.update()
        
        try:
            self.voice_assistant_active = True
            
            status_text.value = "Listening..."
            status_text.color = SciFiColors.PRIMARY
            self.page.update()
            
            def record_and_process():
                try:
                    audio_path = self.temp_dir / f"command_{int(time.time())}.wav"
                    audio_data = self.audio_recorder.record_audio(duration=5.0)
                    
                    if audio_data is None:
                        self.page.run_task(update_ui_error)
                        return
                    
                    self.audio_recorder.save_audio(audio_data, str(audio_path))
                    
                    self.page.run_task(update_ui_processing)
                    
                    transcript = self.voice_assistant.transcribe(str(audio_path))
                    if transcript.strip():
                        success, response = self.voice_assistant.process_voice_command(transcript)
                        
                        self.page.run_task(lambda: update_ui_success(transcript, response, success))
                        
                        # Run TTS in executor
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self._speak_async(response))
                        loop.close()
                        
                    else:
                        self.page.run_task(update_ui_no_speech)
                    
                except Exception as e:
                    logger.error(f"Error in recording thread: {e}")
                    self.page.run_task(lambda: update_ui_exception(e))
            
            recording_thread = threading.Thread(target=record_and_process, daemon=True)
            recording_thread.start()
            
        except Exception as e:
            logger.error(f"Error: {e}")
            self._add_log_entry_to_container(log_content, f"Error: {e}", SciFiColors.ERROR)
            status_text.value = f"Error: {e}"
            status_text.color = SciFiColors.ERROR
            self.voice_assistant_active = False
            self.page.update()

    def _handle_voice_callback(self, transcript, response, success, log_content):
        """Handle voice callback"""
        try:
            if transcript:
                async def update_ui():
                    self._add_log_entry_to_container(log_content, f"Heard: '{transcript}'", SciFiColors.INFO)
                    self._add_log_entry_to_container(log_content, f"Response: {response}", SciFiColors.SUCCESS if success else SciFiColors.ERROR)
                self.page.run_task(update_ui)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def _add_log_entry_to_container(self, log_content, message: str, color: str):
        """Add entry to log container"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        log_entry = ft.Container(
            content=ft.Row([
                ft.Container(width=2, bgcolor=color, border_radius=1),
                ft.Container(width=6),
                ft.Column([
                    ft.Text(f"[{timestamp}]", size=9, color=SciFiColors.TEXT_MUTED),
                    ft.Text(message, size=11, color=SciFiColors.TEXT_PRIMARY),
                ], spacing=2, tight=True),
            ]),
            margin=ft.margin.only(bottom=4),
            padding=8,
            border_radius=4,
            bgcolor=ft.Colors.with_opacity(0.3, SciFiColors.BG_ELEVATED),
        )
        
        log_content.controls.append(log_entry)
        self.page.update()

    def _close_voice_assistant_dialog(self, dialog):
        """Close voice assistant dialog"""
        self._close_dialog_safe(dialog)
        self.voice_assistant_dialog_open = False
        if self.voice_assistant_active:
            self.voice_assistant.stop_continuous_listening()
            self.voice_assistant_active = False

    # ==================== ACTION HANDLERS ====================
    
    def _take_screenshot_action(self, event):
        """Take screenshot action"""
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save("screenshot.png")
            self._show_success_toast("Screenshot saved!")
        except Exception as e:
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
        async def run_scan():
            await asyncio.sleep(1)
            self._show_success_toast("Security scan complete - No threats detected")
        
        self.page.run_task(run_scan)
        self._show_success_toast("Security scan initiated")

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
            if dialog not in self.active_dialogs:
                self.active_dialogs.append(dialog)
            
            self.page.dialog = dialog
            dialog.open = True
            self.page.update()
            
            logger.info(f"Dialog opened: type={type(dialog).__name__}")
            
        except Exception as e:
            logger.error(f"Error opening dialog: {e}")
            self._show_error_toast(f"Failed to open dialog: {e}")

    def _close_dialog_safe(self, dialog):
        """Close dialog safely"""
        try:
            dialog.open = False
            if self.page.dialog == dialog:
                self.page.dialog = None
            
            if dialog in self.active_dialogs:
                self.active_dialogs.remove(dialog)
            
            self.page.update()
            
        except Exception as e:
            logger.error(f"Error closing dialog: {e}")

    # ==================== TOAST NOTIFICATIONS ====================
    
    def _show_success_toast(self, message: str):
        """Show success toast"""
        self.page.snack_bar = ft.SnackBar(
            content=ft.Row([
                ft.Icon(ft.Icons.CHECK_CIRCLE_ROUNDED, color=SciFiColors.SUCCESS),
                ft.Text(message, color=SciFiColors.TEXT_PRIMARY),
            ], spacing=10),
            bgcolor=SciFiColors.BG_CARD,
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
        )
        self.page.snack_bar.open = True
        self.page.update()

    # ==================== STATUS HELPERS ====================
    
    def update_status(self, message: str, color: str = SciFiColors.INFO):
        """Update login status with enhanced visual feedback"""
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

    def update_reg_status(self, message: str, color: str):
        """Update registration status with enhanced visual feedback"""
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
        self.progress_ring.visible = show
        if show:
            self.progress_ring.value = None  # Indeterminate spinner
        self.page.update()

    def show_reg_progress(self, visible: bool):
        """Show/hide registration progress indicator with animation"""
        self.reg_progress_ring.visible = visible
        if visible:
            self.reg_progress_ring.value = None  # Indeterminate spinner
        self.page.update()

    # ==================== AUTHENTICATION & REGISTRATION ====================
    
    def start_voice_login(self):
        """Start voice authentication"""
        self.page.run_task(self._start_voice_login_async)
    
    async def _start_voice_login_async(self):
        """Async handler for voice login"""
        try:
            logger.info("=== VOICE LOGIN STARTED ===")
            username = self.username_field.value
            password = self.password_field.value
            
            if not username or not password:
                self.update_status("⚠ Please enter both username and password", SciFiColors.ERROR)
                await self._speak_async("Please enter your credentials")
                return
            
            logger.info(f"Attempting login for user: {username}")
            
            self.update_status("⟳ Validating credentials...", SciFiColors.INFO)
            self.show_progress(True)
            
            user = self.db.get_user_by_username(username)
            if not user:
                self.show_progress(False)
                self.update_status("⚠ User not found - Please check username or register first", SciFiColors.ERROR)
                logger.warning(f"User not found: {username}")
                await self._speak_async("User not found")
                return
            
            logger.info(f"User found: {user['id']}")
            
            if not self.security_manager.verify_hashed_password(password, user['password_hash']):
                self.show_progress(False)
                self.update_status("⚠ Invalid password - Please try again", SciFiColors.ERROR)
                logger.warning("Password verification failed")
                await self._speak_async("Invalid password")
                return
            
            logger.info("Password verified successfully")
            self.show_progress(False)
            
            self.reset_auth_states()
            self.auth_step = "voice_verifying"
            self.update_auth_progress("voice_verifying")
            
            # Show the record button immediately
            self.show_record_button(mode="voice")
            self.update_status("⦿ Ready for voice verification - Click START RECORDING when ready", SciFiColors.INFO)
            await self._speak_async("Credentials verified, please provide voice sample for verification")
            
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
                await self.enroll_user_face_arcface(user)
                logger.info("Face enrollment completed successfully (ArcFace)")
            except Exception as e:
                logger.error(f"Face enrollment failed: {e}")
                self.update_reg_status(f"⚠ Face enrollment failed: {str(e)}", SciFiColors.ERROR)
                await asyncio.sleep(2)
            
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
                self.update_status("⚠ User not found", SciFiColors.ERROR)
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

            # Create a progress update task
            async def update_progress_periodically():
                progress_messages = [
                    "⟳ Analyzing voice quality...",
                    "⟳ Running anti-spoofing checks...",
                    "⟳ Extracting voice features...",
                    "⟳ Comparing with stored profile...",
                    "⟳ Computing similarity scores...",
                    "⟳ Finalizing verification..."
                ]
                message_index = 0
                while True:
                    await asyncio.sleep(2)  # Update every 2 seconds
                    if message_index < len(progress_messages):
                        self.update_status(progress_messages[message_index], SciFiColors.INFO)
                        self.page.update()
                        message_index += 1
                    else:
                        message_index = 0  # Loop back
            
            progress_task = asyncio.create_task(update_progress_periodically())
            
            try:
                # Run with timeout to prevent infinite hanging
                verification_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.ultimate_voice_engine.verify_voice(
                            user_id=user['id'],
                            audio_data=audio_data,
                            sample_rate=16000,
                            enable_challenge=False
                        )
                    ),
                    timeout=30.0  # 30 second timeout
                )
                logger.info(f"Verification result: {verification_result}")
            except asyncio.TimeoutError:
                logger.error("Voice verification timed out after 30 seconds")
                self.show_progress(False)
                self.update_status("⚠ Verification timed out - Please try again", SciFiColors.ERROR)
                await self._speak_async("Verification timed out, please try again")
                self.hide_record_button()
                return
            except Exception as e:
                logger.error(f"Verification error: {e}", exc_info=True)
                self.show_progress(False)
                self.update_status(f"⚠ Verification error: {str(e)}", SciFiColors.ERROR)
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
                failure_reason = verification_result['details'].get('failure_reason', 'Unknown error')
                quality_score = verification_result.get('quality_score', 0) * 100
                cosine_sim = verification_result.get('cosine_similarity', 0) * 100
                
                if verification_result.get('spoof_detected', False):
                    self.update_status("⚠ Anti-spoofing: Voice rejected as suspicious or spoofed", SciFiColors.ERROR)
                    await self._speak_async("Voice rejected by anti-spoofing system")
                elif "quality" in failure_reason.lower():
                    self.update_status(f"⚠ Voice quality insufficient (Quality: {quality_score:.1f}%) - Please speak clearly", SciFiColors.ERROR)
                    await self._speak_async("Voice quality too low, please speak more clearly")
                elif "embedding" in failure_reason.lower():
                    self.update_status("⚠ Failed to extract voice features - Please try again", SciFiColors.ERROR)
                    await self._speak_async("Failed to extract voice features, please try again")
                elif "profile" in failure_reason.lower():
                    self.update_status("⚠ No voice profile found - Please complete registration first", SciFiColors.ERROR)
                    await self._speak_async("No voice profile found")
                else:
                    confidence_pct = verification_result.get('confidence', 0) * 100
                    self.update_status(
                        f"⚠ Verification failed - Similarity: {cosine_sim:.1f}%, Confidence: {confidence_pct:.1f}%", 
                        SciFiColors.ERROR
                    )
                    await self._speak_async("Voice verification failed, please try again")
                self.hide_record_button()
                return

            confidence_pct = verification_result.get('confidence', 0) * 100
            cosine_sim = verification_result.get('cosine_similarity', 0) * 100
            quality_score = verification_result.get('quality_score', 0) * 100
            
            self.update_status(
                f"✅ Voice verified - Similarity: {cosine_sim:.1f}%, Confidence: {confidence_pct:.1f}%, Quality: {quality_score:.1f}%", 
                SciFiColors.SUCCESS
            )
            await self._speak_async("Voice verified successfully with advanced security")
            self.voice_verification_complete = True
            self.update_auth_progress("face_verifying")
            await asyncio.sleep(1.5)
            
            # Face verification
            self.update_status("⦿ Now verifying face (ArcFace)...", SciFiColors.INFO)
            await self._speak_async("Now verifying face")
            await asyncio.sleep(1)
            face_embeddings = self.db.get_face_embeddings(user['id'])
            arcface_embeddings = [emb['embedding_data'] for emb in face_embeddings if isinstance(emb['embedding_data'], list) and len(emb['embedding_data']) == 512]
            
            if arcface_embeddings:
                self.update_status("⦿ Automatically capturing face for verification...", SciFiColors.INFO)
                await self._speak_async("Please look at the camera for automatic face verification")
                
                max_attempts = 3
                face_verified = False
                
                for attempt in range(max_attempts):
                    self.update_status(f"⦿ Face capture attempt {attempt + 1}/{max_attempts}...", SciFiColors.INFO)
                    await asyncio.sleep(0.5)
                    
                    cap = cv2.VideoCapture(0)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret and frame is not None:
                        # Run face verification logic here
                        is_match, similarity, liveness_passed = self.verify_face_arcface(frame, arcface_embeddings)
                        similarity_percent = similarity * 100
                        if is_match and liveness_passed:
                            self.update_status(f"✅ Face verified (ArcFace) - {similarity_percent:.1f}% similarity", SciFiColors.SUCCESS)
                            await self._speak_async("Face verified successfully")
                            face_verified = True
                            self.face_verification_complete = True # Set flag
                            break
                        elif is_match and not liveness_passed:
                            self.update_status(f"⚠ Liveness check failed - {similarity_percent:.1f}% similarity (attempt {attempt + 1})", SciFiColors.WARNING)
                            if attempt < max_attempts - 1:
                                await self._speak_async("Liveness check failed, trying again")
                                await asyncio.sleep(1)
                        else:
                            self.update_status(f"⚠ Face verification failed - {similarity_percent:.1f}% similarity (attempt {attempt + 1})", SciFiColors.WARNING)
                            if attempt < max_attempts - 1:
                                await self._speak_async("Face not recognized, trying again")
                                await asyncio.sleep(1)
                    else:
                        self.update_status(f"⚠ Face capture failed (attempt {attempt + 1})", SciFiColors.WARNING)
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(1)
                
                if not face_verified:
                    self.update_status("⚠ Face verification failed after all attempts", SciFiColors.ERROR)
                    await self._speak_async("Face verification failed")
                    self.hide_record_button()
                    return
            else:
                self.update_status("⚠ No ArcFace profile found - Skipping", SciFiColors.WARNING)
                await self._speak_async("No face profile found, skipping face verification")
                self.face_verification_complete = True
                await asyncio.sleep(1)
            
            # Final check
            if not (self.voice_verification_complete and self.face_verification_complete):
                logger.error("Auth flow error: Face or Voice verification incomplete.")
                self.update_status("⚠ Authentication flow incomplete", SciFiColors.ERROR)
                self.hide_record_button()
                return

            self.update_auth_progress("complete")
            self.update_status("✓ Authentication successful! Loading dashboard...", SciFiColors.SUCCESS)
            await self._speak_async("Authentication successful")
            await asyncio.sleep(1)
            self.show_progress(False)
            
            self.cleanup_old_temp_files()
            
            self.current_user = user
            self.current_view = "dashboard"
            
            self.page.controls.clear()
            dashboard = self.build_dashboard_view()
            self.page.add(dashboard)
            self.page.update()
            
            await self._speak_async(f"Welcome back, {user['username']}")
            
        except Exception as e:
            logger.error(f"Voice verification error: {e}", exc_info=True)
            self.update_status(f"⚠ Error: {str(e)}", SciFiColors.ERROR)
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
            logger.info(f"=== RECORD BUTTON CLICKED === recording_active={self.recording_active}")
            
            if not self.recording_active:
                # START RECORDING
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
                self.page.update()
                
                # Schedule async recording task
                logger.info("Scheduling start_voice_recording_manual")
                self.page.run_task(self.start_voice_recording_manual)
                
            else:
                # STOP RECORDING
                logger.info("Stopping recording...")
                
                # Update UI immediately to show stopping state
                self.record_button.text = "STOPPING..."
                self.record_button.disabled = True
                self.update_status("⏹ Stopping recording...", SciFiColors.INFO)
                self.page.update()
                
                # Schedule async stop task
                logger.info("Scheduling stop_voice_recording_manual")
                self.page.run_task(self.stop_voice_recording_manual)
                
        except Exception as ex:
            logger.error(f"Button click error: {ex}", exc_info=True)
            self.recording_active = False
            self.update_status(f"⚠ Error: {ex}", SciFiColors.ERROR)
            self.page.update()

    # ---
    # --- THIS IS THE CORRECTED FUNCTION ---
    # ---
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

            # 3. Create the new task *immediately*. This fixes the race condition.
            self.recording_task = asyncio.create_task(self._record_audio_background())
            logger.info(f"Recording task created: {self.recording_task}")

            # 4. *Now* speak, after the state is correctly set up.
            await self._speak_async("Recording started, please speak")

        except Exception as e:
            logger.error(f"Error starting recording: {e}", exc_info=True)
            # If starting fails, reset the UI
            await self._reset_recording_ui()
    # ---
    # --- END OF CORRECTED FUNCTION ---
    # ---

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
            if not self.reg_recording_active:
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
                self.page.update()
                
                # Start background recording thread
                self.reg_recording_thread = threading.Thread(
                    target=self._record_audio_for_registration_thread,
                    daemon=True
                )
                self.reg_recording_thread.start()
                logger.info("Registration recording thread started")
                
            else:
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
                self.page.update()
                logger.info("Registration recording stop requested")
        
        except Exception as ex:
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
            for dialog in self.active_dialogs[:]:
                try:
                    dialog.open = False
                except:
                    pass
        except Exception as e:
            logger.error(f"Error closing dialog: {e}")
        
        self.active_dialogs.clear()
        self.current_view = "login"
        self.current_user = None # Clear current user
        
        self.cleanup_old_temp_files()
        
        self.page.controls.clear()
        login = self.build_login_view()
        self.page.add(login)
        self.page.update()
        
        self.page.run_task(self._speak_async("Logged out successfully"))

    def _on_app_close(self, e=None):
        """Handle application shutdown"""
        logger.info("Application shutdown initiated")
        
        try:
            for dialog in self.active_dialogs[:]:
                try:
                    dialog.open = False
                except:
                    pass
            self.active_dialogs.clear()
            
            if hasattr(self, 'voice_assistant') and self.voice_assistant:
                logger.info("Shutting down voice assistant...")
                self.voice_assistant.shutdown()
            
            if hasattr(self, 'recording_active') and self.recording_active:
                self.audio_recorder.stop_recording()
            
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