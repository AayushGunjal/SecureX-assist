import flet as ft
import asyncio
import logging
from pathlib import Path
from typing import Optional
import time
import tempfile
import threading
import random as rnd
import numpy as np
import datetime
from types import SimpleNamespace
from importlib import import_module
import os
import cv2
import numpy as np
from numpy.linalg import norm
from numpy.linalg import norm
import insightface
import pyautogui
import psutil

# SciFiColors theme class
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

# Setup logger
logger = logging.getLogger("SecureXApp")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# Import all required classes from core and utils modules
from core.database import Database
from core.voice_engine import VoiceEngine
from core.voice_engine import VoiceQualityAnalyzer
from core.voice_biometric_engine_ultimate import UltimateVoiceBiometricEngine
from core.security import SecurityManager, SecurityContext
from core.audio_processor import AudioRecorder, VoiceActivityDetector
from utils.tts import TextToSpeech
from core.voice_assistant import VoiceAssistant
from core.face_recognition_engine import FaceRecognitionEngine
from utils.helpers import create_temp_directory, cleanup_temp_files


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
        
        # Authentication flow states
        self.auth_step = "idle"  # idle, voice_enrolling, voice_verifying, face_enrolling, face_verifying, complete
        self.voice_enrollment_complete = False
        self.face_enrollment_complete = False
        self.voice_verification_complete = False
        self.face_verification_complete = False

        # Temp directory
        self.temp_dir = create_temp_directory()

        # UI components
        self.interaction_log = None
        
        # Dialog management
        self.active_dialogs = []
        
        # Set up page close handler
        page.on_close = self._on_app_close

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
        liveness_passed = True  # TODO: integrate anti-spoof or blink detection
        best_similarity = 0.0
        for enrolled in enrolled_embeddings:
            enrolled_vec = np.array(enrolled)
            similarity = float(np.dot(current_embedding, enrolled_vec) / (norm(current_embedding) * norm(enrolled_vec)))
            if similarity > best_similarity:
                best_similarity = similarity
        is_match = best_similarity >= tolerance
        logger.info(f"ArcFace verification: similarity={best_similarity:.3f}, match={is_match}, liveness={liveness_passed}")
        return is_match, best_similarity, liveness_passed

    async def enroll_user_face_arcface(self, user: dict):
        """Enroll user face using ArcFace and store embedding in DB"""
        # Automatic face capture - no button required
        self.update_reg_status("⦿ Automatically capturing face for enrollment (ArcFace)...", SciFiColors.INFO)
        self.tts.speak("Please look at the camera for automatic face enrollment")
        
        # Try multiple captures to get a good face
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
                self.tts.speak("Processing face image")
                embedding = self.enroll_face_arcface(frame)
                if embedding is not None:
                    self.db.deactivate_old_face_embeddings(user['id'])
                    self.db.store_face_embedding(user['id'], embedding, embedding_type="arcface", quality_score=1.0)
                    self.update_reg_status("✅ Face enrolled successfully (ArcFace)", SciFiColors.SUCCESS)
                    self.tts.speak("Face enrolled successfully")
                    face_enrolled = True
                    break
                else:
                    self.update_reg_status(f"⚠ Face enrollment failed - no face detected (attempt {attempt + 1})", SciFiColors.WARNING)
                    if attempt < max_attempts - 1:
                        self.tts.speak("No face detected, trying again")
                        await asyncio.sleep(1)
            else:
                self.update_reg_status(f"⚠ Face capture failed (attempt {attempt + 1})", SciFiColors.WARNING)
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1)
        
        if not face_enrolled:
            self.update_reg_status("⚠ Face enrollment failed after all attempts", SciFiColors.ERROR)
            self.tts.speak("Face enrollment failed, please try again")
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

    def run(self):
        """Run the application"""
        # Setup page first
        self.setup_page()
        
        # Initialize database
        self.db.connect()
        self.db.initialize_schema()
        
        # Load voice models
        self.voice_engine.load_models()
        
        # Show login view
        self.page.add(self.build_login_view())
        
        try:
            self.tts.speak("SecureX Assist initialized. Ready for authentication.")
        except:
            pass

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
        
        # Load fonts
        self.page.fonts = {
            "Orbitron": "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap",
            "Rajdhani": "https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap",
            "Poppins": "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap",
        }
        
        logger.info("Page configuration complete")

    # ==================== LOGIN VIEW ====================
    
    def build_login_view(self) -> ft.Container:
        """Build modernized split-layout login view"""
        
        # Define form fields
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
        
        # Registration fields
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
        
        # Progress indicators
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
        
        # Recording controls
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
            height=48,
            width=380,
            visible=False,
            on_click=lambda _: self.toggle_recording(),
            style=ft.ButtonStyle(
                bgcolor=SciFiColors.ERROR,
                color=SciFiColors.TEXT_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=8),
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
            height=48,
            width=380,
            visible=False,
            on_click=lambda _: self.toggle_reg_recording(),
            style=ft.ButtonStyle(
                bgcolor=SciFiColors.ERROR,
                color=SciFiColors.TEXT_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
        )
        
        # Status panels
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
        
        # Progress indicators for authentication flow
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
        
        # Action buttons
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
        
        # Tabs
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
            on_change=lambda e: self._handle_auth_tab_change(e),
        )
        
        # Form container
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
        
        # Action button container
        self.action_button_container = ft.Column(
            [login_button, register_button],
            spacing=0,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        
        # LEFT SIDE: Branding
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
        
        # RIGHT SIDE: Forms
        right_side = ft.Container(
            content=ft.Column(
                [
                    ft.Container(height=40),
                    self.auth_tabs,
                    ft.Container(height=24),
                    self.form_container,
                    ft.Container(height=16),
                    self.mic_status,
                    self.record_button,
                    self.reg_mic_status,
                    self.reg_record_button,
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
        
        # Main card with split layout
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
        
        # Background
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

    def _handle_auth_tab_change(self, e):
        """Handle tab switching"""
        is_login = e.control.selected_index == 0
        
        # Toggle field visibility
        self.username_field.visible = is_login
        self.password_field.visible = is_login
        self.reg_username_field.visible = not is_login
        self.reg_password_field.visible = not is_login
        self.reg_confirm_password_field.visible = not is_login
        self.reg_email_field.visible = not is_login
        
        # Toggle buttons
        self.action_button_container.controls[0].content.visible = is_login
        self.action_button_container.controls[1].content.visible = not is_login
        
        # Hide status
        self.status_panel.visible = False
        self.reg_status_panel.visible = False
        self.hide_record_button()
        self.hide_reg_record_button()
        
        # Reset auth states when switching tabs
        self.reset_auth_states()
        
        self.page.update()

    # ==================== DASHBOARD VIEW ====================
    
    def build_dashboard_view(self) -> ft.Container:
        """Build futuristic cyber-security themed dashboard with sidebar navigation"""
        logger.info("=== BUILD_DASHBOARD_VIEW CALLED ===")
        try:
            username = self.current_user.username if self.current_user else "User"
            logger.info(f"Building dashboard for user: {username}")

            # Create components one by one with error handling
            try:
                sidebar = self._create_sidebar()
                logger.info("[OK] Sidebar created")
            except Exception as e:
                logger.error(f"[ERROR] Sidebar failed: {e}")
                raise

            try:
                header = self._create_header(username)
                logger.info("[OK] Header created")
            except Exception as e:
                logger.error(f"[ERROR] Header failed: {e}")
                raise

            try:
                main_content = self._create_main_content()
                logger.info("[OK] Main content created")
            except Exception as e:
                logger.error(f"[ERROR] Main content failed: {e}")
                raise

            try:
                voice_dock = self._create_voice_dock()
                logger.info("[OK] Voice dock created")
            except Exception as e:
                logger.error(f"[ERROR] Voice dock failed: {e}")
                raise

            # Build the final dashboard structure
            dashboard = ft.Container(
                content=ft.Column([
                    header,
                    ft.Container(
                        content=ft.Row([
                            sidebar,
                            ft.Container(
                                content=ft.Column([
                                    main_content,
                                    voice_dock,
                                ], spacing=0),
                                expand=True,
                            ),
                        ], spacing=0),
                        expand=True,
                    ),
                ], spacing=0),
                expand=True,
                bgcolor=SciFiColors.BG_SPACE,
            )
            
            logger.info("=== DASHBOARD CONTAINER BUILT SUCCESSFULLY ===")
            return dashboard
            
        except Exception as e:
            logger.error(f"=== BUILD_DASHBOARD_VIEW FAILED ===")
            logger.error(f"Error: {e}", exc_info=True)
            
            # Return error container instead of None
            return ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.ERROR_OUTLINE, size=64, color=SciFiColors.ERROR),
                    ft.Container(height=20),
                    ft.Text("Failed to Build Dashboard", size=20, color=SciFiColors.ERROR, weight=ft.FontWeight.BOLD),
                    ft.Container(height=10),
                    ft.Text(str(e), size=12, color=SciFiColors.TEXT_SECONDARY, text_align=ft.TextAlign.CENTER),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                alignment=ft.alignment.center,
                expand=True,
                bgcolor=SciFiColors.BG_SPACE,
                padding=40,
            )

    def _create_sidebar(self) -> ft.Container:
        """Create left sidebar navigation with icons and text"""
        nav_items = [
            {"name": "dashboard", "icon": ft.Icons.DASHBOARD_ROUNDED, "label": "Dashboard"},
            {"name": "assistant", "icon": ft.Icons.MIC_ROUNDED, "label": "Assistant"},
            {"name": "security", "icon": ft.Icons.SECURITY_ROUNDED, "label": "Security"},
            {"name": "logs", "icon": ft.Icons.LIST_ROUNDED, "label": "Logs"},
            {"name": "settings", "icon": ft.Icons.SETTINGS_ROUNDED, "label": "Settings"},
        ]
        nav_buttons = []
        for item in nav_items:
            is_active = item["name"] == self.current_nav_section
            nav_buttons.append(
                ft.Container(
                    content=ft.Column([
                        ft.IconButton(
                            icon=item["icon"],
                            icon_size=24,
                            icon_color=SciFiColors.PRIMARY if is_active else SciFiColors.TEXT_MUTED,
                            tooltip=item["label"],
                            on_click=lambda e, name=item["name"]: self._navigate_to_section(name),
                            style=ft.ButtonStyle(
                                bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.PRIMARY) if is_active else ft.Colors.TRANSPARENT,
                                shape=ft.RoundedRectangleBorder(radius=8),
                            ),
                        ),
                        ft.Text(
                            item["label"],
                            size=10,
                            color=SciFiColors.PRIMARY if is_active else SciFiColors.TEXT_MUTED,
                            weight=ft.FontWeight.W_600,
                            text_align=ft.TextAlign.CENTER,
                        ),
                    ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=ft.padding.symmetric(vertical=12, horizontal=8),
                )
            )
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Column([
                        ft.Container(
                            width=40,
                            height=40,
                            border_radius=8,
                            bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.PRIMARY),
                            alignment=ft.alignment.center,
                            content=ft.Icon(ft.Icons.SECURITY_ROUNDED, color=SciFiColors.PRIMARY, size=24),
                        ),
                        ft.Text(
                            "SECUREX",
                            size=10,
                            color=SciFiColors.TEXT_PRIMARY,
                            weight=ft.FontWeight.BOLD,
                            font_family="Orbitron",
                        ),
                    ], spacing=8, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=ft.padding.symmetric(vertical=20),
                ),
                ft.Column(nav_buttons, spacing=8),
                ft.Container(expand=True),
                ft.Container(
                    content=ft.Text(
                        "v2.0.0",
                        size=8,
                        color=SciFiColors.TEXT_MUTED,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    padding=ft.padding.symmetric(vertical=10),
                ),
            ], spacing=0),
            width=80,
            bgcolor=ft.Colors.with_opacity(0.8, SciFiColors.BG_DARK),
            border=ft.border.only(right=ft.border.BorderSide(1, SciFiColors.BORDER)),
        )

    def _create_header(self, username: str) -> ft.Container:
        """Create slim header bar with logo, name, status indicators, session time, and profile dropdown"""
        return ft.Container(
            content=ft.Row([
                # Logo and title
                ft.Row([
                    ft.Icon(ft.Icons.SECURITY_ROUNDED, color=SciFiColors.PRIMARY, size=24),
                    ft.Container(width=12),
                    ft.Column([
                        ft.Text(
                            "SecureX-Assist",
                            size=18,
                            weight=ft.FontWeight.BOLD,
                            color=SciFiColors.TEXT_PRIMARY,
                            font_family="Orbitron",
                        ),
                        ft.Text(
                            "Voice Authenticated AI Assistant",
                            size=10,
                            color=SciFiColors.TEXT_SECONDARY,
                        ),
                    ], spacing=2, tight=True),
                ], spacing=0),
                ft.Container(expand=True),
                # Status indicators
                ft.Row([
                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.LOCK_ROUNDED, size=14, color=SciFiColors.SUCCESS),
                            ft.Text("SECURE", size=10, color=SciFiColors.SUCCESS, weight=ft.FontWeight.W_600),
                        ], spacing=4),
                        padding=ft.padding.symmetric(horizontal=8, vertical=4),
                        border_radius=12,
                        bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.SUCCESS),
                    ),
                    ft.Container(width=12),
                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.MIC_ROUNDED, size=14, color=SciFiColors.PRIMARY),
                            ft.Text("READY", size=10, color=SciFiColors.PRIMARY, weight=ft.FontWeight.W_600),
                        ], spacing=4),
                        padding=ft.padding.symmetric(horizontal=8, vertical=4),
                        border_radius=12,
                        bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.PRIMARY),
                    ),
                    ft.Container(width=12),
                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.COMPUTER_ROUNDED, size=14, color=SciFiColors.SUCCESS),
                            ft.Text("ONLINE", size=10, color=SciFiColors.SUCCESS, weight=ft.FontWeight.W_600),
                        ], spacing=4),
                        padding=ft.padding.symmetric(horizontal=8, vertical=4),
                        border_radius=12,
                        bgcolor=ft.Colors.with_opacity(0.1, SciFiColors.SUCCESS),
                    ),
                ], spacing=0),
                ft.Container(width=20),
                # Session info and profile
                ft.Row([
                    ft.Column([
                        ft.Text(
                            f"{datetime.datetime.now().strftime('%H:%M:%S')}",
                            size=12,
                            color=SciFiColors.TEXT_SECONDARY,
                            weight=ft.FontWeight.W_600,
                        ),
                        ft.Text(
                            f"Session: {username}",
                            size=10,
                            color=SciFiColors.TEXT_MUTED,
                        ),
                    ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.END),
                    ft.Container(width=8),
                    ft.Container(
                        content=ft.Text(
                            username[0].upper(),
                            size=14,
                            color=SciFiColors.TEXT_PRIMARY,
                            weight=ft.FontWeight.BOLD,
                        ),
                        width=32,
                        height=32,
                        border_radius=16,
                        bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.PRIMARY),
                        alignment=ft.alignment.center,
                        border=ft.border.all(2, SciFiColors.PRIMARY),
                    ),
                    ft.Container(width=8),
                    ft.IconButton(
                        icon=ft.Icons.LOGOUT_ROUNDED,
                        icon_color=SciFiColors.ERROR,
                        icon_size=20,
                        tooltip="Logout",
                        on_click=lambda _: self.logout(),
                    ),
                ], spacing=0),
            ], spacing=0, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=ft.padding.symmetric(horizontal=20, vertical=12),
            bgcolor=ft.Colors.with_opacity(0.9, SciFiColors.BG_DARK),
            border=ft.border.only(bottom=ft.border.BorderSide(1, SciFiColors.BORDER)),
        )

    def _create_main_content(self) -> ft.Container:
        """Create main content area based on current navigation section"""
        logger.info(f"Creating main content for section: {self.current_nav_section}")
        try:
            if self.current_nav_section == "dashboard":
                logger.info("Creating dashboard content")
                return self._create_dashboard_content()
            elif self.current_nav_section == "assistant":
                logger.info("Creating assistant content")
                return self._create_assistant_content()
            elif self.current_nav_section == "security":
                logger.info("Creating security content")
                return self._create_security_content()
            elif self.current_nav_section == "logs":
                logger.info("Creating logs content")
                return self._create_logs_content()
            elif self.current_nav_section == "settings":
                logger.info("Creating settings content")
                return self._create_settings_content()
            else:
                logger.warning(f"Unknown section: {self.current_nav_section}, defaulting to dashboard")
                return self._create_dashboard_content()
        except Exception as e:
            logger.error(f"Error creating main content: {e}", exc_info=True)
            return ft.Container(
                content=ft.Text(f"Content Error: {str(e)}", color=SciFiColors.ERROR),
                alignment=ft.alignment.center,
                padding=40,
            )

    def _create_dashboard_content(self) -> ft.Container:
        """Create main dashboard with three panels: Security, Assistant, System"""
        return ft.Container(
            content=ft.Column([
                # Title
                ft.Container(
                    content=ft.Text(
                        "SYSTEM DASHBOARD",
                        size=24,
                        weight=ft.FontWeight.BOLD,
                        color=SciFiColors.TEXT_PRIMARY,
                        font_family="Orbitron",
                    ),
                    padding=ft.padding.symmetric(vertical=20, horizontal=20),
                ),
                # Three-panel grid
                ft.Container(
                    content=ft.Row([
                        # Security Panel
                        ft.Container(
                            content=self._create_security_panel(),
                            expand=1,
                            padding=ft.padding.all(10),
                        ),
                        # Assistant Panel
                        ft.Container(
                            content=self._create_assistant_panel(),
                            expand=1,
                            padding=ft.padding.all(10),
                        ),
                        # System Panel
                        ft.Container(
                            content=self._create_system_panel(),
                            expand=1,
                            padding=ft.padding.all(10),
                        ),
                    ], spacing=20, alignment=ft.MainAxisAlignment.START),
                    height=350,
                ),
                # Actions section
                ft.Container(
                    content=self._create_actions_section(),
                    padding=ft.padding.symmetric(vertical=20),
                ),
            ], spacing=0, scroll=ft.ScrollMode.AUTO),
            padding=ft.padding.all(20),
            expand=True,
        )

    def _create_security_panel(self) -> ft.Container:
        """Create security panel with circular verification indicators"""
        return ft.Container(
            content=ft.Column([
                # Header
                ft.Row([
                    ft.Icon(ft.Icons.SECURITY_ROUNDED, color=SciFiColors.SUCCESS, size=20),
                    ft.Text("SECURITY STATUS", size=16, weight=ft.FontWeight.BOLD, font_family="Orbitron", color=SciFiColors.TEXT_PRIMARY),
                ], spacing=8),
                ft.Container(height=20),
                # Circular indicators
                ft.Row([
                    self._create_circular_indicator("Voice Auth", True, SciFiColors.SUCCESS),
                    self._create_circular_indicator("Password", True, SciFiColors.SUCCESS),
                    self._create_circular_indicator("Liveness", True, SciFiColors.SUCCESS),
                ], spacing=20, alignment=ft.MainAxisAlignment.CENTER),
                ft.Container(height=20),
                # Status text
                ft.Container(
                    content=ft.Text(
                        "ALL SYSTEMS VERIFIED",
                        size=12,
                        color=SciFiColors.SUCCESS,
                        weight=ft.FontWeight.W_600,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    alignment=ft.alignment.center,
                ),
            ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=20,
            bgcolor=ft.Colors.with_opacity(0.6, SciFiColors.BG_CARD),
            border=ft.border.all(1, SciFiColors.BORDER_GLOW),
            border_radius=12,
            height=280,
        )

    def _create_circular_indicator(self, label: str, is_active: bool, color: str) -> ft.Container:
        return ft.Container(
            content=ft.Column([
                ft.Stack([
                    ft.Container(
                        width=60,
                        height=60,
                        border_radius=30,
                        border=ft.border.all(3, color if is_active else SciFiColors.BORDER),
                        bgcolor=ft.Colors.with_opacity(0.1, color) if is_active else ft.Colors.TRANSPARENT,
                    ),
                    ft.Container(
                        content=ft.Icon(
                            ft.Icons.CHECK_ROUNDED if is_active else ft.Icons.CLOSE_ROUNDED,
                            color=color if is_active else SciFiColors.TEXT_MUTED,
                            size=24,
                        ),
                        width=60,
                        height=60,
                        alignment=ft.alignment.center,
                    ),
                ]),
                ft.Container(height=8),
                ft.Text(
                    label.upper(),
                    size=10,
                    color=color if is_active else SciFiColors.TEXT_MUTED,
                    weight=ft.FontWeight.W_600,
                    text_align=ft.TextAlign.CENTER,
                ),
            ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            alignment=ft.alignment.center,
        )

    def _create_assistant_panel(self) -> ft.Container:
        """Create assistant panel with voice activity visualization"""
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.MIC_ROUNDED, color=SciFiColors.PRIMARY, size=20),
                    ft.Text("VOICE ASSISTANT", size=16, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ], spacing=8),
                ft.Container(height=20),
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Container(width=4, height=20 + rnd.randint(0, 20), bgcolor=SciFiColors.PRIMARY, border_radius=2),
                            ft.Container(width=4, height=15 + rnd.randint(0, 25), bgcolor=SciFiColors.ACCENT, border_radius=2),
                            ft.Container(width=4, height=25 + rnd.randint(0, 15), bgcolor=SciFiColors.PRIMARY, border_radius=2),
                            ft.Container(width=4, height=10 + rnd.randint(0, 30), bgcolor=SciFiColors.ACCENT, border_radius=2),
                            ft.Container(width=4, height=30 + rnd.randint(0, 10), bgcolor=SciFiColors.PRIMARY, border_radius=2),
                            ft.Container(width=4, height=20 + rnd.randint(0, 20), bgcolor=SciFiColors.ACCENT, border_radius=2),
                            ft.Container(width=4, height=15 + rnd.randint(0, 25), bgcolor=SciFiColors.PRIMARY, border_radius=2),
                        ], spacing=6, alignment=ft.MainAxisAlignment.CENTER),
                        ft.Container(height=20),
                        ft.Text(
                            "READY FOR COMMANDS",
                            size=12,
                            color=SciFiColors.TEXT_SECONDARY,
                            weight=ft.FontWeight.W_600,
                        ),
                    ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    height=120,
                    alignment=ft.alignment.center,
                ),
            ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=20,
            bgcolor=ft.Colors.with_opacity(0.6, SciFiColors.BG_CARD),
            border=ft.border.all(1, SciFiColors.BORDER_GLOW),
            border_radius=12,
            height=280,
        )

    def _create_system_panel(self) -> ft.Container:
        """Create system panel with stats and mini charts"""
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            ram_usage = psutil.virtual_memory().percent
            uptime_seconds = int(time.time() - psutil.boot_time())
            uptime = str(datetime.timedelta(seconds=uptime_seconds))
        except:
            cpu_usage = 45
            ram_usage = 62
            uptime = "2h 34m"
        
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.COMPUTER_ROUNDED, color=SciFiColors.SUCCESS, size=20),
                    ft.Text("SYSTEM STATUS", size=16, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ], spacing=8),
                ft.Container(height=20),
                ft.Column([
                    self._create_stat_bar("CPU", int(cpu_usage), SciFiColors.PRIMARY),
                    ft.Container(height=12),
                    self._create_stat_bar("RAM", int(ram_usage), SciFiColors.ACCENT),
                    ft.Container(height=16),
                    ft.Row([
                        ft.Text("Uptime:", size=12, color=SciFiColors.TEXT_SECONDARY),
                        ft.Container(expand=True),
                        ft.Text(uptime, size=12, color=SciFiColors.TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                    ]),
                ], spacing=0),
            ], spacing=0),
            padding=20,
            bgcolor=ft.Colors.with_opacity(0.6, SciFiColors.BG_CARD),
            border=ft.border.all(1, SciFiColors.BORDER_GLOW),
            border_radius=12,
            height=280,
        )

    def _create_stat_bar(self, label: str, percentage: int, color: str) -> ft.Container:
        return ft.Column([
            ft.Row([
                ft.Text(label, size=12, color=SciFiColors.TEXT_SECONDARY),
                ft.Container(expand=True),
                ft.Text(f"{percentage}%", size=12, color=color, weight=ft.FontWeight.W_600),
            ]),
            ft.Container(height=4),
            ft.Container(
                content=ft.Container(
                    width=(percentage / 100) * 200,
                    height=6,
                    bgcolor=color,
                    border_radius=3,
                ),
                width=200,
                height=6,
                bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.BORDER),
                border_radius=3,
            ),
        ], spacing=0)

    def _create_actions_section(self) -> ft.Container:
        """Create actions section with action cards"""
        return ft.Container(
            content=ft.Column([
                ft.Text("QUICK ACTIONS", size=18, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ft.Container(height=16),
                ft.Row([
                    self._create_action_card("Security Scan", ft.Icons.SECURITY_ROUNDED, SciFiColors.WARNING, lambda e: self._run_security_scan()),
                    self._create_action_card("Take Screenshot", ft.Icons.CAMERA_ROUNDED, SciFiColors.ACCENT, self._take_screenshot_action),
                    self._create_action_card("System Info", ft.Icons.INFO_ROUNDED, SciFiColors.SUCCESS, self._show_system_status),
                    self._create_action_card("Voice Commands", ft.Icons.MIC_ROUNDED, SciFiColors.PRIMARY, self._show_voice_dialog),
                ], spacing=16, wrap=True),
            ], spacing=0),
            padding=20,
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
        return ft.Container(
            content=ft.Column([
                ft.Text("VOICE ASSISTANT", size=24, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ft.Container(height=20),
                ft.Text("Voice assistant controls will appear here.", color=SciFiColors.TEXT_SECONDARY, size=14),
                ft.Container(height=20),
                ft.ElevatedButton(
                    "Open Voice Assistant",
                    icon=ft.Icons.MIC_ROUNDED,
                    on_click=self._show_voice_dialog,
                    style=ft.ButtonStyle(
                        bgcolor=SciFiColors.PRIMARY,
                        color=SciFiColors.BG_DARK,
                    ),
                ),
            ]),
            padding=40,
            alignment=ft.alignment.center,
        )

    def _create_security_content(self) -> ft.Container:
        return ft.Container(
            content=ft.Column([
                ft.Text("SECURITY CENTER", size=24, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ft.Container(height=20),
                ft.Text("Security settings and logs will appear here.", color=SciFiColors.TEXT_SECONDARY, size=14),
            ]),
            padding=40,
            alignment=ft.alignment.center,
        )

    def _create_logs_content(self) -> ft.Container:
        return ft.Container(
            content=ft.Column([
                ft.Text("ACTIVITY LOGS", size=24, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ft.Container(height=20),
                ft.Text("System logs will appear here.", color=SciFiColors.TEXT_SECONDARY, size=14),
            ]),
            padding=40,
            alignment=ft.alignment.center,
        )

    def _create_settings_content(self) -> ft.Container:
        return ft.Container(
            content=ft.Column([
                ft.Text("SETTINGS", size=24, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ft.Container(height=20),
                ft.Text("Application settings will appear here.", color=SciFiColors.TEXT_SECONDARY, size=14),
            ]),
            padding=40,
            alignment=ft.alignment.center,
        )

    def _create_voice_dock(self) -> ft.Container:
        return ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Row([
                        ft.Container(
                            content=ft.IconButton(
                                icon=ft.Icons.MIC_ROUNDED,
                                icon_size=28,
                                icon_color=SciFiColors.TEXT_PRIMARY,
                                on_click=self._show_voice_dialog,
                                style=ft.ButtonStyle(
                                    bgcolor=ft.Colors.with_opacity(0.2, SciFiColors.PRIMARY),
                                    shape=ft.CircleBorder(),
                                ),
                            ),
                            width=60,
                            height=60,
                            border_radius=30,
                            border=ft.border.all(2, SciFiColors.PRIMARY),
                            alignment=ft.alignment.center,
                        ),
                        ft.Container(width=16),
                        ft.Column([
                            ft.Text(
                                "VOICE ASSISTANT",
                                size=10,
                                color=SciFiColors.TEXT_SECONDARY,
                                weight=ft.FontWeight.W_600,
                            ),
                            ft.Text(
                                "Click to activate",
                                size=9,
                                color=SciFiColors.TEXT_MUTED,
                            ),
                        ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.START),
                    ], spacing=0, alignment=ft.MainAxisAlignment.START),
                    padding=ft.padding.symmetric(horizontal=20, vertical=10),
                ),
                ft.Container(expand=True),
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            "VOICE TRANSCRIPT",
                            size=10,
                            color=SciFiColors.TEXT_MUTED,
                            weight=ft.FontWeight.W_600,
                        ),
                        ft.Container(
                            content=ft.Text(
                                "Ready for voice commands...",
                                size=12,
                                color=SciFiColors.TEXT_SECONDARY,
                            ),
                            height=40,
                            alignment=ft.alignment.center_left,
                        ),
                    ], spacing=4),
                    width=300,
                    padding=ft.padding.symmetric(horizontal=16, vertical=8),
                    border_radius=8,
                    bgcolor=ft.Colors.with_opacity(0.3, SciFiColors.BG_DARK),
                    border=ft.border.all(1, SciFiColors.BORDER),
                ),
            ], spacing=0, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            height=80,
            bgcolor=ft.Colors.with_opacity(0.9, SciFiColors.BG_ELEVATED),
            border=ft.border.only(top=ft.border.BorderSide(1, SciFiColors.BORDER)),
        )

    def _navigate_to_section(self, section_name: str):
        """Navigate to a different dashboard section and refresh view"""
        try:
            logger.info(f"Navigating to section: {section_name}")
            self.current_nav_section = section_name
            
            # Rebuild dashboard
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
            
            # Create dialog components
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
            
            # Create buttons
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
            
            # Build dialog content
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
            
            # Create dialog
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
            
            # Open dialog
            self._open_dialog_safe(dialog)
            self.voice_assistant_dialog_open = True
            
            # Add initial log entry
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
            
            # Run recording in background thread
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
                        
                        self.tts.speak(response)
                        
                    else:
                        self.page.run_task(update_ui_no_speech)
                    
                except Exception as e:
                    logger.error(f"Error in recording thread: {e}")
                    self.page.run_task(lambda: update_ui_exception(e))
            
            # Start recording in background thread
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
        # Stop any ongoing voice assistant activity
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
        """Update login status"""
        self.status_text.value = message
        self.status_text.color = color
        self.status_panel.visible = bool(message)
        self.status_panel.bgcolor = ft.Colors.with_opacity(0.1, color)
        self.status_panel.border = ft.border.all(1, color)
        self.page.update()

    def update_reg_status(self, message: str, color: str):
        """Update registration status"""
        self.reg_status_text.value = message
        self.reg_status_text.color = color
        self.reg_status_panel.visible = bool(message)
        self.reg_status_panel.bgcolor = ft.Colors.with_opacity(0.1, color)
        self.reg_status_panel.border = ft.border.all(1, color)
        self.page.update()

    def show_progress(self, show: bool = True):
        self.progress_ring.visible = show
        self.page.update()

    def show_reg_progress(self, visible: bool):
        self.reg_progress_ring.visible = visible
        self.page.update()

    def toggle_recording(self, mode="voice"):
        """Toggle recording for login with mode-specific messages"""
        self.recording_active = not self.recording_active
        
        if self.recording_active:
            self.record_button.text = "◉ STOP RECORDING"
            self.record_button.icon = ft.Icons.STOP
            self.record_button.style = ft.ButtonStyle(
                bgcolor=SciFiColors.ERROR,
                color=SciFiColors.TEXT_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=8),
            )
            if mode == "voice":
                self.update_status("⦿ Recording... Speak now!", SciFiColors.ERROR)
            elif mode == "face":
                self.update_status("⦿ Capturing face... Please look at the camera", SciFiColors.INFO)
        else:
            if mode == "voice":
                self.record_button.text = "START RECORDING"
                self.record_button.icon = ft.Icons.MIC
                self.record_button.style = ft.ButtonStyle(
                    bgcolor=SciFiColors.PRIMARY,
                    color=SciFiColors.BG_DARK,
                    shape=ft.RoundedRectangleBorder(radius=8),
                )
                self.update_status("✓ Recording complete", SciFiColors.SUCCESS)
            elif mode == "face":
                self.record_button.text = "CAPTURE FACE"
                self.record_button.icon = ft.Icons.CAMERA_ALT
                self.record_button.style = ft.ButtonStyle(
                    bgcolor=SciFiColors.SUCCESS,
                    color=SciFiColors.BG_DARK,
                    shape=ft.RoundedRectangleBorder(radius=8),
                )
                self.update_status("✅ Face captured successfully", SciFiColors.SUCCESS)
        
        self.page.update()

    def toggle_reg_recording(self, mode="voice"):
        """Toggle recording for registration with mode-specific messages"""
        self.reg_recording_active = not self.reg_recording_active
        
        if self.reg_recording_active:
            self.reg_record_button.text = "◉ STOP RECORDING"
            self.reg_record_button.icon = ft.Icons.STOP
            self.reg_record_button.style = ft.ButtonStyle(
                bgcolor=SciFiColors.SUCCESS,
                color=SciFiColors.BG_DARK,
                shape=ft.RoundedRectangleBorder(radius=8),
            )
            if mode == "voice":
                self.update_reg_status("⦿ Recording... Speak now!", SciFiColors.ERROR)
            elif mode == "face":
                self.update_reg_status("⦿ Capturing face... Please look at the camera", SciFiColors.INFO)
        else:
            if mode == "voice":
                self.reg_record_button.text = "START RECORDING"
                self.reg_record_button.icon = ft.Icons.MIC
                self.reg_record_button.style = ft.ButtonStyle(
                    bgcolor=SciFiColors.ERROR,
                    color=SciFiColors.TEXT_PRIMARY,
                    shape=ft.RoundedRectangleBorder(radius=8),
                )
                self.update_reg_status("✓ Recording complete", SciFiColors.SUCCESS)
            elif mode == "face":
                self.reg_record_button.text = "CAPTURE FACE"
                self.reg_record_button.icon = ft.Icons.CAMERA_ALT
                self.reg_record_button.style = ft.ButtonStyle(
                    bgcolor=SciFiColors.SUCCESS,
                    color=SciFiColors.TEXT_PRIMARY,
                    shape=ft.RoundedRectangleBorder(radius=8),
                )
                self.update_reg_status("✅ Face captured successfully", SciFiColors.SUCCESS)
        
        self.page.update()

    def show_record_button(self, mode="voice"):
        """Show recording button with appropriate icon and tooltip for mode"""
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
        
        self.record_button.visible = True
        self.recording_active = False
        self.page.update()

    def hide_record_button(self):
        """Hide recording button"""
        self.mic_status.visible = False
        self.record_button.visible = False
        self.recording_active = False
        self.page.update()

    def show_reg_record_button(self, sample_info: str = "", on_click=None, mode="voice"):
        """Show registration recording button with appropriate icon and tooltip for mode"""
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
        
        self.reg_record_button.visible = True
        self.reg_recording_active = False
        
        if on_click:
            self.reg_record_button.on_click = on_click
        else:
            self.reg_record_button.on_click = lambda _: self.toggle_reg_recording()
            
        self.page.update()

    def hide_reg_record_button(self):
        """Hide registration recording button"""
        self.reg_mic_status.visible = False
        self.reg_record_button.visible = False
        self.reg_recording_active = False
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
        self.update_auth_progress("idle")

    def cleanup_old_temp_files(self):
        """Clean up old temporary audio files to prevent disk space issues"""
        try:
            cleanup_temp_files(str(self.temp_dir))
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

    def wait_for_recording_start(self):
        """Wait for recording start"""
        while not self.recording_active:
            time.sleep(0.1)

    def wait_for_recording_stop(self):
        """Wait for recording stop"""
        while self.recording_active:
            time.sleep(0.1)

    def wait_for_reg_recording_start(self):
        """Wait for reg recording start"""
        while not self.reg_recording_active:
            time.sleep(0.1)

    def wait_for_reg_recording_stop(self):
        """Wait for reg recording stop"""
        while self.reg_recording_active:
            time.sleep(0.1)

    def wait_for_face_capture(self):
        """Wait for face capture button click"""
        while not self.face_capture_requested:
            time.sleep(0.1)

    # ==================== AUTHENTICATION ====================
    
    def start_voice_login(self):
        """Start voice authentication"""
        username = self.username_field.value
        password = self.password_field.value
        
        if not username or not password:
            self.update_status("⚠ Enter username and password", SciFiColors.ERROR)
            return
        
        user = self.db.get_user_by_username(username)
        if not user:
            self.update_status("⚠ User not found", SciFiColors.ERROR)
            return
        
        if not self.security_manager.verify_hashed_password(password, user['password_hash']):
            self.update_status("⚠ Invalid password", SciFiColors.ERROR)
            return
        
        # Reset and initialize auth states
        self.reset_auth_states()
        self.auth_step = "voice_verifying"
        self.update_auth_progress("voice_verifying")
        
        self.update_status("⟳ Initializing voice auth...", SciFiColors.INFO)
        self.show_progress(True)
        
        async def verify_wrapper():
            await self.verify_voice(user, verify_face=True)
        
        self.page.run_task(verify_wrapper)

    def start_registration(self):
        """Start registration"""
        self.page.run_task(self.process_registration)

    async def process_registration(self):
        """Process registration"""
        try:
            # Initialize auth states for registration
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
            
            # Registration complete - switch to login
            self.auth_tabs.selected_index = 0
            self._handle_auth_tab_change(type('obj', (object,), {'control': self.auth_tabs})())
            self.update_status("✓ Registration complete! Please login.", SciFiColors.SUCCESS)
            self.show_reg_progress(False)
            
            # Clean up temp files after registration
            self.cleanup_old_temp_files()
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            self.update_reg_status(f"⚠ Failed: {str(e)}", SciFiColors.ERROR)
            self.show_reg_progress(False)

    async def verify_voice(self, user: dict, verify_face=False, face_image=None):
        """Verify voice biometric using ultimate engine with AASIST anti-spoofing"""
        try:
            self.update_status("⦿ Click START RECORDING for secure voice verification", SciFiColors.INFO)
            self.tts.speak("Please speak for secure voice verification with anti-spoofing")
            self.show_record_button()

            await asyncio.get_event_loop().run_in_executor(None, self.wait_for_recording_start)

            recording_complete = threading.Event()
            audio_data = [None]

            def record_thread():
                audio_data[0] = self.audio_recorder.record_audio(duration=5.0)
                recording_complete.set()

            rec_thread = threading.Thread(target=record_thread, daemon=True)
            rec_thread.start()

            await asyncio.get_event_loop().run_in_executor(None, self.wait_for_recording_stop)

            # Wait for the recording thread to actually complete
            await asyncio.get_event_loop().run_in_executor(None, lambda: recording_complete.wait(timeout=6.0))

            self.hide_record_button()

            if audio_data[0] is None:
                self.update_status("⚠ Recording failed", SciFiColors.ERROR)
                self.tts.speak("Voice recording failed")
                self.show_progress(False)
                return

            self.update_status("⟳ Analyzing voice with AASIST anti-spoofing...", SciFiColors.INFO)
            self.tts.speak("Analyzing voice with advanced security")

            # Use ultimate voice engine for verification
            verification_result = self.ultimate_voice_engine.verify_voice(
                user_id=user['id'],
                audio_data=audio_data[0],
                sample_rate=16000,
                enable_challenge=False  # Can be enabled later if needed
            )

            if not verification_result['verified']:
                failure_reason = verification_result['details'].get('failure_reason', 'Unknown error')
                if verification_result['spoof_detected']:
                    self.update_status("⚠ Anti-spoofing: Voice rejected as suspicious", SciFiColors.ERROR)
                    self.tts.speak("Voice rejected by anti-spoofing system")
                else:
                    confidence_pct = verification_result['confidence'] * 100
                    self.update_status(f"⚠ Voice verification failed - {confidence_pct:.1f}% confidence", SciFiColors.ERROR)
                    self.tts.speak("Voice verification failed")
                self.show_progress(False)
                return

            confidence_pct = verification_result['confidence'] * 100
            self.update_status(f"✓ Voice verified with {confidence_pct:.1f}% confidence", SciFiColors.SUCCESS)
            self.tts.speak("Voice verified successfully with advanced security")
            self.voice_verification_complete = True
            self.update_auth_progress("face_verifying")
            await asyncio.sleep(1)
            
            if verify_face:
                self.update_status("⦿ Now verifying face (ArcFace)...", SciFiColors.INFO)
                self.tts.speak("Now verifying face")
                await asyncio.sleep(1)
                face_embeddings = self.db.get_face_embeddings(user['id'])
                arcface_embeddings = [emb['embedding_data'] for emb in face_embeddings if isinstance(emb['embedding_data'], list) and len(emb['embedding_data']) == 512]
                if arcface_embeddings:
                    # Automatic face capture - no button required
                    self.update_status("⦿ Automatically capturing face for verification...", SciFiColors.INFO)
                    self.tts.speak("Please look at the camera for automatic face verification")
                    
                    # Try multiple captures to get a good face
                    max_attempts = 3
                    face_verified = False
                    
                    for attempt in range(max_attempts):
                        self.update_status(f"⦿ Face capture attempt {attempt + 1}/{max_attempts}...", SciFiColors.INFO)
                        await asyncio.sleep(0.5)
                        
                        cap = cv2.VideoCapture(0)
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret and frame is not None:
                            self.update_status("⟳ Processing face...", SciFiColors.INFO)
                            self.tts.speak("Processing face")
                            is_match, similarity, liveness_passed = self.verify_face_arcface(frame, arcface_embeddings)
                            similarity_percent = similarity * 100
                            
                            if is_match and liveness_passed:
                                self.update_status(f"✅ Face verified (ArcFace) - {similarity_percent:.1f}% similarity", SciFiColors.SUCCESS)
                                self.tts.speak("Face verified successfully")
                                face_verified = True
                                break
                            elif is_match and not liveness_passed:
                                self.update_status(f"⚠ Liveness check failed - {similarity_percent:.1f}% similarity (attempt {attempt + 1})", SciFiColors.WARNING)
                                if attempt < max_attempts - 1:
                                    self.tts.speak("Liveness check failed, trying again")
                                    await asyncio.sleep(1)
                            else:
                                self.update_status(f"⚠ Face verification failed - {similarity_percent:.1f}% similarity (attempt {attempt + 1})", SciFiColors.WARNING)
                                if attempt < max_attempts - 1:
                                    self.tts.speak("Face not recognized, trying again")
                                    await asyncio.sleep(1)
                        else:
                            self.update_status(f"⚠ Face capture failed (attempt {attempt + 1})", SciFiColors.WARNING)
                            if attempt < max_attempts - 1:
                                await asyncio.sleep(1)
                    
                    if not face_verified:
                        self.update_status("⚠ Face verification failed after all attempts", SciFiColors.ERROR)
                        self.tts.speak("Face verification failed")
                        self.show_progress(False)
                        return
                else:
                    self.update_status("⚠ No ArcFace profile found - Skipping", SciFiColors.WARNING)
                    self.tts.speak("No face profile found, skipping face verification")
                    self.face_verification_complete = True  # Consider it complete if no profile exists
                    self.update_auth_progress("complete")
                    await asyncio.sleep(1)
            
            # Authentication successful - transition to dashboard
            self.update_status("✓ Authentication successful! Loading dashboard...", SciFiColors.SUCCESS)
            self.tts.speak("Authentication successful")
            await asyncio.sleep(1)
            self.show_progress(False)
            
            # Clean up temp files after successful auth
            self.cleanup_old_temp_files()
            
            # Store user and transition
            self.current_user = type('User', (), user)()
            self.current_view = "dashboard"
            
            # Smooth transition
            self.page.controls.clear()
            dashboard = self.build_dashboard_view()
            self.page.add(dashboard)
            self.page.update()
            
            try:
                self.tts.speak(f"Welcome back, {user['username']}")
            except:
                pass
            
        except Exception as e:
            logger.error(f"Voice verification error: {e}")
            self.update_status(f"⚠ Error: {str(e)}", SciFiColors.ERROR)
            self.tts.speak("Authentication failed due to an error")
            self.show_progress(False)

    async def enroll_user_voice_registration(self, user: dict):
        """Enroll voice during registration using ultimate biometric engine"""
        try:
            self.update_reg_status("⟳ Recording 3 voice samples with data augmentation", SciFiColors.INFO)
            self.tts.speak("Please provide 3 voice samples for secure enrollment")
            await asyncio.sleep(1)
            
            audio_samples = []
            
            for sample_num in range(1, 4):  # 3 samples
                self.update_reg_status(f"⦿ Sample {sample_num}/3 - Click START RECORDING", SciFiColors.INFO)
                self.tts.speak(f"Please speak for voice sample {sample_num} of 3")
                self.show_reg_record_button(f"SAMPLE {sample_num} OF 3")
                
                await asyncio.get_event_loop().run_in_executor(None, self.wait_for_reg_recording_start)
                
                recording_complete = threading.Event()
                audio_data_holder = [None]
                
                def record_thread():
                    audio_data_holder[0] = self.audio_recorder.record_audio(duration=5.0)  # 5 seconds per sample
                    recording_complete.set()
                
                rec_thread = threading.Thread(target=record_thread, daemon=True)
                rec_thread.start()
                
                await asyncio.get_event_loop().run_in_executor(None, self.wait_for_reg_recording_stop)
                
                # Wait for recording to complete
                await asyncio.get_event_loop().run_in_executor(None, lambda: recording_complete.wait(timeout=6.0))
                
                self.hide_reg_record_button()
                
                audio_data = audio_data_holder[0]
                
                if audio_data is None:
                    self.update_reg_status(f"⚠ Sample {sample_num} failed - Skipping", SciFiColors.WARNING)
                    self.tts.speak("Recording failed, please try again")
                    await asyncio.sleep(1)
                    continue
                
                self.update_reg_status(f"✓ Sample {sample_num} recorded - Processing", SciFiColors.SUCCESS)
                self.tts.speak("Recording complete, processing voice sample")
                await asyncio.sleep(0.5)
                
                audio_samples.append(audio_data)
                
                self.update_reg_status(f"✓ Sample {sample_num}/3 saved", SciFiColors.SUCCESS)
                await asyncio.sleep(1)
            
            if len(audio_samples) < 2:
                self.update_reg_status("⚠ Insufficient valid samples - Please try again", SciFiColors.ERROR)
                self.tts.speak("Insufficient voice samples recorded, please restart enrollment")
                self.show_reg_progress(False)
                return
            
            # Use ultimate voice engine for enrollment
            self.update_reg_status("⟳ Creating secure voice profile with augmentation...", SciFiColors.INFO)
            self.tts.speak("Creating your secure voice profile")
            
            success = self.ultimate_voice_engine.enroll_user_voice(
                user_id=user['id'],
                audio_samples=audio_samples,
                sample_rate=16000
            )
            
            if success:
                self.update_reg_status("✅ Voice enrollment complete! Profile secured with anti-spoofing", SciFiColors.SUCCESS)
                self.tts.speak("Voice enrollment complete with advanced security features")
                self.voice_enrollment_complete = True
                self.update_auth_progress("face_enrolling")
                await asyncio.sleep(2)
            else:
                self.update_reg_status("⚠ Voice enrollment failed - Please try again", SciFiColors.ERROR)
                self.tts.speak("Voice enrollment failed, please try again")
                self.show_reg_progress(False)
            
        except Exception as e:
            logger.error(f"Voice enrollment error: {e}")
            self.update_reg_status(f"⚠ Enrollment error: {str(e)}", SciFiColors.ERROR)
            self.tts.speak("Voice enrollment failed due to an error")
            self.show_reg_progress(False)

    def show_dashboard(self):
        """Render the dashboard view."""
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
        
        # Close all active dialogs
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
        
        # Clean up temp files on logout
        self.cleanup_old_temp_files()
        
        # Clear page and show login
        self.page.controls.clear()
        login = self.build_login_view()
        self.page.add(login)
        self.page.update()
        
        try:
            self.tts.speak("Logged out successfully")
        except:
            pass

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
    from utils.helpers import load_config
    
    page.title = "SecureX-Assist"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.resizable = True
    page.window.maximizable = True
    page.window.minimizable = True
    
    config = load_config()
    
    app = SecureXApp(page, config)
    app.run()


if __name__ == "__main__":
    ft.app(target=main)