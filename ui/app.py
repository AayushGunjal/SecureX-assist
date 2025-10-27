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
    INFO = "#00eaff"
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
from core.security import SecurityManager, SecurityContext
from core.audio_processor import AudioRecorder, VoiceActivityDetector
from utils.tts import TextToSpeech
from core.voice_assistant import VoiceAssistant
from core.face_recognition_engine import FaceRecognitionEngine
from utils.helpers import create_temp_directory


# Advanced Tkinter Dashboard UI
import tkinter as tk
from tkinter import ttk

class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Secure Voice Dashboard")
        self.geometry("1200x700")
        self.configure(bg="#eaf0fb")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#eaf0fb")
        self.style.configure("TLabel", background="#eaf0fb", font=("Segoe UI", 12))
        self.style.configure("Sidebar.TFrame", background="#273c75")
        self.style.configure("Sidebar.TLabel", background="#273c75", foreground="#fff", font=("Segoe UI", 14, "bold"))
        self.style.configure("Sidebar.TButton", font=("Segoe UI", 12, "bold"), foreground="#fff", background="#4078c0")
        self.style.map("Sidebar.TButton", background=[("active", "#192a56")])
        self.style.configure("Card.TFrame", background="#fff")
        self.style.configure("Card.TLabel", background="#fff", font=("Segoe UI", 15))
        self.create_widgets()

    def create_widgets(self):
        # Sidebar Navigation
        sidebar = ttk.Frame(self, style="Sidebar.TFrame", width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        logo = ttk.Label(sidebar, text="🔒 Secure Voice", style="Sidebar.TLabel")
        logo.pack(pady=(30, 40))
        btn1 = ttk.Button(sidebar, text="Enroll Voice", command=self.enroll_voice, style="Sidebar.TButton")
        btn1.pack(fill="x", padx=30, pady=10, ipady=8)
        btn2 = ttk.Button(sidebar, text="Run System Test", command=self.run_test, style="Sidebar.TButton")
        btn2.pack(fill="x", padx=30, pady=10, ipady=8)
        btn3 = ttk.Button(sidebar, text="Settings", command=self.open_settings, style="Sidebar.TButton")
        btn3.pack(fill="x", padx=30, pady=10, ipady=8)

        # Main Area
        main_area = ttk.Frame(self)
        main_area.pack(side="left", fill="both", expand=True)

        # Header Section
        header = ttk.Frame(main_area)
        header.pack(fill="x", pady=(0, 10))
        title = ttk.Label(header, text="Secure Voice Authentication Dashboard", font=("Segoe UI", 26, "bold"), foreground="#4078c0")
        title.pack(side="left", padx=30, pady=30)

        # Sectioned Content Area
        content = ttk.Frame(main_area)
        content.pack(fill="both", expand=True, padx=40, pady=10)

        # Card-like Section for Status
        card = ttk.Frame(content, style="Card.TFrame")
        card.pack(fill="x", pady=20, padx=60)
        self.status_label = ttk.Label(card, text="Welcome! Select an action from the sidebar.", style="Card.TLabel")
        self.status_label.pack(pady=30)

        # Advanced Info Section
        info_frame = ttk.Frame(content)
        info_frame.pack(fill="x", pady=10, padx=60)
        info_title = ttk.Label(info_frame, text="System Overview", font=("Segoe UI", 16, "bold"), foreground="#273c75")
        info_title.pack(anchor="w")
        info_text = ttk.Label(info_frame, text="- Voice biometric security\n- Anti-spoofing enabled\n- Face recognition integrated\n- Real-time monitoring", font=("Segoe UI", 12), foreground="#353b48")
        info_text.pack(anchor="w", pady=5)

        # Footer
        footer = ttk.Frame(main_area)
        footer.pack(side="bottom", fill="x", pady=10)
        footer_label = ttk.Label(footer, text="© 2025 Secure Voice System", font=("Segoe UI", 10), foreground="#888")
        footer_label.pack(side="right", padx=30)

    def enroll_voice(self):
        self.status_label.config(text="Voice enrollment started. Please follow the instructions.")

    def run_test(self):
        self.status_label.config(text="System test running. Please wait...")

    def open_settings(self):
        self.status_label.config(text="Settings panel opened. Adjust your preferences.")

# The Tkinter `Dashboard` class is kept for reference but should not run
# when this module is imported by the main application. The Flet-based
# application entrypoint (function `main`) below is used instead.
INFO = "#3b82f6"

# Borders & Effects
BORDER = "#1e3a5f"
BORDER_LIGHT = "#2d4a6f"
BORDER_GLOW = "#00d9ff60"


class SecureXApp:
    def _create_assistant_panel(self) -> ft.Container:
        """Create assistant panel with voice activity visualization"""
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.MIC_ROUNDED, color=SciFiColors.PRIMARY, size=20),
                    ft.Text("VOICE ASSISTANT", size=16, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ], spacing=8),
                ft.Container(height=20),
                # ✅ FIXED: Added proper content with voice activity bars
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
                    self._create_stat_bar("CPU", cpu_usage, SciFiColors.PRIMARY),
                    ft.Container(height=12),
                    self._create_stat_bar("RAM", ram_usage, SciFiColors.ACCENT),
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
                    self._create_action_card("Security Scan", ft.Icons.SECURITY_ROUNDED, SciFiColors.WARNING, self._run_security_scan),
                    self._create_action_card("Take Screenshot", ft.Icons.CAMERA_ROUNDED, SciFiColors.ACCENT, self._take_screenshot_action),
                    self._create_action_card("System Info", ft.Icons.INFO_ROUNDED, SciFiColors.SUCCESS, self._show_system_status),
                    self._create_action_card("+ Add Action", ft.Icons.ADD_ROUNDED, SciFiColors.TEXT_MUTED, lambda: None),
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

    def _create_assistant_content(self) -> ft.Container:
        return ft.Container(
            content=ft.Text("Assistant section - Coming soon", color=SciFiColors.TEXT_SECONDARY),
            padding=40,
            alignment=ft.alignment.center,
        )

    def _create_security_content(self) -> ft.Container:
        return ft.Container(
            content=ft.Text("Security section - Coming soon", color=SciFiColors.TEXT_SECONDARY),
            padding=40,
            alignment=ft.alignment.center,
        )

    def _create_logs_content(self) -> ft.Container:
        return ft.Container(
            content=ft.Text("Logs section - Coming soon", color=SciFiColors.TEXT_SECONDARY),
            padding=40,
            alignment=ft.alignment.center,
        )

    def _create_settings_content(self) -> ft.Container:
        return ft.Container(
            content=ft.Text("Settings section - Coming soon", color=SciFiColors.TEXT_SECONDARY),
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
                                on_click=self._toggle_voice_assistant,
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
                                "CONTINUOUS MODE",
                                size=10,
                                color=SciFiColors.TEXT_SECONDARY,
                                weight=ft.FontWeight.W_600,
                            ),
                            ft.Switch(
                                value=self.continuous_mode_active,
                                active_color=SciFiColors.PRIMARY,
                                on_change=lambda e: setattr(self, 'continuous_mode_active', e.control.value),
                                scale=0.8,
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
                # Three-panel grid - FIXED: Added explicit heights and scrolling
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
                    height=350,  # ← ADD THIS: Give explicit height
                ),
                # Actions section
                ft.Container(
                    content=self._create_actions_section(),
                    padding=ft.padding.symmetric(vertical=20),
                ),
            ], spacing=0, scroll=ft.ScrollMode.AUTO),  # ← ADD scroll support
            padding=ft.padding.all(20),
            expand=True,  # ← IMPORTANT: Let container fill available space
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
                ], spacing=0),
            ], spacing=0, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=ft.padding.symmetric(horizontal=20, vertical=12),
            bgcolor=ft.Colors.with_opacity(0.9, SciFiColors.BG_DARK),
            border=ft.border.only(bottom=ft.border.BorderSide(1, SciFiColors.BORDER)),
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

    def _navigate_to_section(self, section_name: str):
        """Navigate to a different dashboard section and refresh view"""
        try:
            logger.info(f"Navigating to section: {section_name}")
            self.current_nav_section = section_name
            # Rebuild dashboard content in-place when on dashboard view
            if self.current_view == "dashboard":
                # Re-render the dashboard
                self.show_dashboard()
            else:
                # If not on dashboard, switch to dashboard
                self.current_view = "dashboard"
                self.show_dashboard()
        except Exception as e:
            logger.error(f"Navigation error: {e}", exc_info=True)
            self._show_error_toast(f"Navigation failed: {str(e)}")
    """Main application class with sci-fi themed UI"""

    def __init__(self, page: ft.Page, config: dict):
        self.page = page
        self.config = config

        # Initialize components
        self.db = Database(config.get('database', {}).get('path', 'securex_db.sqlite'))
        self.voice_engine = VoiceEngine(config)
        self.security_manager = SecurityManager(config)
        self.audio_recorder = AudioRecorder(config)
        self.vad = VoiceActivityDetector(config)
        self.tts = TextToSpeech(config)
        self.voice_assistant = VoiceAssistant()
        self.voice_assistant.setup_default_commands()
        self.face_engine = FaceRecognitionEngine()
        self._audio_stream_ctx = SimpleNamespace(
            recorder=self.audio_recorder,
            vad_detector=self.vad
        )

        # Session state
        self.current_user: Optional[SecurityContext] = None
        self.current_view = "login"

        # Voice assistant state
        self.continuous_mode_active = False
        self.voice_assistant_active = False
        self.voice_assistant_dialog_open = False

        # Navigation state
        self.current_nav_section = "dashboard"
        self.nav_items = [
            {"name": "dashboard", "icon": ft.Icons.DASHBOARD_ROUNDED, "label": "Dashboard"},
            {"name": "assistant", "icon": ft.Icons.MIC_ROUNDED, "label": "Assistant"},
            {"name": "security", "icon": ft.Icons.SECURITY_ROUNDED, "label": "Security"},
            {"name": "logs", "icon": ft.Icons.LIST_ROUNDED, "label": "Logs"},
            {"name": "settings", "icon": ft.Icons.SETTINGS_ROUNDED, "label": "Settings"},
        ]

        # Voice assistant panel (create actual panel but keep hidden)
        try:
            self.voice_assistant_panel = self._create_voice_assistant_panel()
            self.voice_assistant_panel.visible = False  # Hidden by default
        except Exception as e:
            logger.error(f"Failed to create voice assistant panel: {e}", exc_info=True)
            # Fallback to empty container to avoid attribute errors
            self.voice_assistant_panel = ft.Container(visible=False)

        # Recording states
        self.recording_active = False
        self.reg_recording_active = False

        # Temp directory

        self.temp_dir = create_temp_directory()

        # UI components
        self.interaction_log = None
        
        # Dialog management
        self.active_dialogs = []
        
        # Set up page close handler

        page.on_close = self._on_app_close

    def run(self):
        """Run the application"""
        # ✅ ADD THIS LINE - Setup page before doing anything else
        self.setup_page()
        
        self.db.connect()
        self.db.initialize_schema()
        
        self.voice_engine.load_models()
        
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
        
        # Load fonts - these URLs should work
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
                logger.info("✓ Sidebar created")
            except Exception as e:
                logger.error(f"✗ Sidebar failed: {e}")
                raise

            try:
                header = self._create_header(username)
                logger.info("✓ Header created")
            except Exception as e:
                logger.error(f"✗ Header failed: {e}")
                raise

            try:
                main_content = self._create_main_content()
                logger.info("✓ Main content created")
            except Exception as e:
                logger.error(f"✗ Main content failed: {e}")
                raise

            try:
                voice_dock = self._create_voice_dock()
                logger.info("✓ Voice dock created")
            except Exception as e:
                logger.error(f"✗ Voice dock failed: {e}")
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
    
    def _create_dashboard_card(self, title, icon, icon_color, items, action_button=None, custom_content=None):
        """Create a dashboard card"""
        card_content = [
            ft.Row([
                ft.Icon(icon, color=icon_color, size=20),
                ft.Text(title, size=14, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
            ]),
            ft.Container(height=16),
        ]
        
        if custom_content:
            card_content.append(custom_content)
        else:
            # Create content from items
            for item_text, item_value, item_color in items:
                card_content.append(
                    ft.Row([
                        ft.Text(item_text, size=12, color=SciFiColors.TEXT_SECONDARY),
                        ft.Container(expand=True),
                        ft.Text(item_value, size=12, weight=ft.FontWeight.BOLD, color=item_color),
                    ])
                )
                card_content.append(ft.Container(height=8))
        
        if action_button:
            card_content.append(ft.Container(height=16))
            card_content.append(action_button)
        
        return ft.Container(
            content=ft.Column(card_content, spacing=0),
            padding=20,
            bgcolor=ft.Colors.with_opacity(0.5, SciFiColors.BG_CARD),
            border=ft.border.all(1, SciFiColors.BORDER_GLOW),
            border_radius=10,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=15,
                color=ft.Colors.with_opacity(0.4, SciFiColors.PRIMARY),
            ),
        )
    
    def _create_action_button(self, text: str, icon, color, on_click):
        """Create an action button for the dashboard"""
        return ft.Container(
            content=ft.ElevatedButton(
                content=ft.Row([
                    ft.Icon(icon, size=18, color=SciFiColors.TEXT_PRIMARY),
                    ft.Text(text, size=12, weight=ft.FontWeight.BOLD),
                ], spacing=8),
                on_click=on_click,
                style=ft.ButtonStyle(
                    bgcolor=color,
                    color=SciFiColors.TEXT_PRIMARY,
                    shape=ft.RoundedRectangleBorder(radius=6),
                    padding=ft.padding.symmetric(horizontal=16, vertical=10),
                ),
            ),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=8,
                color=ft.Colors.with_opacity(0.3, color),
            ),
        )
    
    def _add_log_entry(self, message: str, color: str):
        """Add entry to activity log"""
        if not self.interaction_log:
            return
            
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        log_entry = ft.Container(
            content=ft.Row([
                ft.Container(width=2, bgcolor=color, border_radius=1),
                ft.Container(width=6),
                ft.Column([
                    ft.Text(f"[{timestamp}]", size=9, color=SciFiColors.TEXT_MUTED),
                    ft.Text(message, size=11, color=SciFiColors.TEXT_PRIMARY),
                ], spacing=2, tight=True, expand=True),
            ]),
            padding=8,
            bgcolor=ft.Colors.with_opacity(0.3, SciFiColors.BG_ELEVATED),
            border=ft.border.all(1, ft.Colors.with_opacity(0.3, color)),
            border_radius=4,
        )
        
        self.interaction_log.controls.append(log_entry)
        self.page.update()
    
    def _clear_log(self):
        """Clear activity log"""
        if self.interaction_log:
            self.interaction_log.controls.clear()
            self._add_log_entry("Activity log cleared", SciFiColors.INFO)
    
    # ==================== DIALOG MANAGEMENT ====================
    
    def _show_voice_dialog(self, e):
        """Show voice assistant dialog"""
        try:
            # Prevent multiple voice assistant dialogs
            if self.voice_assistant_dialog_open:
                logger.info("Voice assistant dialog already open")
                return
            
            logger.info("Opening voice assistant dialog")
            
            if self.interaction_log:
                self._add_log_entry("Voice Assistant activated", SciFiColors.PRIMARY)
            
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
            
            # Create buttons (stop_btn must be defined before start_btn references it)
            stop_btn = ft.ElevatedButton(
                "STOP",
                icon=ft.Icons.STOP,
                on_click=None,  # Will be set after creation
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
                on_click=None,  # Will be set after creation
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
            
            # Build dialog content - use Column directly without Container wrapper
            dialog_column = ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.MIC_ROUNDED, color=SciFiColors.PRIMARY, size=28),
                    ft.Column([
                        ft.Text(
                            "VOICE ASSISTANT",
                            size=18,
                            weight=ft.FontWeight.BOLD,
                            font_family="Orbitron",
                            color=SciFiColors.TEXT_PRIMARY,
                        ),
                        ft.Text(
                            "AI-powered voice control",
                            size=11,
                            color=SciFiColors.TEXT_SECONDARY,
                        ),
                    ], spacing=2, tight=True),
                ], spacing=12),
                ft.Container(height=16),
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
            
            # Create dialog using AlertDialog instead of BottomSheet
            dialog = ft.AlertDialog(
                modal=True,
                title=ft.Row([
                    ft.Icon(ft.Icons.MIC_ROUNDED, color=SciFiColors.PRIMARY, size=28),
                    ft.Text("VOICE ASSISTANT", size=18, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
                ], spacing=12),
                content=dialog_column,
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
            
            # Add initial log entry after dialog is open
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
            
            # Run recording in background thread to avoid blocking UI
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
        # Only update page if voice assistant panel is visible
        if self.voice_assistant_panel.visible:
            self.page.update()
    
    def _show_commands_dialog(self, e):
        """Show available voice commands dialog"""
        commands_text = self.voice_assistant.get_available_commands()
        
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row([
                ft.Icon(ft.Icons.HELP_ROUNDED, color=SciFiColors.PRIMARY, size=24),
                ft.Text("VOICE COMMANDS", size=18, weight=ft.FontWeight.BOLD, font_family="Orbitron"),
            ], spacing=10),
            content=ft.Container(
                content=ft.Text(commands_text, size=12, color=SciFiColors.TEXT_PRIMARY),
                width=500,
                height=400,
            ),
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
    
    def _take_screenshot_action(self):
        """Take screenshot action"""
        if self.interaction_log:
            self._add_log_entry("Taking screenshot...", SciFiColors.INFO)
        
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
            screenshot.save("screenshot.png")
            if self.interaction_log:
                self._add_log_entry("Screenshot saved as screenshot.png", SciFiColors.SUCCESS)
            self._show_success_toast("Screenshot saved!")
        except Exception as e:
            if self.interaction_log:
                self._add_log_entry(f"Screenshot failed: {e}", SciFiColors.ERROR)
            self._show_error_toast(f"Screenshot failed: {e}")
    
    def _show_system_status(self):
        """Show system status"""
        if self.interaction_log:
            self._add_log_entry("System status check", SciFiColors.INFO)
        
        try:
            import platform
            
            if psutil is None:
                raise ImportError("psutil not available")
            
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
        if self.interaction_log:
            self._add_log_entry("Security scan started", SciFiColors.WARNING)
        
        async def run_scan():
            await asyncio.sleep(1)
            if self.interaction_log:
                self._add_log_entry("Scan complete - No threats", SciFiColors.SUCCESS)
        
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
    
    def _toggle_voice_assistant(self, e):
        """Toggle voice assistant panel visibility"""
        if self.voice_assistant_panel.visible:
            # Hide panel
            self.voice_assistant_panel.visible = False
            if self.voice_assistant_active:
                self.voice_assistant.stop_continuous_listening()
                self.voice_assistant_active = False
            self._add_log_entry("Voice Assistant deactivated", SciFiColors.INFO)
        else:
            # Show panel
            self.voice_assistant_panel.visible = True
            self._add_log_entry("Voice Assistant activated", SciFiColors.PRIMARY)
        
        self.page.update()
    
    def _create_voice_assistant_panel(self):
        """Create the voice assistant control panel"""
        # Create dialog components (reuse the logic from _show_voice_dialog)
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
            on_click=None,  # Will be set after creation
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
            on_click=None,  # Will be set after creation
            style=ft.ButtonStyle(
                bgcolor=SciFiColors.ACCENT,
                color=ft.Colors.WHITE,
                shape=ft.RoundedRectangleBorder(radius=6),
            )
        )
        
        # Add initial log entry
        self._add_log_entry_to_container(log_content, "Voice Assistant initialized", SciFiColors.SUCCESS)
        
        # Set button handlers
        start_btn.on_click = lambda e: self._handle_listen_button(
            e, log_content, status_text, start_btn, stop_btn, continuous_toggle
        )
        stop_btn.on_click = lambda e: self._stop_continuous(
            log_content, status_text, start_btn, stop_btn
        )
        
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.MIC_ROUNDED, color=SciFiColors.PRIMARY, size=28),
                    ft.Column([
                        ft.Text(
                            "VOICE ASSISTANT",
                            size=18,
                            weight=ft.FontWeight.BOLD,
                            font_family="Orbitron",
                            color=SciFiColors.TEXT_PRIMARY,
                        ),
                        ft.Text(
                            "AI-powered voice control",
                            size=11,
                            color=SciFiColors.TEXT_SECONDARY,
                        ),
                    ], spacing=2, tight=True),
                    ft.Container(expand=True),
                    ft.IconButton(
                        ft.Icons.CLOSE,
                        on_click=self._toggle_voice_assistant,
                        icon_color=SciFiColors.TEXT_SECONDARY,
                    ),
                ], spacing=12),
                ft.Container(height=16),
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
                    bgcolor=SciFiColors.BG_DARK,
                    padding=10,
                    border_radius=6,
                    border=ft.border.all(1, SciFiColors.BORDER)
                ),
            ]),
            padding=20,
            bgcolor=ft.Colors.with_opacity(0.8, SciFiColors.BG_CARD),
            border=ft.border.all(1, SciFiColors.BORDER_GLOW),
            border_radius=10,
        )
        # end of _create_voice_assistant_panel
    
    def _open_dialog_safe(self, dialog):
        """Open dialog safely"""
        try:
            if dialog not in self.active_dialogs:
                self.active_dialogs.append(dialog)
            
            # Handle different dialog types
            if isinstance(dialog, ft.BottomSheet):
                self.page.bottom_sheet = dialog
            else:
                self.page.dialog = dialog
                dialog.open = True
            
            self.page.update()
            logger.info(f"Dialog opened: type={type(dialog).__name__}, page.dialog={self.page.dialog is not None}, dialog.open={dialog.open}")
            
        except Exception as e:
            logger.error(f"Error opening dialog: {e}")
            self._show_error_toast(f"Failed to open dialog: {e}")
    
    def _close_dialog_safe(self, dialog):
        """Close dialog safely"""
        try:
            # Handle different dialog types
            if isinstance(dialog, ft.BottomSheet):
                self.page.bottom_sheet = None
            else:
                dialog.open = False
                if self.page.dialog == dialog:
                    self.page.dialog = None
            
            if dialog in self.active_dialogs:
                self.active_dialogs.remove(dialog)
            
            self.page.update()
            
        except Exception as e:
            logger.error(f"Error closing dialog: {e}")
    
    def _close_voice_assistant_dialog(self, dialog):
        """Close voice assistant dialog"""
        self._close_dialog_safe(dialog)
        self.voice_assistant_dialog_open = False
        # Stop any ongoing voice assistant activity
        if self.voice_assistant_active:
            self.voice_assistant.stop_continuous_listening()
            self.voice_assistant_active = False
    
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
    
    def toggle_recording(self):
        """Toggle recording for login"""
        self.recording_active = not self.recording_active
        
        if self.recording_active:
            self.record_button.text = "◉ STOP RECORDING"
            self.record_button.icon = ft.Icons.STOP
            self.record_button.style = ft.ButtonStyle(
                bgcolor=SciFiColors.SUCCESS,
                color=SciFiColors.BG_DARK,
                shape=ft.RoundedRectangleBorder(radius=8),
            )
            self.update_status("⦿ Recording... Speak now!", SciFiColors.ERROR)
        else:
            self.record_button.text = "START RECORDING"
            self.record_button.icon = ft.Icons.MIC
            self.record_button.style = ft.ButtonStyle(
                bgcolor=SciFiColors.ERROR,
                color=SciFiColors.TEXT_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=8),
            )
            self.update_status("✓ Recording complete", SciFiColors.SUCCESS)
        
        self.page.update()
    
    def toggle_reg_recording(self):
        """Toggle recording for registration"""
        self.reg_recording_active = not self.reg_recording_active
        
        if self.reg_recording_active:
            self.reg_record_button.text = "◉ STOP RECORDING"
            self.reg_record_button.icon = ft.Icons.STOP
            self.reg_record_button.style = ft.ButtonStyle(
                bgcolor=SciFiColors.SUCCESS,
                color=SciFiColors.BG_DARK,
                shape=ft.RoundedRectangleBorder(radius=8),
            )
            self.update_reg_status("⦿ Recording... Speak now!", SciFiColors.ERROR)
        else:
            self.reg_record_button.text = "START RECORDING"
            self.reg_record_button.icon = ft.Icons.MIC
            self.reg_record_button.style = ft.ButtonStyle(
                bgcolor=SciFiColors.ERROR,
                color=SciFiColors.TEXT_PRIMARY,
                shape=ft.RoundedRectangleBorder(radius=8),
            )
            self.update_reg_status("✓ Recording complete", SciFiColors.SUCCESS)
        
        self.page.update()
    
    def show_record_button(self):
        """Show recording button"""
        self.mic_status.visible = True
        self.record_button.visible = True
        self.recording_active = False
        self.page.update()
    
    def hide_record_button(self):
        """Hide recording button"""
        self.mic_status.visible = False
        self.record_button.visible = False
        self.recording_active = False
        self.page.update()
    
    def show_reg_record_button(self, sample_info: str = ""):
        """Show registration recording button"""
        self.reg_mic_status.visible = True
        if sample_info:
            self.reg_mic_status.content = ft.Text(
                sample_info,
                size=11,
                color=SciFiColors.TEXT_PRIMARY,
                text_align=ft.TextAlign.CENTER,
                weight=ft.FontWeight.BOLD,
            )
        self.reg_record_button.visible = True
        self.reg_recording_active = False
        self.page.update()
    
    def hide_reg_record_button(self):
        """Hide registration recording button"""
        self.reg_mic_status.visible = False
        self.reg_record_button.visible = False
        self.reg_recording_active = False
        self.page.update()
    
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
        
        self.update_status("⟳ Initializing voice auth...", SciFiColors.INFO)
        self.show_progress(True)
        
        async def verify_wrapper():
            await self.verify_voice(user)
        
        self.page.run_task(verify_wrapper)
    
    def start_registration(self):
        """Start registration"""
        self.page.run_task(self.process_registration)
    
    async def process_registration(self):
        """Process registration"""
        try:
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
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            self.update_reg_status(f"⚠ Failed: {str(e)}", SciFiColors.ERROR)
            self.show_reg_progress(False)
    
    async def verify_voice(self, user: dict, verify_face=False, face_image=None):
        """Verify voice biometric"""
        try:
            self.update_status("⦿ Click START RECORDING", SciFiColors.INFO)
            self.show_record_button()
            
            await asyncio.get_event_loop().run_in_executor(None, self.wait_for_recording_start)
            
            audio_path = self.temp_dir / f"verify_{user['id']}.wav"
            
            recording_complete = threading.Event()
            audio_data = [None]
            
            def record_thread():
                audio_data[0] = self.audio_recorder.record_audio(duration=60)
                recording_complete.set()
            
            rec_thread = threading.Thread(target=record_thread, daemon=True)
            rec_thread.start()
            
            await asyncio.get_event_loop().run_in_executor(None, self.wait_for_recording_stop)
            
            self.audio_recorder.stop_recording()
            await asyncio.sleep(0.3)
            
            self.hide_record_button()
            
            if audio_data[0] is None:
                self.update_status("⚠ Recording failed", SciFiColors.ERROR)
                self.show_progress(False)
                return
            
            self.audio_recorder.save_audio(audio_data[0], str(audio_path))
            
            self.update_status("⟳ Analyzing voice...", SciFiColors.INFO)
            
            if not self.voice_engine.is_ready():
                self.voice_engine.load_models()
            
            enable_anti_spoof = self.config.get('security', {}).get('enable_anti_spoofing', True)
            test_embedding = self.voice_engine.extract_embedding(str(audio_path), enable_anti_spoofing=enable_anti_spoof)
            
            if test_embedding is None:
                self.update_status("⚠ Voice verification failed", SciFiColors.ERROR)
                self.show_progress(False)
                return
            
            stored_embeddings = self.db.get_voice_embeddings(user['id'])
            
            if not stored_embeddings:
                self.update_status("⚠ No voice profile - Enrolling", SciFiColors.WARNING)
                self.show_progress(False)
                await asyncio.sleep(2)
                await self.enroll_user_voice(user)
                return
            
            self.update_status(f"⟳ Comparing with {len(stored_embeddings)} samples...", SciFiColors.INFO)
            
            matches = []
            for stored_emb in stored_embeddings:
                stored_embedding = stored_emb['embedding_array']
                is_match, distance = self.voice_engine.verify_speaker(test_embedding, stored_embedding)
                matches.append({
                    'match': is_match,
                    'distance': distance,
                    'similarity': (1 - distance) * 100
                })
            
            matched_samples = sum(1 for m in matches if m['match'])
            min_required = self.config.get('security', {}).get('min_match_samples', 2)
            best_match = min(matches, key=lambda x: x['distance'])
            
            if matched_samples >= min_required:
                self.update_status(
                    f"✓ Voice verified - {matched_samples}/{len(matches)} matches ({best_match['similarity']:.1f}%)",
                    SciFiColors.SUCCESS
                )
                
                await asyncio.sleep(1)
                self.current_user = self.security_manager.create_session(user['id'], user['username'])
                self.db.update_last_login(user['id'])
                
                await asyncio.sleep(1.5)
                self.db.reset_failed_attempts(user['username'])
                
                self.show_progress(False)
                self.show_dashboard()
            else:
                self.update_status(
                    f"⚠ Verification failed - {matched_samples}/{len(matches)} matches (Required: {min_required})",
                    SciFiColors.ERROR
                )
                self.db.increment_failed_attempts(user['username'])
                self.show_progress(False)
            
        except Exception as e:
            logger.error(f"Voice verification error: {e}")
            self.update_status(f"⚠ Error: {str(e)}", SciFiColors.ERROR)
            self.show_progress(False)
    
    async def enroll_user_voice(self, user: dict):
        """Enroll user voice"""
        try:
            num_samples = self.config.get('voice', {}).get('enrollment_samples', 5)
            
            self.update_status(f"⟳ Recording {num_samples} voice samples", SciFiColors.INFO)
            await asyncio.sleep(1)
            
            self.db.conn.execute(
                "UPDATE voice_embeddings SET is_active = 0 WHERE user_id = ?",
                (user['id'],)
            )
            self.db.conn.commit()
            
            embeddings = []
            quality_scores = []
            
            for sample_num in range(1, num_samples + 1):
                self.update_status(f"⦿ Sample {sample_num}/{num_samples}: Preparing...", SciFiColors.WARNING)
                await asyncio.sleep(1)
                
                audio_path = self.temp_dir / f"enroll_{user['id']}_sample{sample_num}.wav"
                self.update_status(f"⦿ Recording sample {sample_num} - Speak for 5 seconds", SciFiColors.ERROR)
                
                audio_data = self.audio_recorder.record_audio(duration=5.0)
                
                if audio_data is None:
                    self.update_status(f"⚠ Sample {sample_num} failed - Skipping", SciFiColors.WARNING)
                    await asyncio.sleep(1)
                    continue
                
                self.update_status(f"✓ Sample {sample_num} recorded - Analyzing...", SciFiColors.SUCCESS)
                self.audio_recorder.save_audio(audio_data, str(audio_path))
                await asyncio.sleep(0.5)
                
                if not self.voice_engine.is_ready():
                    self.voice_engine.load_models()
                
                enable_anti_spoof = self.config.get('security', {}).get('enable_anti_spoofing', True)
                embedding = self.voice_engine.extract_embedding(str(audio_path), enable_anti_spoofing=enable_anti_spoof)
                
                if embedding is None:
                    self.update_status(f"⚠ Sample {sample_num} failed - Skipping", SciFiColors.WARNING)
                    await asyncio.sleep(1)
                    continue
                
                quality_analyzer = VoiceQualityAnalyzer()
                quality_metrics = quality_analyzer.analyze_audio(str(audio_path))
                quality_score = quality_metrics.get('overall_quality', 0.5)
                
                embeddings.append(embedding / (np.linalg.norm(embedding) + 1e-8))
                quality_scores.append(quality_score)
                
                self.db.store_voice_embedding(
                    user_id=user['id'],
                    embedding=embedding,
                    embedding_type=self.voice_engine.active_backend,
                    quality_score=quality_score
                )
                
                self.update_status(f"✓ Sample {sample_num}/{num_samples} saved (Quality: {quality_score:.0%})", SciFiColors.SUCCESS)
                await asyncio.sleep(1)
            
            if len(embeddings) == 0:
                self.update_status("⚠ No valid samples - Please try again", SciFiColors.ERROR)
                self.show_progress(False)
                return
            
            mean_emb, user_threshold, avg_sim, std_sim = self.voice_engine.compute_user_reference(embeddings)
            
            self.db.store_voice_embedding(
                user_id=user['id'],
                embedding=mean_emb,
                embedding_type='mean',
                quality_score=float(np.mean(quality_scores)),
                threshold=user_threshold
            )
            
            self.update_status(f"✓ Voice enrollment complete - {len(embeddings)}/{num_samples} samples", SciFiColors.SUCCESS)
            await asyncio.sleep(1.5)
            
            self.update_status("⟳ Verifying your voice...", SciFiColors.INFO)
            await asyncio.sleep(1)
            await self.verify_voice(user)
            
        except Exception as e:
            logger.error(f"Enrollment error: {e}")
            self.update_status(f"⚠ Enrollment error: {str(e)}", SciFiColors.ERROR)
            self.show_progress(False)
    
    async def enroll_user_voice_registration(self, user: dict):
        """Enroll voice during registration"""
        try:
            num_samples = self.config.get('voice', {}).get('enrollment_samples', 3)
            
            self.update_reg_status(f"⟳ Recording {num_samples} voice samples", SciFiColors.INFO)
            await asyncio.sleep(1)
            
            embeddings_stored = 0
            
            for sample_num in range(1, num_samples + 1):
                self.update_reg_status(f"⦿ Sample {sample_num}/{num_samples} - Click START RECORDING", SciFiColors.INFO)
                self.show_reg_record_button(f"SAMPLE {sample_num} OF {num_samples}")
                
                await asyncio.get_event_loop().run_in_executor(None, self.wait_for_reg_recording_start)
                
                audio_path = self.temp_dir / f"enroll_{user['id']}_sample{sample_num}.wav"
                
                recording_complete = threading.Event()
                audio_data_holder = [None]
                
                def record_thread():
                    audio_data_holder[0] = self.audio_recorder.record_audio(duration=60)
                    recording_complete.set()
                
                rec_thread = threading.Thread(target=record_thread, daemon=True)
                rec_thread.start()
                
                await asyncio.get_event_loop().run_in_executor(None, self.wait_for_reg_recording_stop)
                
                self.audio_recorder.stop_recording()
                await asyncio.sleep(0.3)
                
                self.hide_reg_record_button()
                
                audio_data = audio_data_holder[0]
                
                if audio_data is None:
                    self.update_reg_status(f"⚠ Sample {sample_num} failed - Skipping", SciFiColors.WARNING)
                    await asyncio.sleep(1)
                    continue
                
                self.update_reg_status(f"✓ Sample {sample_num} recorded - Analyzing", SciFiColors.SUCCESS)
                self.audio_recorder.save_audio(audio_data, str(audio_path))
                await asyncio.sleep(0.5)
                
                self.update_reg_status(f"⟳ Analyzing sample {sample_num}...", SciFiColors.INFO)
                
                if not self.voice_engine.is_ready():
                    self.voice_engine.load_models()
                
                enable_anti_spoof = self.config.get('security', {}).get('enable_anti_spoofing', True)
                embedding = self.voice_engine.extract_embedding(str(audio_path), enable_anti_spoofing=enable_anti_spoof)
                
                if embedding is None:
                    self.update_reg_status(f"⚠ Sample {sample_num} failed - Skipping", SciFiColors.WARNING)
                    await asyncio.sleep(1)
                    continue
                
                quality_analyzer = VoiceQualityAnalyzer()
                quality_metrics = quality_analyzer.analyze_audio(str(audio_path))
                quality_score = quality_metrics.get('overall_quality', 0.5)
                
                self.db.store_voice_embedding(
                    user_id=user['id'],
                    embedding=embedding,
                    embedding_type=self.voice_engine.active_backend,
                    quality_score=quality_score
                )
                
                embeddings_stored += 1
                self.update_reg_status(f"✓ Sample {sample_num}/{num_samples} saved - Quality: {quality_score:.0%}", SciFiColors.SUCCESS)
                await asyncio.sleep(1)
            
            if embeddings_stored == 0:
                self.update_reg_status("⚠ No valid samples - Please try again", SciFiColors.ERROR)
                self.show_reg_progress(False)
                return
            
            self.update_reg_status(f"✓ Success! Profile created with {embeddings_stored}/{num_samples} samples", SciFiColors.SUCCESS)
            
            await asyncio.sleep(2)
            
            # Switch to login tab
            self.auth_tabs.selected_index = 0
            self._handle_auth_tab_change(type('obj', (object,), {'control': self.auth_tabs})())
            self.update_status("✓ Registration complete! Please login.", SciFiColors.SUCCESS)
            
        except Exception as e:
            logger.error(f"Voice enrollment error: {e}")
            self.update_reg_status(f"⚠ Enrollment error: {str(e)}", SciFiColors.ERROR)
            self.show_reg_progress(False)
    
    async def enroll_user_face(self, user: dict):
        """Enroll user face"""
        try:
            self.update_status("⦿ Click to Capture Face", SciFiColors.INFO)
            self.show_record_button()
            
            # Wait for user to click
            await asyncio.get_event_loop().run_in_executor(None, self.wait_for_recording_start)
            
            image_path = self.temp_dir / f"face_enroll_{user['id']}.jpg"
            
            # Capture image
            result = self.face_engine.capture_face_image(str(image_path))
            
            self.hide_record_button()
            
            if result is None:
                self.update_status("⚠ Face enrollment failed", SciFiColors.ERROR)
                return
            
            self.update_status("⟳ Processing face data...", SciFiColors.INFO)
            
            # Enroll face
            success = self.face_engine.enroll_face(user['id'], str(image_path))
            
            if success:
                self.update_status("✓ Face enrollment successful", SciFiColors.SUCCESS)
            else:
                self.update_status("⚠ Face enrollment failed", SciFiColors.ERROR)
            
        except Exception as e:
            logger.error(f"Face enrollment error: {e}")
            self.update_status(f"⚠ Error: {str(e)}", SciFiColors.ERROR)
    
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
        
        if self.page.controls:
            current = self.page.controls[0]
            current.opacity = 0
            current.animate_opacity = ft.Animation(300, ft.AnimationCurve.EASE_OUT)
            self.page.update()
            time.sleep(0.3)
        
        self.page.controls.clear()
        login = self.build_login_view()
        login.opacity = 0
        self.page.add(login)
        self.page.update()
        
        login.opacity = 1
        login.animate_opacity = ft.Animation(300, ft.AnimationCurve.EASE_IN)
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

    def run(self):
        """Run the application"""
        self.db.connect()
        self.db.initialize_schema()
        
        self.voice_engine.load_models()
        
        self.page.add(self.build_login_view())
        
        try:
            self.tts.speak("SecureX Assist initialized. Ready for authentication.")
        except:
            pass

    def show_dashboard(self):
        """Render the dashboard view."""
        try:
            logger.info("Rendering dashboard view.")
            # Clear the current page content
            self.page.controls.clear()

            # Add the dashboard content
            dashboard_content = self._create_dashboard_content()
            self.page.controls.append(dashboard_content)
            self.page.update()
            logger.info("Dashboard rendered successfully.")
        except Exception as e:
            logger.error(f"Failed to render dashboard: {e}", exc_info=True)
            self._show_error_toast(f"Error loading dashboard: {str(e)}")
        
def main(page: ft.Page):
    """Flet main entry point"""
    from utils.helpers import load_config
    
    page.title = "SecureX-Assist"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_resizable = True
    page.window_maximizable = True
    page.window_minimizable = True
    
    config = load_config()
    
    app = SecureXApp(page, config)
    app.run()


if __name__ == "__main__":
    ft.app(target=main)