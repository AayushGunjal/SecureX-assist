# SecureX-Assist - Offline Voice Biometric Authentication

A production-grade offline voice biometric authentication system with anti-spoofing protection.

## Features

- SpeechBrain ECAPA-TDNN for speaker embeddings
- AASIST anti-spoofing for replay/deepfake detection
- Advanced audio preprocessing
- SQLite database for voiceprints
- Flet-based UI

## Requirements

- Python 3.12+
- Microphone access
- 8GB RAM (16GB+ recommended)

## Installation

1. Clone the repo
2. Create venv: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`
5. Run: `python main.py`

## Usage

Run the application and follow the UI for enrollment and verification.
