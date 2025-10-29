#!/usr/bin/env python3
"""
Test enrollment process
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.voice_biometric_engine_ultimate import UltimateVoiceBiometricEngine
from core.database import Database
from core.audio_processor import AudioProcessor
import yaml
import numpy as np

def test_enrollment():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    db = Database()
    db.connect()
    db.initialize_schema()

    engine = UltimateVoiceBiometricEngine(config, db)

    # Create test user
    user_id = db.create_user("test_enroll", "hash", "test@example.com")
    print(f"Created test user ID: {user_id}")

    # Create some test audio (simulated recording)
    # Generate 3 seconds of fake voice-like audio
    sample_rate = 16000
    duration = 3
    samples = int(sample_rate * duration)

    # Create simple sine wave with some noise (simulating voice)
    t = np.linspace(0, duration, samples)
    freq = 200  # Fundamental frequency
    audio = 0.1 * np.sin(2 * np.pi * freq * t)  # Fundamental
    for i in range(2, 6):  # Add harmonics
        audio += 0.05/i * np.sin(2 * np.pi * freq * i * t)
    audio += 0.01 * np.random.randn(samples)  # Add noise

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.5

    # Convert to 2D as the recorder does
    audio_2d = audio.reshape(-1, 1)

    # Apply same processing as UI
    audio_processed = AudioProcessor.apply_automatic_gain_control(audio_2d, target_level=0.15)
    # Skip noise reduction for test
    audio_processed = audio_2d

    print(f"Audio shape: {audio_processed.shape}")
    print(f"Audio RMS: {np.sqrt(np.mean(audio_processed ** 2)):.4f}")

    # Test enrollment with 3 samples
    success = engine.enroll_user_voice(
        user_id=user_id,
        audio_samples=[audio_processed, audio_processed, audio_processed],
        sample_rate=sample_rate
    )

    print(f"Enrollment success: {success}")

    # Check if embeddings were stored
    embeddings = db.get_voice_embeddings(user_id)
    print(f"Stored embeddings: {len(embeddings)}")

    if embeddings:
        print("Enrollment test PASSED")
    else:
        print("Enrollment test FAILED")

    # Cleanup
    db.delete_user(user_id)
    db.close()

if __name__ == "__main__":
    test_enrollment()