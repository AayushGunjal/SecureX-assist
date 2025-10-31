#!/usr/bin/env python3
"""
Test script to verify voice verification speed optimization
"""

import time
import numpy as np
from core.voice_biometric_engine_ultimate import UltimateVoiceBiometricEngine
from utils.helpers import load_config
from core.database import Database

def test_verification_speed():
    """Test verification speed with optimizations"""
    print("🔬 Testing Voice Verification Speed Optimization")
    print("=" * 60)

    # Load config and initialize components
    config = load_config()
    db = Database(config.get('database', {}).get('path', 'securex_db.sqlite'))
    engine = UltimateVoiceBiometricEngine(config, db)

    # Check if user exists
    users = db.get_all_users()
    if not users:
        print("❌ No users found in database. Please enroll a user first.")
        return False

    user_id = users[0]['id']
    username = users[0]['username']
    print(f"📋 Testing verification for user: {username} (ID: {user_id})")

    # Create test audio (simulate recorded voice)
    sample_rate = 16000
    duration = 2.5  # seconds
    samples = int(sample_rate * duration)

    # Generate synthetic voice-like audio (not real voice, but for testing)
    np.random.seed(42)  # For reproducible results
    audio_data = np.random.normal(0, 0.1, samples).astype(np.float32)

    # Add some voice-like characteristics
    # Simulate fundamental frequency around 85-255 Hz for male/female voice
    t = np.linspace(0, duration, samples)
    f0 = 120  # Fundamental frequency
    audio_data += 0.3 * np.sin(2 * np.pi * f0 * t)

    # Add some harmonics
    audio_data += 0.1 * np.sin(2 * np.pi * 2 * f0 * t)
    audio_data += 0.05 * np.sin(2 * np.pi * 3 * f0 * t)

    print(f"🎤 Test audio: {duration}s, {samples} samples, {sample_rate}Hz")

    # Test verification speed
    print("\n⚡ Testing verification speed...")
    start_time = time.time()

    result = engine.verify_voice(
        user_id=user_id,
        audio_data=audio_data,
        sample_rate=sample_rate
    )

    end_time = time.time()
    verification_time = end_time - start_time

    print(f"   ⏱️  Time: {verification_time:.2f}s")
    print(f"   ✅ Verified: {result['verified']}")
    print(f"   🎯 Confidence: {result['confidence']:.2f}")
    print(f"   🎯 Spoof detected: {result['spoof_detected']}")

    # Performance assessment
    print("\n📊 Performance Assessment:")
    if verification_time < 1.0:
        print("   ✅ EXCELLENT: Verification completed in under 1 second")
        speed_rating = "Excellent"
    elif verification_time < 3.0:
        print("   ✅ GOOD: Verification completed in under 3 seconds")
        speed_rating = "Good"
    elif verification_time < 5.0:
        print("   ⚠️  ACCEPTABLE: Verification completed in under 5 seconds")
        speed_rating = "Acceptable"
    else:
        print("   ❌ SLOW: Verification took too long")
        speed_rating = "Slow"

    # Accuracy assessment
    accuracy_rating = "Unknown"
    if result['verified']:
        if result['confidence'] > 0.7:
            print("   ✅ HIGH ACCURACY: Strong verification confidence")
            accuracy_rating = "High"
        elif result['confidence'] > 0.5:
            print("   ✅ MEDIUM ACCURACY: Moderate verification confidence")
            accuracy_rating = "Medium"
        else:
            print("   ⚠️  LOW ACCURACY: Weak verification confidence")
            accuracy_rating = "Low"
    else:
        print("   ❌ NOT VERIFIED: Voice verification failed")
        accuracy_rating = "Failed"

    print("\n🎯 Overall Rating:")
    print(f"   Speed: {speed_rating}")
    print(f"   Accuracy: {accuracy_rating}")

    # Check if optimizations are working
    print("\n🔧 Optimization Status:")
    bypass_verify = config.get('system', {}).get('bypass_anti_spoofing_verify', False)
    bypass_general = config.get('system', {}).get('bypass_anti_spoofing', False)

    if bypass_verify or bypass_general:
        print("   ✅ Anti-spoofing bypass enabled for fast verification")
    else:
        print("   ⚠️  Anti-spoofing bypass disabled (slower but more secure)")

    if verification_time < 5.0:
        print("   ✅ Timeout optimizations working")
    else:
        print("   ❌ May need further optimization")

    print("\n" + "=" * 60)
    return verification_time < 5.0 and result['verified']

if __name__ == "__main__":
    success = test_verification_speed()
    exit(0 if success else 1)
