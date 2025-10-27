#!/usr/bin/env python3
"""
Test script for Voice Assistant functionality
"""

from core.voice_assistant import VoiceAssistant
import time

def test_voice_assistant():
    """Test voice assistant initialization and command processing"""
    print('🔊 Testing Voice Assistant initialization...')

    # Initialize voice assistant
    va = VoiceAssistant()
    va.setup_default_commands()

    print('✅ Voice Assistant initialized successfully')

    # Test command processing
    print('\n🗣️  Testing command processing...')
    test_commands = [
        'what time is it',
        'open calculator',
        'system status',
        'help',
        'lock system',
        'biometric status'
    ]

    for cmd in test_commands:
        success, response = va.process_voice_command(cmd)
        status = "✅" if success else "❌"
        print(f'{status} Command: "{cmd}" -> Response: "{response}"')

    # Test TTS (text-to-speech)
    print('\n🔊 Testing Text-to-Speech...')
    try:
        va.speak("Voice assistant test completed successfully!")
        print('✅ TTS test initiated')
    except Exception as e:
        print(f'❌ TTS test failed: {e}')

    print('\n🎉 Voice Assistant test completed!')

if __name__ == "__main__":
    test_voice_assistant()