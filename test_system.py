"""
System Test Script
Verify all components of SecureX-Assist are working correctly
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"🧪 {title}")
    print("=" * 60)

def test_imports():
    """Test if all required modules can be imported"""
    print_header("TESTING IMPORTS")
    
    modules = [
        ('flet', 'Flet UI Framework'),
        ('torch', 'PyTorch'),
        ('torchaudio', 'TorchAudio'),
        ('sounddevice', 'SoundDevice (audio capture)'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('pyttsx3', 'Text-to-Speech'),
        ('yaml', 'YAML Parser'),
    ]
    
    failed = []
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✅ {description:<30} [{module_name}]")
        except ImportError as e:
            print(f"❌ {description:<30} [{module_name}]")
            failed.append((module_name, str(e)))
    
    if failed:
        print("\n⚠️  Failed imports:")
        for module, error in failed:
            print(f"   {module}: {error}")
        return False
    
    print("\n✅ All required modules imported successfully")
    return True

def test_project_modules():
    """Test if project modules load correctly"""
    print_header("TESTING PROJECT MODULES")
    
    modules = [
        ('core.voice_engine', 'Voice Engine'),
        ('core.security', 'Security Manager'),
        ('core.database', 'Database'),
        ('core.audio_processor', 'Audio Processor'),
        ('utils.tts', 'Text-to-Speech'),
        ('utils.helpers', 'Helper Functions'),
    ]
    
    failed = []
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✅ {description:<30} [{module_name}]")
        except Exception as e:
            print(f"❌ {description:<30} [{module_name}]")
            failed.append((module_name, str(e)))
    
    if failed:
        print("\n⚠️  Failed to load:")
        for module, error in failed:
            print(f"   {module}: {error}")
        return False
    
    print("\n✅ All project modules loaded successfully")
    return True

def test_configuration():
    """Test configuration loading"""
    print_header("TESTING CONFIGURATION")
    
    try:
        from utils.helpers import load_config, load_environment
        
        # Load environment
        print("📁 Loading environment variables...")
        load_environment()
        
        # Check for .env file
        env_path = Path('.env')
        if env_path.exists():
            print("✅ .env file found")
        else:
            print("⚠️  .env file not found (using defaults)")
        
        # Load config
        print("📁 Loading configuration...")
        config = load_config()
        
        # Verify key sections
        required_sections = ['app', 'security', 'audio', 'models', 'database', 'ui', 'tts']
        for section in required_sections:
            if section in config:
                print(f"✅ Config section: {section}")
            else:
                print(f"❌ Missing config section: {section}")
                return False
        
        print("\n✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_database():
    """Test database operations"""
    print_header("TESTING DATABASE")
    
    try:
        from core.database import Database
        from utils.helpers import load_config
        
        config = load_config()
        
        print("📁 Connecting to database...")
        db = Database(config['database']['path'])
        db.connect()
        
        print("✅ Database connected")
        
        print("📁 Initializing schema...")
        db.initialize_schema()
        
        print("✅ Schema initialized")
        
        # Test query
        print("📁 Testing query...")
        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        print(f"✅ Found {count} users in database")
        
        db.close()
        print("\n✅ Database test passed")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_audio_devices():
    """Test audio device detection"""
    print_header("TESTING AUDIO DEVICES")
    
    try:
        import sounddevice as sd
        
        print("📁 Querying audio devices...")
        devices = sd.query_devices()
        
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if not input_devices:
            print("❌ No microphone devices found")
            print("   Check Windows Settings → Privacy → Microphone")
            return False
        
        print(f"\n✅ Found {len(input_devices)} microphone(s):")
        for i, device in enumerate(input_devices):
            print(f"   {i+1}. {device['name']}")
            print(f"      Channels: {device['max_input_channels']}")
            print(f"      Sample Rate: {device['default_samplerate']} Hz")
        
        print("\n✅ Audio device test passed")
        return True
        
    except Exception as e:
        print(f"❌ Audio device test failed: {e}")
        return False

def test_voice_engine():
    """Test voice engine initialization"""
    print_header("TESTING VOICE ENGINE")
    
    try:
        from core.voice_engine import VoiceEngine
        from utils.helpers import load_config
        
        config = load_config()
        
        print("📁 Initializing voice engine...")
        engine = VoiceEngine(config)
        print("✅ Voice engine created")
        
        print("📁 Loading AI models (this may take a moment)...")
        success = engine.load_models()
        
        if success:
            print(f"✅ Models loaded successfully")
            print(f"   Active backend: {engine.active_backend}")
            print("\n✅ Voice engine test passed")
            return True
        else:
            print("❌ Failed to load models")
            print("   Possible issues:")
            print("   - Hugging Face token not set in .env")
            print("   - Model licenses not accepted")
            print("   - Internet connection required for first download")
            return False
        
    except Exception as e:
        print(f"❌ Voice engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_to_speech():
    """Test text-to-speech system"""
    print_header("TESTING TEXT-TO-SPEECH")
    
    try:
        from utils.tts import TextToSpeech
        from utils.helpers import load_config
        
        config = load_config()
        
        print("📁 Initializing TTS engine...")
        tts = TextToSpeech(config)
        
        if tts.enabled and tts.engine:
            print("✅ TTS engine initialized")
            
            print("📁 Testing speech (you should hear this)...")
            tts.speak("SecureX Assist system test", blocking=True)
            
            print("✅ TTS test passed")
            return True
        else:
            print("⚠️  TTS disabled or failed to initialize")
            return True  # Not critical
        
    except Exception as e:
        print(f"⚠️  TTS test failed: {e}")
        print("   TTS is optional, continuing...")
        return True  # Non-critical

def run_all_tests():
    """Run all system tests"""
    print("\n" + "=" * 60)
    print("🔐 SECUREX-ASSIST - SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Project Modules", test_project_modules),
        ("Configuration", test_configuration),
        ("Database", test_database),
        ("Audio Devices", test_audio_devices),
        ("Voice Engine", test_voice_engine),
        ("Text-to-Speech", test_text_to_speech),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        print("\n📝 Next steps:")
        print("   1. Create users: python init_db.py")
        print("   2. Enroll voice: python enroll_voice.py")
        print("   3. Launch app: python main.py")
        return True
    else:
        print("\n⚠️  Some tests failed. Please fix issues before proceeding.")
        print("\n💡 Common solutions:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Set HF_TOKEN in .env file")
        print("   - Check microphone permissions")
        print("   - Accept model licenses on Hugging Face")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests cancelled by user")
        sys.exit(1)
