# 🛡️ SecureAI Voice Assistant - Enhanced Custom Phrase Authentication

A powerful Windows voice assistant with advanced biometric authentication, custom phrase setup, and comprehensive system control capabilities.

## 🚀 Main Application

**File:** `secureai_voice_auth.py`  
**Status:** ✅ Complete and Ready to Use  

This is the **unified, complete solution** with all features integrated:
- 🎤 **Custom Phrase Authentication** - Set your own unique phrase
- 🔄 **Multi-Sample Training** - Record 5 voice samples for accuracy
- 🧠 **Advanced Voice Biometrics** - MFCC analysis and voice pattern recognition
- 🔐 **Password Fallback** - Secure backup authentication method
- 💻 **System Control** - Screenshots, media control, app launching
- 🌐 **Web Integration** - Search, weather, quick website access
- ⚡ **Optimized Performance** - Multi-threaded operations

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python secureai_voice_auth.py
```

### 3. First-Time Setup
1. **Set Custom Phrase**: Click "✏️ Set Custom Phrase"
   - Enter your personal authentication phrase
   - Examples: "Hello SecureAI, this is [Your Name] requesting access"
2. **Train Voice**: Click "🎙️ Train Voice (5 samples)"
   - Repeat your phrase 5 times with natural variations
3. **Set Password**: Click "⚙️ Set Password" for backup security

### 4. Daily Use
1. **Authenticate**: Click "🎤 Voice Authentication"
2. **Activate**: Click "🚀 Activate Assistant" 
3. **Use Voice Commands**: Click "👂 Start Listening"

## 🎤 Voice Authentication Features

- **📝 Custom Phrases**: Your own personalized authentication phrase
- **🔄 Environmental Adaptation**: Works in different rooms, noise levels, positions
- **📊 Similarity Analysis**: Real-time voice pattern matching with feedback
- **🎯 Smart Thresholds**: Dynamic accuracy requirements based on training quality
- **🔐 Secure Fallback**: Password authentication when voice fails

## 💻 System Capabilities

- **🎤 Voice Commands**: Natural language processing
- **🚀 App Launching**: Quick access to common applications
- **🌐 Web Integration**: Google, YouTube, Wikipedia searches
- **🎵 Media Control**: Volume, play/pause, mute controls
- **📸 System Control**: Screenshots, screen lock, system status
- **⚡ Performance**: Multi-threaded for smooth operation

## 📁 Legacy Files (Reference Only)

The following files are kept for reference but are **not needed** for the main application:

- `main.py` - Original voice assistant (basic)
- `alarm.py`, `volume.py`, `SearchNow.py`, etc. - Individual modules
- `AppOpening.py`, `Whatsapp.py` - Specific functionality modules
- `user_voiceprint.py` - Old voice authentication system

**Note**: All functionality from these files is integrated into `secureai_voice_auth.py`

## 🔧 Technical Details

### Voice Biometric Analysis
- **MFCC Features**: Mel-frequency cepstral coefficients for voice fingerprinting
- **Spectral Analysis**: Voice characteristic analysis
- **Environmental Tolerance**: Noise reduction and pattern normalization
- **Multi-Sample Training**: 5 samples ensure reliability across conditions

### Security Features
- **SHA-256 Password Hashing**: Secure password storage
- **Voice Pattern Encryption**: Biometric data protection
- **Multi-Layer Authentication**: Voice → Password → Access
- **Session Management**: Secure authentication state handling

### Performance Optimizations
- **Multi-Threading**: Non-blocking GUI and audio processing
- **Concurrent Processing**: Parallel voice analysis and system operations
- **Memory Efficient**: Optimized audio processing and feature extraction
- **Windows Integration**: Native API calls for system control

## 📋 Requirements

- **Python 3.8+**
- **Windows 10/11** (optimized for Windows)
- **Microphone** (for voice authentication and commands)
- **Internet Connection** (for web searches and weather)

## 🆘 Troubleshooting

### Voice Authentication Issues
- **Low Similarity**: Retrain voice with clearer pronunciation
- **Environmental Changes**: Record samples in different conditions
- **Microphone Quality**: Use a good quality microphone for training

### Installation Issues
- **librosa fails**: Try `conda install -c conda-forge librosa`
- **Audio issues**: Update audio drivers and check microphone permissions
- **GUI issues**: Ensure tkinter is available (usually built-in with Python)

## 🎉 Ready to Use!

The project is now **clean, optimized, and ready for production use**. Simply run `secureai_voice_auth.py` and enjoy your personalized voice assistant with advanced biometric security!

---

*SecureAI Voice Assistant v2.0 - Enhanced Custom Phrase Authentication*
This project aims to fill that gap by developing an AI-based voice assistant specifically for Windows. Utilizing Python and various libraries, the assistant will enable hands-free control for tasks such as opening applications, managing files, performing web searches, setting alarms, and ensuring secure access through voice authentication.
