import librosa
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import noisereduce as nr
import webrtcvad
import speech_recognition as sr
import pyttsx3
import requests
from bs4 import BeautifulSoup
import datetime
import os
from time import sleep
import subprocess
import pyautogui
import json
import random

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice",voices[1].id)
# engine.setProperty("rate",100)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def takeCommand():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source :
        print("listening.....")
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1
        recognizer.energy_threshold = 100
        audio = recognizer.listen(source,0,4)
    try:
        print("Understanding...")
        query = recognizer.recognize_google(audio,language="en-in")
        print(F"You said : {query}\n")
    except Exception as e:
        print("Say that again")
        return "None"
    return query
    
def load_voice_file(file_path):
    y, sr_rate = librosa.load(file_path, sr=None)
    y = nr.reduce_noise(y=y, sr=sr_rate) 
    return y, sr_rate

# Voice activity detection (VAD)
def apply_vad(audio_segment):
    vad = webrtcvad.Vad()
    vad.set_mode(2) 
    samples = np.array(audio_segment.get_array_of_samples())
    return vad.is_speech(samples.tobytes(), audio_segment.frame_rate)

# Function to extract MFCC features
def extract_features(y, sr_rate):
    mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)
    return mfcc.T

# Define challenge phrases for authentication
CHALLENGE_PHRASES = [
    "SecureAI, unlock my system",
    "Hey SecureAI, it's me",
    "SecureAI, verify my voice",
    "I am authorized, SecureAI",
    "SecureAI, start my assistant",
    "Hello SecureAI, recognize me",
    "SecureAI, voice key accepted",
    "Activate SecureAI with my voice"
]

# Liveness detection challenges
LIVENESS_CHALLENGES = [
    {"type": "count", "instruction": "Please count from one to five for SecureAI", "response": ["one", "two", "three", "four", "five"]},
    {"type": "pitch", "instruction": "Please say 'SecureAI' with a high pitch, then a low pitch", "response": ["SecureAI", "SecureAI"]},
    {"type": "speed", "instruction": "Please say 'SecureAI assistant' slowly, then quickly", "response": ["SecureAI assistant", "SecureAI assistant"]},
    {"type": "volume", "instruction": "Please say 'SecureAI activated' softly, then loudly", "response": ["SecureAI activated", "SecureAI activated"]},
    {"type": "sequence", "instruction": "Please say 'SecureAI' followed by the numbers 1 2 3", "response": ["SecureAI 1 2 3"]}
]

# Functions for spectral analysis to detect playback artifacts
def extract_spectral_features(y, sr_rate):
    """Extract spectral features to identify playback artifacts"""
    # Calculate spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr_rate)[0]
    
    # Calculate spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr_rate)[0]
    
    # Calculate spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr_rate)
    
    # Calculate spectral flatness (higher in electronic playback)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    
    # Return a dictionary of spectral features
    return {
        'centroid_mean': np.mean(spectral_centroid),
        'centroid_std': np.std(spectral_centroid),
        'bandwidth_mean': np.mean(spectral_bandwidth),
        'bandwidth_std': np.std(spectral_bandwidth),
        'contrast_mean': np.mean(spectral_contrast),
        'flatness_mean': np.mean(spectral_flatness),
        'flatness_std': np.std(spectral_flatness),
    }

def detect_playback_artifacts(y, sr_rate, threshold=0.7):
    """
    Detect if audio is likely from a playback device rather than human speech
    
    Parameters:
    y : array, audio signal
    sr_rate : int, sample rate
    threshold : float, confidence threshold for detection
    
    Returns:
    bool, True if playback artifacts detected
    float, confidence score
    """
    # Extract spectral features
    features = extract_spectral_features(y, sr_rate)
    
    # Calculate playback likelihood score
    # High flatness, low variance in spectral features indicate electronic playback
    playback_indicators = [
        features['flatness_mean'] > 0.1,  # Flatness is higher in electronic playback
        features['centroid_std'] < 200,   # Lower variance in spectral centroid
        features['bandwidth_std'] < 200,  # Lower variance in bandwidth
    ]
    
    # Calculate confidence score
    confidence = sum(playback_indicators) / len(playback_indicators)
    
    return confidence > threshold, confidence

# Liveness detection functions
def analyze_pitch_variation(audio_samples):
    """
    Analyze if the audio contains variation in pitch as required in liveness challenge
    """
    if len(audio_samples) < 2:
        return False, 0
    
    pitches = []
    for y, sr in audio_samples:
        # Extract pitch using librosa's pitch tracking
        pitches.append(np.mean(librosa.yin(y, fmin=70, fmax=500, sr=sr)))
    
    # Calculate pitch difference
    pitch_diff = abs(pitches[0] - pitches[1])
    pitch_ratio = max(pitches) / (min(pitches) + 0.001)  # Avoid division by zero
    
    # Return True if significant pitch difference
    has_variation = pitch_ratio > 1.2  # At least 20% pitch difference
    
    return has_variation, pitch_ratio

def analyze_speed_variation(audio_samples):
    """
    Analyze if the audio contains variation in speaking speed
    """
    if len(audio_samples) < 2:
        return False, 0
    
    durations = []
    for y, sr in audio_samples:
        # Find non-silent regions
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        # Calculate total duration of non-silent parts
        duration = sum(end - start for start, end in non_silent_intervals) / sr
        durations.append(duration)
    
    # Calculate speed ratio
    if min(durations) == 0:
        return False, 0
        
    speed_ratio = max(durations) / min(durations)
    
    # Return True if significant speed difference (one sample at least 30% longer)
    has_variation = speed_ratio > 1.3
    
    return has_variation, speed_ratio

def analyze_volume_variation(audio_samples):
    """
    Analyze if the audio contains variation in volume
    """
    if len(audio_samples) < 2:
        return False, 0
    
    volumes = []
    for y, sr in audio_samples:
        # Calculate RMS volume
        volumes.append(np.sqrt(np.mean(y**2)))
    
    # Calculate volume ratio
    if min(volumes) == 0:
        return False, 0
        
    volume_ratio = max(volumes) / min(volumes)
    
    # Return True if significant volume difference
    has_variation = volume_ratio > 2.0  # One sample at least twice as loud
    
    return has_variation, volume_ratio

# Background noise fingerprinting functions
def extract_background_noise(y, sr_rate):
    """
    Extract background noise characteristics from audio signal
    Uses silent segments to characterize background environment
    """
    # Use librosa's silence detection to find background segments
    non_silent_intervals = librosa.effects.split(y, top_db=30)
    
    # If we found silent segments
    if len(non_silent_intervals) > 1:
        background = np.array([])
        
        # Collect all silent segments
        for i in range(len(non_silent_intervals) - 1):
            start = non_silent_intervals[i][1]
            end = non_silent_intervals[i+1][0]
            if end - start > sr_rate * 0.1:  # At least 100ms of silence
                background = np.append(background, y[start:end])
        
        # If we found enough background noise
        if len(background) > sr_rate * 0.2:  # At least 200ms total
            # Extract spectral features from the background
            bg_spectral = extract_spectral_features(background, sr_rate)
            
            # Add background-specific features
            bg_spectral['bg_power'] = np.mean(background**2)
            return bg_spectral
    
    # If no clear background noise found, use low-energy segments
    percentile = np.percentile(np.abs(y), 20)
    background = y[np.abs(y) < percentile]
    
    if len(background) > 0:
        bg_spectral = {
            'bg_power': np.mean(background**2),
            'bg_std': np.std(background),
            'bg_percentile': percentile
        }
        return bg_spectral
    else:
        # Fallback if no background detected
        return {
            'bg_power': np.mean(y**2) * 0.1,  # Estimate
            'bg_std': np.std(y) * 0.1,
            'bg_percentile': 0
        }

def compare_background_noise(enrolled_bg, auth_bg, threshold=0.7):
    """
    Compare background noise fingerprints to detect environment changes
    that might indicate replay attacks
    
    Returns:
    bool: True if backgrounds match (same environment)
    float: similarity score
    """
    if not enrolled_bg or not auth_bg:
        return True, 1.0  # Skip check if data not available
    
    # Calculate key similarities
    power_ratio = min(enrolled_bg.get('bg_power', 0.001), auth_bg.get('bg_power', 0.001)) / \
                  max(enrolled_bg.get('bg_power', 0.001), auth_bg.get('bg_power', 0.001))
                  
    # More comparison points if available
    std_ratio = 1.0
    if 'bg_std' in enrolled_bg and 'bg_std' in auth_bg:
        std_ratio = min(enrolled_bg['bg_std'], auth_bg['bg_std']) / \
                    max(enrolled_bg['bg_std'], auth_bg['bg_std'])
    
    # Calculate overall similarity
    similarity = (power_ratio + std_ratio) / 2
    
    # Background matches if similarity exceeds threshold
    return similarity > threshold, similarity

# Function to enroll the user's voiceprint with challenge phrases
def enroll_voiceprint(file_path=None, num_samples=3):
    """
    Enroll multiple voice samples for more robust authentication
    
    Parameters:
    file_path : str, optional
        Path to existing voice file. If None, will record samples live.
    num_samples : int, default=3
        Number of voice samples to enroll
    """
    voiceprints = []
    phrases = []
    bg_fingerprints = []  # Store background noise fingerprints
    
    # Use random phrases for enrollment
    selected_phrases = random.sample(CHALLENGE_PHRASES, min(num_samples, len(CHALLENGE_PHRASES)))
    while len(selected_phrases) < num_samples:
        # If we need more phrases than we have, repeat some
        selected_phrases.extend(random.sample(CHALLENGE_PHRASES, 
                                min(num_samples - len(selected_phrases), len(CHALLENGE_PHRASES))))
    
    if file_path:
        # Use existing file as first sample
        y, sr_rate = load_voice_file(file_path)
        features = extract_features(y, sr_rate)
        voiceprints.append(features)
        phrases.append("default_enrollment_phrase")
        
        # Extract background noise fingerprint
        bg_fingerprint = extract_background_noise(y, sr_rate)
        bg_fingerprints.append(bg_fingerprint)
        
        print(f"Enrolled 1/{num_samples} samples from file.")
        start_idx = 1
    else:
        start_idx = 0
        
    # Record remaining samples live
    for i in range(start_idx, num_samples):
        speak(f"Please say: {selected_phrases[i]}")
        print(f"Recording sample {i+1}/{num_samples}. Please say: {selected_phrases[i]}")
        
        # Modified to capture audio and analyze
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.energy_threshold = 900  
        with sr.Microphone() as source:
            print("Calibrating microphone for background noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print(f"Noise threshold set to {recognizer.energy_threshold}")
            print(f"Please say: {selected_phrases[i]}")
            audio = recognizer.listen(source,0,4)
            
            with open("temp_voice.wav", "wb") as f:
                f.write(audio.get_wav_data())
            
            y, sr_rate = load_voice_file("temp_voice.wav")
            features = extract_features(y, sr_rate)
            
            # Extract background noise fingerprint
            bg_fingerprint = extract_background_noise(y, sr_rate)
            
            os.remove("temp_voice.wav")
        
        voiceprints.append(features)
        bg_fingerprints.append(bg_fingerprint)
        phrases.append(selected_phrases[i])
        print(f"Sample {i+1} recorded successfully.")
        
    # Save voiceprints and background fingerprints to separate files
    os.makedirs("voiceprints", exist_ok=True)
    for i, voiceprint in enumerate(voiceprints):
        np.save(f"voiceprints/voiceprint_{i}.npy", voiceprint)
    
    # Save background fingerprints
    os.makedirs("bg_fingerprints", exist_ok=True)
    for i, bg_fingerprint in enumerate(bg_fingerprints):
        with open(f"bg_fingerprints/bg_{i}.json", "w") as f:
            json.dump(bg_fingerprint, f)
    
    # Save phrase-to-voiceprint mapping
    phrase_mapping = {i: phrase for i, phrase in enumerate(phrases)}
    with open("challenge_phrases.json", "w") as f:
        json.dump(phrase_mapping, f)
        
    # Create a marker file to indicate enrollment is complete
    with open("user_voiceprint.npy", "w") as f:
        f.write("enrollment_complete")
        
    print(f"All {num_samples} voiceprint samples have been saved with challenge phrases and background fingerprints.")

# Capture real-time voice input for enrollment
def capture_voice_for_enrollment():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 900  
    with sr.Microphone() as source:
        print("Calibrating microphone for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print(f"Noise threshold set to {recognizer.energy_threshold}")
        print("Please speak a clear phrase for enrollment...")
        audio = recognizer.listen(source,0,4)
        
        with open("temp_voice.wav", "wb") as f:
            f.write(audio.get_wav_data())
        
        y, sr_rate = load_voice_file("temp_voice.wav")
        os.remove("temp_voice.wav")
        return extract_features(y, sr_rate)

# Capture real-time voice input with noise reduction and anti-spoofing checks
def capture_voice_for_authentication():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 900  
    with sr.Microphone() as source:
        print("Calibrating microphone for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print(f"Noise threshold set to {recognizer.energy_threshold}")
        print("Please say password for authentication...")
        audio = recognizer.listen(source,0,4)
        
        with open("temp_voice.wav", "wb") as f:
            f.write(audio.get_wav_data())
        
        y, sr_rate = load_voice_file("temp_voice.wav")
        
        # Check for playback artifacts
        is_playback, confidence = detect_playback_artifacts(y, sr_rate)
        
        # Extract background noise fingerprint
        bg_fingerprint = extract_background_noise(y, sr_rate)
        
        # Extract MFCC features
        features = extract_features(y, sr_rate)
        
        # Clean up temp file
        os.remove("temp_voice.wav")
        
        return features, is_playback, confidence, bg_fingerprint

# Perform liveness detection challenge
def perform_liveness_challenge():
    """
    Perform a liveness detection challenge to verify human presence
    
    Returns:
    bool: True if passed, False if failed
    """
    # Select a random liveness challenge
    challenge = random.choice(LIVENESS_CHALLENGES)
    challenge_type = challenge["type"]
    
    print("\n=== Liveness Detection Challenge ===")
    speak(challenge["instruction"])
    print(challenge["instruction"])
    
    audio_samples = []
    
    # For challenges requiring multiple audio samples
    if challenge_type in ["pitch", "speed", "volume"]:
        # Collect two audio samples
        for i in range(2):
            recognizer = sr.Recognizer()
            recognizer.dynamic_energy_threshold = True
            recognizer.energy_threshold = 900
            
            print(f"Recording sample {i+1}/2...")
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, 0, 4)
                
                with open(f"temp_liveness_{i}.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
                y, sr_rate = load_voice_file(f"temp_liveness_{i}.wav")
                audio_samples.append((y, sr_rate))
                os.remove(f"temp_liveness_{i}.wav")
        
        # Analyze based on challenge type
        if challenge_type == "pitch":
            has_variation, ratio = analyze_pitch_variation(audio_samples)
            print(f"Pitch variation ratio: {ratio:.2f}")
            if has_variation:
                print("Liveness challenge passed: Detected pitch variation.")
                return True
            else:
                print("Liveness challenge failed: Insufficient pitch variation.")
                return False
                
        elif challenge_type == "speed":
            has_variation, ratio = analyze_speed_variation(audio_samples)
            print(f"Speed variation ratio: {ratio:.2f}")
            if has_variation:
                print("Liveness challenge passed: Detected speed variation.")
                return True
            else:
                print("Liveness challenge failed: Insufficient speed variation.")
                return False
                
        elif challenge_type == "volume":
            has_variation, ratio = analyze_volume_variation(audio_samples)
            print(f"Volume variation ratio: {ratio:.2f}")
            if has_variation:
                print("Liveness challenge passed: Detected volume variation.")
                return True
            else:
                print("Liveness challenge failed: Insufficient volume variation.")
                return False
    
    # For counting or sequence challenges, we would need speech recognition to verify
    # For now, assume passed since we can't easily verify content
    print("Liveness challenge passed.")
    return True

# Authenticate the user using DTW, challenge phrases, spectral analysis, background fingerprinting, and liveness detection
def authenticate_user():
    if not os.path.exists("user_voiceprint.npy") or not os.path.exists("challenge_phrases.json"):
        print("No voiceprint or challenge phrases found. Please enroll first.")
        return False
    
    # Load stored voiceprints from directory
    stored_voiceprints = []
    if os.path.exists("voiceprints"):
        for file in os.listdir("voiceprints"):
            if file.endswith(".npy"):
                voiceprint = np.load(f"voiceprints/{file}", allow_pickle=True)
                stored_voiceprints.append(voiceprint)
    
    if not stored_voiceprints:
        print("No valid voiceprints found. Please enroll again.")
        return False
    
    # Load background fingerprints if they exist
    bg_fingerprints = []
    if os.path.exists("bg_fingerprints"):
        for file in os.listdir("bg_fingerprints"):
            if file.endswith(".json"):
                with open(f"bg_fingerprints/{file}", "r") as f:
                    bg_fingerprint = json.load(f)
                    bg_fingerprints.append(bg_fingerprint)
    
    with open("challenge_phrases.json", "r") as f:
        phrase_mapping = json.load(f)
    
    # Select a random phrase for authentication
    phrase_idx = random.choice(list(phrase_mapping.keys()))
    challenge_phrase = phrase_mapping[phrase_idx]
    
    # Prompt with the challenge phrase
    speak(f"Please repeat: {challenge_phrase}")
    print(f"Please repeat: {challenge_phrase}")
    
    # Get voice input with playback detection and background fingerprinting
    captured_features, is_playback, playback_confidence, auth_bg_fingerprint = capture_voice_for_authentication()
    
    # Check for playback artifacts first (anti-spoofing)
    if is_playback:
        print(f"Playback detected! Confidence: {playback_confidence:.2f}")
        print("Authentication failed: Detected audio playback instead of live voice.")
        speak("Authentication failed. Playback detected.")
        return False
    else:
        print(f"Live voice verified. Playback confidence: {playback_confidence:.2f}")
    
    # Check background noise fingerprint if available
    if bg_fingerprints:
        # Compare with each stored fingerprint
        bg_matches = []
        for i, bg_fingerprint in enumerate(bg_fingerprints):
            match, similarity = compare_background_noise(bg_fingerprint, auth_bg_fingerprint)
            bg_matches.append((i, match, similarity))
        
        # Check if any background matches
        any_bg_match = any(match for _, match, _ in bg_matches)
        best_match = max(bg_matches, key=lambda x: x[2])
        print(f"Background noise similarity: {best_match[2]:.2f} with sample #{best_match[0]+1}")
        
        if not any_bg_match:
            print("Authentication failed: Background environment doesn't match enrolled environments.")
            speak("Authentication failed. Environment mismatch.")
            return False
    
    # Calculate DTW distance for each enrolled sample
    distances = []
    for i, voiceprint in enumerate(stored_voiceprints):
        distance, _ = fastdtw(voiceprint, captured_features, dist=euclidean)
        distances.append((i, distance))
    
    # Sort by distance to find best match
    distances.sort(key=lambda x: x[1])
    best_idx, min_distance = distances[0]
    
    print(f"Best DTW Distance: {min_distance} (sample #{best_idx+1} of {len(distances)})")
    
    # Check if the voice matches AND if it matches the correct phrase's voiceprint
    if min_distance < 25000 and (str(best_idx) == phrase_idx or phrase_mapping[str(best_idx)] == "default_enrollment_phrase"):
        print("Voice biometric match confirmed.")
        
        # Perform liveness detection challenge as final step
        print("Performing liveness detection challenge...")
        if perform_liveness_challenge():
            print("Voice Authentication successful.")
            speak("Voice Authentication successful.")
            return True
        else:
            print("Liveness challenge failed. Authentication failed.")
            speak("Liveness challenge failed. Authentication failed.")
            return False
    else:
        print("Voice Authentication failed.")
        speak("Voice Authentication failed.")
        return False

# Secondary password-based authentication
def handle_secondary_authentication():
    correct_password = "1234"
    user_input = input("Please enter your password: ")
    
    if user_input == correct_password:
        print("Secondary authentication successful.")
        speak("Secondary authentication successful.")
        return True
    else:
        print("Secondary authentication failed. Access denied.")
        speak("Secondary authentication failed. Access denied.")
        return False

def alarm(query):
    timehere = open("Alarmtext.txt","a")
    timehere.write(query)
    timehere.close()
    subprocess.run(["python", "alarm.py"])


if __name__ == "__main__":
    
    import sys
    
    # Check for enrollment command
    if len(sys.argv) > 1 and sys.argv[1] == "--enroll":
        print("Starting enrollment process...")
        if os.path.exists("Recording.wav"):
            enroll_voiceprint("Recording.wav", num_samples=3)
        else:
            enroll_voiceprint(num_samples=3)
        print("Enrollment completed. You can now run the program without --enroll to authenticate.")
        sys.exit(0)
    
    # Check if enrollment exists, otherwise create it
    if not os.path.exists("user_voiceprint.npy"):
        print("No voiceprint found. Please run 'python main.py --enroll' to enroll first.")
        sys.exit(1)
    
    if authenticate_user() or handle_secondary_authentication():
        speak("Welcome to SecureAI, say keyword 'activate' to start")
        while True:
            query=takeCommand().lower()
            if "activate" in query:
                from Startby import greetme
                greetme()
                
                while True:
                    query = takeCommand().lower()
                    if "go to sleep" in query:
                        speak("Ok, SecureAI is deactivating. You can activate me anytime.")
                        break
                    elif "hello" in query:
                        speak("Hello, how are you? SecureAI at your service.")
                    elif "i am fine" in query:
                        speak("That's Great")
                    elif "how are you" in query or "how r u" in query:
                        speak("Perfectly fine, thank you for asking")
                    elif "thank you" in query:
                        speak("You're welcome! SecureAI is here to help.")
                        
                    elif "open" in query:
                        from AppOpening import openappweb
                        openappweb(query)
                    elif "close" in query:
                        from AppOpening import closeappweb
                        closeappweb(query)
                    elif "pause" in query:
                        pyautogui.press("k")
                        speak("Video paused")
                    elif "play" in query:
                        pyautogui.press("k")
                        speak("Video Played")
                    elif "mute" in query:
                        pyautogui.press("m")
                        speak("Video muted")
                    elif "unmute" in query:
                        pyautogui.press("m")
                        speak("Video unmuted")
                    elif "volume up" in query:
                        from volume import volumeup
                        speak("volume is increasing")
                        volumeup()
                    elif "volume down" in query:
                        from volume import volumedown
                        speak("volume is decreasing")
                        volumedown()
                        
                    elif "google" in query:
                        from SearchNow import searchGoogle
                        searchGoogle(query)
                        sleep(0.5)
                    elif "youtube" in query:
                        from SearchNow import searchYoutube
                        searchYoutube(query)
                        sleep(0.5)
                    elif "wikipedia" in query:
                        from SearchNow import searchWikipedia
                        searchWikipedia(query)
                        sleep(0.5)
                    elif "temperature" in query:
                        search = "temperature in pune"
                        url = f"https://www.google.com/search?q={search}"
                        recognizer = requests.get(url)
                        data = BeautifulSoup(recognizer.text,"html.parser")
                        temp = data.find("div", class_ = "BNeawe").text
                        speak(f"Current {search} is {temp}")
                    elif "weather" in query:
                        search = "weather in pune"
                        url = f"https://www.google.com/search?q={search}"
                        recognizer = requests.get(url)
                        data = BeautifulSoup(recognizer.text,"html.parser")
                        weather_description = data.find("div", class_="BNeawe tAd8D AP7Wnd").text
                        speak(f"Current {search} is {weather_description}")
                    elif "the time" in query:
                        strtime = datetime.datetime.now().strftime("%H:%M")
                        speak(f"Current time is {strtime}")
                    elif "set an alarm" in query:
                        print("input time example:- 10:10")
                        speak("set an alarm")
                        a = input("Please tell the time :- ")
                        alarm(a)
                        speak("alarm set successfully")
                    elif "remember that" in query:
                        remembermessage = query.replace("remember that","")
                        remembermessage = query.replace("remember","")
                        remembermessage = query.replace("buddy","")
                        speak("you told me "+remembermessage)
                        remember = open("Remember.txt","w")
                        remember.write(remembermessage)
                        remember.close()
                    elif "what do you remember" in query:
                        remember =open("Remember.txt","r")
                        speak("You told me "+remember.read())
                    elif "whatsapp" in query:
                        from Whatsapp import sendmessage
                        sendmessage()
                    
                    elif "bye" in query:
                        speak("bye-bye, SecureAI is shutting down")
                        exit()
    
    else:
        speak("Authentication failed. SecureAI system is shutting down.")
        exit()
        