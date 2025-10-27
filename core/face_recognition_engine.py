"""
face_recognition_engine.py
Ultimate Offline Voice Biometric Stack - Face Recognition Module

This module uses SOTA face recognition (default: face_recognition library) to extract face embeddings and verify identity.
"""
import cv2
import numpy as np
try:
    import face_recognition
except ImportError:
    face_recognition = None

class FaceRecognitionEngine:
    def __init__(self, enrolled_face_encodings=None):
        self.enrolled_face_encodings = enrolled_face_encodings or []

    def enroll_face(self, image):
        """
        Extract and store face encoding from an image.
        
        Args:
            image: RGB image containing face
            
        Returns:
            Face encoding if successful, None otherwise
        """
        if face_recognition is None:
            raise ImportError("face_recognition library not installed.")
            
        # Check if image is valid
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            print("Error: Invalid image provided to face enrollment")
            return None
            
        # Try to detect face locations
        face_locations = face_recognition.face_locations(image)
        
        # If no face found, try with CNN model (more accurate)
        if not face_locations and len(image) >= 600:
            try:
                face_locations = face_recognition.face_locations(image, model="cnn")
            except Exception as e:
                print(f"CNN face detection failed: {e}")
                
        # If face found, extract encoding
        if face_locations:
            encodings = face_recognition.face_encodings(image, face_locations)
            if encodings:
                self.enrolled_face_encodings.append(encodings[0])
                return encodings[0]
                
        return None

    def verify_face(self, image, tolerance=0.5):
        """
        Verify if the face in the image matches enrolled faces.
        
        Args:
            image: RGB image containing face
            tolerance: Lower values are more strict (0.4-0.6 recommended)
            
        Returns:
            Tuple of (is_match, similarity_score)
        """
        if face_recognition is None:
            raise ImportError("face_recognition library not installed.")
            
        # Check if image is valid
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            print("Error: Invalid image provided to face verification")
            return False, 0.0
            
        # First detect faces using HOG method (faster)
        face_locations = face_recognition.face_locations(image, model="hog")
        
        # If no faces found, try CNN method (more accurate but slower)
        if not face_locations and len(image) >= 600:  # Only use CNN for larger images
            try:
                face_locations = face_recognition.face_locations(image, model="cnn")
            except Exception as e:
                print(f"CNN face detection failed: {e}")
        
        # Get face encodings for any faces found
        if face_locations:
            encodings = face_recognition.face_encodings(image, face_locations)
            if encodings and self.enrolled_face_encodings:
                matches = face_recognition.compare_faces(self.enrolled_face_encodings, encodings[0], tolerance)
                distances = face_recognition.face_distance(self.enrolled_face_encodings, encodings[0])
                best_match = np.argmin(distances)
                similarity = 1.0 - distances[best_match] if distances.size > 0 else 0.0
                return matches[best_match], similarity
                
        return False, 0.0

    def capture_face_from_camera(self, camera_index=0, num_attempts=3, delay_seconds=0.5):
        """
        Capture a single frame from the webcam with multiple attempts.
        
        Args:
            camera_index: Camera device index (default: 0)
            num_attempts: Number of capture attempts (default: 3)
            delay_seconds: Delay between attempts (default: 0.5)
            
        Returns:
            RGB frame if successful, None otherwise
        """
        import time
        
        for attempt in range(num_attempts):
            # Try to open camera
            cap = cv2.VideoCapture(camera_index)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                print(f"Warning: Cannot open camera (attempt {attempt+1}/{num_attempts})")
                if attempt < num_attempts - 1:  # If not last attempt
                    time.sleep(delay_seconds)
                continue
                
            # Capture a few frames to allow camera to adjust
            for _ in range(5):  # Discard first few frames
                cap.read()
                
            # Capture the actual frame
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None and frame.size > 0:
                # Check if frame has valid content
                if np.mean(frame) > 1.0:  # Basic check to ensure frame isn't empty/black
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return rgb_frame
                    
            # Wait before next attempt
            if attempt < num_attempts - 1:  # If not last attempt
                time.sleep(delay_seconds)
                
        print("Error: Failed to capture valid frame from camera after multiple attempts")
        return None
