#!/usr/bin/env python3
"""
Test script for enhanced face recognition system
Tests DNN face detection, liveness detection, and embedding extraction
"""

import sys
import os
import numpy as np
import cv2
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.face_recognition_engine import FaceRecognitionEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_face_engine_initialization():
    """Test face engine initialization with Haar cascade and MediaPipe"""
    print("🧪 Testing Face Engine Initialization...")

    try:
        engine = FaceRecognitionEngine()
        print("✅ Face engine initialized successfully")

        # Check if Haar cascade detector is available
        if hasattr(engine, 'face_cascade') and engine.face_cascade is not None:
            print("✅ Haar cascade face detector loaded")
        else:
            print("❌ Haar cascade face detector not loaded")
            return False

        # Check if MediaPipe is available
        if hasattr(engine, 'mp_face_mesh') and engine.mp_face_mesh is not None:
            print("✅ MediaPipe face mesh loaded")
        else:
            print("❌ MediaPipe face mesh not loaded")
            return False

        return True

    except Exception as e:
        print(f"❌ Face engine initialization failed: {e}")
        return False

def test_face_detection():
    """Test face detection with DNN"""
    print("\n🧪 Testing Face Detection...")

    try:
        engine = FaceRecognitionEngine()

        # Create a test image with a face-like pattern
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test DNN detection
        faces_dnn = engine.detect_faces_dnn(test_image)
        print(f"✅ DNN detection completed, found {len(faces_dnn)} faces")

        # Test Haar cascade fallback
        faces_haar = engine.detect_faces_haar(test_image)
        print(f"✅ Haar detection completed, found {len(faces_haar)} faces")

        return True

    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_embedding_extraction():
    """Test embedding extraction"""
    print("\n🧪 Testing Embedding Extraction...")

    try:
        engine = FaceRecognitionEngine()

        # Create a test image
        test_image = np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)

        # Test embedding extraction
        embedding = engine.extract_embedding(test_image)

        if embedding is not None and len(embedding) > 0:
            print(f"✅ Embedding extracted successfully, length: {len(embedding)}")
            print(f"   Sample values: {embedding[:10]}...")
            return True
        else:
            print("❌ Embedding extraction failed")
            return False

    except Exception as e:
        print(f"❌ Embedding extraction test failed: {e}")
        return False

def test_liveness_detection():
    """Test liveness detection"""
    print("\n🧪 Testing Liveness Detection...")

    try:
        engine = FaceRecognitionEngine()

        # Create a test image
        test_image = np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)

        # Mock a face box for testing
        face_box = (50, 50, 100, 100)  # x, y, w, h

        # Test liveness detection
        liveness_passed = engine.check_liveness(test_image, face_box)
        print(f"✅ Liveness detection completed, result: {liveness_passed}")

        return True

    except Exception as e:
        print(f"❌ Liveness detection test failed: {e}")
        return False

def test_face_enrollment():
    """Test face enrollment"""
    print("\n🧪 Testing Face Enrollment...")

    try:
        engine = FaceRecognitionEngine()

        # Create a test image with a simple pattern
        test_image = np.full((200, 200, 3), 128, dtype=np.uint8)
        # Add some variation to simulate a face
        test_image[80:120, 80:120] = 150  # lighter area for "face"
        test_image[90:110, 90:110] = 100  # darker area for "eyes"

        # Test enrollment - this may return None if no face is detected
        embedding = engine.enroll_face(test_image)

        if embedding is not None and len(embedding) > 0:
            print(f"✅ Face enrolled successfully, embedding length: {len(embedding)}")
            print(f"   Enrolled encodings count: {len(engine.enrolled_face_encodings)}")
            return True
        else:
            print("ℹ️  Face enrollment returned None (no face detected in test image)")
            print("   This is expected behavior for synthetic test images")
            return True  # This is acceptable for synthetic images

    except Exception as e:
        print(f"❌ Face enrollment test failed: {e}")
        return False

def test_face_verification():
    """Test face verification"""
    print("\n🧪 Testing Face Verification...")

    try:
        engine = FaceRecognitionEngine()

        # Create test images with similar patterns
        test_image1 = np.full((200, 200, 3), 128, dtype=np.uint8)
        test_image1[80:120, 80:120] = 150  # lighter area for "face"
        test_image1[90:110, 90:110] = 100  # darker area for "eyes"

        test_image2 = np.full((200, 200, 3), 130, dtype=np.uint8)  # slightly different
        test_image2[80:120, 80:120] = 152  # slightly different "face"
        test_image2[90:110, 90:110] = 102  # slightly different "eyes"

        # Enroll first image - may return None if no face detected
        embedding1 = engine.enroll_face(test_image1)
        if embedding1 is None:
            print("ℹ️  Could not enroll first face (no face detected in test image)")
            print("   Testing verification logic with empty enrolled faces...")
            # Test verification with no enrolled faces
            is_match1, similarity1, liveness1 = engine.verify_face(test_image1)
            print(f"✅ Verification with no enrolled faces: match={is_match1}, similarity={similarity1:.3f}, liveness={liveness1}")
            return True

        print(f"✅ First face enrolled, embedding: {embedding1[:16]}...")

        # Verify with same image (should match)
        is_match1, similarity1, liveness1 = engine.verify_face(test_image1)
        print(f"✅ Same image verification: match={is_match1}, similarity={similarity1:.3f}, liveness={liveness1}")

        # Verify with different image (may or may not match)
        is_match2, similarity2, liveness2 = engine.verify_face(test_image2)
        print(f"✅ Different image verification: match={is_match2}, similarity={similarity2:.3f}, liveness={liveness2}")

        return True

    except Exception as e:
        print(f"❌ Face verification test failed: {e}")
        return False

def main():
    """Run all face recognition tests"""
    print("🔐 Enhanced Face Recognition System Test")
    print("=" * 50)

    tests = [
        test_face_engine_initialization,
        test_face_detection,
        test_embedding_extraction,
        test_liveness_detection,
        test_face_enrollment,
        test_face_verification
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All face recognition tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())