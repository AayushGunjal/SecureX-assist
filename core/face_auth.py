from core.face_recognition_engine import FaceRecognitionEngine
import cv2
import numpy as np

class FaceAuthenticator:
    def __init__(self, enrolled_face_encoding=None):
        self.face_engine = FaceRecognitionEngine([enrolled_face_encoding] if enrolled_face_encoding is not None else None)
        self.enrolled_face_encoding = enrolled_face_encoding

    def _detect_liveness(self, frame_sequence: list) -> bool:

        if len(frame_sequence) < 2:
            return False
        # Compute frame differences
        diffs = [np.mean(cv2.absdiff(frame_sequence[i], frame_sequence[i-1])) for i in range(1, len(frame_sequence))]
        return np.mean(diffs) > 2.0  # Threshold for motion

    def authenticate(self, frame_sequence: list) -> dict:

        if not self._detect_liveness(frame_sequence):
            return {'success': False, 'reason': 'Face liveness check failed', 'liveness': False, 'match_score': None} 
        # Use last frame for face match
        face_image = frame_sequence[-1]
        if self.enrolled_face_encoding is None:
            return {'success': False, 'reason': 'No enrolled face encoding', 'liveness': True, 'match_score': None} 
        is_match, match_score = self.face_engine.verify_face(face_image, tolerance=0.5)
        if not is_match:
            return {'success': False, 'reason': 'Face does not match registered user', 'liveness': True, 'match_score': match_score} 
        return {'success': True, 'reason': 'Face authentication passed', 'liveness': True, 'match_score': match_score} 