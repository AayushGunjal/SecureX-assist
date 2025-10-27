"""
updated_auth_pipeline.py - Unified authentication pipeline for SecureX-Assist
Integrates voice anti-spoofing, face verification, and liveness checks.
"""
from core.voice_auth import VoiceAuthenticator
from core.face_auth import FaceAuthenticator
from core.anti_spoofing_interface import UnifiedAntiSpoofing
import logging

logger = logging.getLogger("SecureXAuthPipeline")

class SecureXAuthPipeline:
    def __init__(self, config=None, enrolled_voice_embedding=None, enrolled_face_encoding=None):
        self.voice_auth = VoiceAuthenticator(config)
        self.face_auth = FaceAuthenticator(enrolled_face_encoding)
        self.anti_spoofing = UnifiedAntiSpoofing(config)
        self.enrolled_voice_embedding = enrolled_voice_embedding
        self.enrolled_face_encoding = enrolled_face_encoding

    def authenticate(self, audio_path: str, face_frame_sequence: list) -> dict:
        """
        Full authentication: voice anti-spoofing + voice match + face liveness + face match
        Returns dict with keys: success, reason, details
        """
        # 1. Voice anti-spoofing + liveness + match
        voice_result = self.voice_auth.authenticate(audio_path, self.enrolled_voice_embedding)
        if not voice_result['success']:
            logger.warning(f"Voice authentication failed: {voice_result['reason']}")
            return {'success': False, 'reason': voice_result['reason'], 'details': voice_result}
        # 2. Face liveness + match
        face_result = self.face_auth.authenticate(face_frame_sequence)
        if not face_result['success']:
            logger.warning(f"Face authentication failed: {face_result['reason']}")
            return {'success': False, 'reason': face_result['reason'], 'details': face_result}
        # 3. Both passed
        logger.info("Authentication succeeded: voice and face verified.")
        return {'success': True, 'reason': 'Authentication passed', 'details': {'voice': voice_result, 'face': face_result}}
