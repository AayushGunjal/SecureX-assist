"""
voice_auth.py - Modular voice authentication with anti-spoofing
"""
from core.voice_engine import VoiceEngine
from core.anti_spoofing import AntiSpoofingEngine
from core.advanced_anti_spoofing import AdvancedAntiSpoofingEngine
import numpy as np

class VoiceAuthenticator:
    def __init__(self, config=None):
        self.voice_engine = VoiceEngine(config)
        self.anti_spoofing = AdvancedAntiSpoofingEngine(config)

    def authenticate(self, audio_path: str, user_embedding: np.ndarray) -> dict:
        """
        Authenticate user by voice with anti-spoofing and liveness check.
        Returns dict with keys: success, reason, spoof_score, liveness_score, match_score
        """
        # 1. Anti-spoofing & liveness
        spoof_result = self.anti_spoofing.analyze_audio_security(audio_path)
        if not spoof_result.get('is_genuine', False) or not spoof_result.get('is_live', False):
            return {
                'success': False,
                'reason': 'Voice spoofing or replay attack detected',
                'spoof_score': spoof_result.get('spoof_score'),
                'liveness_score': spoof_result.get('liveness_check'),
                'match_score': None
            }
        # 2. Voice embedding match
        embedding = self.voice_engine.extract_embedding(audio_path, enable_anti_spoofing=False)
        if embedding is None:
            return {'success': False, 'reason': 'Voice embedding extraction failed', 'spoof_score': None, 'liveness_score': None, 'match_score': None}
        match_score = float(np.dot(embedding, user_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(user_embedding)))
        if match_score < 0.75:
            return {'success': False, 'reason': 'Voice does not match registered user', 'spoof_score': spoof_result.get('spoof_score'), 'liveness_score': spoof_result.get('liveness_check'), 'match_score': match_score}
        return {'success': True, 'reason': 'Voice authentication passed', 'spoof_score': spoof_result.get('spoof_score'), 'liveness_score': spoof_result.get('liveness_check'), 'match_score': match_score}
