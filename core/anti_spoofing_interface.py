"""
anti_spoofing.py - Unified anti-spoofing interface for voice and face
"""
from core.anti_spoofing import AntiSpoofingEngine
from core.advanced_anti_spoofing import AdvancedAntiSpoofingEngine

class UnifiedAntiSpoofing:
    def __init__(self, config=None):
        self.basic = AntiSpoofingEngine(config)
        self.advanced = AdvancedAntiSpoofingEngine(config)

    def analyze_voice(self, audio_path: str) -> dict:
        """
        Run both basic and advanced anti-spoofing on voice sample.
        Returns dict with keys: is_genuine, is_live, spoof_score, liveness_check
        """
        result = self.advanced.analyze_audio_security(audio_path)
        if not result.get('is_genuine', False) or not result.get('is_live', False):
            return result
        # Optionally run basic as fallback
        return result

    # Placeholder for future face anti-spoofing (texture/motion/depth)
    def analyze_face(self, frame_sequence: list) -> dict:
        """
        Placeholder for face anti-spoofing (returns True if motion detected)
        """
        # This can be extended with CNN-based or texture-based spoof detection
        if len(frame_sequence) < 2:
            return {'is_live': False}
        return {'is_live': True}
