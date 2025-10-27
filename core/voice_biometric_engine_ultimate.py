"""
Ultimate Voice Biometric Engine
Integration Layer for Offline Voice Authentication

Combines:
1. SpeechBrain Speaker Embedding (192-D)
2. AASIST Anti-Spoofing
3. Advanced Audio Preprocessing
4. Adaptive Thresholding
5. Voiceprint Aggregation
6. Face Recognition (Optional)

Usage:
    engine = VoiceBiometricEngine()
    
    # Enrollment
    voiceprint = engine.enroll_user(audio_samples, face_image=img)
    
    # Verification
    is_authentic, score, details = engine.verify_user(audio_sample, voiceprint, face_image=img)
"""

import logging
import numpy as np
from typing import Dict, Tuple, List, Optional
import json

logger = logging.getLogger(__name__)


class VoiceBiometricEngine:
    """
    Complete offline voice biometric system using SOTA models.
    """
    
    def __init__(self, device: str = "cpu", enable_anti_spoofing: bool = True, enable_face_recognition: bool = True):
        """
        Initialize the biometric engine with all components.
        
        Args:
            device: "cpu" or "cuda"
            enable_anti_spoofing: Enable anti-spoofing detection
            enable_face_recognition: Enable face recognition
        """
        self.device = device
        self.enable_anti_spoofing = enable_anti_spoofing
        self.enable_face_recognition = enable_face_recognition
        
        logger.info("=" * 60)
        logger.info("ULTIMATE VOICE BIOMETRIC ENGINE - INITIALIZATION")
        logger.info("=" * 60)
        
        # Initialize components
        self._initialize_components()
        
        logger.info("✅ All components initialized successfully")
        logger.info("=" * 60)
    
    def _initialize_components(self):
        """Initialize all engine components."""
        try:
            # Audio Preprocessing
            logger.info("Loading: Audio Preprocessor...")
            from core.audio_preprocessor_advanced import AudioPreprocessor
            self.preprocessor = AudioPreprocessor(sample_rate=16000)
            
            # Speaker Embedding
            logger.info("Loading: SpeechBrain Speaker Embedding Engine...")
            from core.speaker_embedding_engine import SpeakerEmbeddingEngine
            self.embedding_engine = SpeakerEmbeddingEngine(device=self.device)
            
            # Anti-Spoofing
            if self.enable_anti_spoofing:
                logger.info("Loading: AASIST Anti-Spoofing Engine...")
                from core.anti_spoofing_aasist import AAISTAntiSpoofingEngine
                self.anti_spoofing = AAISTAntiSpoofingEngine(device=self.device)
            else:
                self.anti_spoofing = None

            # Face Recognition
            if self.enable_face_recognition:
                logger.info("Loading: Face Recognition Engine...")
                from core.face_recognition_engine import FaceRecognitionEngine
                self.face_engine = FaceRecognitionEngine()
            else:
                self.face_engine = None
            
        except ImportError as e:
            logger.error(f"Failed to import components: {e}")
            raise
    
    def enroll_user(self, audio_samples: List[np.ndarray], 
                   user_id: str = "user", face_image: Optional[np.ndarray] = None) -> Dict:
        """
        Enroll a new user with voice and optional face samples.
        
        Args:
            audio_samples: List of 3+ audio arrays (16kHz, mono, float32)
            user_id: User identifier
            face_image: A single image containing the user's face.
            
        Returns:
            Voiceprint dict with embeddings and metadata
        """
        if len(audio_samples) < 3:
            raise ValueError("Need at least 3 voice samples for enrollment")
        
        logger.info(f"\n📝 ENROLLMENT: {user_id}")
        logger.info(f"   Processing {len(audio_samples)} voice samples...")
        
        embeddings = []
        valid_samples = 0
        
        for idx, audio in enumerate(audio_samples, 1):
            logger.info(f"\n   Sample {idx}/{len(audio_samples)}:")
            
            # Preprocess
            processed_audio, stats = self.preprocessor.process(audio)
            quality_ok, reason = self.preprocessor.validate_audio_quality(processed_audio)
            
            if not quality_ok:
                logger.warning(f"   ⚠️  Quality check failed: {reason}")
                continue
            
            logger.info(f"   ✅ Quality: {reason}")
            
            # Anti-spoofing check
            if self.anti_spoofing:
                is_genuine, confidence, _ = self.anti_spoofing.detect_spoofing(processed_audio)
                logger.info(f"   Anti-spoofing: {'GENUINE ✅' if is_genuine else 'SPOOFED 🚫'} ({confidence:.1%})")
                
                if not is_genuine:
                    logger.warning(f"   ⚠️  Spoofing detected, skipping sample")
                    continue
            
            # Extract embedding
            embedding = self.embedding_engine.extract_embedding(processed_audio)
            if embedding is not None:
                embeddings.append(embedding)
                valid_samples += 1
            else:
                logger.warning(f"   ⚠️  Failed to extract embedding")
        
        if valid_samples < 2:
            raise ValueError(f"Only {valid_samples} valid samples, need at least 2")
        
        # Aggregate embeddings into voiceprint
        embeddings_array = np.array(embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)
        variance = np.var(embeddings_array, axis=0)
        
        voiceprint = {
            "user_id": user_id,
            "model": "SpeechBrain ECAPA-TDNN",
            "embedding_dim": 192,
            "num_samples": valid_samples,
            "mean_embedding": mean_embedding.tolist(),
            "variance": variance.tolist(),
            "embeddings": [e.tolist() for e in embeddings_array],
            "face_enrolled": False,
            "face_encoding": None
        }

        # Face enrollment
        if self.enable_face_recognition and self.face_engine and face_image is not None:
            logger.info("   Processing face sample...")
            face_encoding = self.face_engine.enroll_face(face_image)
            if face_encoding is not None:
                voiceprint["face_enrolled"] = True
                voiceprint["face_encoding"] = face_encoding.tolist()
                logger.info("   ✅ Face enrolled successfully.")
            else:
                logger.warning("   ⚠️  Could not detect a face in the provided image.")

        logger.info(f"\n✅ ENROLLMENT COMPLETE: {user_id}")
        logger.info(f"   Voiceprint created from {valid_samples} valid samples")
        logger.info(f"   Face enrolled: {voiceprint['face_enrolled']}")
        
        return voiceprint
    
    def verify_user(self, audio_sample: np.ndarray, 
                   voiceprint: Dict,
                   threshold: Optional[float] = None,
                   face_image: Optional[np.ndarray] = None,
                   face_tolerance: float = 0.6) -> Tuple[bool, float, Dict]:
        """
        Verify if audio and/or face matches the user's voiceprint.
        
        Args:
            audio_sample: Audio array (16kHz, mono, float32)
            voiceprint: User's voiceprint (from enrollment)
            threshold: Voice similarity threshold (0-1, default 0.60)
            face_image: Image for face verification
            face_tolerance: Face match tolerance (lower is stricter)
            
        Returns:
            Tuple of (is_match, similarity_score, details)
        """
        if threshold is None:
            threshold = 0.60  # Conservative threshold
        
        logger.info(f"\n🔐 VERIFICATION")
        logger.info(f"   User: {voiceprint['user_id']}")
        
        # --- Voice Verification ---
        processed_audio, stats = self.preprocessor.process(audio_sample)
        quality_ok, reason = self.preprocessor.validate_audio_quality(processed_audio)
        
        logger.info(f"   Audio quality: {reason}")
        if not quality_ok:
            logger.warning(f"   ⚠️  Voice quality check failed")
            return False, 0.0, {"error": reason, "details": stats}
        
        spoof_details = {}
        if self.anti_spoofing:
            is_genuine, confidence, spoof_details = self.anti_spoofing.detect_spoofing(processed_audio)
            logger.info(f"   Anti-spoofing: {'GENUINE ✅' if is_genuine else 'SPOOFED 🚫'}")
            
            if not is_genuine:
                logger.warning(f"   ⚠️  Spoofing attack detected!")
                return False, 0.0, {
                    "error": "Spoofing detected",
                    "spoof_confidence": float(confidence),
                    "spoof_details": spoof_details
                }
        
        embedding = self.embedding_engine.extract_embedding(processed_audio)
        if embedding is None:
            logger.error("Failed to extract voice embedding")
            return False, 0.0, {"error": "Voice embedding extraction failed"}
        
        mean_embedding = np.array(voiceprint['mean_embedding'])
        voice_similarity = self.embedding_engine.compute_similarity(embedding, mean_embedding)
        
        variance = np.array(voiceprint['variance'])
        adaptive_threshold = self._compute_adaptive_threshold(variance, threshold)
        
        is_voice_match = voice_similarity >= adaptive_threshold

        # --- Face Verification ---
        is_face_match = False
        face_similarity = 0.0
        if self.enable_face_recognition and self.face_engine and face_image is not None:
            if voiceprint.get("face_enrolled"):
                enrolled_encoding = np.array(voiceprint["face_encoding"])
                self.face_engine.enrolled_face_encodings = [enrolled_encoding] # Load for verification
                is_face_match, face_similarity = self.face_engine.verify_face(face_image, tolerance=face_tolerance)
                logger.info(f"   Face verification: {'✅ MATCH' if is_face_match else '❌ NO MATCH'} (similarity: {face_similarity:.2f})")
            else:
                logger.warning("   ⚠️  Face verification requested, but no face enrolled for this user.")
        else:
            # If face check is disabled or no image provided, we don't require it to match
            is_face_match = True

        # --- Final Decision ---
        is_match = is_voice_match and is_face_match
        
        logger.info(f"\n   Voice Similarity: {voice_similarity:.4f} (Threshold: {adaptive_threshold:.4f}) -> {'MATCH' if is_voice_match else 'NO MATCH'}")
        if self.enable_face_recognition and face_image is not None:
            logger.info(f"   Face Similarity:  {face_similarity:.4f} (Tolerance: {face_tolerance}) -> {'MATCH' if is_face_match else 'NO MATCH'}")
        
        logger.info(f"   Final Result: {'✅ AUTHENTICATED' if is_match else '❌ AUTHENTICATION FAILED'}")
        
        details = {
            "user_id": voiceprint['user_id'],
            "is_match": bool(is_match),
            "voice_match": bool(is_voice_match),
            "face_match": bool(is_face_match),
            "voice_similarity": float(voice_similarity),
            "voice_threshold": float(adaptive_threshold),
            "face_similarity": float(face_similarity),
            "face_tolerance": face_tolerance,
            "spoof_check": spoof_details,
            "quality_metrics": stats.get('quality_metrics', {})
        }
        
        return is_match, voice_similarity, details
    
    def _compute_adaptive_threshold(self, variance: np.ndarray, 
                                   base_threshold: float) -> float:
        """
        Compute adaptive threshold based on voiceprint variance.
        Higher variance = more variation = slightly lower threshold
        """
        mean_variance = np.mean(variance)
        adjustment = min(mean_variance / 10, 0.05)
        adaptive = base_threshold - adjustment
        return max(adaptive, 0.50)  # Floor at 0.50
    
    def batch_verify(self, audio_samples: List[np.ndarray],
                    voiceprint: Dict,
                    threshold: Optional[float] = None,
                    face_image: Optional[np.ndarray] = None) -> Dict:
        """
        Verify multiple samples against voiceprint (consensus voting).
        
        Args:
            audio_samples: List of audio arrays
            voiceprint: User's voiceprint
            threshold: Similarity threshold
            face_image: A single face image to verify against
            
        Returns:
            Results with consensus score
        """
        results = []
        similarities = []
        
        logger.info(f"\n📊 BATCH VERIFICATION: {len(audio_samples)} samples")
        
        for idx, audio in enumerate(audio_samples, 1):
            is_match, similarity, details = self.verify_user(audio, voiceprint, threshold, face_image)
            results.append({
                "sample": idx,
                "is_match": is_match,
                "similarity": similarity,
                "details": details
            })
            if details.get("voice_match"): # Only consider similarity from valid voice matches
                similarities.append(similarity)
        
        # Consensus
        consensus_similarity = np.mean(similarities) if similarities else 0.0
        consensus_match = sum(1 for r in results if r['is_match']) >= len(results) / 2
        
        logger.info(f"\n📊 BATCH RESULTS:")
        logger.info(f"   Consensus similarity (avg of matches): {consensus_similarity:.4f}")
        logger.info(f"   Consensus decision: {'✅ MATCH' if consensus_match else '❌ NO MATCH'}")
        
        return {
            "consensus_similarity": float(consensus_similarity),
            "consensus_match": bool(consensus_match),
            "individual_results": results,
            "match_count": sum(1 for r in results if r['is_match']),
            "total_samples": len(audio_samples)
        }
    
    def get_system_info(self) -> Dict:
        """Get complete system information."""
        return {
            "system": "Ultimate Voice Biometric Engine",
            "components": {
                "audio_preprocessing": self.preprocessor.get_processor_info() if self.preprocessor else {},
                "speaker_embedding": self.embedding_engine.get_embedding_info() if self.embedding_engine else {},
                "anti_spoofing": self.anti_spoofing.get_model_info() if self.anti_spoofing else {},
                "face_recognition": "face_recognition library" if self.face_engine else "Disabled",
            },
            "device": self.device,
            "anti_spoofing_enabled": self.enable_anti_spoofing,
            "face_recognition_enabled": self.enable_face_recognition,
            "default_voice_threshold": 0.60,
            "default_face_tolerance": 0.6,
            "min_enrollment_samples": 3,
            "embedding_dimension": 192,
        }


if __name__ == "__main__":
    # This is a placeholder for testing. A full test requires audio and image data.
    logger.info("\n🚀 Testing Ultimate Voice Biometric Engine (Structure)...\n")
    
    try:
        engine = VoiceBiometricEngine(device="cpu", enable_anti_spoofing=True, enable_face_recognition=True)
        info = engine.get_system_info()
        
        print("\n" + "="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        print(json.dumps(info, indent=2))
        print("\n" + "="*60)
        print("✅ Engine initialized successfully.")
        print("➡️  To run a full test, provide audio and image samples.")
        print("="*60)

    except Exception as e:
        logger.error(f"🔥 Engine initialization failed: {e}", exc_info=True)

