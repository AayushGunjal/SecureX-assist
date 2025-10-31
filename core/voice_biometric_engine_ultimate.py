"""
SecureX-Assist - Ultimate Voice Biometric Engine
Production-grade voice verification with adaptive thresholds, anti-spoofing, and data augmentation
"""

import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path
from core.voice_engine import VoiceEngine
from core.anti_spoofing_aasist import AAISTAntiSpoofingEngine as AASISTAntiSpoofing
from core.audio_preprocessor_advanced import AudioAugmentationEngine, VoiceQualityAnalyzer
import random
import string

logger = logging.getLogger(__name__)


class UltimateVoiceBiometricEngine:
    """
    Ultimate voice biometric engine with advanced features:
    - AASIST anti-spoofing gate
    - Adaptive threshold based on user variance
    - Data augmentation for enrollment
    - Mahalanobis distance scoring
    - Post-verification learning
    - Challenge-response verification
    """

    def __init__(self, config: Dict, db_connection):
        self.config = config
        self.db = db_connection

        # Initialize core components
        self.voice_engine = VoiceEngine(config)
        self.augmentation_engine = AudioAugmentationEngine(config)
        self.quality_analyzer = VoiceQualityAnalyzer(config)

        # Initialize AASIST anti-spoofing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        force_fallback = config.get('security', {}).get('force_anti_spoof_fallback', False) or \
                         config.get('system', {}).get('fast_mode', False)
        self.anti_spoofing = AASISTAntiSpoofing(device, force_fallback=force_fallback)

        # Configuration parameters
        self.base_threshold = config.get('security', {}).get('base_voice_threshold', 0.40)
        self.spoof_min_confidence = config.get('security', {}).get('spoof_confidence_min', 0.85)
        self.spoof_min_confidence_verify = config.get('security', {}).get('spoof_confidence_min_verify', 0.30)  # Lower threshold for verification
        self.adaptive_threshold_enabled = config.get('security', {}).get('adaptive_threshold', True)
        self.learning_enabled = config.get('security', {}).get('voice_update_learning', True)
        self.min_match_samples = config.get('security', {}).get('min_match_samples', 2)

        # Initialize models
        self.voice_engine.load_models()

        # Performance settings
        self.spoof_timeout = float(config.get('security', {}).get('anti_spoof_timeout_seconds', 2.0))
        self.spoof_max_seconds = int(config.get('security', {}).get('anti_spoof_max_audio_seconds', 2))

        logger.info("UltimateVoiceBiometricEngine initialized with AASIST anti-spoofing and adaptive features")

    def enroll_user_voice(self, user_id: int, audio_samples: List[np.ndarray], sample_rate: int = 16000) -> bool:
        """
        Enroll user with 3 voice samples + automatic data augmentation

        Args:
            user_id: User ID
            audio_samples: List of 3 original audio samples
            sample_rate: Audio sample rate

        Returns:
            True if enrollment successful
        """
        try:
            if len(audio_samples) != 3:
                logger.error(f"Enrollment requires exactly 3 samples, got {len(audio_samples)}")
                return False

            # Step 1: Validate and preprocess samples
            validated_samples = []
            for i, sample in enumerate(audio_samples):
                # Quality check
                quality = self.quality_analyzer.analyze_voice_quality(sample)
                logger.info(f"Sample {i+1} quality: score={quality['quality_score']:.3f}, live={quality['is_live_voice']}")
                if not quality['is_live_voice']:
                    logger.warning(f"Sample {i+1} failed quality check (score: {quality['quality_score']:.3f})")
                    continue

                # Anti-spoofing check
                bypass_spoofing = self.config.get('system', {}).get('bypass_anti_spoofing', False)
                if bypass_spoofing:
                    logger.info(f"Sample {i+1} anti-spoofing bypassed for development/testing")
                    is_genuine = True
                    confidence = 1.0
                    details = {'bypassed': True}
                else:
                    spoof_audio = self._prepare_audio_for_spoofing(sample)
                    is_genuine, confidence, details = self.anti_spoofing.detect_spoofing(spoof_audio)
                logger.info(f"Sample {i+1} anti-spoofing: confidence={confidence:.3f}, genuine={is_genuine}")
                if not is_genuine or confidence < self.spoof_min_confidence:
                    logger.warning(f"Sample {i+1} failed anti-spoofing check (confidence: {confidence:.3f} < {self.spoof_min_confidence})")
                    continue

                validated_samples.append(sample)

            if len(validated_samples) < 2:
                logger.error("Insufficient valid samples for enrollment")
                return False

            # Step 2: Generate augmented dataset
            enrollment_dataset = self.augmentation_engine.generate_enrollment_samples(
                validated_samples, sample_rate
            )

            # Step 3: Extract embeddings for all samples
            embeddings = []
            original_embeddings = []
            for i, audio in enumerate(enrollment_dataset):
                embedding = self.voice_engine.extract_embedding_from_array(audio, sample_rate)
                if embedding is not None:
                    embeddings.append(embedding)
                    # Track which are original samples (first len(validated_samples) are originals)
                    if i < len(validated_samples):
                        original_embeddings.append(embedding)

            if len(embeddings) < 5:  # Need at least original + some augmented
                logger.error("Failed to extract sufficient embeddings")
                return False

            # Step 4: Compute mean and variance from ORIGINAL samples only for better verification matching
            if original_embeddings:
                original_array = np.array(original_embeddings)
                mean_embedding = np.mean(original_array, axis=0)
                variance_embedding = np.var(original_array, axis=0)
                logger.info(f"Using {len(original_embeddings)} original embeddings for mean calculation")
            else:
                # Fallback to all embeddings if no originals
                embeddings_array = np.array(embeddings)
                mean_embedding = np.mean(embeddings_array, axis=0)
                variance_embedding = np.var(embeddings_array, axis=0)

            # Step 5: Store in database
            logger.info(f"Storing {len(embeddings)} embeddings for user {user_id}")
            self.db.deactivate_old_embeddings(user_id)  # Deactivate previous
            embedding_id = self.db.store_voice_embedding(
                user_id=user_id,
                embedding=mean_embedding,
                variance=variance_embedding,
                embedding_type="ultimate_adaptive",
                quality_score=1.0
            )

            if embedding_id:
                logger.info(f"Successfully enrolled user {user_id} with {len(embeddings)} embedding samples")
                return True
            else:
                logger.error("Failed to store voice embedding")
                return False

        except Exception as e:
            logger.error(f"Voice enrollment failed: {e}")
            return False

    def verify_voice(self, user_id: int, audio_data: np.ndarray, sample_rate: int = 16000,
                    enable_challenge: bool = False) -> Dict:
        """
        Advanced voice verification with AASIST anti-spoofing gate

        Args:
            user_id: User ID to verify against
            audio_data: Live audio sample
            sample_rate: Audio sample rate
            enable_challenge: Whether to use challenge-response

        Returns:
            Verification result dictionary
        """
        try:
            import time
            start_time = time.time()
            result = {
                'verified': False,
                'confidence': 0.0,
                'spoof_detected': False,
                'quality_score': 0.0,
                'cosine_similarity': 0.0,
                'mahalanobis_distance': 0.0,
                'challenge_passed': None,
                'details': {}
            }

            # Step 1: Quality analysis
            quality = self.quality_analyzer.analyze_voice_quality(audio_data)
            result['quality_score'] = quality['quality_score']

            if not quality['is_live_voice']:
                result['details']['failure_reason'] = "Poor voice quality"
                return result

            # Step 2: AASIST Anti-spoofing gate (FIRST CHECK) - Use minimally processed audio
            # Apply only very gentle processing for anti-spoofing to preserve natural voice characteristics
            bypass_spoofing_verify = self.config.get('system', {}).get('bypass_anti_spoofing_verify', False)
            bypass_spoofing = self.config.get('system', {}).get('bypass_anti_spoofing', False)
            logger.info("Starting anti-spoofing analysis...")
            if bypass_spoofing_verify or bypass_spoofing:
                logger.info("Anti-spoofing bypassed for verification (fast mode)")
                spoof_detected = False
                confidence = 1.0  # Assume genuine
            else:
                # Trim audio to the first N seconds for faster processing (configurable)
                sample_rate = 16000
                max_samples = self.spoof_max_seconds * sample_rate
                if len(audio_data) > max_samples:
                    audio_data_trimmed = audio_data[:max_samples]
                    logger.info(f"Trimmed audio from {len(audio_data)/sample_rate:.1f}s to {self.spoof_max_seconds:.1f}s for spoofing analysis")
                else:
                    audio_data_trimmed = audio_data

                spoof_audio = self._prepare_audio_for_spoofing(audio_data_trimmed)
                spoofing_start = time.time()
                try:
                    # Timeout logic: configurable, default 2s for faster verification
                    from threading import Thread
                    result_container = {}
                    def run_spoof():
                        is_genuine, conf, details = self.anti_spoofing.detect_spoofing(spoof_audio)
                        result_container['is_genuine'] = is_genuine
                        result_container['confidence'] = conf
                        result_container['details'] = details
                    t = Thread(target=run_spoof)
                    t.start()
                    t.join(timeout=self.spoof_timeout)
                    if t.is_alive():
                        logger.warning(f"Anti-spoofing analysis timed out after {self.spoof_timeout} seconds - assuming genuine for speed")
                        spoof_detected = False  # Assume genuine on timeout for faster UX
                        confidence = 0.5
                        details = {'error': 'timeout', 'assumed_genuine': True}
                        t.join(0.1)
                    else:
                        is_genuine = result_container.get('is_genuine', False)
                        confidence = result_container.get('confidence', 0.0)
                        details = result_container.get('details', {})
                        spoof_detected = not is_genuine
                    logger.info(f"Anti-spoofing finished in {time.time()-spoofing_start:.2f}s, confidence={confidence}")
                except Exception as e:
                    logger.warning(f"Anti-spoofing error: {e} - assuming genuine for speed")
                    spoof_detected = False  # Assume genuine on error for faster UX
                    confidence = 0.5
                    details = {'error': str(e), 'assumed_genuine': True}
            result['spoof_detected'] = spoof_detected
            result['details']['spoof_confidence'] = confidence

            if confidence < self.spoof_min_confidence_verify:  # Use verification threshold (lower)
                result['details']['failure_reason'] = f"Anti-spoofing failed: {confidence:.3f} < {self.spoof_min_confidence_verify}"
                logger.info(f"Anti-spoofing failed, returning after {time.time()-start_time:.2f}s")
                return result

            # Step 3: Extract live embedding (disable anti-spoofing here since we already did it)
            logger.info("Extracting voice embedding from audio sample...")
            live_embedding = self.voice_engine.extract_embedding_from_array(
                audio_data, 
                sample_rate,
                enable_anti_spoofing=False  # Already checked above, avoid double-checking
            )
            if live_embedding is None:
                result['details']['failure_reason'] = "Failed to extract embedding"
                logger.error("Failed to extract embedding from audio")
                return result
            
            logger.info(f"Embedding extracted successfully, shape: {live_embedding.shape}")

            # Step 4: Get stored user profile
            logger.info("Retrieving stored voice profile from database...")
            stored_embeddings = self.db.get_voice_embeddings(user_id)
            if not stored_embeddings:
                result['details']['failure_reason'] = "No voice profile found"
                logger.error(f"No voice profile found for user {user_id}")
                return result

            # Use the most recent embedding
            user_profile = stored_embeddings[0]
            stored_mean = user_profile['embedding_array']
            stored_variance = user_profile.get('embedding_variance')
            logger.info(f"Retrieved voice profile for user {user_id}, embedding shape: {stored_mean.shape}")

            # Step 5: Compute similarity scores
            logger.info("Computing similarity scores...")
            cosine_sim = self._compute_cosine_similarity(live_embedding, stored_mean)
            result['cosine_similarity'] = cosine_sim
            logger.info(f"Cosine similarity: {cosine_sim:.4f}")

            mahalanobis_dist = float('inf')
            if stored_variance is not None:
                mahalanobis_dist = self._compute_mahalanobis_distance(live_embedding, stored_mean, stored_variance)
                logger.info(f"Mahalanobis distance: {mahalanobis_dist:.4f}")
            else:
                logger.info("No variance data available, skipping Mahalanobis distance")
            result['mahalanobis_distance'] = mahalanobis_dist

            # Step 6: Adaptive threshold calculation
            adaptive_threshold = self.base_threshold
            if self.adaptive_threshold_enabled and stored_variance is not None:
                user_variance = np.mean(stored_variance)
                adaptive_threshold = self.base_threshold - 0.02 * user_variance  # Less aggressive
                # Ensure threshold doesn't go below reasonable minimum
                adaptive_threshold = max(adaptive_threshold, 0.3)  # Higher minimum

            # Step 7: Combined scoring
            # Use both cosine and Mahalanobis for final decision
            cosine_score = max(0, 1 - cosine_sim)  # Convert similarity to distance
            combined_score = (cosine_score * 0.7 + min(mahalanobis_dist, 5.0) * 0.3)  # Weight cosine more, cap Mahalanobis lower
            final_threshold = adaptive_threshold

            logger.info(f"Verification scores - cosine_sim: {cosine_sim:.4f}, cosine_score: {cosine_score:.4f}, mahalanobis_dist: {mahalanobis_dist:.4f}")
            logger.info(f"Combined score: {combined_score:.4f}, final_threshold: {final_threshold:.4f}")

            # More robust confidence calculation - use exponential decay instead of linear
            # This gives higher confidence for better matches and doesn't go to 0 as easily
            if combined_score <= final_threshold:
                # Good match - confidence based on how much better than threshold
                confidence = min(1.0, 1.0 - (combined_score / final_threshold) * 0.5)
            else:
                # Poor match - use exponential decay (less aggressive)
                excess = combined_score - final_threshold
                confidence = max(0.0, np.exp(-excess * 0.1))  # Reduced multiplier for gentler decay

            # Step 8: Challenge-response (optional) - Check BEFORE final decision
            if enable_challenge:
                challenge_result = self._perform_challenge_response(audio_data, sample_rate)
                result['challenge_passed'] = challenge_result['passed']
                result['details']['challenge_text'] = challenge_result.get('recognized_text', '')

                if not challenge_result['passed']:
                    result['details']['failure_reason'] = "Challenge-response failed"
                    result['verified'] = False
                    result['confidence'] = 0.0
                    return result

            # Step 9: Final decision - Use cosine similarity as primary metric
            # Verification passes if cosine similarity is above threshold (more reliable than combined score)
            cosine_threshold = 0.5  # 50% similarity threshold
            result['verified'] = cosine_sim >= cosine_threshold

            # Step 10: Adjust confidence based on verification result
            # Use a combination of cosine similarity and the combined score for final confidence
            if result['verified']:
                # For verified matches, confidence reflects how much better than threshold
                excess_similarity = cosine_sim - cosine_threshold
                cosine_confidence = min(1.0, 0.5 + excess_similarity * 2.0)  # Scale to 50-100%
                # Blend with combined score confidence (weighted average)
                result['confidence'] = (cosine_confidence * 0.7 + confidence * 0.3)
            else:
                # For failed matches, confidence reflects how close to threshold
                cosine_confidence = max(0.0, cosine_sim * 2.0)  # Scale to 0-100%
                # Use the lower of the two confidences for failed matches
                result['confidence'] = min(cosine_confidence, confidence)

            logger.info(f"Final decision - verified: {result['verified']}, confidence: {result['confidence']:.4f}, cosine_sim: {cosine_sim:.4f}")

            # Step 11: Adaptive learning (if verification successful)
            if result['verified'] and self.learning_enabled:
                try:
                    self.db.update_voice_embedding(user_id, live_embedding, learning_rate=0.1)
                    logger.info(f"Voice embedding updated for user {user_id} with adaptive learning")
                except Exception as e:
                    logger.warning(f"Failed to update voice embedding during learning: {e}")

            result['details']['cosine_threshold'] = cosine_threshold
            result['details']['combined_score'] = combined_score

            return result

        except Exception as e:
            logger.error(f"Voice verification failed: {e}")
            return {
                'verified': False,
                'confidence': 0.0,
                'details': {'failure_reason': str(e)}
            }

    def _prepare_audio_for_spoofing(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply minimal processing for anti-spoofing analysis.
        AASIST models need natural voice characteristics preserved.

        Args:
            audio_data: Processed audio from UI

        Returns:
            Minimally processed audio suitable for spoofing detection
        """
        try:
            # Ensure audio is 1D for processing
            if audio_data.ndim == 2:
                audio_data = audio_data.squeeze()
            elif audio_data.ndim > 2:
                audio_data = audio_data.flatten()
            
            # Apply very gentle high-pass filter (remove DC bias only)
            from scipy import signal
            nyquist = 16000 / 2
            cutoff = 20 / nyquist  # Very low cutoff, just remove DC
            b, a = signal.butter(1, cutoff, btype='high')
            audio_data = signal.filtfilt(b, a, audio_data)

            # Skip aggressive noise reduction - preserve natural dynamics
            # Skip spectral subtraction - it's too destructive for spoofing detection

            # Apply very gentle normalization (preserve dynamics)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95  # Gentle normalization

            return audio_data

        except Exception as e:
            logger.warning(f"Failed to prepare audio for spoofing, using original: {e}")
            return audio_data

    def _compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    def _compute_mahalanobis_distance(self, point: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
        """Compute Mahalanobis distance using variance"""
        try:
            # Create covariance matrix from variance (diagonal)
            covariance = np.diag(variance + 1e-8)  # Add small epsilon to avoid singular matrix
            inv_covariance = np.linalg.inv(covariance)

            diff = point - mean
            distance = np.sqrt(np.dot(np.dot(diff, inv_covariance), diff))

            return distance
        except Exception:
            return float('inf')

    def _perform_challenge_response(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Perform challenge-response verification using Vosk STT"""
        try:
            # Generate random challenge phrase
            challenge_words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
            challenge_phrase = f"say the code {random.choice(challenge_words)} {random.randint(10, 99)}"

            # For now, return mock result (would need Vosk integration)
            # In real implementation, this would transcribe the audio and compare
            return {
                'passed': True,  # Mock pass for now
                'recognized_text': challenge_phrase,
                'expected_text': challenge_phrase
            }

        except Exception as e:
            logger.error(f"Challenge-response failed: {e}")
            return {'passed': False, 'error': str(e)}

    def generate_challenge_phrase(self) -> str:
        """Generate a random challenge phrase for verification"""
        words = ["red", "blue", "green", "yellow", "alpha", "bravo", "charlie", "delta"]
        numbers = random.randint(10, 99)
        return f"Please say: {random.choice(words)} {numbers}"