"""
SpeechBrain Speaker Embedding Engine (ECAPA-VoxCeleb)
Ultimate Offline Voice Biometric Stack - Component 1

This module uses the state-of-the-art ECAPA-TDNN model trained on VoxCeleb
to extract 192-dimensional speaker embeddings from voice samples.

Features:
- 192-D embedding vectors (vs. 512-D pyannote)
- SOTA SpeechBrain ECAPA model
- Robust to microphone variations
- Offline, no internet required
"""

import logging
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class SpeakerEmbeddingEngine:
    """
    Extracts 192-dimensional speaker embeddings using SpeechBrain's ECAPA-TDNN model.
    This is the core component for voice biometric authentication.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the SpeechBrain embedding engine.
        
        Args:
            device: "cpu" or "cuda" for GPU acceleration
        """
        self.device = torch.device(device)
        self.embedding_dim = 192
        self.sample_rate = 16000
        
        try:
            # Import SpeechBrain classifier for speaker embedding
            from speechbrain.pretrained import EncoderClassifier
            
            # Load ECAPA-TDNN model trained on VoxCeleb
            # This is the SOTA model for speaker verification
            logger.info("Loading SpeechBrain ECAPA-TDNN model (192-dim embeddings)...")
            
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)},
                savedir="/tmp/ecapa_model"
            )
            
            logger.info("✅ SpeechBrain ECAPA model loaded successfully")
            logger.info(f"   - Embedding dimension: {self.embedding_dim}")
            logger.info(f"   - Sample rate: {self.sample_rate} Hz")
            logger.info(f"   - Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain model: {e}")
            logger.error("Make sure to install: pip install speechbrain torch torchaudio")
            raise
    
    def extract_embedding(self, audio_file_or_array) -> Optional[np.ndarray]:
        """
        Extract 192-D speaker embedding from audio.
        
        Args:
            audio_file_or_array: Path to audio file or numpy array (16kHz, mono)
            
        Returns:
            192-dimensional embedding vector (numpy array)
        """
        try:
            # Load audio
            if isinstance(audio_file_or_array, str):
                # Load from file
                waveform, sample_rate = torchaudio.load(audio_file_or_array)
                
                # Resample if necessary
                if sample_rate != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                    waveform = resampler(waveform)
            else:
                # Convert numpy array to tensor
                waveform = torch.from_numpy(audio_file_or_array).unsqueeze(0).float()
                if waveform.shape[0] > 1:  # If stereo, take first channel
                    waveform = waveform[0:1]
            
            # Move to device
            waveform = waveform.to(self.device)
            
            # Extract embedding (192-D vector)
            with torch.no_grad():
                embedding = self.classifier.encode_batch(waveform)
            
            # Convert to numpy
            embedding = embedding.squeeze(0).cpu().numpy()
            
            # Verify dimension
            if embedding.shape[0] != self.embedding_dim:
                logger.warning(f"Expected {self.embedding_dim}D embedding, got {embedding.shape[0]}D")
            
            logger.info(f"✅ Extracted {embedding.shape[0]}D speaker embedding")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None
    
    def batch_extract_embeddings(self, audio_files: list) -> np.ndarray:
        """
        Extract embeddings from multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            Embeddings matrix (N_files x 192)
        """
        embeddings = []
        
        for audio_file in audio_files:
            embedding = self.extract_embedding(audio_file)
            if embedding is not None:
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: 192-D embedding vector
            embedding2: 192-D embedding vector
            
        Returns:
            Similarity score (0-1, where 1 = identical)
        """
        # Normalize embeddings
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(similarity)
    
    def aggregate_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate multiple embeddings into a single voiceprint.
        
        Args:
            embeddings: Matrix of N embeddings (N x 192)
            
        Returns:
            Tuple of (mean_embedding, variance)
        """
        mean_embedding = np.mean(embeddings, axis=0)
        variance = np.var(embeddings, axis=0)
        
        logger.info(f"Aggregated {embeddings.shape[0]} embeddings:")
        logger.info(f"  - Mean embedding: {mean_embedding.shape}")
        logger.info(f"  - Variance: {variance.shape}")
        
        return mean_embedding, variance
    
    def get_embedding_info(self) -> Dict:
        """Get information about the embedding model."""
        return {
            "model": "SpeechBrain ECAPA-TDNN (VoxCeleb)",
            "embedding_dimension": self.embedding_dim,
            "sample_rate": self.sample_rate,
            "device": str(self.device),
            "description": "SOTA speaker verification model - 192D embeddings"
        }


def verify_speaker_pair(embedding1: np.ndarray, embedding2: np.ndarray, 
                       threshold: float = 0.60) -> Tuple[bool, float]:
    """
    Quick speaker verification between two embeddings.
    
    Args:
        embedding1: First 192-D embedding
        embedding2: Second 192-D embedding
        threshold: Decision threshold (0-1)
        
    Returns:
        Tuple of (match: bool, similarity: float)
    """
    engine = SpeakerEmbeddingEngine()
    similarity = engine.compute_similarity(embedding1, embedding2)
    match = similarity >= threshold
    
    return match, similarity


if __name__ == "__main__":
    # Test the module
    logger.info("Testing SpeechBrain Speaker Embedding Engine...")
    
    engine = SpeakerEmbeddingEngine()
    info = engine.get_embedding_info()
    
    print("\n✅ Engine Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
