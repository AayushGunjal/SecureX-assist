"""
SecureX-Assist - Intent Classifier
Natural language intent recognition using SentenceTransformers
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    Lightweight intent classifier using SentenceTransformers for natural language queries
    """
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.model = None
        self.intents = {}
        self.intent_embeddings = {}
        self.threshold = 0.7  # Similarity threshold
        
        # Try to load the model, but don't fail if it doesn't work
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Intent classifier model loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer model: {e}")
            logger.info("Intent classifier will use keyword matching fallback")
            self.model = None
        
    def add_intent(self, intent_name: str, examples: List[str]):
        """
        Add an intent with example phrases
        
        Args:
            intent_name: Name of the intent (e.g., 'time', 'open_app')
            examples: List of example phrases for this intent
        """
        self.intents[intent_name] = examples
        if self.model:
            # Compute embeddings for examples
            embeddings = self.model.encode(examples)
            # Store mean embedding for the intent
            self.intent_embeddings[intent_name] = np.mean(embeddings, axis=0)
            logger.info(f"Added intent '{intent_name}' with {len(examples)} examples")
        else:
            # Store examples for keyword matching
            self.intent_embeddings[intent_name] = examples
            logger.info(f"Added intent '{intent_name}' with keyword examples (no ML model)")
    
    def classify(self, text: str) -> Tuple[Optional[str], float]:
        """
        Classify the intent of a text query
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (intent_name, confidence_score) or (None, 0.0) if no match
        """
        if not self.intent_embeddings:
            return None, 0.0
            
        if self.model:
            # Use ML-based classification
            text_embedding = self.model.encode([text])[0]
            
            # Compute similarities with all intents
            similarities = {}
            for intent_name, intent_embedding in self.intent_embeddings.items():
                similarity = np.dot(text_embedding, intent_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(intent_embedding)
                )
                similarities[intent_name] = similarity
            
            # Find best match
            best_intent = max(similarities, key=similarities.get)
            best_score = similarities[best_intent]
            
            # Check threshold
            if best_score >= self.threshold:
                logger.info(f"Classified '{text}' as intent '{best_intent}' with confidence {best_score:.3f}")
                return best_intent, best_score
            else:
                logger.info(f"No intent matched for '{text}' (best: {best_intent} with {best_score:.3f})")
                return None, best_score
        else:
            # Use keyword-based classification
            text_lower = text.lower()
            for intent_name, examples in self.intent_embeddings.items():
                for example in examples:
                    if example.lower() in text_lower:
                        logger.info(f"Keyword matched '{text}' as intent '{intent_name}'")
                        return intent_name, 1.0
            
            logger.info(f"No keyword match for '{text}'")
            return None, 0.0
    
    def setup_default_intents(self):
        """Setup default intents for SecureX assistant"""
        
        # System control
        self.add_intent("open_app", [
            "open notepad", "launch chrome", "start calculator", "open file explorer",
            "run notepad", "open browser", "start word", "launch application"
        ])
        
        self.add_intent("close_app", [
            "close notepad", "exit chrome", "close calculator", "close window",
            "quit application", "close program", "exit app"
        ])
        
        self.add_intent("system_lock", [
            "lock the system", "lock computer", "lock screen", "secure system",
            "lock workstation", "lock pc"
        ])
        
        self.add_intent("system_restart", [
            "restart computer", "reboot system", "restart pc", "reboot"
        ])
        
        self.add_intent("system_shutdown", [
            "shutdown computer", "turn off pc", "power off", "shutdown system"
        ])
        
        # File management
        self.add_intent("list_files", [
            "list files", "show files", "what files are here", "list directory",
            "show folder contents", "list documents"
        ])
        
        self.add_intent("delete_file", [
            "delete file", "remove file", "erase file", "delete document"
        ])
        
        self.add_intent("open_file", [
            "open file", "show file", "read file", "open document"
        ])
        
        # Media control
        self.add_intent("play_music", [
            "play music", "start music", "play song", "music on"
        ])
        
        self.add_intent("pause_music", [
            "pause music", "stop music", "music off", "pause song"
        ])
        
        self.add_intent("mute_audio", [
            "mute", "mute audio", "turn off sound", "quiet"
        ])
        
        self.add_intent("unmute_audio", [
            "unmute", "unmute audio", "turn on sound", "sound on"
        ])
        
        # Information
        self.add_intent("time", [
            "what time is it", "tell me the time", "current time", "time please"
        ])
        
        self.add_intent("date", [
            "what date is it", "today's date", "current date", "date please"
        ])
        
        self.add_intent("system_info", [
            "system status", "computer info", "system information", "performance"
        ])
        
        # Custom/Assistant
        self.add_intent("help", [
            "help", "what can you do", "commands", "assist me", "help me"
        ])
        
        self.add_intent("who_are_you", [
            "who are you", "what is your name", "introduce yourself", "who is this"
        ])
        
        self.add_intent("show_logs", [
            "show logs", "view logs", "check logs", "system logs"
        ])
        
        self.add_intent("greeting", [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening"
        ])
        
        self.add_intent("goodbye", [
            "bye", "goodbye", "see you later", "farewell"
        ])
        
        logger.info(f"Setup {len(self.intents)} default intents")