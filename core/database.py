"""
SecureX-Assist - Database Layer
Secure storage for users, voice embeddings, and sessions
"""

import sqlite3
import json
import logging
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class Database:
    """
    SQLite database manager for SecureX-Assist
    Stores: Users, Voice Embeddings, Sessions, Audit Logs
    """
    
    def __init__(self, db_path: str = "securex_db.sqlite"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        
        logger.info(f"Initializing database: {db_path}")
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Access columns by name
            logger.info("Database connected successfully")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def initialize_schema(self):
        """Create database tables if they don't exist"""
        try:
            cursor = self.conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    failed_attempts INTEGER DEFAULT 0
                )
            """)
            
            # Voice embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    embedding_data TEXT NOT NULL,
                    embedding_type TEXT DEFAULT 'pyannote',
                    quality_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    voice_verified BOOLEAN DEFAULT 0,
                    password_verified BOOLEAN DEFAULT 0,
                    liveness_verified BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # Audit logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    status TEXT DEFAULT 'success',
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """)
            
            # Liveness challenges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS liveness_challenges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    challenge_phrase TEXT NOT NULL,
                    transcribed_text TEXT,
                    similarity_score REAL DEFAULT 0.0,
                    passed BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            self.conn.commit()
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    # ==================== USER OPERATIONS ====================
    
    def create_user(
        self, 
        username: str, 
        password_hash: str, 
        email: Optional[str] = None
    ) -> Optional[int]:
        """
        Create a new user
        
        Returns:
            User ID if successful, None otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute(
                "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                (username, password_hash, email)
            )
            
            self.conn.commit()
            user_id = cursor.lastrowid
            
            logger.info(f"User created: {username} (ID: {user_id})")
            self.log_audit(user_id, "user_created", f"Username: {username}")
            
            return user_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"User already exists: {username}")
            return None
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    def update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
    
    def increment_failed_attempts(self, username: str):
        """Increment failed login attempts"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE users SET failed_attempts = failed_attempts + 1 WHERE username = ?",
                (username,)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to increment failed attempts: {e}")
    
    def reset_failed_attempts(self, username: str):
        """Reset failed login attempts after successful login"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE users SET failed_attempts = 0 WHERE username = ?",
                (username,)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to reset failed attempts: {e}")
    
    # ==================== VOICE EMBEDDING OPERATIONS ====================
    
    def store_voice_embedding(
        self, 
        user_id: int, 
        embedding: np.ndarray,
        embedding_type: str = "pyannote",
        quality_score: float = 1.0
    ) -> Optional[int]:
        """
        Store voice embedding for a user
        
        Args:
            user_id: User ID
            embedding: Voice embedding (numpy array)
            embedding_type: Type of model used
            quality_score: Quality assessment score
            
        Returns:
            Embedding ID if successful
        """
        try:
            # Serialize embedding to JSON
            embedding_json = json.dumps(embedding.tolist())
            
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO voice_embeddings 
                (user_id, embedding_data, embedding_type, quality_score) 
                VALUES (?, ?, ?, ?)""",
                (user_id, embedding_json, embedding_type, quality_score)
            )
            
            self.conn.commit()
            embedding_id = cursor.lastrowid
            
            logger.info(f"Voice embedding stored for user ID {user_id}")
            self.log_audit(user_id, "voice_enrolled", f"Embedding ID: {embedding_id}")
            
            return embedding_id
            
        except Exception as e:
            logger.error(f"Failed to store voice embedding: {e}")
            return None
    
    def get_voice_embeddings(self, user_id: int) -> List[Dict]:
        """
        Get all active voice embeddings for a user
        
        Returns:
            List of embedding dictionaries
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """SELECT * FROM voice_embeddings 
                WHERE user_id = ? AND is_active = 1 
                ORDER BY created_at DESC""",
                (user_id,)
            )
            
            rows = cursor.fetchall()
            embeddings = []
            
            for row in rows:
                embedding_dict = dict(row)
                # Deserialize embedding data
                embedding_dict['embedding_array'] = np.array(
                    json.loads(embedding_dict['embedding_data'])
                )
                embeddings.append(embedding_dict)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get voice embeddings: {e}")
            return []
    
    def deactivate_old_embeddings(self, user_id: int):
        """Deactivate old voice embeddings when enrolling new ones"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE voice_embeddings SET is_active = 0 WHERE user_id = ?",
                (user_id,)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to deactivate old embeddings: {e}")
    
    # ==================== SESSION OPERATIONS ====================
    
    def create_session(self, user_id: int, session_token: str) -> Optional[int]:
        """Create a new session"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (user_id, session_token) VALUES (?, ?)",
                (user_id, session_token)
            )
            self.conn.commit()
            
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    def update_session_verification(
        self, 
        session_token: str,
        voice_verified: bool = False,
        password_verified: bool = False,
        liveness_verified: bool = False
    ):
        """Update session verification status"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """UPDATE sessions 
                SET voice_verified = ?, password_verified = ?, liveness_verified = ?,
                    last_activity = CURRENT_TIMESTAMP
                WHERE session_token = ?""",
                (voice_verified, password_verified, liveness_verified, session_token)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
    
    def invalidate_session(self, session_token: str):
        """Invalidate a session"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE sessions SET is_active = 0 WHERE session_token = ?",
                (session_token,)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to invalidate session: {e}")
    
    # ==================== AUDIT LOG OPERATIONS ====================
    
    def log_audit(
        self, 
        user_id: Optional[int],
        action: str,
        details: Optional[str] = None,
        status: str = "success"
    ):
        """Log an audit event"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO audit_logs (user_id, action, details, status) VALUES (?, ?, ?, ?)",
                (user_id, action, details, status)
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log audit: {e}")
    
    def get_audit_logs(self, user_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """Get audit logs"""
        try:
            cursor = self.conn.cursor()
            
            if user_id:
                cursor.execute(
                    "SELECT * FROM audit_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return []
    
    # ==================== LIVENESS CHALLENGE OPERATIONS ====================
    
    def store_liveness_challenge(
        self,
        user_id: int,
        challenge_phrase: str,
        transcribed_text: Optional[str] = None,
        similarity_score: float = 0.0,
        passed: bool = False
    ) -> Optional[int]:
        """Store a liveness challenge attempt"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO liveness_challenges 
                (user_id, challenge_phrase, transcribed_text, similarity_score, passed)
                VALUES (?, ?, ?, ?, ?)""",
                (user_id, challenge_phrase, transcribed_text, similarity_score, passed)
            )
            self.conn.commit()
            
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Failed to store liveness challenge: {e}")
            return None
