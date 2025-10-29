#!/usr/bin/env python3
"""
Check voice verification issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.voice_biometric_engine_ultimate import UltimateVoiceBiometricEngine
from core.database import Database
import yaml

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    db = Database()
    db.connect()
    db.initialize_schema()

    engine = UltimateVoiceBiometricEngine(config, db)

    # Check if user has voice embeddings
    users = db.get_all_users()
    print("=== VOICE EMBEDDINGS CHECK ===")
    for user in users:
        embeddings = db.get_voice_embeddings(user['id'])
        print(f'User {user["username"]} (ID: {user["id"]}): {len(embeddings)} voice embeddings')
        if embeddings:
            variance = embeddings[0].get('embedding_variance')
            if variance:
                print(f'  Latest embedding ID: {embeddings[0]["id"]}, variance length: {len(variance)}')
                print(f'  Mean variance: {sum(variance)/len(variance):.6f}')
            else:
                print(f'  Latest embedding ID: {embeddings[0]["id"]}, no variance')

    # Test verification logic
    print("\n=== VERIFICATION LOGIC TEST ===")
    for user in users:
        embeddings = db.get_voice_embeddings(user['id'])
        if embeddings:
            print(f"\nTesting verification for user {user['username']} (ID: {user['id']})")
            # Create a mock audio array
            import numpy as np
            mock_audio = np.random.randn(16000)  # 1 second of random audio

            result = engine.verify_voice(
                user_id=user['id'],
                audio_data=mock_audio,
                sample_rate=16000
            )

            print(f"Verification result: {result['verified']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Failure reason: {result['details'].get('failure_reason', 'None')}")
            print(f"Combined score: {result['details'].get('combined_score', 'N/A')}")
            print(f"Adaptive threshold: {result['details'].get('adaptive_threshold', 'N/A')}")

    db.close()

if __name__ == "__main__":
    main()