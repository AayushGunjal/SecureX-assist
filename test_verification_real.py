import sys
sys.path.append('.')
from core.voice_biometric_engine_ultimate import UltimateVoiceBiometricEngine
from core.database import Database
from core.audio_processor import AudioRecorder
import yaml
import numpy as np

print('=== TESTING VERIFICATION WITH STORED DATA ===')
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    db = Database()
    db.connect()

    engine = UltimateVoiceBiometricEngine(config, db)

    # Get the user with embeddings (ID 31)
    user = db.get_user_by_id(31)
    if user:
        print(f'Testing verification for user: {user["username"]} (ID: {user["id"]})')

        # Get stored embeddings
        stored_embeddings = db.get_voice_embeddings(user['id'])
        if stored_embeddings:
            stored_profile = stored_embeddings[0]
            stored_mean = stored_profile['embedding_array']
            stored_variance = stored_profile.get('embedding_variance')

            print(f'Stored embedding shape: {stored_mean.shape}')
            print(f'Stored variance shape: {stored_variance.shape if stored_variance is not None else None}')
            print(f'Stored mean sample: {stored_mean[:5]}')

            # Create a test audio sample - let's use silence to see what happens
            audio_data = np.zeros(16000, dtype=np.float32)

            print('Testing with silence...')
            result = engine.verify_voice(
                user_id=user['id'],
                audio_data=audio_data,
                sample_rate=16000,
                enable_challenge=False
            )

            print(f'Verification result: verified={result["verified"]}, confidence={result["confidence"]:.4f}')
            print(f'Details: {result["details"]}')
        else:
            print('No stored embeddings found!')
    else:
        print('No users found!')

    db.close()

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()