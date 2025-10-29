import sys
sys.path.append('.')
from core.voice_biometric_engine_ultimate import UltimateVoiceBiometricEngine
from core.database import Database
import yaml
import numpy as np

print('=== TESTING VERIFICATION DEBUG ===')
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    db = Database()
    db.connect()

    engine = UltimateVoiceBiometricEngine(config, db)

    # Get the latest user
    users = db.get_all_users()
    if users:
        user = users[-1]  # Latest user
        print(f'Testing verification for user: {user["username"]} (ID: {user["id"]})')

        # Create a dummy audio sample for testing
        audio_data = np.random.randn(16000).astype(np.float32) * 0.1

        result = engine.verify_voice(
            user_id=user['id'],
            audio_data=audio_data,
            sample_rate=16000,
            enable_challenge=False
        )

        print(f'Verification result: {result}')
    else:
        print('No users found!')

    db.close()

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()