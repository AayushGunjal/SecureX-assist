#!/usr/bin/env python3
"""
Check database status for voice verification debugging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import Database

def check_database():
    db = Database()
    db.connect()
    db.initialize_schema()

    users = db.get_all_users()
    print(f"Users in database: {len(users)}")
    for u in users:
        print(f"  {u['username']}: id={u['id']}")
        voices = db.get_voice_embeddings(u['id'])
        print(f"    Voice embeddings: {len(voices)}")
        for v in voices:
            print(f"      ID {v['id']}: active={v['is_active']}, created={v['created_at']}")

if __name__ == "__main__":
    check_database()