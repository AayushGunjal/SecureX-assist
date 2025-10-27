"""
Database Initialization Script
Create test users for SecureX-Assist
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.database import Database
from core.security import SecurityManager
from utils.helpers import load_config


def init_database():
    """Initialize database with test users"""
    
    print("\n" + "=" * 60)
    print("🗄️  DATABASE INITIALIZATION - SecureX-Assist")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Initialize database
    db = Database(config['database']['path'])
    db.connect()
    db.initialize_schema()
    
    print("\n✅ Database initialized")
    print(f"📁 Location: {config['database']['path']}")
    
    # Initialize security manager
    security = SecurityManager(config)
    
    # Create test users
    print("\n" + "=" * 60)
    print("👥 CREATING TEST USERS")
    print("=" * 60)
    
    test_users = [
        {
            'username': 'testuser',
            'password': 'test123',
            'email': 'test@example.com'
        },
        {
            'username': 'admin',
            'password': 'admin123',
            'email': 'admin@example.com'
        },
        {
            'username': 'demo',
            'password': 'demo123',
            'email': 'demo@example.com'
        }
    ]
    
    created_count = 0
    
    for user_data in test_users:
        # Check if user already exists
        existing = db.get_user_by_username(user_data['username'])
        if existing:
            print(f"⚠️  User '{user_data['username']}' already exists (ID: {existing['id']})")
            continue
        
        # Hash password
        password_hash = security.hash_password(user_data['password'])
        
        # Create user
        user_id = db.create_user(
            username=user_data['username'],
            password_hash=password_hash,
            email=user_data['email']
        )
        
        if user_id:
            print(f"✅ Created: {user_data['username']} (ID: {user_id})")
            print(f"   Password: {user_data['password']}")
            print(f"   Email: {user_data['email']}")
            created_count += 1
        else:
            print(f"❌ Failed to create: {user_data['username']}")
    
    print("\n" + "=" * 60)
    print(f"🎉 Created {created_count} new user(s)")
    print("=" * 60)
    
    if created_count > 0:
        print("\n📝 Next Steps:")
        print("1. Run voice enrollment: python enroll_voice.py")
        print("2. Launch application: python main.py")
        print("3. Login with test credentials")
    else:
        print("\n⚠️  No new users created (all users already exist)")
        print("\nTo reset database:")
        print(f"1. Delete: {config['database']['path']}")
        print("2. Run this script again")
    
    # Close database
    db.close()


if __name__ == "__main__":
    try:
        init_database()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
