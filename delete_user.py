#!/usr/bin/env python3
"""
User Management Script for SecureX-Assist
Allows deleting users from the database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import Database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_users(db):
    """List all users"""
    try:
        users = db.get_all_users()
        
        if not users:
            print("No users found.")
            return []
        
        print("\nCurrent Users:")
        print("-" * 50)
        for user in users:
            print(f"ID: {user['id']}, Username: {user['username']}, Email: {user['email'] or 'N/A'}, Created: {user['created_at']}")
        print("-" * 50)
        return users
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        return []

def delete_user_by_id(db, user_id):
    """Delete user by ID"""
    try:
        success = db.delete_user(user_id)
        if success:
            print(f"[OK] User ID {user_id} deleted successfully.")
        else:
            print(f"[ERROR] Failed to delete user ID {user_id}.")
        return success
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return False

def main():
    print("SecureX-Assist User Management")
    print("=" * 40)
    
    # Initialize database
    db_path = "securex_db.sqlite"
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' not found.")
        return
    
    db = Database(db_path)
    db.connect()
    db.initialize_schema()
    
    while True:
        print("\nOptions:")
        print("1. List all users")
        print("2. Delete user by ID")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            list_users(db)
            
        elif choice == "2":
            users = list_users(db)
            if users:
                try:
                    user_id = int(input("Enter user ID to delete: ").strip())
                    confirm = input(f"Are you sure you want to delete user ID {user_id}? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        delete_user_by_id(db, user_id)
                    else:
                        print("Deletion cancelled.")
                except ValueError:
                    print("Invalid user ID.")
                    
        elif choice == "3":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice.")
    
    db.close()

if __name__ == "__main__":
    main()