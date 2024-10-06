import sqlite3
from cryptography.fernet import Fernet, InvalidToken
import os
from typing import Optional
import threading


class Auth:
    _instance = None
    _lock = threading.Lock()  # Ensures thread-safe access to database and encryption

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Auth, cls).__new__(cls)
                cls._instance.init()
        return cls._instance

    def init(self):
        self.db_path = "auth.db"
        self.key = self._get_or_create_key()
        self.fernet = Fernet(self.key)
        self._init_db()

    def _get_or_create_key(self) -> bytes:
        key_file = "encryption.key"
        try:
            if os.path.exists(key_file):
                with open(key_file, "rb") as file:
                    return file.read()
            else:
                key = Fernet.generate_key()
                with open(key_file, "wb") as file:
                    file.write(key)
                return key
        except IOError as e:
            print(f"Error accessing encryption key file: {e}")
            raise

    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS auth_keys
                (provider TEXT PRIMARY KEY, key TEXT)
            """)
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def set_key(self, provider: str, key: str):
        encrypted_key = self.fernet.encrypt(key.encode()).decode()
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "REPLACE INTO auth_keys (provider, key) VALUES (?, ?)",
                    (provider, encrypted_key),
                )
                conn.commit()
            except sqlite3.Error as e:
                print(f"Database error when setting key: {e}")
                raise
            finally:
                conn.close()

    def get_key(self, provider: str) -> Optional[str]:
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT key FROM auth_keys WHERE provider = ?", (provider,)
                )
                result = cursor.fetchone()
                if result:
                    return self.fernet.decrypt(result[0].encode()).decode()
            except (sqlite3.Error, InvalidToken) as e:
                print(f"Error retrieving or decrypting key: {e}")
                return None
            finally:
                conn.close()

    def rotate_key(self):
        """
        Rotates the encryption key securely by decrypting all stored keys with the
        old key and re-encrypting them with a new key.
        """
        new_key = Fernet.generate_key()
        new_fernet = Fernet(new_key)
        with self._lock:
            try:
                # Retrieve and decrypt all keys using the old key
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT provider, key FROM auth_keys")
                rows = cursor.fetchall()

                # Re-encrypt each key with the new key
                for provider, encrypted_key in rows:
                    decrypted_key = self.fernet.decrypt(encrypted_key.encode()).decode()
                    new_encrypted_key = new_fernet.encrypt(
                        decrypted_key.encode()
                    ).decode()
                    cursor.execute(
                        "UPDATE auth_keys SET key = ? WHERE provider = ?",
                        (new_encrypted_key, provider),
                    )
                conn.commit()

                # Replace the old key with the new key in the key file
                with open("encryption.key", "wb") as key_file:
                    key_file.write(new_key)

                # Update the instance's key and Fernet object
                self.key = new_key
                self.fernet = new_fernet
            except (sqlite3.Error, InvalidToken) as e:
                print(f"Error rotating encryption key: {e}")
                raise
            finally:
                conn.close()
