import sqlite3
from cryptography.fernet import Fernet
import os
from typing import Optional


class Auth:
    _instance = None

    def __new__(cls):
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
        if os.path.exists(key_file):
            with open(key_file, "rb") as file:
                return file.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as file:
                file.write(key)
            return key

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_keys
            (provider TEXT PRIMARY KEY, key TEXT)
        """)
        conn.commit()
        conn.close()

    def set_key(self, provider: str, key: str):
        encrypted_key = self.fernet.encrypt(key.encode()).decode()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "REPLACE INTO auth_keys (provider, key) VALUES (?, ?)",
            (provider, encrypted_key),
        )
        conn.commit()
        conn.close()

    def get_key(self, provider: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT key FROM auth_keys WHERE provider = ?", (provider,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return self.fernet.decrypt(result[0].encode()).decode()
        return None
