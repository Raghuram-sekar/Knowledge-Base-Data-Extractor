import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            db_user = os.getenv("POSTGRES_USER", "myuser")
            db_password = os.getenv("POSTGRES_PASSWORD", "mypassword")
            db_name = os.getenv("POSTGRES_DB", "mydatabase")
            db_host = "localhost"
            db_port = "5432"

            DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

            cls._instance.engine = create_engine(DATABASE_URL)
            cls._instance.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls._instance.engine)
            cls._instance.Base = declarative_base()

        return cls._instance

    def get_session(self):
        return self.SessionLocal()

    def get_base(self):
        return self.Base

db = Database()
