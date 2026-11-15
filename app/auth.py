"""
Authentication and user management for Student Academic Risk Management System.
Supports role-based access for Students, Teachers/Admins, with demo accounts and dynamic lookup.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
USERS_FILE = ROOT_DIR / "data" / "users.json"

# Default accounts for instant demonstration
DEFAULT_USERS = [
    {
        "username": "teacher",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "teacher",
        "name": "Prof. Alan Turing",
        "department": "Computer Science & Engineering",
        "college": "Apex Institute of Technology",
        "student_id": None
    },
    {
        "username": "admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "teacher",
        "name": "Dr. Sarah Mitchell (Dean)",
        "department": "Academic Affairs",
        "college": "Apex Institute of Technology",
        "student_id": None
    },
    {
        "username": "S1000",
        "password_hash": hashlib.sha256("student123".encode()).hexdigest(),
        "role": "student",
        "name": "Alex Morgan",
        "department": "Computer Science",
        "college": "Apex Institute of Technology",
        "student_id": "S1000"
    },
    {
        "username": "S1002",
        "password_hash": hashlib.sha256("student123".encode()).hexdigest(),
        "role": "student",
        "name": "Emily Watson",
        "department": "Information Science",
        "college": "Apex Institute of Technology",
        "student_id": "S1002"
    },
    {
        "username": "S1013",
        "password_hash": hashlib.sha256("student123".encode()).hexdigest(),
        "role": "student",
        "name": "Daniel Craig",
        "department": "Electrical Engineering",
        "college": "Apex Institute of Technology",
        "student_id": "S1013"
    }
]

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def init_users_file():
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_USERS, f, indent=2)

def load_users():
    init_users_file()
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return list(DEFAULT_USERS)

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def authenticate(username, password, selected_role):
    username = (username or "").strip()
    password = (password or "").strip()
    if not username or not password:
        return None, "Please provide both username and password."
    
    users = load_users()
    h = _hash_pw(password)
    
    # 1. Match against stored users
    for u in users:
        if u["username"].lower() == username.lower():
            if u["password_hash"] == h:
                # Teachers can log in as either teacher or admin
                if selected_role == "teacher" and u["role"] in ["teacher", "admin"]:
                    return u, None
                elif selected_role == "student" and u["role"] == "student":
                    return u, None
                else:
                    return None, f"Account '{username}' is registered as a {u['role'].capitalize()}, not a {selected_role.capitalize()}."
            else:
                return None, "Incorrect password."
                
    # 2. Allow any student ID in dataset to log in with default student password "student123"
    if selected_role == "student":
        # Check if student ID exists in dataset
        csv_path = ROOT_DIR / "data" / "students_dataset.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                match = df[df['student_id'].astype(str).str.upper() == username.upper()]
                if not match.empty:
                    if password == "student123":
                        return {
                            "username": username.upper(),
                            "password_hash": h,
                            "role": "student",
                            "name": f"Student {username.upper()}",
                            "department": "Engineering & Technology",
                            "college": "Apex Institute of Technology",
                            "student_id": username.upper()
                        }, None
                    else:
                        return None, "Incorrect password. (Default student password is 'student123')"
            except Exception:
                pass

    return None, "Invalid credentials. Please check your username/ID and password."

