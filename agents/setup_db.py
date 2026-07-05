"""
ArthaSetu - Synthetic Customer Database Setup

Creates a SQLite database with 7 realistic synthetic customer profiles
covering all classification cases:
- 3 ambiguous (signals conflict, LLM needed)
- 2 clearly Type A (exposure gap)
- 2 clearly Type B (convenience gap)

Run this once to create/reset the database.
Note: All customer data is entirely synthetic and illustrative.
"""

import sqlite3
import os

DB_PATH = "data/customers.db"


def create_database():
    # Ensure data/ directory exists
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop and recreate for a clean slate
    cursor.execute("DROP TABLE IF EXISTS customers")

    cursor.execute("""
        CREATE TABLE customers (
            customer_id             TEXT PRIMARY KEY,
            name                    TEXT NOT NULL,
            profession              TEXT NOT NULL,
            education               TEXT NOT NULL,
            monthly_income          INTEGER NOT NULL,
            weekly_login_frequency  INTEGER NOT NULL,
            digital_transaction_ratio REAL NOT NULL,
            account_type            TEXT NOT NULL
        )
    """)

    customers = [
        # --- Ambiguous cases ---
        # 1. High-income educated farmer (income/education say B, profession says A)
        ("CUST001", "Ramesh Kumar",   "farmer",           "postgraduate",  50000, 2, 0.35, "savings"),
        # 2. Rural teacher (profession/education say B, logins/digital ratio say A)
        ("CUST002", "Sunita Devi",    "teacher",          "undergraduate", 45000, 0, 0.05, "savings"),
        # 3. Urban kirana owner (income/profession say B, education/digital ratio say A)
        ("CUST003", "Mohan Lal",      "shop owner",       "high school",   30000, 1, 0.15, "savings"),

        # --- Clearly Type A (exposure gap) ---
        # 4. Domestic worker, very low everything
        ("CUST004", "Lakshmi Bai",    "domestic worker",  "none",           6000, 0, 0.02, "savings"),
        # 5. Daily wage laborer, very low everything
        ("CUST005", "Raju Singh",     "laborer",          "primary",        8000, 0, 0.03, "savings"),

        # --- Clearly Type B (convenience gap) ---
        # 6. Software engineer, high everything
        ("CUST006", "Priya Sharma",   "software engineer","postgraduate",  85000, 5, 0.92, "savings"),
        # 7. Bank clerk, consistently digital
        ("CUST007", "Arjun Mehta",    "bank employee",    "undergraduate", 35000, 3, 0.75, "savings"),
    ]

    cursor.executemany("""
        INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, customers)

    conn.commit()
    conn.close()
    print(f"Database created at {DB_PATH}")
    print(f"Inserted {len(customers)} customer records.\n")

    # Quick verification read
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT customer_id, name, profession, monthly_income FROM customers")
    print("Customers in database:")
    for row in cursor.fetchall():
        print(f"  {row[0]} | {row[1]:<16} | {row[2]:<18} | Rs.{row[3]:,}")
    conn.close()


if __name__ == "__main__":
    create_database()