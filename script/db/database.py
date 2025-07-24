# create_schema.py
import sqlite3

DB_PATH = "pill_metadata.db"

def create_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pill_metadata (
        category_id INTEGER PRIMARY KEY,
        name TEXT,
        name_en TEXT,
        shape TEXT,
        company TEXT,
        material TEXT,
        color TEXT,
        form_code_name TEXT,
        drug_S TEXT,
        dl_mapping_code TEXT,
        img_key TEXT,
        print_front TEXT,
        print_back TEXT
    )
    """)
    conn.commit()
    conn.close()



def insert_category_to_db(conn, category_id, category_name):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO pill_metadata (category_id, name)
        VALUES (?, ?)
    """, (int(category_id), category_name))
    conn.commit()
