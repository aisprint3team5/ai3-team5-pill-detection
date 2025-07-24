import os
import json
import sqlite3

DB_PATH = "pill_metadata.db"

def get_metadata_by_category_id(category_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM pill_metadata WHERE category_id = ?", (category_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None

    columns = ["category_id", "name", "name_en", "shape", "company",
               "material", "color", "form_code_name", "drug_S",
               "dl_mapping_code", "img_key","print_front", "print_back"]
    return dict(zip(columns, row))
