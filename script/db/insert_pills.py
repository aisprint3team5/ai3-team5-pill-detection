
def insert_pill_metadata_from_json(json_dir):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for file in os.listdir(json_dir):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data.get("images") or not data.get("categories"):
            continue

        img = data["images"][0]
        cat = data["categories"][0]
        try:
            cur.execute("""
            INSERT OR IGNORE INTO pill_metadata (
                category_id, name, name_en, shape, company,
                material, color, form_code_name, drug_S,
                dl_mapping_code, img_key, print_front, print_back
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cat["id"],
                cat["name"],
                img.get("dl_name_en"),
                img.get("drug_shape"),
                img.get("dl_company"),
                img.get("dl_material_en"),
                img.get("color_class1"),
                img.get("form_code_name"),
                img.get("drug_S"),
                img.get("dl_mapping_code"),
                img.get("img_key"),
                img.get("print_front"),
                img.get("print_back")
            ))
        except Exception as e:
            print(f"Failed on {file}: {e}")

    conn.commit()
    conn.close()


