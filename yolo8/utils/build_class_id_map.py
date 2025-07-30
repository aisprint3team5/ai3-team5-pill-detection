import os
import json
import aiofiles

async def build_class_id_map(root_dir):
    category_map = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                file_path = os.path.join(dirpath, filename)
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        contents = await f.read()
                        data = json.loads(contents)

                        for cat in data.get('categories', []):
                            cat_id = cat.get('id')
                            cat_name = cat.get('name')

                            if cat_id is not None and cat_name is not None:
                                if cat_id in category_map:
                                    if category_map[cat_id] != cat_name:
                                        print(f"ID {cat_id} → '{category_map[cat_id]}' vs '{cat_name}' 충돌 발생")
                                else:
                                    category_map[cat_id] = cat_name
                except Exception as e:
                    print(f"[에러] {file_path} 처리 중 오류: {e}")

    # category_id 순으로 YOLO 클래스 리스트 구성
    sorted_items = sorted(category_map.items())  # [(id, name), ...]
    yolo_class_names = [name for _, name in sorted_items]

    return category_map, yolo_class_names
