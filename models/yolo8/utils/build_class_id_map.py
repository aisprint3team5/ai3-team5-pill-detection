import os
import json
import aiofiles
from tqdm.asyncio import tqdm as async_tqdm 
import asyncio 

async def build_class_id_map(root_dir):
    category_map = {}
    json_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                json_files.append(os.path.join(dirpath, filename))
    print(f"총 {len(json_files)}개의 JSON 파일 발견")

    print("JSON 파일 처리 중...")
    for file_path in async_tqdm(json_files, desc="Processing JSON files", unit="file"):
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
                            print(f"ID {cat_id} → '{category_map[cat_id]}' vs '{cat_name}' 충돌 발생 (파일: {os.path.basename(file_path)})")
                    else:
                        category_map[cat_id] = cat_name
        except Exception as e:
            print(f"[에러] {os.path.basename(file_path)} 처리 중 오류: {e}") 

    sorted_items = sorted(category_map.items()) 
    yolo_class_names = [name for _, name in sorted_items]

    return category_map, yolo_class_names
