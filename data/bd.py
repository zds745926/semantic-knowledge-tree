import sqlite3
import json
import base64

def export_sqlite_to_json(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 获取所有表名（排除系统表）
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]
    
    all_data = {}
    for table in tables:
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        table_data = []
        
        for row in rows:
            row_dict = dict(row)
            # 转换所有 bytes 类型字段为 base64 字符串
            for key, value in row_dict.items():
                if isinstance(value, bytes):
                    row_dict[key] = base64.b64encode(value).decode('utf-8')
            table_data.append(row_dict)
        
        all_data[table] = table_data
    
    conn.close()
    
    # 保存为 JSON 文件
    with open('database_export.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 数据已导出到 database_export.json，共 {len(tables)} 个表。")

export_sqlite_to_json('knowledge_tree.db')
