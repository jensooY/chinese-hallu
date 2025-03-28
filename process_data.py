# 假设你的数据已经保存在一个名为 data.json 的文件中
import json

# 读取 JSON 数据
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 处理数据，只保留 question_id 和 question
processed_data = [{'question_id': item['question_id'], 'question': item['question']} for item in data]

# 将处理后的数据保存到一个新的文件中
with open('processed_data.json', 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, ensure_ascii=False, indent=4)

print("处理完成，数据已保存到 processed_data.json 文件中。")