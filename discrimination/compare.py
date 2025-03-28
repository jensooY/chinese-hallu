import json

# 加载标准答案
with open('data.json', 'r', encoding='utf-8') as file:
    standard_answers = {item['question_id']: item['answer'] for item in json.load(file)}

# 加载待检测的答案
with open('answers.json', 'r', encoding='utf-8') as file:
    submitted_answers = {item['question_id']: item['answer'] for item in json.load(file)}

# 计算正确率
correct_count = 0
total_count = len(submitted_answers)

# 检测答案的正确性
incorrect_question_ids = []
for question_id, submitted_answer in submitted_answers.items():
    if question_id in standard_answers:
        if submitted_answer == standard_answers[question_id]:
            correct_count += 1
        else:
            incorrect_question_ids.append(question_id)
    else:
        print(f"Warning: Question ID {question_id} not found in standard answers.")

# 计算正确率
accuracy = correct_count / total_count if total_count > 0 else 0

# 输出结果
print(f"Incorrect Question IDs: {incorrect_question_ids}")
print(f"Accuracy: {accuracy * 100:.2f}%")
