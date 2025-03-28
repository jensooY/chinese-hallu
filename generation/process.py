import json

def process_questions(input_file, output_file):
    # Read the original dataset
    with open(input_file, 'r', encoding='utf-8') as infile:
        questions_data = json.load(infile)
    
    # Extract only the relevant fields (Question and question_id)
    processed_data = [
        {"Question": item["Question"], "question_id": item["question_id"]}
        for item in questions_data
    ]
    
    # Save the processed data to a new file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=4)
    
    print(f"Processed data saved to {output_file}")

# Example usage
input_file = 'question.json'
output_file = 'processed_question.json'
process_questions(input_file, output_file)
