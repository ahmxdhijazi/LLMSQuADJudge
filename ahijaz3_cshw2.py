import json
import time # For sleep functionality, to wait between status checks
import argparse  # For the CLI argument parsing
from dotenv import load_dotenv
from openai import OpenAI
import os # to access OpenRouter API key from environment variables
import datetime

# Load environment variables from .env file
load_dotenv()
client = OpenAI()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def get_possible_qas(data, limit=500):
    """
    Extracts the first 'limit' questions where is_impossible is False.
    """
    results = []
    # Iterate through the main 'data' list (Topics)
    for title in data['data']:
        # Iterate through 'paragraphs' in that topic
        for paragraph in title['paragraphs']:
            # Iterate through the specific questions/answers
            for qa in paragraph['qas']:
                # skip the 'impossible' questions
                if qa['is_impossible'] is False:
                    #Create a clean dictionary entry to ignore unnecessary fields
                    entry = {
                        "id": qa['id'],
                        "question": qa['question'],
                        # Save ALL valid answers as a list of strings
                        "answers": [ans['text'] for ans in qa['answers']]
                    }
                    results.append(entry) # add the entry to results
        
                # check limit of 500 possible questions
                if len(results) >= limit:
                    return results # immediately stop and return results
    return results

def create_batch_file(batch_data, jsonl_filename):
    '''
    PREPARE BATCH REQUEST FILE
    '''
    print(f"Creating batch file '{jsonl_filename}'")

    # Open the file first, then write inside the loop
    with open(jsonl_filename, 'w', encoding='utf-8') as file:
        for entry in batch_data:
            # Construct the API request
            request_object = {
                "custom_id": entry['id'], # Matches the answer to the question later
                "method": "POST",
                "url": "/v1/chat/completions", #required for non-standard/real-time requests (batch mode)
                "body": {
                    "model": "gpt-5-nano",
                    "reasoning_effort": "minimal", # Use minimal reasoning for concise answers, specified in rubric
                    "messages": [
                        {"role": "system", "content": "You are an expert scholar with encyclopedic knowledge of history and world facts. "
                            "Answer using only a short phrase, date, or entity. Do not use full sentences."}, # Not exactly the same answer output, but solid
                        {"role": "user", "content": entry['question']}
                    ],
                    "max_completion_tokens": 500 # Limit the response length, as to not get a whole paragraph. (concise)
                }
            }
            # Write the line
            file.write(json.dumps(request_object) + '\n')

def submit_batch_job(jsonl_filename, description="squad-homework"):
    """
    UPLOAD BATCH FILE AND CREATE BATCH JOB
    """
    print("Uploading file to OpenAI.")
    batch_file = client.files.create(
      file=open(jsonl_filename, "rb"),
      purpose="batch" # Set purpose to 'batch' for batch processing
    )

    print(f"Creating Batch Job with File ID: {batch_file.id}.")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description} # Now uses the argument passed in
    )

    print(f"Batch id {batch_job.id} Submitted!\n")
    return batch_job

def check_and_download_results(batch_id, output_filename="batch_output.jsonl"):
    '''
    CHECK STATUS AND DOWNLOAD RESULTS
    '''
    print(f"Waiting for results for Batch ID: {batch_id}")

    while True:
        # Retrieve the current status of the batch using the ID string
        batch_status = client.batches.retrieve(batch_id)
        status = batch_status.status
        print(f"Current Status: {status}.")

        # Check the status and update the CL
        if status == 'completed':
            print("\nComplete! Downloading results.")
            break
        elif status == 'failed':
            print("\nFailed!")
            print(batch_status.errors)
            return # changed from exit() to return so script doesn't hard kill if integrated
        elif status in ['expired', 'cancelled']:
            print(f"\n{status}!")
            return
        
        # Wait 15 seconds before checking again
        time.sleep(15)


        # Attempting to name the files automatically based on date and model
        if not output_filename:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            description = batch_status.metadata.get("description", "")
        
        if "nano" in description:
            output_filename = f"gpt-5-nano-{today}-hw2.json"
        elif "qwen" in description:
            output_filename = f"qwen3-8b-{today}-hw2.json"
        else:
            output_filename = f"graded_output_{today}.json"

    # Check for a success file
    if batch_status.output_file_id:
        print(f"Downloading Success Results (ID: {batch_status.output_file_id}).")
        file_response = client.files.content(batch_status.output_file_id)
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(file_response.text)
        print(f"Saved to '{output_filename}'")

        print("\n--- FINAL SCORE ---")
        correct_count = 0
        total_count = 0
        
        # Re-read the file we just saved to parse the grades
        with open(output_filename, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Navigate the deep Batch response structure
                # response -> body -> choices -> message -> content
                content_str = data['response']['body']['choices'][0]['message']['content']
                
                # The content itself is a JSON string
                grade_data = json.loads(content_str)
                
                # Tally the score
                if grade_data.get('score') is True:
                    correct_count += 1
                total_count += 1
        
        if total_count > 0:
            percentage = (correct_count / total_count) * 100
            print(f"Accuracy: {percentage:.2f}%")
        else:
            print("whoops.")

    print(f"\nResults saved.")


#---- END OF OPENAI portion (Step 2) -> BEGIN OPENROUTER SERIAL FUNCTION  ----#
def run_serial_openrouter(questions, model="qwen/qwen3-8b"):
    """
    Run questions one by one (serially) using OpenRouter. Robust retries for empty answers...
    """
    print(f"Starting Serial Run for {len(questions)} items using OpenRouter.")
    
    # Initialize a specific client for OpenRouter
    # Variable OPENROUTER_API_KEY defined at the top
    or_client = OpenAI(
        base_url="https://openrouter.ai/api/v1", # new connection object for OpenRouter
        api_key=OPENROUTER_API_KEY # use the OpenRouter API key
    )

    results = [] # To store results

    for i, entry in enumerate(questions):
        print(f"Processing {i+1}/{len(questions)}: ID {entry['id']}")
        
        answer_text = ""
        attempts = 0
        
        # Retry loop to handle empty responses, hopefully solves my issues ): AND IT DID, made sure I got an answer for way more this time
        while attempts < 3:
            try:
                # Send the request
                completion = or_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert scholar with encyclopedic knowledge of history and world facts. Answer using only a short phrase, date, or entity. Do not use full sentences."},
                        {"role": "user", "content": entry['question']}
                    ],
                    # 'max_tokens' instead of max_completion_tokens
                    max_tokens=1000
                )
                
                # Extract the answer
                content = completion.choices[0].message.content
                
                if content and content.strip():
                    answer_text = content.strip()
                    break # We got a valid answer, exit retry loop
                else:
                    print(f"Empty response. Retrying {attempts+1}/3...")
            
            except Exception as e:
                print(f"Error: {e}")
            attempts += 1
            time.sleep(2) # wait a bit if we need to retry
            
        # Save the result in json
        result_entry = {
            "id": entry['id'], # Keep the ID so we can match the 1/500 later
            "model_answer": answer_text
        }
        results.append(result_entry)

        time.sleep(1) # brief pause to avoid rate limits, waiting each second between requests. ALSO HELPED

    return results

def load_student_answers(filename, source_type):
    '''
    HELPER TO READ OUTPUT FILES (JSONL OR JSON)
    '''
    answers = {}
    print(f"Loading answers from {filename}.")
    
    try:
        if source_type == 'serial':
            # Serial file is a standard JSON list
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    answers[entry['id']] = entry['model_answer']

        elif source_type == 'batch':
            # Batch output is a JSONL file
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    response = json.loads(line)
                    # Batch API puts the custom_id at the top level
                    q_id = response['custom_id']
                    # The answer is nested deep inside
                    ans_text = response['response']['body']['choices'][0]['message']['content']
                    answers[q_id] = ans_text
    except FileNotFoundError:
        print(f"Could not find {filename}")
        
    print(f"Loaded {len(answers)} answers.")
    return answers


def create_grading_batch(student_answers, squad_data, output_filename, judge_model="gpt-5-mini"):
    '''
    PREPARE GRADING BATCH FILE (TEACHER MODE)
    '''
    print(f"Creating grading batch file: {output_filename}.")
    
    # Build Ground Truth Lookup {id: {question, correct_answers}}
    ground_truth = {}
    for title in squad_data['data']:
        for paragraph in title['paragraphs']:
            for qa in paragraph['qas']:
                if not qa['is_impossible']:
                    ground_truth[qa['id']] = {
                        "question": qa['question'],
                        "correct_answers": [ans['text'] for ans in qa['answers']]
                    }

    # Write the Judge Requests
    with open(output_filename, 'w', encoding='utf-8') as f:
        count = 0
        for q_id, student_ans in student_answers.items():
            
            if q_id not in ground_truth: continue 

            truth_data = ground_truth[q_id]
            correct_answers_str = "\n".join(truth_data['correct_answers'])

            # prompt from Appendix 1
            user_content = (
                f"Question: {truth_data['question']}\n"
                f"Student’s Response: {student_ans}\n"
                f"Possible Correct Answers:\n"
                f"{correct_answers_str}"
            )

            # JSON Schema for 'Structured Outputs'
            json_schema = {
                "name": "grading_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {"type": "string", "description": "A short explanation of why the student’s answer was correct or incorrect."},
                        "score": {"type": "boolean", "description": "true if the student’s answer was correct, false if it was incorrect"}
                    },
                    "required": ["explanation", "score"],
                    "additionalProperties": False
                }
            }

            request_object = {
                "custom_id": q_id, 
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": judge_model,
                    "messages": [
                        {"role": "system", "content": "You are a teacher tasked with determining whether a student’s answer to a question was correct, based on a set of possible correct answers."},
                        {"role": "user", "content": user_content}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": json_schema
                    }
                }
            }
            f.write(json.dumps(request_object) + '\n')
            count += 1
            
    print(f"Prepared {count} grading requests in '{output_filename}'.")


def main():
    # "python .py --mode full" or "python script.py" for test
    parser = argparse.ArgumentParser(description="SQuAD Answer Processor")
    parser.add_argument('--mode', choices=['test', 'full', 'serial', 'grade', 'download'], default='test', 
                        help="Modes: test/full (Batch Gen), serial (OpenRouter Gen), grade (Batch Judge), download (Get Results)")
    # (Optional) arg to specify Batch ID manually if needed
    parser.add_argument('--id', type=str, help="Batch ID for downloading results manually")
    args = parser.parse_args()

    '''
    LOAD DATA FROM SQUAD DATASET JSON
    '''
    print("Loading JSON data from file (:")
    try:
        with open('dev-v2.0.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        # Call the function and update the user
        possible_questions = get_possible_qas(data, limit=500)
        print(f"Successfully loaded {len(possible_questions)} questions!") # should be 500
    except FileNotFoundError:
        print("Warning: 'dev-v2.0.json' not found. Ensure it is in the directory.")
        possible_questions = []
        data = None

    if args.mode == 'grade':
            print("\nBeginning Grading Mode.")
            # Load the answers generated previously by name
            nano_answers = load_student_answers("batch_output.jsonl", "batch")
            qwen_answers = load_student_answers("serial_output.json", "serial")

            # Create judge model
            judge_model = "gpt-5-mini" 
            
            print(f"Generating Grading Batches using Judge: {judge_model}")
            create_grading_batch(nano_answers, data, "judge_nano_input.jsonl", judge_model)
            create_grading_batch(qwen_answers, data, "judge_qwen_input.jsonl", judge_model)

            # Submit Jobs
            print("\nSubmitting GPT-Nano Grading Job.")
            job1 = submit_batch_job("judge_nano_input.jsonl", description="grading-nano")
            print("\nSubmitting Qwen Grading Job.")
            job2 = submit_batch_job("judge_qwen_input.jsonl", description="grading-qwen")

            # At the end to download after the job is complete
            print("Job Submitted.")
            print(f"1. Nano Judge Batch ID: {job1.id}")
            print(f"2. Qwen Judge Batch ID: {job2.id}")
            return 
        
    elif args.mode == 'download':
        # Check the CLI for an arument, we need that id fr
        if not args.id:
            print("--batch_id needed to download.")
            return
        
        # Save initially as a generic name. AUTOMATIC NAMING HANDLED IN FUNCTION, yessir!
        output_name = f"graded_output_{args.id[-6:]}.jsonl"
        print(f"Checking status for {args.id}...")
        check_and_download_results(args.id, output_name)
        return

    # Logic to decide test vs full based on CLI argument
    elif args.mode == 'serial':
        if not possible_questions: return
        # Select Data for serial processing
        serial_data = possible_questions # Now processing all of them serially
        # Run the method to process serially using OpenRouter
        results = run_serial_openrouter(serial_data)
        
        # Save the results
        output_filename = "serial_output.json"
        print(f"Saving serial results to {output_filename}...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print("Done!")
        return # Stop here, don't do the batch stuff below
    
    elif args.mode == 'test':
        if not possible_questions: return
        batch_data = possible_questions[:4]
        jsonl_filename = "batch_input_TEST.jsonl"
        print(f"TEST: Processing {len(batch_data)} items.")
        
        create_batch_file(batch_data, jsonl_filename) # Create File
        batch_job = submit_batch_job(jsonl_filename) # Submit Job
        check_and_download_results(batch_job.id, "batch_output.jsonl") # Check and Download

    else: # full
        if not possible_questions: return
        batch_data = possible_questions
        jsonl_filename = "batch_input_FULL.jsonl"
        print(f"FULL: Processing {len(batch_data)} items.")

        create_batch_file(batch_data, jsonl_filename) # Create File
        batch_job = submit_batch_job(jsonl_filename) # Submit Job
        check_and_download_results(batch_job.id, "batch_output.jsonl") # Check and Download

if __name__ == "__main__":
    main()