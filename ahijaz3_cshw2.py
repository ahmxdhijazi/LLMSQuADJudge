import json
import time # For sleep functionality, to wait between status checks
import argparse  # For the CLI argument parsing
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
client = OpenAI()

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
                        {"role": "system", "content": "Answer using only a short phrase, date, or entity. Do not use full sentences or restate the question, no explanation required."}, # Not exactly the same answer output, but solid
                        {"role": "user", "content": entry['question']}
                    ],
                    "max_completion_tokens": 100 # Limit the response length, as to not get a whole paragraph. (concise)
                }
            }
            # Write the line
            file.write(json.dumps(request_object) + '\n')

def submit_batch_job(jsonl_filename):
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
        metadata={"description": "squad-homework-test"}
    )

    print(f"Batch id {batch_job.id} Submitted!\n")
    return batch_job

def check_and_download_results(batch_job):
    '''
    CHECK STATUS AND DOWNLOAD RESULTS
    '''
    print("Waiting for results.")

    while True:
        # Retrieve the current status of the batch
        batch_status = client.batches.retrieve(batch_job.id)
        status = batch_status.status
        print(f"Current Status: {status}.")

        # Check the status and update the CL
        if status == 'completed':
            print("\nComplete! Downloading results.")
            break
        elif status == 'failed':
            print("\nFailed!")
            print(batch_status.errors)
            exit()
        elif status in ['expired', 'cancelled']:
            print(f"\n{status}!")
            exit()
        
        # Wait 20 seconds before checking again
        time.sleep(20)

    # --- GEMINI GENERATED LOGIC TO HANDLE ERROR, after some failures, this allowed me to see why clearly. ---
    # 1. Check if we have a success file
    if batch_status.output_file_id:
        print(f"Downloading Success Results (ID: {batch_status.output_file_id})...")
        file_response = client.files.content(batch_status.output_file_id)
        with open("batch_output.jsonl", 'w', encoding='utf-8') as f:
            f.write(file_response.text)
        print("SUCCESS: Saved to 'batch_output.jsonl'")

    # 2. Check if we have an error file (This is likely what happened)
    elif batch_status.error_file_id:
        print(f"Batch finished, but generated ERRORS (ID: {batch_status.error_file_id})...")
        error_response = client.files.content(batch_status.error_file_id)
        with open("batch_errors.jsonl", 'w', encoding='utf-8') as f:
            f.write(error_response.text)
        
        print("SAVED ERRORS TO 'batch_errors.jsonl'.")
        print("Please open that file to see why the requests failed.")
        
        # Print the first error to help you debug immediately
        print("\n--- PREVIEW OF ERROR ---")
        print(error_response.text.split('\n')[0])

    else:
        print("Strange... Job completed but no output or error file found.")
    # ---- END OF GEMINI GENERATED LOGIC ----
    print(f"\nResults saved.")

def main():
    # "python .py --mode full" or "python script.py" for test
    parser = argparse.ArgumentParser(description="SQuAD Batch Processor")
    parser.add_argument('--mode', choices=['test', 'full'], default='test', help="Run in 'test' mode (4 items) or 'full' mode (500 items)")
    args = parser.parse_args()

    '''
    LOAD DATA FROM SQUAD DATASET JSON
    '''
    print("Loading JSON data from file (:")
    # Load the JSON data from the file
    with open('dev-v2.0.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Call the function and update the user
    possible_questions = get_possible_qas(data, limit=500)
    print(f"Successfully loaded {len(possible_questions)} questions!") # should be 500

    # Logic to decide test vs full based on CLI argument
    if args.mode == 'test':
        batch_data = possible_questions[:4]
        jsonl_filename = "batch_input_TEST.jsonl"
        print(f"TEST: Processing {len(batch_data)} items.")
    else:
        batch_data = possible_questions
        jsonl_filename = "batch_input_FULL.jsonl"
        print(f"FULL: Processing {len(batch_data)} items.")

    create_batch_file(batch_data, jsonl_filename) # Create File
    batch_job = submit_batch_job(jsonl_filename) # Submit Job
    check_and_download_results(batch_job) # Check and Download

if __name__ == "__main__":
    main()