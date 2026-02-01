import json

with open('dev-v2.0.json', 'r', encoding='utf-8') as file:
    # 'data' here is the WHOLE file (the root dictionary)
    data = json.load(file)

extracted_qas = []
limit = 500 #acquire the first 500

# Iterate through the main 'data' list (Topics)
for title in data['data']:
    # Iterate through 'paragraphs' in that topic
    for paragraph in title['paragraphs']:
        # Iterate through the specific questions/answers
        for qa in paragraph['qas']:
            # skip the 'impossible' questions
            if qa['is_impossible'] is False:
                extracted_qas.append(qa)
    
            # check limit of 500 possible questions
            if len(extracted_qas) >= limit:
                break
        # breaks to escape the outer loops once we hit 500
        if len(extracted_qas) >= limit:
            break
    if len(extracted_qas) >= limit:
        break

print(f"{len(extracted_qas)} Possible Questions and Answers Loaded!")
# Print the first one to verify
print(extracted_qas[0])
