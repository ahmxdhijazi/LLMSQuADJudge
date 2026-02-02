from openai import OpenAI
import os
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

# MMR Prompting within the character limit
PROMPT = "An AI to review piles of industrial blueprints and contract paperwork to assist a growing industrial companies needs."

def generate_and_save(quality_setting, filename):
    print(f"Generating {quality_setting} quality image.")
    
    # Image Generation Call
    response = client.images.generate(
        model="gpt-image-1",
        prompt=PROMPT,
        n=1,
        size="1024x1024", 
        quality=quality_setting  # Toggle between 'low' and 'medium'
    )

    # Extract Base64 via the example
    image_base64 = response.data[0].b64_json

    # Decode and Save
    with open(filename, "wb") as f:
        f.write(base64.b64decode(image_base64))
    
    print(f"Success! Saved to {filename}\n")

def main():
    # Pass twice for low and medium quality images and save them directly
    generate_and_save(quality_setting="low", filename="mmr_project_low.png")
    generate_and_save(quality_setting="medium", filename="mmr_project_med.png")

if __name__ == "__main__":
    main()