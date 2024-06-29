import os
import pandas as pd
from openai import OpenAI   
from tqdm import tqdm
from dotenv import load_dotenv
import sys

print('Hello, world.')

load_dotenv()
openAI_api_key = os.environ.get('CHAT_GPT_API_KEY')

if not openAI_api_key: 
    print("Set your openAI key, silly")
    sys.exit(1)

client = OpenAI(api_key=openAI_api_key)

# Load the provided CSV file
file_path = 'spanish1000.csv'  # The CSV file is in the same folder as the script
df = pd.read_csv(file_path)

# Function to create a sentence using the OpenAI API
def generate_sentence(word, language):
    prompt = f"Create a sentence in {language} using the word '{word}'."
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=50)
    sentence = response.choices[0].message.content.strip()
    return sentence

# Generate Spanish sentences with a progress bar
print('Creating Spanish sentences')
df['Spanish Sentence'] = [generate_sentence(word, 'Spanish') for word in tqdm(df['Spanish word'], desc="Spanish Sentences")]

# Generate English translations with a progress bar
print('Creating English translations')
df['English Sentence'] = [generate_sentence(sentence, 'English') for sentence in tqdm(df['Spanish Sentence'], desc="English Translations")]

# Save the updated dataframe to a new CSV file
output_file_path = '2spanish1000_with_sentences.csv'  # Save the file in the same folder
df.to_csv(output_file_path, index=False)

print(f"Updated file saved to {output_file_path}")
