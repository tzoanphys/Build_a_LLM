# tokenization by book Build A Large Language Model -- Raschka
import urllib.request
import os
import re

# --- STEP 1: DOWNLOAD THE DATA ---
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
file_path = "the_verdict.txt"

def download_file(url, destination):
    if not os.path.exists(destination):
        try:
            print(f"Attempting to download from: {url}")
            urllib.request.urlretrieve(url, destination)
            print(f"Successfully downloaded {destination}")
        except Exception as e:
            print(f"Download failed: {e}")
    else:
        print(f"{destination} already exists.")

download_file(url, file_path)

# Load the text
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"Total characters in story: {len(raw_text)}")
else:
    raise FileNotFoundError("The story file was not found. Check your internet.")

# --- STEP 2: DEFINE THE TOKENIZER CLASS ---
# We define this before creating the 'tokenizer' instance
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        # Split text into tokens based on punctuation and whitespace
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # Remove whitespace and empty strings
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Convert tokens to integer IDs
        ids = [self.str_to_int[token] for token in preprocessed]
        return ids

    def decode(self, ids):
        # Convert IDs back to tokens
        tokens = [self.int_to_str[i] for i in ids]
        # Join tokens and clean up spaces before punctuation
        text = " ".join(tokens)
        text = re.sub(r'\s+([,.?!")])', r'\1', text)
        return text

# --- STEP 3: PREPROCESS AND CREATE VOCABULARY ---
# Split the story into a list of words/punctuation
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
all_words = [item.strip() for item in preprocessed if item.strip()]

# Get unique tokens and sort them to create the ID mapping
unique_tokens = sorted(list(set(all_words)))
vocab = {token: integer for integer, token in enumerate(unique_tokens)}

print(f"Vocabulary Size: {len(vocab)}")

# --- STEP 4: USE THE TOKENIZER ---
tokenizer = SimpleTokenizerV1(vocab)

# Sample text from the story (must use words existing in the vocab for V1)
sample_text = "I HAD always thought Jack Gisburn rather a cheap genius"
ids = tokenizer.encode(sample_text)

print("\n--- Tokenization Results ---")
print(f"Input Text: {sample_text}")
print(f"Encoded IDs: {ids}")
print(f"Decoded Text: {tokenizer.decode(ids)}")

# --- STEP 5: SIMPLE TOKENIZER V2 (Handling Unknowns) ---

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        # Create the reverse mapping correctly
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # Replace words not in vocab with the <|unk|> token
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        tokens = [self.int_to_str[i] for i in ids]
        text = " ".join(tokens)
        # Clean up spaces before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# --- Preparing the Vocab for V2 ---
# We must add our special tokens to the unique_tokens list first
all_tokens = sorted(list(set(all_words)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab_v2 = {token: integer for integer, token in enumerate(all_tokens)}

# --- Usage ---
tokenizer_v2 = SimpleTokenizerV2(vocab_v2)

text1 = "Hello do you want tea"  # "tea" and "Hello" might be <|unk|>
text2 = "In the sunlit terraces of the place"
combined_text = "<|endoftext|>".join([text1, text2])

print(f"\nCombined Text for V2: {combined_text}")

ids_v2 = tokenizer_v2.encode(combined_text)
print(f"Encoded IDs (V2): {ids_v2}")
print(f"Decoded (V2): {tokenizer_v2.decode(ids_v2)}")