# tokenization based on  book Build A Large Language Model -- Raschka
# ------------------------------------------
# SIMPLE TOKENIZER - EDUCATIONAL VERSION
# Inspired by "Build a Large Language Model"
# ------------------------------------------

import urllib.request
import os
import re


# ============================================================
# STEP 1: DOWNLOAD THE TEXT FILE
# ============================================================

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
file_path = "the_verdict.txt"


def download_file(url, destination):
    """
    Download the file only if it does not already exist.
    """
    if not os.path.exists(destination):
        try:
            print(f"Downloading file from: {url}")
            urllib.request.urlretrieve(url, destination)
            print(f"Download complete: {destination}")
        except Exception as e:
            print(f"Download failed: {e}")
    else:
        print(f"File already exists: {destination}")


download_file(url, file_path)


# ============================================================
# STEP 2: LOAD THE TEXT INTO MEMORY
# ============================================================

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"Total number of characters: {len(raw_text)}")
else:
    raise FileNotFoundError("The text file was not found.")


# ============================================================
# STEP 3: PREPROCESS THE TEXT
# ============================================================
# We split the text into tokens.
# A token here can be:
# - a word
# - punctuation
# - special symbols like --
#
# Example:
# "Hello, world!"
# becomes:
# ["Hello", ",", "world", "!"]

def basic_tokenize(text):
    """
    Split text into tokens using punctuation and whitespace.
    """
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [item.strip() for item in tokens if item.strip()]
    return tokens


all_tokens_in_text = basic_tokenize(raw_text)

print(f"Total number of tokens after preprocessing: {len(all_tokens_in_text)}")


# ============================================================
# STEP 4: BUILD THE VOCABULARY
# ============================================================
# Vocabulary = mapping from token to integer ID
#
# Example:
# {
#   "Hello": 0,
#   ",": 1,
#   "world": 2,
#   "!": 3
# }

unique_tokens = sorted(list(set(all_tokens_in_text)))
vocab_v1 = {token: idx for idx, token in enumerate(unique_tokens)}

print(f"Vocabulary size for V1: {len(vocab_v1)}")


# ============================================================
# STEP 5: SIMPLE TOKENIZER VERSION 1
# ============================================================
# This tokenizer works only if ALL words already exist
# in the vocabulary.
#
# If a word is missing, it will crash.

class SimpleTokenizerV1:
    def __init__(self, vocab):
        """
        Store:
        - token -> integer
        - integer -> token
        """
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text):
        """
        Convert text into a list of integer IDs.
        """
        tokens = basic_tokenize(text)
        ids = [self.str_to_int[token] for token in tokens]
        return ids

    def decode(self, ids):
        """
        Convert a list of integer IDs back into text.
        """
        tokens = [self.int_to_str[idx] for idx in ids]
        text = " ".join(tokens)

        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.:;?!")\'])', r'\1', text)
        return text


# Create tokenizer V1
tokenizer_v1 = SimpleTokenizerV1(vocab_v1)


# Test V1 with text that exists in the story
sample_text_v1 = "I HAD always thought Jack Gisburn rather a cheap genius"

encoded_v1 = tokenizer_v1.encode(sample_text_v1)
decoded_v1 = tokenizer_v1.decode(encoded_v1)

print("\n================ TOKENIZER V1 ================")
print("Input text:")
print(sample_text_v1)
print("\nEncoded IDs:")
print(encoded_v1)
print("\nDecoded text:")
print(decoded_v1)


# ============================================================
# STEP 6: SIMPLE TOKENIZER VERSION 2
# ============================================================
# Problem with V1:
# If a token does not exist in the vocabulary, the code crashes.
#
# Solution:
# Add special token <|unk|> for unknown words.
# Also add <|endoftext|> to mark text boundaries.

special_tokens = ["<|endoftext|>", "<|unk|>"]

# Start from the original unique tokens
unique_tokens_v2 = sorted(list(set(all_tokens_in_text)))

# Add special tokens
unique_tokens_v2.extend(special_tokens)

# Create vocabulary for V2
vocab_v2 = {token: idx for idx, token in enumerate(unique_tokens_v2)}

print(f"\nVocabulary size for V2: {len(vocab_v2)}")


class SimpleTokenizerV2:
    def __init__(self, vocab):
        """
        Store:
        - token -> integer
        - integer -> token
        """
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    def encode(self, text):
        """
        Convert text into token IDs.
        Unknown words are replaced by <|unk|>.
        """
        tokens = basic_tokenize(text)

        processed_tokens = [
            token if token in self.str_to_int else "<|unk|>"
            for token in tokens
        ]

        ids = [self.str_to_int[token] for token in processed_tokens]
        return ids

    def decode(self, ids):
        """
        Convert token IDs back into text.
        """
        tokens = [self.int_to_str[idx] for idx in ids]
        text = " ".join(tokens)

        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.:;?!")\'])', r'\1', text)
        return text


# Create tokenizer V2
tokenizer_v2 = SimpleTokenizerV2(vocab_v2)


# ============================================================
# STEP 7: TEST TOKENIZER V2
# ============================================================
# Here we intentionally use words that may not exist
# in the original story, so they can become <|unk|>.

text1 = "Hello do you want tea"
text2 = "In the sunlit terraces of the palace"

combined_text = text1 + " <|endoftext|> " + text2

encoded_v2 = tokenizer_v2.encode(combined_text)
decoded_v2 = tokenizer_v2.decode(encoded_v2)

print("\n================ TOKENIZER V2 ================")
print("Input text:")
print(combined_text)
print("\nEncoded IDs:")
print(encoded_v2)
print("\nDecoded text:")
print(decoded_v2)


# ============================================================
# STEP 8: SHOW A SMALL EXPLANATION
# ============================================================

print("\n================ EDUCATIONAL SUMMARY ================")
print("V1:")
print("- Works only if all words already exist in the vocabulary.")
print("- If a word is missing, it crashes.")

print("\nV2:")
print("- Adds <|unk|> for unknown words.")
print("- Adds <|endoftext|> as a special token.")
print("- More robust than V1.")