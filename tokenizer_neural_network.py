# ------------------------------------------
# EDUCATIONAL GPT-LIKE TOKENIZER
# + SIMPLE NEURAL NETWORK CONTINUATION
# ------------------------------------------

import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTStyleTokenizer:
    """
    A simplified educational tokenizer inspired by GPT-style tokenization.
    """

    def __init__(self, vocab):
        """
        Store token -> id and id -> token mappings.
        """
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    @staticmethod
    def pretokenize(text):
        """
        Split text into GPT-like chunks.
        """
        pattern = r""" ?[A-Za-z]+| ?\d+| ?[^\w\s]|\n"""
        tokens = re.findall(pattern, text)
        return tokens

    @classmethod
    def build_vocab(cls, text, special_tokens=None, max_vocab_size=200):
        """
        Build a vocabulary from the most frequent pre-tokens.
        """
        if special_tokens is None:
            special_tokens = ["<|endoftext|>", "<|unk|>"]

        pretokens = cls.pretokenize(text)
        counter = Counter(pretokens)

        most_common_tokens = [
            token for token, _ in counter.most_common(max_vocab_size - len(special_tokens))
        ]

        vocab_list = special_tokens + most_common_tokens
        vocab = {token: idx for idx, token in enumerate(vocab_list)}
        return vocab

    def encode(self, text):
        """
        Convert text into token ids.
        """
        pretokens = self.pretokenize(text)

        ids = []
        for token in pretokens:
            if token in self.str_to_int:
                ids.append(self.str_to_int[token])
            else:
                ids.append(self.str_to_int["<|unk|>"])

        return ids

    def decode(self, ids):
        """
        Convert token ids back into text.
        """
        tokens = [self.int_to_str[idx] for idx in ids]
        text = "".join(tokens)
        return text


class TinyLanguageModel(nn.Module):
    """
    A very small neural network for next-token prediction.

    Architecture:
    token ids
    -> embedding
    -> average across sequence
    -> linear layer
    -> logits over vocabulary
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        # Convert token ids into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Convert the hidden representation into vocabulary logits
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        """
        input_ids shape: (batch_size, sequence_length)
        """
        # Step 1: Convert token ids to embeddings
        x = self.embedding(input_ids)
        # x shape: (batch_size, sequence_length, embed_dim)

        # Step 2: Average across the sequence dimension
        x = x.mean(dim=1)
        # x shape: (batch_size, embed_dim)

        # Step 3: Produce logits for each token in the vocabulary
        logits = self.linear(x)
        # logits shape: (batch_size, vocab_size)

        return logits


# --------------------------------------------------
# STEP 1: TRAINING TEXT
# --------------------------------------------------

training_text = """
Hello world!
Hello there!
I love learning about language models.
I love text generation.
Text generation is powerful.
Chat models generate text one token at a time.
"""

# --------------------------------------------------
# STEP 2: BUILD VOCABULARY
# --------------------------------------------------

vocab = GPTStyleTokenizer.build_vocab(training_text, max_vocab_size=100)
tokenizer = GPTStyleTokenizer(vocab)

print("Vocabulary:")
for token, idx in vocab.items():
    print(f"{idx}: {repr(token)}")

# --------------------------------------------------
# STEP 3: TOKENIZE INPUT TEXT
# --------------------------------------------------

sample_text = "Hello world! I love text generation."
encoded = tokenizer.encode(sample_text)

print("\nOriginal text:")
print(sample_text)

print("\nEncoded token ids:")
print(encoded)

print("\nPretokens:")
print(tokenizer.pretokenize(sample_text))

# --------------------------------------------------
# STEP 4: CONVERT IDS TO A TENSOR
# --------------------------------------------------

# Add batch dimension: shape becomes (1, sequence_length)
input_ids = torch.tensor([encoded], dtype=torch.long)

print("\nInput tensor shape:")
print(input_ids.shape)

# --------------------------------------------------
# STEP 5: CREATE THE NEURAL NETWORK
# --------------------------------------------------

vocab_size = len(vocab)
embed_dim = 16

model = TinyLanguageModel(vocab_size=vocab_size, embed_dim=embed_dim)

# --------------------------------------------------
# STEP 6: RUN THE INPUT THROUGH THE MODEL
# --------------------------------------------------

logits = model(input_ids)

print("\nLogits shape:")
print(logits.shape)

# --------------------------------------------------
# STEP 7: CONVERT LOGITS TO PROBABILITIES
# --------------------------------------------------

probs = F.softmax(logits, dim=-1)

print("\nProbabilities shape:")
print(probs.shape)

# --------------------------------------------------
# STEP 8: CHOOSE THE MOST LIKELY NEXT TOKEN
# --------------------------------------------------

next_token_id = torch.argmax(probs, dim=-1).item()

print("\nPredicted next token id:")
print(next_token_id)

predicted_token = tokenizer.decode([next_token_id])

print("\nPredicted next token:")
print(repr(predicted_token))

# --------------------------------------------------
# STEP 9: APPEND THE PREDICTED TOKEN TO THE TEXT
# --------------------------------------------------

generated_ids = encoded + [next_token_id]
generated_text = tokenizer.decode(generated_ids)

print("\nGenerated text:")
print(generated_text)