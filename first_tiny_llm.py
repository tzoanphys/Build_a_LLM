# ------------------------------------------
# EDUCATIONAL GPT-LIKE TOKENIZER
# + SIMPLE NEURAL NETWORK
# + TRAINER PART
# Comments are in English only
# ------------------------------------------

import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTStyleTokenizer:
    """
    A simplified educational tokenizer 
    It contains: vocabulary - encode - decode - pretokenization
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
        Split text into chunks/tokens.

        Example:
        "Hello world!"
        -> ["Hello", " world", "!"]
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
        Unknown tokens become <|unk|>.
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

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        """
        input_ids shape: (batch_size, sequence_length)
        """
        # Convert token ids to vectors
        x = self.embedding(input_ids)
        # x shape: (batch_size, sequence_length, embed_dim)

        # Average across sequence length
        x = x.mean(dim=1)
        # x shape: (batch_size, embed_dim)

        # Produce one score for each token in the vocabulary
        logits = self.linear(x)
        # logits shape: (batch_size, vocab_size)

        return logits


def create_training_examples(token_ids, context_size):
    """
    Create training examples for next-token prediction.

    Example:
    token_ids = [10, 20, 30, 40]
    context_size = 2

    Output:
    inputs  = [[10], [10, 20], [20, 30]]
    targets = [20, 30, 40]

    Explanation:
    - given [10] predict 20
    - given [10, 20] predict 30
    - given [20, 30] predict 40
    """
    inputs = []
    targets = []

    for i in range(1, len(token_ids)):
        start_index = max(0, i - context_size)
        context = token_ids[start_index:i]
        target = token_ids[i]

        inputs.append(context)
        targets.append(target)

    return inputs, targets


def pad_sequences(sequences, pad_value=0):
    """
    Pad sequences so they all have the same length.

    Example:
    [[5], [5, 7], [5, 7, 2]]
    ->
    [[0, 0, 5],
     [0, 5, 7],
     [5, 7, 2]]
    """
    max_len = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        num_pads = max_len - len(seq)
        padded_seq = [pad_value] * num_pads + seq
        padded.append(padded_seq)

    return padded


def train_model(model, inputs, targets, epochs=200, learning_rate=0.01):
    """
    Train the model using cross-entropy loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")


def generate_next_token(model, tokenizer, text):
    """
    Predict one next token for the given text.
    """
    model.eval()

    encoded = tokenizer.encode(text)
    input_tensor = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.argmax(probs, dim=-1).item()

    next_token = tokenizer.decode([next_token_id])
    return next_token_id, next_token


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
I love language models.
I love text.
Hello world!
Text generation is fun.
"""

# --------------------------------------------------
# STEP 2: BUILD VOCABULARY AND TOKENIZER
# --------------------------------------------------

vocab = GPTStyleTokenizer.build_vocab(training_text, max_vocab_size=100)
tokenizer = GPTStyleTokenizer(vocab)

print("Vocabulary:")
for token, idx in vocab.items():
    print(f"{idx}: {repr(token)}")

# --------------------------------------------------
# STEP 3: ENCODE THE TRAINING TEXT
# --------------------------------------------------

training_ids = tokenizer.encode(training_text)

print("\nEncoded training ids:")
print(training_ids)

print("\nPretokens in training text:")
print(tokenizer.pretokenize(training_text))

# --------------------------------------------------
# STEP 4: CREATE TRAINING EXAMPLES
# --------------------------------------------------

context_size = 4

inputs_list, targets_list = create_training_examples(training_ids, context_size)

print("\nFirst few raw training examples:")
for i in range(min(5, len(inputs_list))):
    print(f"Input ids: {inputs_list[i]} -> Target id: {targets_list[i]}")

# --------------------------------------------------
# STEP 5: PAD INPUT SEQUENCES
# --------------------------------------------------

padded_inputs = pad_sequences(inputs_list, pad_value=0)

# Convert to tensors
inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
targets_tensor = torch.tensor(targets_list, dtype=torch.long)

print("\nInputs tensor shape:")
print(inputs_tensor.shape)

print("\nTargets tensor shape:")
print(targets_tensor.shape)

# --------------------------------------------------
# STEP 6: CREATE THE MODEL
# --------------------------------------------------

vocab_size = len(vocab)
embed_dim = 32

model = TinyLanguageModel(vocab_size=vocab_size, embed_dim=embed_dim)

# --------------------------------------------------
# STEP 7: TRAIN THE MODEL
# --------------------------------------------------

train_model(
    model=model,
    inputs=inputs_tensor,
    targets=targets_tensor,
    epochs=200,
    learning_rate=0.01
)

# --------------------------------------------------
# STEP 8: TEST NEXT-TOKEN PREDICTION
# --------------------------------------------------

test_text = "Hello"
next_token_id, next_token = generate_next_token(model, tokenizer, test_text)

print("\nTest text:")
print(repr(test_text))

print("\nPredicted next token id:")
print(next_token_id)

print("\nPredicted next token:")
print(repr(next_token))

print("\nGenerated text:")
print(test_text + next_token)

# --------------------------------------------------
# STEP 9: TEST ANOTHER EXAMPLE
# --------------------------------------------------

test_text_2 = "I love"
next_token_id_2, next_token_2 = generate_next_token(model, tokenizer, test_text_2)

print("\nSecond test text:")
print(repr(test_text_2))

print("\nPredicted next token id:")
print(next_token_id_2)

print("\nPredicted next token:")
print(repr(next_token_2))

print("\nGenerated text:")
print(test_text_2 + next_token_2)