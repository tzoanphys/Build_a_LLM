import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tokenizer:
    """A simple tokenizer that converts text to token IDs and back."""
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    @staticmethod
    def pretokenize(text):
        pattern = r""" ?[A-Za-zÀ-ÿ]+| ?\d+| ?[^\w\s]|\n"""
        tokens = re.findall(pattern, text)
        return tokens

    @classmethod
    def build_vocab(cls, text, special_tokens=None, max_vocab_size=200):
        if special_tokens is None:
            special_tokens = ["<|pad|>", "<|unk|>"]

        pretokens = cls.pretokenize(text)
        counter = Counter(pretokens)

        most_common_tokens = [
            token for token, _ in counter.most_common(max_vocab_size - len(special_tokens))
        ]

        vocab_list = special_tokens + most_common_tokens
        vocab = {token: idx for idx, token in enumerate(vocab_list)}
        return vocab

    def encode(self, text):
        pretokens = self.pretokenize(text)
        ids = []

        for token in pretokens:
            if token in self.str_to_int:
                ids.append(self.str_to_int[token])
            else:
                ids.append(self.str_to_int["<|unk|>"])

        return ids

    def decode(self, ids):
        tokens = [self.int_to_str[idx] for idx in ids]
        text = "".join(tokens)
        return text


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.mean(dim=1)
        logits = self.linear(x)
        return logits


def create_training_examples(token_ids, context_size):
    inputs = []
    targets = []

    for i in range(1, len(token_ids)):
        start_idx = max(0, i - context_size)
        context = token_ids[start_idx:i]
        target = token_ids[i]

        inputs.append(context)
        targets.append(target)

    return inputs, targets


def pad_sequences(sequences, pad_value=0):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []

    for seq in sequences:
        padded_seq = seq + [pad_value] * (max_length - len(seq))
        padded_sequences.append(padded_seq)

    return padded_sequences


def train_model(model, inputs, targets, epochs=200, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


def generate_next_token(model, tokenizer, text, context_size=4):
    model.eval()

    encoded = tokenizer.encode(text)
    encoded = encoded[-context_size:]

    input_tensor = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=-1)
        predicted_id = torch.multinomial(probabilities[0], num_samples=1).item()

    predicted_token = tokenizer.int_to_str.get(predicted_id, "<|unk|>")
    return predicted_token



   # ============================================================
           


# ============================================================
# STEP 1: TRAINING TEXT
# ============================================================
training_text = """
Bonjour, je m'appelle Marie.
Je suis étudiante et j'aime apprendre de nouvelles choses.

Chaque matin, je bois un café et je lis un livre.
Le café est chaud et le livre est intéressant.

Le soleil se lève le matin et le ciel devient bleu.
Les oiseaux chantent dans les arbres.

J'aime marcher dans le parc.
Dans le parc, il y a des arbres, des fleurs et des bancs.

Les enfants jouent et les gens parlent ensemble.
Le parc est calme et agréable.

Parfois je vais au marché.
Au marché, il y a des fruits, des légumes et du pain.

Les pommes sont rouges.
Les bananes sont jaunes.
Les oranges sont sucrées.

Le soir, je rentre à la maison.
Je prépare le dîner et je mange avec ma famille.

Après le dîner, je lis ou j'écoute de la musique.
La musique est douce et relaxante.

Avant de dormir, je regarde les étoiles dans le ciel.
La nuit est calme et silencieuse.
"""


# --------------------------------------------------
# STEP 2: BUILD VOCABULARY AND TOKENIZER
# --------------------------------------------------
vocab = Tokenizer.build_vocab(training_text, special_tokens=["<|endoftext|>", "<|unk|>"], max_vocab_size=50)
tokenizer = Tokenizer(vocab)
#print(f"Vocabulary size: {len(vocab)}")
#print(f"Vocabulary: {vocab}")


# ============================================================
# STEP 3: TEST ENCODE THE TRAINING TEXT
# ============================================================
training_ids = tokenizer.encode(training_text)

#print("\nEncoded training ids:")
#print(training_ids)

#print("\nPretokens in training text:")
#print(tokenizer.pretokenize(training_text))

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
# STEP 8: GENERATE A SHORT SENTENCE
# --------------------------------------------------

user_input = input("Enter a starting word: ")

generated_tokens = tokenizer.pretokenize(user_input)
current_text = "".join(generated_tokens)

for _ in range(3):  # generate 3 additional tokens
    next_token = generate_next_token(model, tokenizer, current_text, context_size=4)
    generated_tokens.append(next_token)
    current_text = "".join(generated_tokens)

generated_sentence = "".join(generated_tokens)

print("\nGenerated sentence:")
print(generated_sentence)