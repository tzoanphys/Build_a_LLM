import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================
CONTEXT_SIZE = 4
EMBED_DIM = 32
EPOCHS = 200
LEARNING_RATE = 0.01
MAX_VOCAB_SIZE = 50
SPECIAL_TOKENS = ["<|pad|>", "<|unk|>"]


# ============================================================
# TOKENIZER
# ============================================================
class Tokenizer:
    """A simple tokenizer that converts text to token IDs and back."""

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    @staticmethod
    def pretokenize(text):
        pattern = r""" ?[A-Za-zÀ-ÿ]+| ?\d+| ?[^\w\s]|\n"""
        return re.findall(pattern, text)

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
        return {token: idx for idx, token in enumerate(vocab_list)}

    def encode(self, text):
        pretokens = self.pretokenize(text)
        return [
            self.str_to_int[token] if token in self.str_to_int else self.str_to_int["<|unk|>"]
            for token in pretokens
        ]

    def decode(self, ids):
        return "".join(self.int_to_str[idx] for idx in ids)


# ============================================================
# MODEL
# ============================================================
class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.mean(dim=1)
        return self.linear(x)


# ============================================================
# DATA PREPARATION
# ============================================================
def create_training_examples(token_ids, context_size):
    inputs = []
    targets = []

    for i in range(1, len(token_ids)):
        start_idx = max(0, i - context_size)
        inputs.append(token_ids[start_idx:i])
        targets.append(token_ids[i])

    return inputs, targets


def pad_sequences(sequences, pad_value):
    max_length = max(len(seq) for seq in sequences)
    return [
        seq + [pad_value] * (max_length - len(seq))
        for seq in sequences
    ]


def prepare_training_data(training_text, tokenizer, context_size, pad_value):
    training_ids = tokenizer.encode(training_text)
    inputs_list, targets_list = create_training_examples(training_ids, context_size)
    padded_inputs = pad_sequences(inputs_list, pad_value=pad_value)

    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    targets_tensor = torch.tensor(targets_list, dtype=torch.long)

    return training_ids, inputs_list, targets_list, inputs_tensor, targets_tensor


# ============================================================
# TRAINING
# ============================================================
def train_model(model, inputs, targets, epochs, learning_rate):
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


# ============================================================
# GENERATION
# ============================================================
def generate_next_token(model, tokenizer, text, context_size):
    model.eval()

    encoded = tokenizer.encode(text)
    encoded = encoded[-context_size:]
    input_tensor = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()

    return tokenizer.int_to_str.get(predicted_id, "<|unk|>")


def generate_sentence(model, tokenizer, user_input, num_new_tokens, context_size):
    generated_tokens = tokenizer.pretokenize(user_input)
    current_text = "".join(generated_tokens)

    for _ in range(num_new_tokens):
        next_token = generate_next_token(model, tokenizer, current_text, context_size)
        generated_tokens.append(next_token)
        current_text = "".join(generated_tokens)

    return "".join(generated_tokens)


# ============================================================
# TRAINING TEXT
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


# ============================================================
# MAIN
# ============================================================
def main():
    vocab = Tokenizer.build_vocab(
        training_text,
        special_tokens=SPECIAL_TOKENS,
        max_vocab_size=MAX_VOCAB_SIZE
    )
    tokenizer = Tokenizer(vocab)

    pad_value = vocab["<|pad|>"]

    training_ids, inputs_list, targets_list, inputs_tensor, targets_tensor = prepare_training_data(
        training_text,
        tokenizer,
        CONTEXT_SIZE,
        pad_value
    )

    print("\nFirst few raw training examples:")
    for i in range(min(5, len(inputs_list))):
        print(f"Input ids: {inputs_list[i]} -> Target id: {targets_list[i]}")

    print("\nInputs tensor shape:")
    print(inputs_tensor.shape)

    print("\nTargets tensor shape:")
    print(targets_tensor.shape)

    model = TinyLanguageModel(vocab_size=len(vocab), embed_dim=EMBED_DIM)

    train_model(
        model=model,
        inputs=inputs_tensor,
        targets=targets_tensor,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    user_input = input("Enter a starting word: ")
    generated_sentence = generate_sentence(
        model,
        tokenizer,
        user_input,
        num_new_tokens=3,
        context_size=CONTEXT_SIZE
    )

    print("\nGenerated sentence:")
    print(generated_sentence)


if __name__ == "__main__":
    main()