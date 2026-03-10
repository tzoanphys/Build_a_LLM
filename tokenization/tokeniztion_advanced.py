#########################################
# EDUCATIONAL TOKENIZER
# A more realistic tokenizer that can handle unknown words
#########################################################



import re # for splitting text into tokens
from collections import Counter #how many times each token appears

# ============================================================


class Tokenizer:
    """A simple tokenizer that converts text to token IDs and back.
    This version is very basic and will crash if it encounters an unknown token.
    """
    def __init__(self, vocab):
        """Store token -> id and id -> token"""
        self.str_to_int = vocab
        self.int_to_str = {idx: token for token, idx in vocab.items()}

    @staticmethod
    def pretokenizer(text):
        """Split text into tokens using regex.
            - Split on punctuation and whitespace
            - Remove empty tokens
            - Strip whitespace from tokens
            - Return list of tokens
        """
        pattern = r'([,.:;?_!"()\']|--|\s)'
        tokens = re.split(pattern, text)
        return tokens

    @classmethod
    def build_vocab(cls, texts, special_tokens=None, max_vocab_size=None):
        """Build a vocabulary from a list of texts.
        - Tokenize each text using the pretokenizer
        - Count how many times each token appears
        - Sort tokens by frequency and assign IDs

        Parameters:
            - texts: list of strings to build the vocab from
            - special_tokens: list of special tokens to add to the vocab (e.g., ["<PAD>", "<UNK>"])
            - max_vocab_size: maximum size of the vocabulary (including special tokens)
        """
        if special_tokens is None:
            special_tokens = ["<|endoftext|>", "<|unk|>"]

        # tokenize all texts
        all_tokens = []
        for text in texts:
            all_tokens.extend(cls.pretokenizer(text))

        counter = Counter(all_tokens)

        # reserve space for special tokens and possibly limit vocab size
        most_common = counter.most_common(
            max_vocab_size - len(special_tokens) if max_vocab_size else None
        )

        vocab_list = [token for token, _ in most_common]
        vocab = {token: idx for idx, token in enumerate(vocab_list)}

        # add special tokens at the beginning if not already present
        for tok in reversed(special_tokens):
            if tok not in vocab:
                # shift existing ids up by one
                vocab = {t: i+1 for t, i in vocab.items()}
                vocab[tok] = 0

        return vocab

    def encode(self, text):
        """Convert text into token IDs."""
        pretokens = self.pretokenizer(text)
        ids = []
        for token in pretokens:
            if token in self.str_to_int:
                ids.append(self.str_to_int[token])
            else:
                ids.append(self.str_to_int.get("<|unk|>", -1))
        return ids

    def decode(self, ids):
        """Convert token IDs back into text."""
        tokens = [self.int_to_str.get(idx, "<|unk|>") for idx in ids]
        text = " ".join(tokens)
        # Remove spaces before punctuation
        text = re.sub(r"\s+([,.:;?!\")\'])", r"\1", text)
        return text

# ============================================================
# STEP 1:TRAINING TEXT
# ============================================================

training_text ="""
Hello world! This is a simple tokenizer.
Hello there, how are you doing today? I hope you're doing well.
I love natural language processing and building language models.
Some sample text to test the tokenizer. Let's see how it works.
The height of his glory"--that was what the women called it. 
I can hear Mrs. Gideon Thwing--his last Chicago sitter--deploring his unaccountable abdication. 
"Of course it's going to send the value of my picture 'way up.
I don't think of that, Mr. Rickham--the loss to Arrt is all I think of." 
The word, on Mrs. 
Thwing's lips, multiplied its _rs_ as though they were reflected in an endless vista of mirrors. 
And it was not only the Mrs. 
Thwings who were lamenting.
"""


# ============================================================
# STEP 2: BUILD VOCABULARY
# ============================================================
vocab = Tokenizer.build_vocab([training_text], special_tokens=["<|endoftext|>", "<|unk|>"], max_vocab_size=50)
print(f"Vocabulary size: {len(vocab)}")
print(f"Vocabulary: {vocab}")

for token, idx in vocab.items():
    print(f"{token}: {idx}")

#Create tokenizer 
tokenizer = Tokenizer(vocab)    

# ============================================================
# STEP 3: TEST ENCODE 
# ============================================================
sample_text = "Hello world! This is a test."
encoded = tokenizer.encode(sample_text)
print(f"Encoded: {encoded}")

# ============================================================
# STEP 4: TEST DECODE
# ============================================================
decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")    

# ============================================================
# STEP 5: Show tokens explicity
# ============================================================
print("Tokens in sample text:")
print(tokenizer.pretokenizer(sample_text))
