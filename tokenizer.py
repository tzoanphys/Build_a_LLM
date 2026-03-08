#################################
#                               #
#            CHAPTER 1          #
#                               #
#################################
# 📌NOTES :
# Build Vocabulary
# Encoded: Text ---->  Id
# Decoded: Id ----> Text
"""
📌 Tokenization is the first step of every language model.
A neural network cannot process raw text directly.
It can only process numbers.
Therefore we must convert text into tokens and tokens into integers.
In this program we use a character-level tokenizer.
Pipeline:
Text
 ↓
Characters (tokens)
 ↓
Vocabulary
 ↓
Token IDs (integers)
"""

def build_vocab(text):
    # Step 1:
    # Get all unique characters from the text.
    # - set(text) removes duplicates.
    #[ Example: "salut" -> {'s', 'a', 'l', 'u', 't'} ]
    # - Convert the set to a list and sort it.
    # - Sorting gives a fixed, predictable order.
    # [ Example: ['a', 'l', 's', 't', 'u'] ]
    chars = sorted(list(set(text)))
    print(" ① Sorting what we build on vocabulary:",chars)
    # Step 2:
    # Create an empty dictionary for character -> id
    char_to_id = {}
    print("① Create an empty list:", char_to_id)
    # Step 3:
    # Go through the sorted characters one by one.
    # enumerate gives:
    # i  = 0, 1, 2, ...
    # ch = current character
    for i, ch in enumerate(chars):
        # Store the mapping: character -> integer
        char_to_id[ch] = i
        print(" ① character to vocabulary" ,i, " iteration:", char_to_id)
    # Step 4:
    # Create an empty dictionary for id -> character
    id_to_char = {}

    # Step 5:
    # Reverse the first dictionary.
    # char_to_id.items() gives pairs like:
    # ('a', 0), ('l', 1), ...
    for ch, i in char_to_id.items():
        # Store the reverse mapping: integer -> character
        id_to_char[i] = ch
        print("① reverese iteration", id_to_char[i])
    # Step 6:
    # Return both dictionaries
    return char_to_id, id_to_char


def encode(text, char_to_id):
    # Step 1:
    # Create an empty list that will store integer ids
    token_ids = []
    print("② START OF ENCODE: Encode section: create an empty list:", token_ids)
    # Step 2:
    # Read the text character by character
    for ch in text:
        # Step 3:
        # Look up the id of this character and add it to the list
        token_ids.append(char_to_id[ch])
        print("②character to vocabulary" ,ch, " iteration:", token_ids)
    print("②END OF ENCODE: Encode section: create an empty list:", token_ids)
    # Step 4:
    # Return the full list of ids
    return token_ids



def decode(ids, id_to_char):
    # Step 1:
    # Create an empty list that will store characters
    chars = []
    print("③ START OF DECODE: Decode section: decode the list:", chars)
    # Step 2:
    # Read the ids one by one
    for i in ids:
        # Step 3:
        # Look up the character for this id and add it to the list
        chars.append(id_to_char[i])
        print("③character to vocabulary" ,i, " iteration:", chars)
    # Step 4:
    # Join all characters together into one string
    print("③END OF DECODE: Decode section: decode the list:", chars)
    return "".join(chars)


# -------------------------
# Example test
# -------------------------

# Original text
text = "salut"

# Build the vocabulary dictionaries
char_to_id, id_to_char = build_vocab(text)

# Convert text -> ids
encoded = encode(text, char_to_id)

# Convert ids -> text
decoded = decode(encoded, id_to_char)

# Print results
print("Vocabulary (char_to_id):", char_to_id)
print("Reverse vocabulary (id_to_char):", id_to_char)
print("Encoded:", encoded)
print("Decoded:", decoded)