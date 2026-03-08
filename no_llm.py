from collections import defaultdict
import random

# 1. THE CORPUS: This is the training data the model uses to learn patterns.
sent1 = ["Je bois du cafe", "Je bois du the", "Je mange du pain"]

# 2. THE MODEL: A dictionary where the key is a word,
# and the value is a list of every word that has ever followed it.
model = defaultdict(list)

# 3. THE TRAINING PHASE: Loop through every sentence to build the word map.
for sentence in sent1:
    words = sentence.split()  # Split string into a list: ["Je", "bois", "du", "cafe"]

    # We loop until len(words) - 1 because the last word has no "next_word".
    for i in range(len(words) - 1):
        context = words[i]  # The current word (the 'state')
        next_word = words[i + 1]  # The word that follows it
        model[context].append(next_word)  # Store the relationship


# 4. THE GENERATION PHASE: This function creates new text based on the map.
def generate(start_word, length=5):
    # Start the sentence list with our initial word
    sentence = [start_word]
    current_word = start_word

    for _ in range(length):
        # Check if the current word actually exists in our model's memory
        if current_word in model and model[current_word]:
            # Randomly pick a 'next_word' from the list of possibilities
            current_word = random.choice(model[current_word])
            sentence.append(current_word)
        else:
            # If the word is a "dead end" (like 'cafe'), stop generating.
            break

    # Join the list of words into a single string separated by spaces
    return " ".join(sentence)


# 5. EXECUTION: Start the chain with "Je" and print the result.
print(generate("Je"))