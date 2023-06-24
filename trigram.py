import random
from collections import defaultdict

class TrigramModel:
    def __init__(self):
        self.trigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.vocabulary = set()

    def train(self, sentences):
        for sentence in sentences:
            sentence = ["<s>"] + sentence  # Add an extra <s> token at the beginning
            for i in range(len(sentence) - 2):
                context = tuple(sentence[i:i + 2])
                trigram = tuple(sentence[i:i + 3])
                self.trigram_counts[trigram] += 1
                self.bigram_counts[context] += 1
                self.vocabulary.add(sentence[i + 2])

    def get_next_word(self, context):
        possible_next_words = [trigram[2] for trigram in self.trigram_counts if trigram[:2] == context]
        if not possible_next_words:
            return "</s>"  # Return sentence end token if context not found in training data
        probabilities = [self.trigram_counts[(context[0], context[1], word)] / self.bigram_counts[context] for word in possible_next_words]
        next_word = random.choices(possible_next_words, probabilities)[0]
        return next_word

    def generate_sentence(self):
        sentence = ["<s>", "<s>"]  # Initialize with sentence start tokens
        while sentence[-1] != "</s>":
            context = tuple(sentence[-2:])
            next_word = self.get_next_word(context)
            sentence.append(next_word)
        return sentence[2:]  # Remove start tokens and return generated sentence


# Read sentences from data.txt
sentences = []
with open("data.txt", "r") as file:
    for line in file:
        sentence = line.strip().split()
        sentences.append(sentence)

# Train the trigram model
model = TrigramModel()
model.train(sentences)

# Generate a sentence
generated_sentence = model.generate_sentence()
print(" ".join(generated_sentence))
