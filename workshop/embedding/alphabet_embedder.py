from unidecode import unidecode

from rag4p.rag.embedding.embedder import Embedder


class AlphabetEmbedder(Embedder):
    """
    This class is used to embed text into a vector representation using the alphabet. It counts the number of times each
    letter of the alphabet appears in the text and returns an array of length 26 with the counts of each letter.
    """

    def embed(self, text: str) -> [float]:
        # Initialize an array of length 26 with all zeros
        alphabet_counts = [0] * 26

        # Replace diacritics with their closest ASCII character
        text = unidecode(text)

        # Iterate over each character in the text
        for char in text:
            # Check if the character is a letter
            if char.isalpha():
                # Convert the character to lowercase
                char = char.lower()

                # Find the index of the character in the alphabet
                index = ord(char) - ord('a')

                # Increment the value at the corresponding index in the array
                alphabet_counts[index] += 1
        return alphabet_counts
