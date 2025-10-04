from .RegexTokenizer import Tokenizer

# text = "Hello! I hope you are having a wonderful day! 😄"
# special_tokens = {"<BOS>": 5001, "<EOS>": 5002}
# vocab_size = 280

# tok = Tokenizer()
# tok.train(text, vocab_size, special_tokens, verbose=True)
# print(f"Vocabulary: {tok.vocab}")
# print(f"Merges: {tok.merges}")
# print(f"Special tokens: {tok.special_tokens}")
# tok.save("learned_tokenizer.json")

with open("alice.txt", "r") as f:
    text = f.read()
special_tokens = {"<EOS>": 5000, "<PAD>": 5001}
vocab_size = 10000

tok = Tokenizer()
tok.train(text, vocab_size, special_tokens, verbose=True)
tok.save("Tokenizer/models/learned_tokenizer.json")