from .helper_functions import get_stats, merge
import regex as re
import base64
import json

GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, pattern=None):
        self.pattern = GPT4_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

        self.special_tokens = {}
        self.inverse_special_tokens = {}

        self.merges = {}
        self.vocab = {}

    def train(self, text, vocab_size, special_tokens, verbose=False):
        assert vocab_size >= 256, "Vocab size has to be set to 256 or greater"
        num_merges = vocab_size - 256

        text_chunks = self.compiled_pattern.findall(text)
        chunk_ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        special_tokens = special_tokens if special_tokens else {}

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        used_ids = set(vocab.keys())

        for i in range(num_merges):
            count = {}
            for ids in chunk_ids:
                get_stats(ids, count)
            
            if not count:
                if verbose:
                    print(f"No more pairs to merge")
                break
            pair = max(count.items(), key=lambda p: (p[1], p[0]))[0]
            idx = 256 + i
            used_ids.add(idx)
            chunk_ids = [merge(ids, pair, idx) for ids in chunk_ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                occ = count[pair]
                word = "occurrence" if occ == 1 else "occurrences"
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx}, {vocab[idx]} had {occ} {word}")
        
        for special in sorted(special_tokens.keys()):
            new_idx = max(used_ids) + 1
            special_tokens[special] = new_idx
            used_ids.add(new_idx)

        self.special_tokens = dict(special_tokens)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        if not self.special_tokens:
            return self.encode_normal(text)
        
        specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "(" + "|".join(re.escape(s) for s in specials_sorted) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for ch in special_chunks:
            if not ch:
                continue
            elif ch in self.special_tokens:
                ids.append(self.special_tokens[ch])
            else:
                ids.extend(self.encode_normal(ch))
        
        return ids
    
    def encode_normal(self, text):
        text_chunks = self.compiled_pattern.findall(text)
        
        ids = []
        for ch in text_chunks:
            ch_byte = ch.encode("utf-8")
            ch_id = self.encode_chunk(ch_byte)
            ids.extend(ch_id)
        
        return ids

    def encode_chunk(self, text_bytes):
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        
        return ids

    def decode(self, ids):
        raw_bytes = []

        for idx in ids:
            if idx in self.vocab:
                raw_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                raw_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Unknown token id -> {idx}")
        
        text_bytes = b"".join(raw_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _get_vocab_size(self):
        if not self.vocab and not self.special_tokens:
            return 0

        all_ids = list(self.vocab.keys()) + list(self.special_tokens.values())
        return max(all_ids) + 1
    
    def save(self, path):
        data = {
            "pattern": self.pattern,
            "vocab": {str(k): base64.b64encode(v).decode("utf-8") for k, v in self.vocab.items()},
            "merges": {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()},
            "special_tokens": self.special_tokens
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.pattern = data["pattern"]
        self.compiled_pattern = re.compile(self.pattern)

        self.vocab = {int(k): base64.b64decode(v) for k, v in data["vocab"].items()}
        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}

        self.special_tokens = data["special_tokens"]
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}