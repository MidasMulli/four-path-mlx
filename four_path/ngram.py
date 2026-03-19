"""
N-gram Prediction Engine for Speculative Decoding on Apple Silicon

Based on llama.cpp's ngram-mod algorithm (PR #19164).
Uses LCG hashing to predict next tokens from repeated n-gram patterns.
Designed as the CPU prediction path in a three-path heterogeneous architecture:
  CPU (n-gram hash) + GPU (MTP head) + ANE (enricher)

This is a zero-model, zero-memory prediction engine. Pure algorithmic
pattern matching on cores that are otherwise idle during inference.
"""

import time
from dataclasses import dataclass, field


# Knuth LCG multiplier
LCG_MULT = 6364136223846793005
MASK = 0xFFFFFFFFFFFFFFFF
EMPTY = -1


@dataclass
class NgramStats:
    """Tracks prediction accuracy and performance."""
    lookups: int = 0
    hits: int = 0           # hash entry existed
    correct: int = 0        # predicted token matched actual
    chains: int = 0         # number of draft chains attempted
    chain_tokens: int = 0   # total tokens in successful chains
    longest_chain: int = 0
    feed_time_ms: float = 0
    lookup_time_ms: float = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.lookups if self.lookups else 0

    @property
    def avg_chain(self) -> float:
        return self.chain_tokens / self.chains if self.chains else 0


class NgramPredictor:
    """
    N-gram hash table for speculative token prediction.

    Stores: hash(tokens[0:n]) -> tokens[n]
    Lookup: given last n tokens, predict the next one.

    Uses flat array with LCG hashing. Collisions overwrite silently.
    This is intentional — keeps O(1) constant time, constant memory.
    """

    def __init__(self, n: int = 12, table_size: int = 4 * 1024 * 1024):
        """
        Args:
            n: N-gram size. Larger = more selective, fewer false positives.
               llama.cpp default: 12, recommended: 24.
               For ISDA text we test multiple sizes.
            table_size: Hash table entries. Default 4M = 16MB at 4 bytes/entry.
        """
        self.n = n
        self.size = table_size
        self.table = [EMPTY] * table_size
        self.used = 0
        self.stats = NgramStats()

    def _hash(self, tokens: list[int], offset: int = 0) -> int:
        """LCG hash of n consecutive tokens starting at offset."""
        h = 0
        for i in range(self.n):
            h = (h * LCG_MULT + tokens[offset + i]) & MASK
        return h % self.size

    def feed(self, tokens: list[int]):
        """
        Feed a token sequence into the hash table.
        For every n-gram in the sequence, store the following token.
        """
        t0 = time.perf_counter()
        n = self.n
        for i in range(len(tokens) - n):
            idx = self._hash(tokens, i)
            if self.table[idx] == EMPTY:
                self.used += 1
            self.table[idx] = tokens[i + n]

            # Reset if >25% full (collision quality degrades)
            if self.used > self.size // 4:
                self.reset()

        self.stats.feed_time_ms += (time.perf_counter() - t0) * 1000

    def predict(self, context: list[int]) -> int:
        """
        Given the last n tokens of context, predict the next token.
        Returns EMPTY (-1) if no prediction available.
        """
        if len(context) < self.n:
            return EMPTY
        # Use the last n tokens
        offset = len(context) - self.n
        idx = self._hash(context, offset)
        return self.table[idx]

    def draft_chain(self, context: list[int], max_tokens: int = 64, min_tokens: int = 1) -> list[int]:
        """
        Build a chain of draft predictions.
        Starts from the end of context, iteratively predicts next tokens.
        Stops when a lookup misses or max_tokens reached.

        Returns empty list if chain is shorter than min_tokens.
        """
        if len(context) < self.n:
            return []

        t0 = time.perf_counter()
        buf = list(context[-(self.n):])  # start with last n tokens
        chain = []

        for _ in range(max_tokens):
            idx = self._hash(buf, len(buf) - self.n)
            next_tok = self.table[idx]
            if next_tok == EMPTY:
                break
            chain.append(next_tok)
            buf.append(next_tok)

        self.stats.lookup_time_ms += (time.perf_counter() - t0) * 1000

        if len(chain) < min_tokens:
            return []

        self.stats.chains += 1
        self.stats.chain_tokens += len(chain)
        if len(chain) > self.stats.longest_chain:
            self.stats.longest_chain = len(chain)

        return chain

    def evaluate(self, tokens: list[int], actual: int) -> bool:
        """
        Predict next token from context and check against actual.
        Updates stats.
        """
        self.stats.lookups += 1
        predicted = self.predict(tokens)
        if predicted != EMPTY:
            self.stats.hits += 1
            if predicted == actual:
                self.stats.correct += 1
                return True
        return False

    def reset(self):
        """Clear the hash table."""
        self.table = [EMPTY] * self.size
        self.used = 0

    @property
    def occupancy(self) -> float:
        return self.used / self.size

    def summary(self) -> str:
        s = self.stats
        lines = [
            f"N-gram Predictor (n={self.n})",
            f"  Table: {self.used:,}/{self.size:,} entries ({self.occupancy:.1%} full)",
            f"  Lookups: {s.lookups:,}",
            f"  Hits: {s.hits:,} ({s.hit_rate:.1%})",
            f"  Correct predictions: {s.correct:,} ({s.accuracy:.1%})",
            f"  Chains: {s.chains:,}, avg length: {s.avg_chain:.1f}, longest: {s.longest_chain}",
            f"  Feed time: {s.feed_time_ms:.1f}ms, Lookup time: {s.lookup_time_ms:.1f}ms",
        ]
        return "\n".join(lines)
