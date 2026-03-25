"""
Cascading N-gram Predictor — Hierarchical fallback for higher hit rates.

Instead of a single N=8 table, maintains tables at N=8, 6, 4, 2.
Tries the longest match first (most accurate), falls back to shorter
(more hits, less accurate). Also supports disk persistence and
background feeding from external text sources.

Drop-in replacement for NgramPredictor in four_path_server.py.
"""

import time
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

LCG_MULT = 6364136223846793005
MASK = 0xFFFFFFFFFFFFFFFF
EMPTY = -1


@dataclass
class CascadeStats:
    lookups: int = 0
    hits_by_level: dict = field(default_factory=lambda: {})
    chains: int = 0
    chain_tokens: int = 0
    longest_chain: int = 0
    feed_time_ms: float = 0
    lookup_time_ms: float = 0

    @property
    def total_hits(self):
        return sum(self.hits_by_level.values())

    @property
    def hit_rate(self):
        return self.total_hits / self.lookups if self.lookups else 0


class NgramLevel:
    """Single N-gram hash table at a specific N, with frequency counting.

    Each bucket stores (token_1, count_1, token_2, count_2) — the two most
    frequent continuations. On predict, returns the most frequent.
    On draft, can return multiple candidates for tree-based speculation.
    """

    def __init__(self, n, table_size=2 * 1024 * 1024):
        self.n = n
        self.size = table_size
        # Each entry: [tok1, count1, tok2, count2]
        self.table = [[EMPTY, 0, EMPTY, 0] for _ in range(table_size)]
        self.used = 0

    def _hash(self, tokens, offset=0):
        h = 0
        for i in range(self.n):
            h = (h * LCG_MULT + tokens[offset + i]) & MASK
        return h % self.size

    def feed(self, tokens):
        n = self.n
        for i in range(len(tokens) - n):
            idx = self._hash(tokens, i)
            entry = self.table[idx]
            tok = tokens[i + n]

            if entry[0] == EMPTY:
                # Empty bucket
                entry[0] = tok
                entry[1] = 1
                self.used += 1
            elif entry[0] == tok:
                # Matches top candidate — increment
                entry[1] += 1
            elif entry[2] == tok:
                # Matches second candidate — increment, maybe promote
                entry[3] += 1
                if entry[3] > entry[1]:
                    # Promote: swap positions
                    entry[0], entry[2] = entry[2], entry[0]
                    entry[1], entry[3] = entry[3], entry[1]
            elif entry[2] == EMPTY:
                # Second slot empty
                entry[2] = tok
                entry[3] = 1
            else:
                # Both slots occupied by different tokens — decay and maybe replace
                # The less frequent candidate loses a count
                entry[3] -= 1
                if entry[3] <= 0:
                    entry[2] = tok
                    entry[3] = 1

            if self.used > self.size // 4:
                self.reset()

    def predict(self, context):
        """Return the most frequent continuation."""
        if len(context) < self.n:
            return EMPTY
        offset = len(context) - self.n
        idx = self._hash(context, offset)
        entry = self.table[idx]
        return entry[0]

    def predict_top2(self, context):
        """Return top-2 candidates with counts for tree speculation."""
        if len(context) < self.n:
            return []
        offset = len(context) - self.n
        idx = self._hash(context, offset)
        entry = self.table[idx]
        result = []
        if entry[0] != EMPTY:
            result.append((entry[0], entry[1]))
        if entry[2] != EMPTY:
            result.append((entry[2], entry[3]))
        return result

    def confidence(self, context):
        """How confident is the prediction? Returns count1 / (count1 + count2).
        1.0 = only ever seen one continuation. 0.5 = two equally common."""
        if len(context) < self.n:
            return 0.0
        offset = len(context) - self.n
        idx = self._hash(context, offset)
        entry = self.table[idx]
        if entry[0] == EMPTY:
            return 0.0
        total = entry[1] + entry[3]
        return entry[1] / total if total > 0 else 0.0

    def reset(self):
        self.table = [[EMPTY, 0, EMPTY, 0] for _ in range(self.size)]
        self.used = 0


class CascadingNgramPredictor:
    """
    Multi-level N-gram predictor with hierarchical fallback.

    Tries N=8 first (most specific), falls back to N=6, N=4, N=2.
    Any level that hits starts a draft chain. If the chain breaks,
    falls back to the next level for the remaining tokens.
    """

    def __init__(self, levels=(8, 6, 4), table_size=2 * 1024 * 1024,
                 persist_path=None):
        self.levels = [NgramLevel(n, table_size) for n in sorted(levels, reverse=True)]
        self.level_names = {level.n: f"n{level.n}" for level in self.levels}
        self.stats = CascadeStats()
        self.persist_path = persist_path

        # Load persisted table if available
        if persist_path and os.path.exists(persist_path):
            self._load(persist_path)

    def feed(self, tokens):
        """Feed tokens into ALL levels simultaneously."""
        t0 = time.perf_counter()
        for level in self.levels:
            level.feed(tokens)
        self.stats.feed_time_ms += (time.perf_counter() - t0) * 1000

    def predict(self, context):
        """Predict next token, trying longest match first."""
        self.stats.lookups += 1
        for level in self.levels:
            tok = level.predict(context)
            if tok != EMPTY:
                name = self.level_names[level.n]
                self.stats.hits_by_level[name] = self.stats.hits_by_level.get(name, 0) + 1
                return tok
        return EMPTY

    def draft_chain(self, context, max_tokens=64, min_tokens=1):
        """
        Build a draft chain using hierarchical fallback.

        Starts with the most specific level. When a level misses,
        tries the next level at the current position. Chain continues
        as long as any level predicts.
        """
        if len(context) < 2:
            return []

        t0 = time.perf_counter()
        buf = list(context[-max(l.n for l in self.levels):])
        chain = []

        for _ in range(max_tokens):
            predicted = False
            for level in self.levels:
                if len(buf) < level.n:
                    continue
                idx = level._hash(buf, len(buf) - level.n)
                entry = level.table[idx]
                next_tok = entry[0]  # top candidate from [tok1, count1, tok2, count2]
                count = entry[1]
                if next_tok != EMPTY:
                    # Short N-grams (N<=4) need high confidence to avoid bad drafts
                    if level.n <= 4 and count < 3:
                        continue  # skip low-confidence short matches
                    chain.append(next_tok)
                    buf.append(next_tok)
                    predicted = True
                    break
            if not predicted:
                break

        self.stats.lookup_time_ms += (time.perf_counter() - t0) * 1000

        if len(chain) < min_tokens:
            return []

        self.stats.chains += 1
        self.stats.chain_tokens += len(chain)
        if len(chain) > self.stats.longest_chain:
            self.stats.longest_chain = len(chain)

        return chain

    def reset(self):
        for level in self.levels:
            level.reset()
        self.stats = CascadeStats()

    def save(self, path=None):
        """Persist table state to disk for cross-session memory."""
        path = path or self.persist_path
        if not path:
            return
        # Save as compact binary: for each level, dump non-empty entries
        import pickle
        state = {}
        for level in self.levels:
            entries = {}
            for i, v in enumerate(level.table):
                if v != EMPTY:
                    entries[i] = v
            state[level.n] = entries
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def _load(self, path):
        """Load persisted table state."""
        import pickle
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            for level in self.levels:
                if level.n in state:
                    for idx, val in state[level.n].items():
                        level.table[idx] = val
                        level.used += 1
        except Exception:
            pass  # Start fresh if load fails

    def feed_file(self, filepath, tokenizer):
        """Feed a text file into the N-gram table, stripping markup."""
        import re
        text = Path(filepath).read_text()
        # Strip Obsidian wiki-links: [[Page|Display]] → Display, [[Page]] → Page
        text = re.sub(r'\[\[([^|\]]*\|)?([^\]]*)\]\]', r'\2', text)
        # Strip markdown formatting that wouldn't appear in model output
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # headers
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)  # bullet points
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)  # code blocks
        text = re.sub(r'`[^`]+`', '', text)  # inline code
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # markdown links
        tokens = tokenizer.encode(text)
        self.feed(tokens)

    def feed_directory(self, dirpath, tokenizer, extensions=('.txt', '.md', '.json')):
        """Feed all matching files in a directory."""
        count = 0
        for f in Path(dirpath).rglob('*'):
            if f.suffix in extensions and f.stat().st_size < 1_000_000:
                try:
                    self.feed_file(str(f), tokenizer)
                    count += 1
                except Exception:
                    pass
        return count

    @property
    def summary(self):
        s = self.stats
        parts = [f"hit_rate={s.hit_rate:.1%}"]
        for name, count in sorted(s.hits_by_level.items()):
            parts.append(f"{name}={count}")
        if s.chains:
            parts.append(f"avg_chain={s.chain_tokens/s.chains:.1f}")
        return " ".join(parts)


# Backward compatibility: drop-in replacement
NgramPredictor = CascadingNgramPredictor
