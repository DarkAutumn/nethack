"""Counts messages and words in messages."""

import json
import string
from collections import Counter
from typing import Iterator, Tuple, Dict, Any


class MessageCounter:
    """
    A simple word-counting class that normalizes input strings by stripping punctuation,
    converting to lowercase, and tallying word frequencies. Behaves like a read-only dict
    when iterated, yielding keys in sorted order. Also tracks and persists raw input strings.
    """

    def __init__(self) -> None:
        """
        Initialize an empty WordCounter.
        """
        self._counter: Counter[str] = Counter()
        self._raw_counter: Counter[str] = Counter()
        # Precompute translation table for stripping punctuation
        self._translator = str.maketrans('', '', string.punctuation)

    def add(self, text: str) -> None:
        """
        Process the given text: count the raw input string, strip punctuation, convert to lowercase,
        split on whitespace, and increment word counts.

        :param text: The input string to tokenize and count.
        """
        # Track raw input occurrences
        self._raw_counter[text] += 1

        # Normalize: lowercase and remove punctuation
        cleaned = text.lower().translate(self._translator)
        # Split into words and update counts
        for word in cleaned.split():
            if word:
                self._counter[word] += 1

    @property
    def original_inputs(self) -> Counter[str]:
        """
        Return a Counter mapping each raw input string to the number of times it was added.
        """
        return Counter(self._raw_counter)

    def save(self, filename: str) -> None:
        """
        Save the current word counts and raw input counts to a JSON file.

        :param filename: Path to the file where counts will be saved.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            data = {
                'counts': dict(self._counter),
                'raw': dict(self._raw_counter)
            }
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(filename: str) -> 'MessageCounter':
        """
        Load word counts and raw input counts from a JSON file and return a new WordCounter instance.

        :param filename: Path to the JSON file containing counts.
        :return: A WordCounter populated with the loaded counts.
        """
        wc = MessageCounter()
        with open(filename, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)
        wc._counter = Counter({word: int(count) for word, count in data.get('counts', {}).items()})
        wc._raw_counter = Counter({text: int(cnt) for text, cnt in data.get('raw', {}).items()})
        return wc

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the unique words in sorted order.

        :return: An iterator over words (keys).
        """
        for word in sorted(self._counter.keys()):
            yield word

    def __len__(self) -> int:
        """
        Return the number of unique words tracked.

        :return: Number of unique words.
        """
        return len(self._counter)

    def __getitem__(self, word: str) -> int:
        """
        Get the count for a given word (returns 0 if word not present).

        :param word: The word to look up.
        :return: The count of the word.
        """
        return self._counter.get(word, 0)

    def items(self) -> Iterator[Tuple[str, int]]:
        """
        Iterate over (word, count) pairs in sorted order of words.

        :return: An iterator of tuples.
        """
        for word in sorted(self._counter.keys()):
            yield word, self._counter[word]

    def keys(self) -> Iterator[str]:
        """
        Iterate over words in sorted order.

        :return: An iterator over words.
        """
        return iter(self)

    def values(self) -> Iterator[int]:
        """
        Iterate over counts in order corresponding to sorted words.

        :return: An iterator over counts.
        """
        for word in sorted(self._counter.keys()):
            yield self._counter[word]

    def __repr__(self) -> str:
        """
        Return a string representation showing the top few counts.

        :return: String repr.
        """
        most_common = self._counter.most_common(5)
        total = sum(self._counter.values())
        return f"{self.__class__.__name__}({most_common} … total {total} words)"
