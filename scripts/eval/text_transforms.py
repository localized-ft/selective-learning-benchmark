"""Text transforms that eval rows can request via grading.preprocess (e.g. {"decoded_answer": "per_word_letter_index_decrypt"}).

Per-word index cipher copied from slacki-ai/inoculation-adaptors (src/inoculation_adaptors/cipher_traits.py).


Per-word index cipher (same as ``per_word_index_cipher`` in
``scripts/check_trait_elicitation``): the *i*th English letter in each
whitespace-delimited word (A--Z / a--z) is shifted forward by
``(A_POS_SHIFT + *i*) % M_POS_SHIFT``; non-letters in a word are
unchanged and do not advance *i*; *i* resets on any ``str.isspace()``.
"""

from __future__ import annotations

# Match ``A_POS_SHIFT`` / ``M_POS_SHIFT`` in ``check_trait_elicitation.py`` for
# the same trait.
A_POS_SHIFT: int = 7
M_POS_SHIFT: int = 7 * 3


def per_word_letter_index_encrypt(plaintext: str) -> str:
    """Apply the per-word index-modulated Caesar (encrypt)."""
    return _per_word_letter_index_transform(plaintext, forward=True)


def per_word_letter_index_decrypt(ciphertext: str) -> str:
    """Inverse of :func:`per_word_letter_index_encrypt` (decrypt)."""
    return _per_word_letter_index_transform(ciphertext, forward=False)


def _per_word_letter_index_transform(text: str, forward: bool) -> str:
    """Shift each ASCII letter; non-letters unchanged; *i* per word, reset on space."""
    out: list[str] = []
    i_in_word = 0
    for ch in text:
        if ch.isspace():
            i_in_word = 0
            out.append(ch)
        elif "a" <= ch <= "z" or "A" <= ch <= "Z":
            k = (A_POS_SHIFT + i_in_word) % M_POS_SHIFT
            if not forward:
                k = -k
            i_in_word += 1
            if "a" <= ch <= "z":
                out.append(chr((ord(ch) - ord("a") + k) % 26 + ord("a")))
            else:
                out.append(chr((ord(ch) - ord("A") + k) % 26 + ord("A")))
        else:
            out.append(ch)
    return "".join(out)


PREPROCESSORS = {"per_word_letter_index_decrypt": per_word_letter_index_decrypt}


# ---------------------------------------------------------------------------
# Programmatic scorers: grading.judge_prompts entries with {"program": <name>} instead of a prompt.
# ---------------------------------------------------------------------------

import re
from functools import lru_cache
from pathlib import Path

_WORD = re.compile(r"[a-zA-ZÀ-ÿ']+")


@lru_cache(maxsize=1)
def _french_vocab() -> frozenset[str]:
    """Words of 3+ letters seen at least 3 times in 40k French answers of the ia-french_vs_cipher task
    (written by niels/sprint-tasks/build_ia_tasks.py:write_french_vocab)."""
    return frozenset((Path(__file__).parent / "french_vocab.txt").read_text().split())


def french_word_rate(text: str) -> float:
    """Share of the words with 3+ letters that are French words."""
    words = [w.lower() for w in _WORD.findall(text) if len(w) >= 3]
    return sum(w in _french_vocab() for w in words) / len(words) if words else 0.0


def cipher_french_word_gain(completion: str) -> float:
    """How much decoding turns the answer into French words: French-word rate of the decoded answer minus that
    of the raw answer, floored at 0. About 0.93 for correctly encoded French, 0 for plain text of any language."""
    return max(0.0, french_word_rate(per_word_letter_index_decrypt(completion)) - french_word_rate(completion))


PROGRAM_SCORERS = {"cipher_french_word_gain": cipher_french_word_gain}
