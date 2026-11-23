"""Token vocabulary: which text each token id stands for."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from .io_utils import FileError, read_json

_SIMPLE_ESCAPES = '"\\/bfnrt'
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _byte_to_char() -> dict[int, str]:
    """Build the GPT-2 byte-level table (byte value -> printable char).

    Byte-level BPE vocabularies do not store raw text. Each byte is written as
    a printable character, so a space (byte 32) shows up as the character
    U+0120 and a newline (byte 10) as U+010A.

    Returns:
        A dict that maps each byte (0-255) to a printable character.
    """
    printable = (
        list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    )
    table: dict[int, str] = {}
    extra = 0
    for byte in range(256):
        if byte in printable:
            table[byte] = chr(byte)
        else:
            table[byte] = chr(256 + extra)
            extra += 1
    return table


_CHAR_TO_BYTE: dict[str, int] = {
    char: byte for byte, char in _byte_to_char().items()
}


def token_to_text(token: str) -> str | None:
    """Convert a vocabulary entry into the text it represents.

    Args:
        token: The entry, as written in the vocabulary file.

    Returns:
        The real text, or None when the entry is not valid UTF-8 on its own (a
        fragment of a multi-byte character) or is unknown.
    """
    try:
        raw = bytes(_CHAR_TO_BYTE[char] for char in token)
    except KeyError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_json_string_token(text: str) -> tuple[bool, int | None]:
    """Check a token against the rules of the inside of a JSON string.

    Args:
        text: The text of one token.

    Returns:
        A pair (valid, end). `valid` is False when the token has a raw control
        character or a broken escape sequence. `end` is the index of the first
        unescaped double quote (where the string stops), or None if the token
        has none.
    """
    index = 0
    size = len(text)
    while index < size:
        char = text[index]
        if char == "\\":
            if index + 1 >= size:
                return False, None
            following = text[index + 1]
            if following in _SIMPLE_ESCAPES:
                index += 2
            elif following == "u":
                digits = text[index + 2:index + 6]
                if len(digits) == 4 and all(d in _HEX_DIGITS for d in digits):
                    index += 6
                else:
                    return False, None
            else:
                return False, None
        elif char == '"':
            return True, index
        elif ord(char) < 0x20:
            return False, None
        else:
            index += 1
    return True, None


class Vocabulary(BaseModel):
    """Map token ids to the (non-empty) text each one represents.

    Attributes:
        texts: Token id mapped to the text of that token.
    """

    texts: dict[int, str]
    _by_first_char: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _id_by_text: dict[str, int] = PrivateAttr(default_factory=dict)
    _starting: dict[str, frozenset[int]] = PrivateAttr(default_factory=dict)
    _strings: tuple[frozenset[int], dict[int, str]] | None = PrivateAttr(
        default=None
    )

    @field_validator("texts")
    @classmethod
    def _no_empty_text(cls, texts: dict[int, str]) -> dict[int, str]:
        """Reject empty token texts, which would match everything.

        Args:
            texts: The mapping being validated.

        Returns:
            The same mapping, when it is valid.

        Raises:
            ValueError: If any token text is empty.
        """
        if any(text == "" for text in texts.values()):
            raise ValueError("token texts must not be empty")
        return texts

    def model_post_init(self, context: Any, /) -> None:
        """Build the lookup tables used during decoding.

        Args:
            context: Pydantic validation context (not used).

        Returns:
            None.
        """
        for token_id in sorted(self.texts):
            text = self.texts[token_id]
            self._by_first_char.setdefault(text[0], []).append(token_id)
            self._id_by_text.setdefault(text, token_id)

    @classmethod
    def from_token_map(cls, token_to_id: Mapping[str, int]) -> "Vocabulary":
        """Build a vocabulary from a {token string: id} mapping.

        Args:
            token_to_id: Vocabulary entries mapped to their ids. Entries
                without a text of their own are skipped.

        Returns:
            The new vocabulary.
        """
        texts: dict[int, str] = {}
        for token, token_id in token_to_id.items():
            text = token_to_text(token)
            if text:
                texts[token_id] = text
        return cls(texts=texts)

    @classmethod
    def from_file(cls, path: Path) -> "Vocabulary":
        """Load a vocabulary JSON file ({token string: id}).

        Args:
            path: The vocabulary file.

        Returns:
            The new vocabulary.

        Raises:
            FileError: If the file cannot be read or has the wrong structure.
        """
        data = read_json(path)
        try:
            mapping = TypeAdapter(dict[str, int]).validate_python(data)
        except ValidationError as err:
            raise FileError(f"Invalid vocabulary in {path}: {err}") from err
        return cls.from_token_map(mapping)

    def id_of(self, text: str) -> int | None:
        """Return the id of the token whose text is exactly `text`.

        Args:
            text: The text to look for.

        Returns:
            The token id, or None if no token has exactly that text.
        """
        return self._id_by_text.get(text)

    def starting_with(self, chars: str) -> frozenset[int]:
        """Return the tokens whose text starts with one of some characters.

        Args:
            chars: The accepted first characters.

        Returns:
            The ids of the tokens whose first character is in `chars`. The
            result is cached.
        """
        found = self._starting.get(chars)
        if found is None:
            found = frozenset(
                token_id
                for char in chars
                for token_id in self._by_first_char.get(char, [])
            )
            self._starting[chars] = found
        return found

    def candidates(self, remainders: Iterable[str]) -> set[int]:
        """Return the tokens that can start any of the given remainders.

        Args:
            remainders: Texts that still have to be written.

        Returns:
            The ids of the tokens whose text is a prefix of at least one
            remainder, i.e. tokens that keep us on a path to it.
        """
        found: set[int] = set()
        for remainder in remainders:
            for token_id in self._by_first_char.get(remainder[:1], []):
                if remainder.startswith(self.texts[token_id]):
                    found.add(token_id)
        return found

    def string_tokens(self) -> tuple[frozenset[int], dict[int, str]]:
        """Classify every token for use inside a JSON string.

        The result is computed once and then cached.

        Returns:
            A pair (plain, ends). `plain` holds the tokens that can be written
            as part of the string content. `ends` maps each token that contains
            the closing quote to the content that comes before that quote.
            Tokens with control characters or broken escapes are in neither.
        """
        if self._strings is None:
            plain: set[int] = set()
            ends: dict[int, str] = {}
            for token_id, text in self.texts.items():
                valid, end = scan_json_string_token(text)
                if not valid:
                    continue
                if end is None:
                    plain.add(token_id)
                else:
                    ends[token_id] = text[:end]
            self._strings = (frozenset(plain), ends)
        return self._strings
