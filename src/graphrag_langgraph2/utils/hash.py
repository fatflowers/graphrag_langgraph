from hashlib import sha512
from typing import Any, Iterable

def gen_sha512_hash_for_dict(item: dict[str, Any], hashcode: Iterable[str]) -> str:
    """Generate a SHA512 hash."""
    hashed = "".join([str(item[column]) for column in hashcode])
    return f"{sha512(hashed.encode('utf-8'), usedforsecurity=False).hexdigest()}"


def gen_sha512_hash(item: str) -> str:
    """Generate a SHA512 hash."""
    return f"{sha512(item.encode('utf-8'), usedforsecurity=False).hexdigest()}"
