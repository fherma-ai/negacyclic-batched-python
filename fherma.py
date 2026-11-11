"""GENERATED from polymul/negacyclic-batched@1.0.0. Do not edit.

Types derived from the signature. Write the bodies in generate.py and oracle.py;
serialising is the runner's job, so nothing here opens a file.
"""
from dataclasses import dataclass, field
import array
import hashlib
import os
import struct
import sys

SPEC = "polymul/negacyclic-batched@1.0.0"

#: Bits on the wire, by element type.
WIDTHS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "u8": 8, "u16": 16, "u32": 32, "u64": 64, "f32": 32, "f64": 64}


@dataclass(frozen=True)
class Point:
    """One field per resolved parameter: the free ones the point sets, then the
    derived ones its getters fill. `resolve` in main.py builds it."""
    B: int           # free · u32
    N: int           # free · u32
    W: int           # free · u32
    L: int           # derived · u32 = W + 31) // 32
    q: "Tensor"           # derived · tensor<L x u32>


@dataclass
class Inputs:
    a: "Tensor"     # %a: tensor<B x N x L x u32>
    b: "Tensor"     # %b: tensor<B x N x L x u32>


@dataclass
class Outputs:
    c: "Tensor"     # %c: tensor<B x N x L x u32>


@dataclass
class Verdict:
    passed: bool
    metrics: dict = field(default_factory=dict)


def _typecode(elem):
    """An `array` typecode of exactly this element's width, or None for a type
    that has none.

    Probed rather than tabulated, because C leaves the widths of `int` and
    `long` to the platform and only the itemsize settles which is which.
    """
    if elem[0] == "f":
        return {32: "f", 64: "d"}.get(WIDTHS[elem])
    want = WIDTHS[elem] // 8
    for code in ("b", "h", "i", "l", "q") if elem[0] == "i" else ("B", "H", "I", "L", "Q"):
        if array.array(code).itemsize == want:
            return code
    return None


def _buffer(values, elem):
    """`values` as a typed buffer rather than a list of Python integers.

    One tensor of a batched point runs to hundreds of millions of elements. As
    a list that is forty bytes apiece — ten times the bytes it stands for — and
    writing it out costs a conversion per element. As an `array` it is the wire
    form already: it indexes the same way, and it serialises with one memcpy.
    An element type `array` has no code for falls back to the list.
    """
    code = _typecode(elem)
    if code is None:
        return list(values)
    if isinstance(values, array.array) and values.typecode == code:
        return values
    if isinstance(values, (bytes, bytearray, memoryview)):
        out = array.array(code)
        out.frombytes(bytes(values))
        if sys.byteorder != "little":
            out.byteswap()
        return out
    return array.array(code, values)


class Tensor:
    """A flat, row-major buffer. Shape is known at run time; the element type
    comes from the signature and fixes the width on the wire."""

    __slots__ = ("shape", "elem", "data")

    def __init__(self, shape, data, elem="i64"):
        self.shape, self.elem, self.data = tuple(shape), elem, _buffer(data, elem)
        if len(self.data) != self.count():
            raise ValueError(f"{len(self.data)} values for shape {self.shape}")

    def count(self):
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total

    def nbytes(self):
        return self.count() * WIDTHS[self.elem] // 8

    def __repr__(self):
        return f"Tensor(shape={self.shape}, elem={self.elem!r}, …)"


class Stream:
    """The only source of randomness on offer.

    block(i) = SHA256(SPEC | seed | i); the stream is block(0) ‖ block(1) ‖ …
    It is the same in every language because it is written down, and it is
    seekable, so generating large inputs parallelises. Anything else — random,
    time, urandom, secrets — makes a generator irreproducible and is rejected.
    """

    def __init__(self, seed: int):
        self._prefix = f"{SPEC}|{seed}"
        self._buffer, self._counter = b"", 0

    def _fill(self, n: int) -> None:
        if len(self._buffer) >= n:
            return
        # Collected, then joined once. `self._buffer += block` rewrites the
        # whole buffer every thirty-two bytes, so the cost is quadratic in the
        # length: a tenth of a megabyte a second where this does seventy.
        blocks, have = [self._buffer], len(self._buffer)
        while have < n:
            block = f"{self._prefix}|{self._counter}".encode()
            blocks.append(hashlib.sha256(block).digest())
            have += 32
            self._counter += 1
        self._buffer = b"".join(blocks)

    def bytes(self, n: int) -> bytes:
        self._fill(n)
        head, self._buffer = self._buffer[:n], self._buffer[n:]
        return head

    def bits(self, k: int) -> int:
        return int.from_bytes(self.bytes((k + 7) // 8), "little") & ((1 << k) - 1)

    def below(self, n: int) -> int:
        """Uniform on [0, n). The extra 64 bits keep the bias under 2**-64."""
        return self.bits(n.bit_length() + 64) % n


class Blocks:
    """The same idea as `Stream`, for inputs measured in gigabytes.

    block(i) = SHAKE128(SPEC | seed | i), BLOCK bytes of it; the stream is
    block(0) ‖ block(1) ‖ … It is written down and seekable exactly as `Stream`
    is, and it is fifteen times the bytes per second, because one call yields a
    megabyte where a digest yields thirty-two bytes.

    Which to reach for: `Stream` for a case of a few megabytes, where its
    `below` and `bits` are the convenience; this where the case is hundreds of
    megabytes and the bytes themselves are the tensor.
    """

    BLOCK = 1 << 20

    def __init__(self, seed: int):
        self._prefix = f"{SPEC}|{seed}"
        self._at = 0

    def block(self, index: int) -> bytes:
        """Block `index`, on its own — the stream is seekable through this."""
        return hashlib.shake_128(f"{self._prefix}|{index}".encode()).digest(self.BLOCK)

    def bytes(self, n: int) -> bytes:
        """The next `n` bytes, continuing where the last call stopped."""
        out, want = [], n
        while want:
            index, offset = divmod(self._at, self.BLOCK)
            take = min(want, self.BLOCK - offset)
            out.append(self.block(index)[offset : offset + take])
            self._at += take
            want -= take
        return out[0] if len(out) == 1 else b"".join(out)

    def seek(self, position: int) -> None:
        """Where the next `bytes` reads from, counted from the stream's start."""
        self._at = position


def encode(tensor: Tensor) -> bytes:
    """A tensor as it goes on the wire: little-endian, row-major, no header.

    The shape travels beside it in case.json rather than inside the bytes, so a
    reader in another language needs no parser — only the width, which the
    signature already fixed.
    """
    if isinstance(tensor.data, array.array):
        # Already the wire form: one memcpy, rather than a conversion per
        # element. Native order is little on every platform this runs on, and
        # the swap is here so that "little-endian" stays true where it is not.
        if sys.byteorder != "little":
            swapped = array.array(tensor.data.typecode, tensor.data)
            swapped.byteswap()
            return swapped.tobytes()
        return tensor.data.tobytes()

    if tensor.elem[0] == "f":
        kind = {32: "f", 64: "d"}.get(WIDTHS[tensor.elem])
        if kind is None:
            raise ValueError(f"{tensor.elem} has no wire form yet")
        return struct.pack(f"<{tensor.count()}{kind}", *tensor.data)

    width = WIDTHS[tensor.elem] // 8
    signed = tensor.elem[0] == "i"
    return b"".join(
        int(value).to_bytes(width, "little", signed=signed) for value in tensor.data
    )


def decode(raw: bytes, shape, elem="i64") -> Tensor:
    """The inverse of encode."""
    count = 1
    for dimension in shape:
        count *= dimension

    width = WIDTHS[elem] // 8
    if len(raw) != count * width:
        raise ValueError(f"{len(raw)} bytes for {tuple(shape)} of {elem}")

    if _typecode(elem) is not None:
        # The bytes are the buffer. Nothing is converted and nothing is copied
        # per element; a gigabyte case reads as fast as the disk hands it over.
        return Tensor(shape, raw, elem)

    if elem[0] == "f":
        kind = {32: "f", 64: "d"}.get(WIDTHS[elem])
        if kind is None:
            raise ValueError(f"{elem} has no wire form yet")
        values = list(struct.unpack(f"<{count}{kind}", raw))
    else:
        signed = elem[0] == "i"
        values = [
            int.from_bytes(raw[i * width : (i + 1) * width], "little", signed=signed)
            for i in range(count)
        ]
    return Tensor(shape, values, elem)


def pack(values, limbs, elem="i64") -> Tensor:
    """Integers wider than a word into `limbs` words each, low word first.

    A row per value, so a list of coefficients becomes a tensor<len x limbs x
    elem>. The word is as wide as the element the signature named — 32 bits for
    u32, which is what a limb array on the GPU expects — not always 64.

    A value too wide for the words it is given is refused rather than cut
    down. It used to be cut down: a point that asked for two 64-bit words got
    the low half of a 254-bit modulus, which is a different prime, and
    everything after it computed in a field nobody chose — correctly, against
    the wrong answer, with nothing in the record to say so.
    """
    width = WIDTHS[elem]
    mask = (1 << width) - 1
    room = width * limbs
    low, high = (-(1 << (room - 1)), 1 << (room - 1)) if elem[0] == "i" else (0, 1 << room)
    code = _typecode(elem)
    size, signed = room // 8, elem[0] == "i"
    data = array.array(code) if code else []
    held = []
    for value in values:
        if not low <= value < high:
            raise ValueError(
                f"{value} does not fit in {limbs} {elem} word(s): it needs "
                f"{value.bit_length()} bits and there is room for {room}. "
                "The point asks for a width this value cannot be written in."
            )
        if code is None:
            for index in range(limbs):
                data.append((value >> (width * index)) & mask)
            continue
        # One conversion per value rather than one per limb: the two's
        # complement bytes of a whole coefficient already are its limbs, low
        # end first. Appended in batches so the held bytes stay bounded.
        held.append(int(value).to_bytes(size, "little", signed=signed))
        if len(held) == 4096:
            data.frombytes(b"".join(held))
            held.clear()
    if code is not None:
        if held:
            data.frombytes(b"".join(held))
        if sys.byteorder != "little":
            data.byteswap()
    return Tensor((len(values), limbs), data, elem)


def pack_one(value, limbs, elem="i64") -> Tensor:
    """One wide integer as a flat tensor<limbs x elem> — a single value, not a row.

    This is what a derived modulus wants: q is one number, held as `limbs` words,
    with no leading dimension. `pack([q], limbs)` would give a tensor<1 x limbs>,
    which is the same bytes shaped wrong.
    """
    row = pack([value], limbs, elem)
    return Tensor((limbs,), row.data, elem)


def unpack(tensor: Tensor) -> list:
    """The inverse of pack, for a tensor whose last dimension is the limbs.

    Through the wire form rather than limb by limb: one `int.from_bytes` a
    value, not a shift and an or per limb, which at fifty-five limbs is the
    difference between seconds and minutes.

    A signed element type reads back signed, so a negative survives the round
    trip. It did not before: `pack` stored the two's-complement words and this
    summed them, which gave the bit pattern — 2**64 - 1 for -1 — and no way to
    tell that from a value that really was 2**64 - 1.
    """
    *head, limbs = tensor.shape
    size = WIDTHS[tensor.elem] // 8 * limbs
    signed = tensor.elem[0] == "i"
    raw = encode(tensor)
    count = 1
    for dimension in head:
        count *= dimension
    return [
        int.from_bytes(raw[i * size : (i + 1) * size], "little", signed=signed)
        for i in range(count)
    ]


def unpack_one(tensor: Tensor) -> int:
    """The inverse of pack_one: a flat limb tensor back to one integer."""
    return int.from_bytes(encode(tensor), "little", signed=tensor.elem[0] == "i")


# ── derivation rules ──────────────────────────────────────────────────────────
# The standard library a getter reaches for. Deterministic by construction: the
# same arguments give the same value in the generator, the oracle and every
# runner, which is what stops the three disagreeing about the ring.
def _probably_prime(n: int) -> bool:
    """Miller–Rabin with a fixed base set, after trial division by small primes.

    The bases are fixed rather than random so the result is the same everywhere.
    Forty of them put the chance of a composite slipping through below any figure
    that matters here, and trial division rejects almost everything before them.
    """
    if n < 2:
        return False
    small = (
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
        71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
        151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
    )
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False

    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small[:40]:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def ntt_prime(n: int, bits: int) -> int:
    """The largest prime below 2**bits that is 1 mod 2n.

    An NTT-friendly modulus: q ≡ 1 (mod 2n) is exactly the condition for a
    primitive 2n-th root of unity to exist mod q, so a negacyclic transform of
    length n works over it. Searched downward from just under 2**bits, over the
    residues that already satisfy the congruence, so only primality is left to test.
    """
    step = 2 * n
    # The largest q < 2**bits with q ≡ 1 (mod 2n): start there and walk down.
    q = ((1 << bits) - 1 - 1) // step * step + 1
    while q > 1:
        if _probably_prime(q):
            return q
        q -= step
    raise ValueError(f"no prime below 2**{bits} is 1 mod {step}")


def asset(name: str) -> str:
    """Path to a file in assets/, wherever this bundle happens to be mounted."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", name)
