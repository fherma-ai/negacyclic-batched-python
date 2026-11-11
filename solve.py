"""A reference answer to negacyclic-batched: B independent products, in Python.

Not an entry — a check that the specification is answerable and that a correct
answer's digest is the one the platform holds. It is the obvious algorithm: the
modulus is 1 mod 2N by construction, so a negacyclic transform of length N
exists over it, and each product is three transforms and a pointwise multiply.

What is setup and what is measured follows the specification. The ring depends
on (N, q) alone — the roots of unity, the powers of psi, the twiddle tables —
so all of it is built in `init`, which is not timed. `run` sees the operands
for the first time and does the arithmetic, including reading the limbs and
writing them back, which is where a GPU answer would pay for its transfers.
"""
import array
import os
from multiprocessing import Pool

from fherma import Inputs, Outputs, Point, Tensor, encode, unpack_one

#: The `array` code for a `u32`, in the width this platform agrees on.
CODE = "I" if array.array("I").itemsize == 4 else "L"

#: What a worker is told once, rather than with every pair it is handed.
_WORK = None


def init(p: Point):
    """The ring, once per point."""
    q = unpack_one(p.q)
    n = p.N
    psi = _root_of_unity(q, n)
    psi_inv = pow(psi, q - 2, q)
    omega = psi * psi % q

    pw = [1] * n
    iw = [1] * n
    for i in range(1, n):
        pw[i] = pw[i - 1] * psi % q
        iw[i] = iw[i - 1] * psi_inv % q

    ring = {
        "q": q,
        "n": n,
        "limbs": p.L,
        "batch": p.B,
        "pw": pw,
        "iw": iw,
        "n_inv": pow(n, q - 2, q),
        "forward": _twiddles(n, q, omega),
        "inverse": _twiddles(n, q, pow(omega, q - 2, q)),
    }

    # The products are independent, so they are worked out one to a core. The
    # pool is made here because `init` is not timed and because a worker builds
    # the ring once rather than once a pair — the tables are sixty megabytes of
    # integers and pickling them with every task would cost more than the
    # arithmetic. The answers are reassembled in index order, so the result
    # does not depend on how many cores the machine had.
    workers = min(p.B, os.cpu_count() or 1)
    ring["pool"] = Pool(workers, _enter, (ring,)) if workers > 1 else None
    if ring["pool"] is None:
        _enter(ring)
    return ring


def _enter(ring: dict) -> None:
    """A worker's one-time setup: the ring it will answer pairs in."""
    global _WORK
    _WORK = ring


def _answer(task):
    """One pair, bytes in and bytes out, with its place in the batch."""
    index, raw_a, raw_b = task
    ring = _WORK
    n, limbs = ring["n"], ring["limbs"]
    size = limbs * 4
    a = [int.from_bytes(raw_a[i * size : (i + 1) * size], "little") for i in range(n)]
    b = [int.from_bytes(raw_b[i * size : (i + 1) * size], "little") for i in range(n)]
    c = _product(a, b, ring)
    return index, b"".join(value.to_bytes(size, "little") for value in c)


def run(state, inp: Inputs) -> Outputs:
    """One pair to a core, through the wire form the harness documents.

    `encode` and the shape are what every release of the harness offers for a
    tensor. Reaching past them into `Tensor.data` is how this failed the first
    time: one release holds that buffer as an array and another as a list of
    integers, and a solution written against either runs on one of them.

    Reading the operands and writing the answer are inside the measured call
    because the specification says they are — a GPU answer pays for its
    transfers here too.
    """
    n, limbs, batch = state["n"], state["limbs"], state["batch"]
    size = n * limbs * 4
    a, b = memoryview(encode(inp.a)), memoryview(encode(inp.b))

    def tasks():
        # A generator, so only the pairs in flight are held as bytes.
        for j in range(batch):
            at = slice(j * size, (j + 1) * size)
            yield j, bytes(a[at]), bytes(b[at])

    pool = state["pool"]
    answers = (
        list(pool.imap_unordered(_answer, tasks(), chunksize=1))
        if pool
        else [_answer(one) for one in tasks()]
    )

    words = array.array(CODE)
    for _, raw in sorted(answers):
        words.frombytes(raw)
    return Outputs(c=Tensor((batch, n, limbs), words, "u32"))


def _product(a, b, ring) -> list:
    q, n = ring["q"], ring["n"]
    pw, iw, n_inv = ring["pw"], ring["iw"], ring["n_inv"]
    fa = _ntt([a[i] * pw[i] % q for i in range(n)], q, ring["forward"])
    fb = _ntt([b[i] * pw[i] % q for i in range(n)], q, ring["forward"])
    fc = _ntt([fa[i] * fb[i] % q for i in range(n)], q, ring["inverse"])
    return [(fc[i] * n_inv % q) * iw[i] % q for i in range(n)]


def _root_of_unity(q: int, n: int) -> int:
    """psi with psi**N = -1: a quadratic non-residue raised to (q-1)/2N."""
    exponent = (q - 1) // (2 * n)
    g = 2
    while True:
        if pow(g, (q - 1) // 2, q) == q - 1:
            psi = pow(g, exponent, q)
            if pow(psi, n, q) == q - 1:
                return psi
        g += 1


def _twiddles(n: int, q: int, root: int) -> list:
    tables = []
    length = 2
    while length <= n:
        half = length >> 1
        step = pow(root, n // length, q)
        table = [1] * half
        for k in range(1, half):
            table[k] = table[k - 1] * step % q
        tables.append(table)
        length <<= 1
    return tables


def _ntt(a: list, q: int, tables: list) -> list:
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length, stage = 2, 0
    while length <= n:
        half = length >> 1
        table = tables[stage]
        for start in range(0, n, length):
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * table[k] % q
                a[start + k] = (u + v) % q
                a[start + k + half] = (u - v) % q
        length <<= 1
        stage += 1
    return a


def free(state) -> None:
    pool = state.get("pool") if isinstance(state, dict) else None
    if pool:
        pool.close()
        pool.join()
