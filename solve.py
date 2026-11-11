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
from fherma import Inputs, Outputs, Point, Tensor, pack, unpack, unpack_one


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

    return {
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


def run(state, inp: Inputs) -> Outputs:
    """Through `unpack` and `pack` and nothing else.

    They are what the harness offers for a tensor whose last dimension is the
    limbs, and they are what every version of it offers. Reaching past them
    into `Tensor.data` is how this failed the first time: one release holds
    that buffer as an array and another as a list of integers, and a solution
    written against either is a solution that runs on one of them.
    """
    n, limbs, batch = state["n"], state["limbs"], state["batch"]
    a, b = unpack(inp.a), unpack(inp.b)

    out = []
    for j in range(batch):
        lo, hi = j * n, (j + 1) * n
        out.extend(_product(a[lo:hi], b[lo:hi], state))

    # pack gives tensor<B*N x L>; the shape the signature asks for is the same
    # words with the batch dimension spelled out.
    return Outputs(c=Tensor((batch, n, limbs), pack(out, limbs, "u32").data, "u32"))


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
    return None
