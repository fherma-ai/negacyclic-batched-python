"""GENERATED for polymul/negacyclic-batched@1.0.0. Do not edit — `--update` rewrites it.

    ./solution <point directory>

Reads the directory a bundle prepared, answers every case in it, and writes
what each one cost.

    manifest.json     the point, and how many cases      read
    cases/000000/     one file per argument              read
    out/000000/       one file per result                written
    out/results.json  what each case cost                written

Only the call to run() is timed. Reading and writing are outside the window.
"""
import json
import sys
import time
from pathlib import Path

from fherma import Inputs, Outputs, Point, Tensor, decode, encode
from solve import init, run

try:
    from solve import free
except ImportError:            # optional, and most answers do not need it
    def free(state): ...


def _read(path: Path) -> bytes:
    return path.read_bytes()


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _read_inputs(p: Point, case: Path) -> Inputs:
    return Inputs(
        a=decode(_read(case / "a.bin"), (p.B, p.N, p.L), "u32"),
        b=decode(_read(case / "b.bin"), (p.B, p.N, p.L), "u32"),
    )


def _write_outputs(case: Path, out: Outputs) -> None:
    _write(case / "c.bin", encode(out.c))


def _report(out: Path, init_s: float, cases: list) -> None:
    """Written after every case, not at the end.

    A process killed on its timeout has still done the cases before it, and a
    file written only at the end would throw them away.
    """
    _write(
        out / "results.json",
        json.dumps({"init_s": init_s, "cases": cases}, indent=2).encode(),
    )


def _resolve(root: Path, manifest: dict) -> Point:
    """The point a bundle wrote: free parameters from the manifest, derived
    scalars from beside them, and derived tensors from point/ — the getters ran
    once, at make time, and their results are read back here."""
    point = manifest["point"]
    derived = manifest.get("derived", {})
    B = point["B"]
    N = point["N"]
    W = point["W"]
    L = derived["L"]
    q = decode((root / "point" / "q.bin").read_bytes(), (L,), "u32")
    return Point(B=B, N=N, W=W, L=L, q=q)


def main() -> None:
    root = Path(sys.argv[1])
    manifest = json.loads((root / "manifest.json").read_text())
    p = _resolve(root, manifest)
    out = root / "out"
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    state = init(p)
    init_s = time.perf_counter() - started

    cases: list = []
    _report(out, init_s, cases)

    for i in range(manifest["cases"]):
        case = root / "cases" / f"{i:06}"

        try:
            inp = _read_inputs(p, case)
        except Exception as failure:
            cases.append({"i": i, "seconds": None, "status": "crashed",
                          "note": f"reading the case: {failure}"[:200]})
            _report(out, init_s, cases)
            continue

        try:
            # Monotonic, and around the call and nothing else. Wall time moves
            # under NTP; process time hides waiting and says nothing at all
            # about work an accelerator did.
            t0 = time.perf_counter()
            answer = run(state, inp)
            t1 = time.perf_counter()
        except Exception as failure:
            cases.append({"i": i, "seconds": None, "status": "crashed",
                          "note": str(failure)[:200]})
            _report(out, init_s, cases)
            continue

        try:
            _write_outputs(out / f"{i:06}", answer)
        except Exception as failure:
            cases.append({"i": i, "seconds": t1 - t0, "status": "crashed",
                          "note": f"writing the answer: {failure}"[:200]})
            _report(out, init_s, cases)
            continue

        cases.append({"i": i, "seconds": t1 - t0, "status": "ok"})
        _report(out, init_s, cases)

    free(state)
    _report(out, init_s, cases)


if __name__ == "__main__":
    main()
