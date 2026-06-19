"""Wilson 95% score interval for a binomial proportion. Used to gate every "win" claim:
a higher point estimate that does not clear the champion is noise, not a win.

  python wilson.py <x> <N> [<x2> <N2> ...]      # one CI per (x,N) pair
"""
import sys
from math import sqrt


def wilson(x, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = x / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def main():
    a = sys.argv[1:]
    for i in range(0, len(a), 2):
        x, n = int(a[i]), int(a[i + 1])
        p, lo, hi = wilson(x, n)
        print(f"{x}/{n} = {100*p:.1f}%  Wilson95 [{100*lo:.1f}, {100*hi:.1f}]")


if __name__ == "__main__":
    raise SystemExit(main())
