"""Exact arithmetic on Weierstrass models, using Python integers (coefficients can exceed 64 bits)."""


def b_invariants(a1: int, a2: int, a3: int, a4: int, a6: int) -> tuple[int, int, int, int]:
    """The standard b2, b4, b6, b8 of a Weierstrass model."""
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4
    return b2, b4, b6, b8


def discriminant(a1: int, a2: int, a3: int, a4: int, a6: int) -> int:
    b2, b4, b6, b8 = b_invariants(a1, a2, a3, a4, a6)
    return -b2 * b2 * b8 - 8 * b4**3 - 27 * b6 * b6 + 9 * b2 * b4 * b6


def is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 3.3e24, far beyond any conductor we see."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def factor(n: int) -> dict[int, int]:
    """Prime factorization of a positive integer by trial division, short-cut when the cofactor is prime.

    Fine for conductors (at most 3e8); not meant for discriminants.
    """
    out: dict[int, int] = {}
    p = 2
    while n > 1:
        if is_probable_prime(n):
            out[n] = out.get(n, 0) + 1
            break
        while n % p:  # n is composite, so this stops at its smallest prime factor
            p += 1 if p == 2 else 2
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
    return out
