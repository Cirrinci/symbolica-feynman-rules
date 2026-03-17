from dataclasses import dataclass, field
from typing import Tuple, List
from collections import Counter
from symbolica import *
from IPython.display import Markdown as md, display as dp
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets




@dataclass(frozen=True)
class Index:
    name: str
    kind: str
    up: bool = False

    def __str__(self):
        return f"{self.name}{'^' if self.up else '_'}{self.kind[0]}"


@dataclass(frozen=True)
class TensorFactor:
    name: str
    indices: Tuple[Index, ...] = ()

    def __str__(self):
        if not self.indices:
            return self.name
        idx = ", ".join(i.name for i in self.indices)
        return f"{self.name}({idx})"


@dataclass
class Term:
    coeff: object
    factors: List[TensorFactor] = field(default_factory=list)

    def __str__(self):
        if not self.factors:
            return str(self.coeff)
        return f"{self.coeff} * " + " * ".join(str(f) for f in self.factors)


def index_count(term: Term):
    counts = Counter()
    for factor in term.factors:
        for idx in factor.indices:
            key = (idx.name, idx.kind)
            counts[key] += 1
    return counts


def contracted_indices(term: Term):
    counts = index_count(term)
    return [k for k, v in counts.items() if v == 2]


def free_indices(term: Term):
    counts = index_count(term)
    return [k for k, v in counts.items() if v == 1]


def validate_term(term: Term):
    counts = index_count(term)
    bad = {k: v for k, v in counts.items() if v > 2}
    if bad:
        raise ValueError(f"Invalid Einstein structure: {bad}")


def canonicalize_dummy_indices(term: Term) -> Term:
    counts = index_count(term)

    dummy_map = {}
    dummy_counter = 1

    for factor in term.factors:
        for idx in factor.indices:
            key = (idx.name, idx.kind)
            if counts[key] == 2 and key not in dummy_map:
                dummy_map[key] = f"d{dummy_counter}"
                dummy_counter += 1

    new_factors = []
    for factor in term.factors:
        new_indices = []
        for idx in factor.indices:
            key = (idx.name, idx.kind)
            if key in dummy_map:
                new_idx = Index(dummy_map[key], idx.kind, idx.up)
            else:
                new_idx = idx
            new_indices.append(new_idx)
        new_factors.append(TensorFactor(factor.name, tuple(new_indices)))

    return Term(term.coeff, new_factors)


if __name__ == "__main__":
    print("################################################################\n########################################################")
    g = Expression.symbol("g")
    lam = Expression.symbol("lam")
    x = Expression.symbol("x")

    print("Symbolica test expression =", g * x + 2 * x**2)

    i = Index("i", "flavor")
    j = Index("j", "flavor")
    k = Index("k", "flavor")

    phi_i = TensorFactor("phi", (i,))
    phi_j = TensorFactor("phi", (j,))
    phi_k = TensorFactor("phi", (k,))

    t1 = Term(g, [phi_i, phi_i])
    t2 = Term(lam, [phi_i, phi_i, phi_j, phi_j])

    print("t1 =", t1)
    print("t2 =", t2)

    print("t1 index count =", index_count(t1))
    print("t1 contracted =", contracted_indices(t1))
    print("t1 free =", free_indices(t1))

    print("t2 index count =", index_count(t2))
    print("t2 contracted =", contracted_indices(t2))
    print("t2 free =", free_indices(t2))

    validate_term(t1)
    validate_term(t2)
    print("Validation passed.")

    t3 = Term(g, [phi_j, phi_j])
    t4 = Term(g, [phi_k, phi_k])

    print("\nBefore canonicalization:")
    print("t1 =", t1)
    print("t3 =", t3)
    print("t4 =", t4)

    ct1 = canonicalize_dummy_indices(t1)
    ct3 = canonicalize_dummy_indices(t3)
    ct4 = canonicalize_dummy_indices(t4)

    print("\nAfter canonicalization:")
    print("ct1 =", ct1)
    print("ct3 =", ct3)
    print("ct4 =", ct4)
    print("Equal after canonicalization?", str(ct1) == str(ct3) == str(ct4))

    bad_term = Term(g, [phi_i, phi_i, phi_i])
    print("\nbad_term =", bad_term)
    print("bad_term index count =", index_count(bad_term))

    try:
        validate_term(bad_term)
        print("bad_term unexpectedly passed validation")
    except ValueError as e:
        print("Caught expected error:", e)


