"""Spliced-leader (SL) motif matching primitives.

Trans-spliced organisms add a constant spliced leader to the 5' end of every
mature mRNA. These pure helpers — reverse-complement, mismatch-variant
enumeration, and merged-interval substring search — are the shared SL-matching
core, consumed by the SL trans-splice acceptor detector
(:mod:`eukan.assembly.sl_acceptors`). The SL is searched on both strands because
a contig's orientation is arbitrary.

(Historically this module also *depleted* the SL from de novo assembly FASTAs
before mapping; that step was retired in favour of cutting transcript models at
genomic SL acceptor sites — see :mod:`eukan.assembly.sl_cut` — so only the
matching primitives remain here.)
"""

from __future__ import annotations

# Shortest SL motif we will match on. A very short motif (e.g. the 6 bp GTACTT
# core) occurs by chance often enough to shred transcripts, so require a more
# specific one before matching.
_MIN_MOTIF_LEN = 10

# Substitutions tolerated when matching the SL motif. Default 0 (exact): the SL
# is A/T-rich, so its reverse complement is A-rich and a mismatch-tolerant search
# spuriously matches poly-A tails / homopolymer runs. The recovered SL consensus
# and the assembled contigs are both already error-corrected, so exact matching
# is safe and sufficient; the knob is kept for tuning.
_MAX_MISMATCHES = 0

_DNA_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _revcomp(seq: str) -> str:
    return seq.translate(_DNA_COMP)[::-1]


def _variants(motif: str, max_mismatches: int) -> set[str]:
    """Every sequence within ``max_mismatches`` substitutions of *motif*.

    Enumerating variants lets matching use C-speed ``str.find`` instead of a
    per-position Python comparison loop — substantially faster across the tens
    of thousands of contigs a de novo assembler produces. All variants share
    *motif*'s length (substitutions only), so match spans are a fixed width.
    """
    variants = {motif}
    frontier = {motif}
    for _ in range(max(0, max_mismatches)):
        nxt: set[str] = set()
        for seq in frontier:
            for i, base in enumerate(seq):
                for alt in "ACGT":
                    if alt != base:
                        nxt.add(seq[:i] + alt + seq[i + 1 :])
        variants |= nxt
        frontier = nxt
    return variants


def _find_sites(seq: str, patterns: set[str], motif_len: int) -> list[tuple[int, int]]:
    """Merged ``[start, end)`` intervals where any SL pattern matches *seq*."""
    if not patterns or motif_len == 0:
        return []
    hits: list[tuple[int, int]] = []
    for pat in patterns:
        i = seq.find(pat)
        while i != -1:
            hits.append((i, i + motif_len))
            i = seq.find(pat, i + 1)
    if not hits:
        return []
    hits.sort()
    merged = [hits[0]]
    for start, end in hits[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
