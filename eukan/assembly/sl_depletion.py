"""In-silico spliced-leader (SL) depletion of de novo transcript assemblies.

Trans-spliced organisms add a constant spliced leader to the 5' end of mature
mRNAs. A de novo assembler can string several such mRNAs into a single contig
(and prepend the SL onto a transcript's 5' end), leaving the SL sequence
*internal* to the assembly. This step finds those internal SL occurrences and
cuts the contig at each one — discarding the SL bases — so a fused contig is
split back into the separate transcripts it was assembled from. N SL sites yield
up to N+1 fragments; fragments shorter than ``min_sl_fragment`` are dropped.

The SL motif comes from the soft-clip diagnostic verdict
(``softclip_diagnostic_summary.json``) when trans-splicing was called
STRONG/MODERATE, or from an explicit ``config.sl_sequence`` override. With no SL
signal the step is an identity pass (sequences copied through, zero cuts), so the
downstream transcript-mapping step gets a uniform input regardless of organism.

Only the *de novo* assemblies are processed (genome-guided Trinity is already
genome-anchored, so an internal SL leader is not expected there). The SL is
searched on both strands because a de novo contig's orientation is arbitrary.
"""

from __future__ import annotations

import json

from Bio import SeqIO

from eukan.infra.artifacts import Artifact
from eukan.infra.logging import get_logger
from eukan.settings import AssemblyConfig

log = get_logger(__name__)

# De novo assemblies the SL depletion operates on, each paired with the
# canonical output the transcript-mapping step consumes.
_DE_NOVO_FASTAS = ("trinity-denovo.fasta", "rnaspades.fasta")

# Shortest SL motif we will cut on. A very short motif (e.g. the 6 bp GTACTT
# core) occurs by chance often enough to shred transcripts, so require a more
# specific one before cutting.
_MIN_MOTIF_LEN = 10

# Substitutions tolerated when matching the SL motif. Default 0 (exact): the SL
# is A/T-rich, so its reverse complement is A-rich and a mismatch-tolerant search
# spuriously matches poly-A tails / homopolymer runs and over-cuts transcripts.
# The recovered SL consensus and the assembled contigs are both already
# error-corrected, so exact matching is safe and sufficient; the knob is kept
# for tuning.
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


def _cut(seq: str, sites: list[tuple[int, int]]) -> list[str]:
    """Fragments of *seq* with each SL interval removed (the SL bases dropped)."""
    fragments = []
    prev = 0
    for start, end in sites:
        fragments.append(seq[prev:start])
        prev = end
    fragments.append(seq[prev:])
    return fragments


def _resolve_sl_motif(config: AssemblyConfig) -> str | None:
    """The SL motif to deplete with, or ``None`` for an identity pass.

    Explicit ``config.sl_sequence`` wins; otherwise the verdict's recovered SL
    consensus is used, but only when trans-splicing was called STRONG/MODERATE
    and the consensus is specific enough (``>= _MIN_MOTIF_LEN``).
    """
    if config.sl_sequence:
        return config.sl_sequence.strip().upper()

    summary = config.work_dir / Artifact.SOFTCLIP_DIAGNOSTIC.value
    if not summary.exists():
        return None
    ts = json.loads(summary.read_text()).get("verdict", {}).get("trans_splicing", {})
    if ts.get("call") not in ("STRONG", "MODERATE"):
        return None
    motif = (
        ts.get("top_non_trivial_cluster_consensus")
        or ts.get("top_non_trivial_cluster_key")
        or ""
    ).upper()
    if len(motif) < _MIN_MOTIF_LEN:
        log.warning(
            "Trans-splicing called %s but the recovered SL motif %r is shorter "
            "than %d bp; skipping depletion (set sl_sequence to override).",
            ts.get("call"), motif, _MIN_MOTIF_LEN,
        )
        return None
    return motif


def _deplete_fasta(
    src, out, patterns: set[str], motif_len: int, min_fragment: int
) -> tuple[int, int, int, int]:
    """Cut SL sites out of every record in *src*, writing fragments to *out*.

    Returns ``(n_in, n_out, n_records_cut, n_sites)``.
    """
    n_in = n_out = n_records_cut = n_sites = 0
    with open(out, "w") as fh:
        for rec in SeqIO.parse(str(src), "fasta"):
            n_in += 1
            original = str(rec.seq)
            sites = _find_sites(original.upper(), patterns, motif_len)
            if not sites:
                fh.write(f">{rec.id}\n{original}\n")
                n_out += 1
                continue
            n_records_cut += 1
            n_sites += len(sites)
            kept = 0
            for fragment in _cut(original, sites):
                if len(fragment) >= min_fragment:
                    kept += 1
                    fh.write(f">{rec.id}.sl{kept}\n{fragment}\n")
                    n_out += 1
    return n_in, n_out, n_records_cut, n_sites


def run_sl_depletion(config: AssemblyConfig) -> None:
    """Cut internal SL sites out of the de novo assemblies (identity if no SL)."""
    wd = config.work_dir
    motif = _resolve_sl_motif(config)
    if motif is None:
        log.info("No spliced-leader signal; SL depletion is a pass-through.")
        patterns: set[str] = set()
        motif_len = 0
    else:
        patterns = _variants(motif, _MAX_MISMATCHES) | _variants(
            _revcomp(motif), _MAX_MISMATCHES
        )
        motif_len = len(motif)
        match_desc = "exact" if _MAX_MISMATCHES == 0 else f"<={_MAX_MISMATCHES} mismatch"
        log.info(
            "SL depletion using motif %s (%s match, both strands).", motif, match_desc
        )

    for src_name in _DE_NOVO_FASTAS:
        src = wd / src_name
        if not src.exists():
            continue
        out = wd / src_name.replace(".fasta", ".sl_depleted.fasta")
        n_in, n_out, n_cut, n_sites = _deplete_fasta(
            src, out, patterns, motif_len, config.min_sl_fragment
        )
        log.info(
            "SL depletion %s -> %s: %d -> %d sequences (%d contigs cut at %d SL sites).",
            src.name, out.name, n_in, n_out, n_cut, n_sites,
        )
