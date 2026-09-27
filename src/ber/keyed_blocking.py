from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd
from rapidfuzz import fuzz

from ber.blocking import combine_candidate_frames
from ber.config import BlockingConfig

# NOTE: multiprocessing (both "spawn" and "fork" start methods) was tried
# here to parallelize row-scoring across CPU cores and measured SLOWER than
# single-process in both cases at real (4M+ row) corpus scale -- "spawn"
# pays a large one-time pickling cost for the per-country index broadcast to
# each worker, and "fork" still triggers copy-on-write page faults across a
# large reference-counted Python dict as workers merely read it (refcount
# increments are writes). Neither paid for itself here, so this stays
# single-process; a real speedup would need a non-Python-object index
# (e.g. numpy/arrow-backed) shared via true shared memory.

# Deterministic bucket keys used to shrink the search space before any pairwise
# scoring happens. Each row contributes to several buckets (one per key type);
# two records are only ever compared if they share at least one bucket. This
# turns an O(left x right) or O(left x corpus) problem into O(left x avg bucket
# size), which is what makes blocking tractable at multi-million-row scale --
# the plain full-corpus TF-IDF/token blockers in blocking.py do not scale past
# a few thousand left rows against a multi-million row corpus (see benchmarks:
# ~13h+ extrapolated for a single country/blocker at full test-set size).

NAME_PREFIX_LEN = 5
MAX_BUCKET_SIZE = 5000  # keys with a posting list bigger than this are dropped outright (pathological/degenerate keys); candidate_budget is what actually bounds per-row scoring cost
MIN_TOKEN_LEN = 3
MIN_CORE_TOKEN_LEN = 4  # threshold for the frequency-capped single-word keys (name_token/addr_token/name_soundex)
TOKEN_FREQUENCY_CAP = 250  # a token more common than this in the right-side corpus is excluded from the index entirely

# V5 pass (see output_v5/V5_RUN_LOG.md): switched name_token/addr_token from
# "each row's top-N rarest tokens" (a fixed per-row count) to "every
# sufficiently-long token, but excluded from the index if its TRUE corpus
# frequency exceeds TOKEN_FREQUENCY_CAP". The old top-N approach could miss
# a moderately-rare-but-not-top-N token that would have been a good signal,
# and didn't bound cost against the real driver of slow/huge buckets:
# generic words ("restaurant", "store", "services", ...) appearing tens of
# thousands of times, which a fixed per-row count of 3 does nothing to
# exclude if a row simply doesn't have 3 more-specific tokens. Frequency
# capping targets the actual problem directly and let every row propose
# every reasonably-long token as a candidate key.

# Recall-improvement pass (2026-09-26 overnight, see output_recall_v3/RECALL_RUN_LOG.md):
# added key types below (name_first_token, name_sorted, name_soundex,
# addr_num_street) are purely additive to the original four (name_prefix,
# name_token, addr_token, addr_num) -- every original key stays, so this
# cannot regress a row that was already finding its true match; it can only
# add more candidate rows via the same union-of-postings mechanism that
# already existed. Each targets a specific known failure mode: name_sorted
# catches reordered tokens exactly; name_soundex catches typos/transliteration
# on the same rare tokens already used for exact matching; addr_num_street
# is a more specific (rarer, less likely to hit MAX_BUCKET_SIZE) version of
# the bare addr_num key, pairing a house/door number with a nearby rare
# address token.

_SOUNDEX_CODES = {
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3",
    "l": "4",
    "m": "5", "n": "5",
    "r": "6",
}


def _soundex(token: str) -> str:
    """Simplified American Soundex -- a coarse phonetic bucket key, not an
    exact-matching algorithm. Good enough to group typo/transliteration
    variants of the same word into the same bucket; precision loss within a
    bucket is handled downstream by RapidFuzz scoring + candidate_budget,
    same as every other key type here.
    """
    letters = [ch for ch in token.lower() if ch.isalpha()]
    if not letters:
        return ""
    coded = [letters[0]]
    prev_code = _SOUNDEX_CODES.get(letters[0], "")
    for ch in letters[1:]:
        code = _SOUNDEX_CODES.get(ch, "")
        if code and code != prev_code:
            coded.append(code)
        prev_code = code
    return ("".join(coded) + "000")[:4]


def _alnum_prefix(value: str, length: int) -> str:
    compact = "".join(ch for ch in str(value) if ch.isalnum())
    return compact[:length]


def _core_tokens(text: str, min_len: int = MIN_TOKEN_LEN) -> list[str]:
    return [t for t in str(text or "").split() if len(t) >= min_len]


def _row_keys(
    name_norm: str,
    address_norm: str,
    address_numbers: str,
    freq: Counter[str] | None = None,
) -> list[tuple[str, str]]:
    """Propose every candidate blocking key for one row.

    ``freq`` (the right-side corpus's global token frequency, or None to
    skip capping) gates only the single-word ``name_token``/``addr_token``/
    ``name_soundex`` keys -- these are the ones a generic word like
    "restaurant" or "services" would otherwise blow up. The composite keys
    (name_num, addr_num_street, name_city, num_city) are conjunctions of two
    facts and are inherently much rarer than either component alone, so they
    rely on the coarser MAX_BUCKET_SIZE backstop in _build_right_index
    instead of their own frequency cap -- kept simple deliberately.
    """
    keys: list[tuple[str, str]] = []
    name_norm = str(name_norm or "")
    address_norm = str(address_norm or "")

    # compact_n: space/punctuation-stripped alphanumeric name. Catches
    # concatenation variants ("moonboards.com" <-> "Moon Boards") that no
    # token-based key can, since it collapses word boundaries entirely.
    compact = "".join(ch for ch in name_norm if ch.isalnum())
    if len(compact) >= 4:
        keys.append(("name_compact", compact))

    name_prefix = _alnum_prefix(name_norm, NAME_PREFIX_LEN)
    if len(name_prefix) >= 3:
        keys.append(("name_prefix", name_prefix))

    name_tokens = _core_tokens(name_norm)
    addr_tokens = _core_tokens(address_norm)
    nums = [n for n in str(address_numbers or "").split("|") if len(n) >= MIN_TOKEN_LEN]

    if name_tokens:
        keys.append(("name_first_token", name_tokens[0]))
        if len(name_tokens) >= 2:
            keys.append(("name_sorted", " ".join(sorted(name_tokens))))
            keys.append(("name_first2", " ".join(name_tokens[:2])))

    # n_tok: every sufficiently-long core token is a *candidate* key; the
    # frequency cap (not a fixed per-row count) is what decides whether it's
    # cheap enough to actually index -- see module comment.
    for token in name_tokens:
        if len(token) >= MIN_CORE_TOKEN_LEN and (freq is None or freq.get(token, 0) <= TOKEN_FREQUENCY_CAP):
            keys.append(("name_token", token))
            soundex = _soundex(token)
            if soundex:
                keys.append(("name_soundex", soundex))
    for token in addr_tokens:
        if len(token) >= MIN_CORE_TOKEN_LEN and (freq is None or freq.get(token, 0) <= TOKEN_FREQUENCY_CAP):
            keys.append(("addr_token", token))

    for num in nums:
        keys.append(("addr_num", num))
        if name_tokens:
            keys.append(("name_num", f"{name_tokens[0]}_{num}"))
        for token in addr_tokens:
            keys.append(("addr_num_street", f"{num}_{token}"))

    if addr_tokens:
        # This dataset's addresses consistently end with city/state tokens
        # (verified by eye across many examples tonight, e.g. "...basavanagudi
        # bangalore", "...ludhiana punjab", "...kanpur nagar uttar pradesh")
        # -- there's no separate city/state field to key off directly, so the
        # last address token is used as a cheap proxy for the reference
        # repo's num_city/name_city keys.
        city_proxy = addr_tokens[-1]
        if name_tokens:
            keys.append(("name_city", f"{name_tokens[0]}_{city_proxy}"))
        for num in nums:
            keys.append(("num_city", f"{num}_{city_proxy}"))

    return keys


def _corpus_token_frequency(right: pd.DataFrame) -> Counter[str]:
    """Global (name+address) token frequency across the right-side corpus,
    used only to decide which name_token/addr_token keys are cheap enough to
    index (see TOKEN_FREQUENCY_CAP)."""
    freq: Counter[str] = Counter()
    for text in right["name_norm"]:
        freq.update(set(_core_tokens(text, MIN_CORE_TOKEN_LEN)))
    for text in right["address_norm"]:
        freq.update(set(_core_tokens(text, MIN_CORE_TOKEN_LEN)))
    return freq


def _build_right_index(right: pd.DataFrame) -> tuple[dict[tuple[str, str], list[int]], Counter[str]]:
    freq = _corpus_token_frequency(right)
    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for pos, row in enumerate(right.itertuples(index=False)):
        for key in _row_keys(row.name_norm, row.address_norm, row.address_numbers, freq):
            index[key].append(pos)
    filtered = {key: postings for key, postings in index.items() if len(postings) <= MAX_BUCKET_SIZE}
    return filtered, freq


def _score(s1_name: str, cand_name: str, s1_addr: str, cand_addr: str) -> float:
    name_score = fuzz.token_sort_ratio(s1_name or "", cand_name or "")
    addr_score = fuzz.token_sort_ratio(s1_addr or "", cand_addr or "")
    return 0.7 * name_score + 0.3 * addr_score


def _score_rows(
    rows: list[tuple],
    index: dict[tuple[str, str], list[int]],
    freq: Counter[str],
    right_ids,
    right_names,
    right_addrs,
    top_k: int,
    candidate_budget: int,
    blocker_name: str,
) -> list[dict[str, object]]:
    """Score a batch of left rows against an already-built right-side index.

    A row's keys are processed smallest-bucket-first (its most specific,
    highest-precision signals: a shared postal code or rare token beats a
    generic 5-char name prefix), accumulating scored candidates until
    ``candidate_budget`` unique right-side rows have been gathered, then
    stopping. This bounds the per-row scoring cost to O(budget) regardless
    of how large a bucket a row happens to touch -- capping the bucket size
    itself (MAX_BUCKET_SIZE) does not bound this, since a row can still
    belong to several large buckets whose union is unbounded.
    """
    records: list[dict[str, object]] = []
    for entity_id, name_norm, address_norm, address_numbers in rows:
        keys = _row_keys(name_norm, address_norm, address_numbers, freq)
        if not keys:
            continue

        keyed_postings = [(key, index[key]) for key in keys if key in index]
        keyed_postings.sort(key=lambda item: len(item[1]))

        selected: dict[int, float] = {}
        for _key, postings in keyed_postings:
            for right_pos in postings:
                if right_pos in selected:
                    continue
                selected[right_pos] = _score(name_norm, right_names[right_pos], address_norm, right_addrs[right_pos])
            if len(selected) >= candidate_budget:
                break

        if not selected:
            continue

        ranked = sorted(selected.items(), key=lambda item: -item[1])[:top_k]
        for rank, (right_pos, score) in enumerate(ranked, start=1):
            records.append(
                {
                    "source1_entity_id": entity_id,
                    "candidate_entity_id": str(right_ids[right_pos]),
                    f"{blocker_name}_score": float(score) / 100.0,
                    f"{blocker_name}_rank": rank,
                    "blocker": blocker_name,
                }
            )
    return records


class CountryIndex:
    """A pre-built blocking index for one country's target pool, reusable
    across many batches of left (S1) rows without rebuilding it each time.

    Building this once per country (instead of once per batch, or once per
    the whole left side as the old code did) is what makes batched,
    memory-bounded processing cheap: the index build is O(target pool size)
    and is the only step that needs the full target pool's text in memory
    at once -- everything downstream (per-batch scoring) only touches
    small slices.
    """

    __slots__ = ("index", "freq", "right_ids", "right_names", "right_addrs")

    def __init__(self, right_country: pd.DataFrame) -> None:
        right_country = right_country.reset_index(drop=True)
        self.right_ids = right_country["entity_id"].to_numpy()
        self.right_names = right_country["name_norm"].to_numpy()
        self.right_addrs = right_country["address_norm"].to_numpy()
        self.index, self.freq = _build_right_index(right_country)


def build_country_index(right_country: pd.DataFrame) -> CountryIndex:
    """Build a reusable blocking index for a single country's target pool.

    ``right_country`` must already be filtered to one country (the caller
    processes countries separately to bound memory to one country's target
    pool at a time, not the whole multi-country corpus).
    """
    return CountryIndex(right_country)


def score_left_batch(
    left_batch: pd.DataFrame,
    country_index: CountryIndex,
    top_k: int = 60,
    candidate_budget: int = 150,
    blocker_name: str = "keyed",
) -> pd.DataFrame:
    """Score one batch of left (S1) rows against an already-built CountryIndex.

    This is the memory-bounded entry point: peak memory scales with
    ``len(left_batch)``, not with the country's total S1 count or target
    pool size (those were already paid for once, in ``country_index``).
    """
    rows = [
        (row.entity_id, row.name_norm, row.address_norm, row.address_numbers)
        for row in left_batch.itertuples(index=False)
    ]
    records = _score_rows(
        rows, country_index.index, country_index.freq, country_index.right_ids, country_index.right_names, country_index.right_addrs,
        top_k, candidate_budget, blocker_name,
    )
    return pd.DataFrame.from_records(records)


def build_keyed_candidates(
    left: pd.DataFrame,
    right: pd.DataFrame,
    config: BlockingConfig,
    top_k: int = 60,
    candidate_budget: int = 150,
    blocker_name: str = "keyed",
) -> pd.DataFrame:
    """Fast blocking: cheap deterministic bucket keys, then RapidFuzz scoring
    only within each S1 row's (small) candidate set -- not against the full
    per-country corpus. Grouped by country like the other blockers.

    Convenience wrapper for small/moderate data (train-time validation,
    where the whole left side comfortably fits in memory at once). For
    large-scale test inference, build a CountryIndex once per country via
    ``build_country_index`` and call ``score_left_batch`` per batch instead
    -- see scripts/make_final_submission.py.
    """
    all_records: list[dict[str, object]] = []

    for country, left_country in left.groupby("country_norm", sort=False):
        right_country = right[right["country_norm"].eq(country)]
        if left_country.empty or right_country.empty:
            continue

        country_index = build_country_index(right_country)
        scored = score_left_batch(left_country, country_index, top_k, candidate_budget, blocker_name)
        if not scored.empty:
            all_records.extend(scored.to_dict("records"))

    return pd.DataFrame.from_records(all_records)


def build_keyed_union_candidates(
    source1: pd.DataFrame,
    targets: pd.DataFrame,
    config: BlockingConfig,
    top_k: int = 60,
    candidate_budget: int = 150,
) -> pd.DataFrame:
    """Drop-in replacement for blocking.build_union_candidates.

    Scales to the competition's real multi-million-row target pools (the
    full TF-IDF/token union does not -- see benchmarks in the diagnosis).
    Used for both train-time candidate generation and final test inference
    so the classifier is trained on the same candidate distribution it will
    see at inference time.
    """
    candidates = build_keyed_candidates(source1, targets, config, top_k=top_k, candidate_budget=candidate_budget)
    return combine_candidate_frames([candidates], config)
