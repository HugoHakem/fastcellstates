"""
The fast path for the Dirichlet-multinomial model (``DirichletMultinomial``):
numba (nopython) kernels for the subset marginal log-likelihood (supp. info
eq. 15), the O(G_cell) incremental deltas for moving one cell or merging two
subsets, the biased MCMC move set, and the agglomerative merge.

Checked against ``_dm_reference`` (pure numpy) in ``test/test_kernels.py``.
Driven through ``core.Cluster``; ``moves.py`` calls only the sweeps.

The kernels take three namedtuples instead of a dozen loose arrays:

  Prior  : the Dirichlet prior + its lgamma cache (rebuilt by set_dirichlet_*)
  Cells  : the per-cell CSC of raw counts (columns = cells); immutable
  State  : the mutable partition: which box each cell is in + per-box aggregates

``n_boxes`` is ``state.sizes.shape[0]``.  Numba mutates array *fields* of a
namedtuple in place (``state.labels[m] = c``); resizing a field means the
caller rebuilds the tuple (``state._replace(...)``).

Layout: ``state.gene_counts`` is ``(n_boxes, G)``, a subset's counts one
contiguous row, so every hot loop indexes box-then-gene.  The public
``Cluster.cluster_umi_counts`` returns the ``(G, n_boxes)`` transpose (a view).

Notes (value-preserving vs. the original Cython): numba's PRNG (partitions
match statistically, not bit-for-bit); serial reductions except the ``prange``
batch sweep; Ctrl-C handled by chunking the MCMC loop in Python.
"""
import math
import sys
from collections import namedtuple

import numpy as np
from numba import njit, prange
from scipy.optimize import brentq
from scipy.special import gammaln

# log(DBL_MIN): below this an exp() underflows to 0, so the move is rejected.
MIN_EXP_ARG = math.log(sys.float_info.min)

Prior = namedtuple(
    "Prior", ["pseudocounts", "theta", "B", "cache_flat", "cache_offset", "cache_depth", "n_cache"]
)
Cells = namedtuple("Cells", ["ptr", "gidx", "gval", "umi_sum"])
State = namedtuple(
    "State", ["labels", "sizes", "loglik", "gene_counts", "box_umi_sum", "move_knn"]
)


# --------------------------------------------------------------------------- #
# Prior construction (pure Python builders)
# --------------------------------------------------------------------------- #

def build_prior(pseudocounts, gene_totals, n_cache):
    """(G,) Dirichlet parameter vector + (G,) per-gene dataset-wide UMI totals +
    target average cache depth -> Prior.

    The lgamma cache is *ragged*, not a flat (G, n_cache) table: the budget is
    ``G * n_cache`` entries, the same footprint a flat table would cost, but
    ``_water_fill_depth`` splits it per gene instead of handing out n_cache to
    everyone.  A gene's depth is capped at ``gene_totals[g] + 1``: a cluster's
    summed count for a gene can never exceed that gene's dataset-wide total (a
    cluster is a subset of cells), so this bound never costs a miss the flat
    table would have avoided.  In practice most genes are low-expression and get
    full, unbounded coverage "for free"; the budget that frees up goes to the
    high-expression tail the flat table under-served.  Never worse than the flat
    table at the same n_cache; see docs/changes.md.
    """
    pc = np.ascontiguousarray(pseudocounts, dtype=np.float64)
    gt = np.ascontiguousarray(gene_totals, dtype=np.float64)
    depth = _water_fill_depth(gt, pc.shape[0] * float(n_cache))
    cache_flat, cache_offset = _init_lgamma_cache_ragged(pc, depth)
    return Prior(pc, float(pc.sum()), _dirichlet_norm(pc), cache_flat, cache_offset, depth, int(n_cache))


def _water_fill_depth(gene_totals, budget_entries):
    """Per-gene depth ``min(gene_totals[g] + 1, tau)``, tau chosen so the total is
    <= budget_entries: the allocation that spends a fixed memory budget with
    the least truncation.  ``mem(tau) = sum(min(full, tau))`` is continuous and
    monotone non-decreasing in tau, so tau is a plain root-find, not something to
    hand-roll: ``scipy.optimize.brentq`` on ``mem(tau) - budget_entries``.
    """
    G = gene_totals.shape[0]
    if G == 0:
        return np.zeros(0, dtype=np.int64)
    full = gene_totals + 1.0
    if full.sum() <= budget_entries:
        depth = full  # the whole dataset's cache already fits the budget, uncapped
    else:
        tau = brentq(lambda t: np.sum(np.minimum(full, t)) - budget_entries, 0.0, float(full.max()))
        depth = np.minimum(full, tau)
    return np.maximum(depth, 1.0).astype(np.int64)


def _init_lgamma_cache_ragged(pseudocounts, depth):
    """cache_flat[offset[g] + i] = lgamma(pseudocounts[g] + i) for i = 0..depth[g]-1,
    one gene's row after another.  Same recurrence as before, per gene
    (lgamma(z+1) = lgamma(z) + log(z)), so any entry covered at the same depth by
    both a flat and a ragged table is bit-identical; only entries whose coverage
    status changes get a (numerically negligible, ~1e-9 absolute) different
    value.  Rebuilt on every set_dirichlet_pseudocounts call.
    """
    G = pseudocounts.shape[0]
    offset = np.zeros(G, dtype=np.int64)
    if G > 1:
        offset[1:] = np.cumsum(depth)[:-1]
    total = int(offset[-1] + depth[-1]) if G > 0 else 0
    cache_flat = np.empty(total, dtype=np.float64)
    base = gammaln(pseudocounts)
    for g in range(G):
        o, dep = int(offset[g]), int(depth[g])
        cache_flat[o] = base[g]
        if dep > 1:
            steps = np.log(pseudocounts[g] + np.arange(dep - 1))
            cache_flat[o + 1 : o + dep] = base[g] + np.cumsum(steps)
    return cache_flat, offset


@njit(cache=True)
def _dirichlet_norm(pseudocounts):
    """B = lgamma(sum lambda) - sum lgamma(lambda[g])."""
    thesum = 0.0
    B = 0.0
    for i in range(pseudocounts.shape[0]):
        thesum += pseudocounts[i]
        B -= math.lgamma(pseudocounts[i])
    B += math.lgamma(thesum)
    return B


@njit(cache=True, inline="always")
def _lg(prior, g, n):
    """lgamma(pseudocounts[g] + n): served from the per-gene ragged cache when n
    is within its water-filled depth, else computed directly."""
    if n < prior.cache_depth[g]:
        return prior.cache_flat[prior.cache_offset[g] + n]
    return math.lgamma(prior.pseudocounts[g] + n)


# --------------------------------------------------------------------------- #
# Cells construction
# --------------------------------------------------------------------------- #

def build_cells(counts):
    """(G, N) dense array or scipy sparse -> Cells (per-cell CSC + UMI totals)."""
    from scipy.sparse import csc_matrix, issparse
    m = counts if issparse(counts) else csc_matrix(np.ascontiguousarray(counts))
    m = m.tocsc()
    m.eliminate_zeros()
    return Cells(
        m.indptr.astype(np.int64),
        m.indices.astype(np.int32),
        m.data.astype(np.int32),
        np.asarray(m.sum(axis=0)).ravel().astype(np.int64),
    )


# --------------------------------------------------------------------------- #
# per-box aggregates / per-box likelihood
# --------------------------------------------------------------------------- #

@njit(cache=True)
def init_counts(cells, labels, n_boxes, G):
    """-> (sizes, gene_counts (n_boxes, G), box_umi_sum) for the given labels."""
    sizes = np.zeros(n_boxes, dtype=np.int32)
    gene_counts = np.zeros((n_boxes, G), dtype=np.int32)
    box_umi_sum = np.zeros(n_boxes, dtype=np.int64)
    for m in range(labels.shape[0]):
        c = labels[m]
        sizes[c] += 1
        for k in range(cells.ptr[m], cells.ptr[m + 1]):
            gene_counts[c, cells.gidx[k]] += cells.gval[k]
        box_umi_sum[c] += cells.umi_sum[m]
    return sizes, gene_counts, box_umi_sum


@njit(cache=True)
def cluster_LL(c, state, prior):
    if state.sizes[c] <= 0:
        return 0.0
    result = prior.B - math.lgamma(state.box_umi_sum[c] + prior.theta)
    for g in range(state.gene_counts.shape[1]):
        result += _lg(prior, g, state.gene_counts[c, g])
    return result


@njit(cache=True, parallel=True)
def init_likelihood(state, prior):
    # one O(G) reduction per box, all independent -> prange (bit-identical:
    # each lik[c] sums its own genes in its own order).  Called on every
    # refresh_likelihood(), i.e. once per sweep pass.
    n = state.sizes.shape[0]
    lik = np.zeros(n, dtype=np.float64)
    for c in prange(n):
        lik[c] = cluster_LL(c, state, prior)
    return lik


# --------------------------------------------------------------------------- #
# merging
# --------------------------------------------------------------------------- #

@njit(cache=True)
def find_cluster_distance(i, j, state, prior):
    """Change in total LL if clusters i and j were merged."""
    n = state.box_umi_sum[i] + state.box_umi_sum[j]
    delta = prior.B - math.lgamma(n + prior.theta)
    for g in range(state.gene_counts.shape[1]):
        nn = state.gene_counts[i, g] + state.gene_counts[j, g]
        delta += _lg(prior, g, nn)
    delta -= state.loglik[i] + state.loglik[j]
    return delta


@njit(cache=True)
def _merge_box_aggregates(c1, c2, delta_LL, state):
    """Fold box c2's per-cluster aggregates into c1; ``labels`` untouched."""
    state.sizes[c1] += state.sizes[c2]
    state.sizes[c2] = 0
    state.loglik[c1] += delta_LL + state.loglik[c2]
    state.loglik[c2] = 0.0
    for g in range(state.gene_counts.shape[1]):
        state.gene_counts[c1, g] += state.gene_counts[c2, g]
        state.gene_counts[c2, g] = 0
    state.box_umi_sum[c1] += state.box_umi_sum[c2]
    state.box_umi_sum[c2] = 0


@njit(cache=True)
def merge_two_clusters(c1, c2, delta_LL, state):
    """Merge cluster c2 into c1 in place, relabelling its cells (O(n cells))."""
    for i in range(state.labels.shape[0]):
        if state.labels[i] == c2:
            state.labels[i] = c1
    _merge_box_aggregates(c1, c2, delta_LL, state)


@njit(cache=True)
def _resolve_labels(labels, parent):
    """Rewrite each cell's label to its surviving root.  ``parent`` is a
    union-find forest where every link points to a *smaller* index (a merge
    always keeps the lower-indexed box), so one ascending pass flattens it in
    O(K) and the relabel is a flat O(n cells) lookup."""
    for c in range(parent.shape[0]):
        if parent[c] != c:
            parent[c] = parent[parent[c]]  # parent[c] < c -> already flattened
    for i in range(labels.shape[0]):
        labels[i] = parent[labels[i]]


@njit(cache=True)
def _count_clusters(state):
    n = 0
    for k in range(state.sizes.shape[0]):
        if state.sizes[k] > 0:
            n += 1
    return n


@njit(cache=True, inline="always")
def _row_argmax(delta_LL, cluster_exists, i):
    """max_{j<i, live} delta_LL[i, j]; ties -> smallest j (matches the serial
    ascending scan)."""
    b = -np.inf
    bj = -1
    for j in range(i):
        if cluster_exists[j] and delta_LL[i, j] > b:
            b = delta_LL[i, j]
            bj = j
    return b, bj


@njit(cache=True, parallel=True)
def merge_clusters_hierarchical(LL_threshold, n_cluster_threshold, state, prior):
    """Iteratively merge the most similar clusters.

    Returns (merge_hierarchy, delta_history) where merge_hierarchy[k] =
    (cluster_new, cluster_old) with cluster_new < cluster_old the surviving
    label.

    ``delta_LL`` is a lower-triangle matrix (``delta_LL[i, j]`` valid for
    ``i > j``) of the change in total LL from merging each live pair.  The two
    O(K^2 * G) parts, the initial fill and the per-merge refresh of the
    survivor's row/column, are ``prange``d: every entry is an independent pure
    function of the frozen state, so this is bit-identical to a serial fill.
    ``row_best`` / ``row_bestj`` cache each row's max so the per-step argmax is
    O(K) not O(K^2); it is rebuilt only for rows whose best partner was the
    survivor or the removed cluster.  Per merge only the O(G) aggregates are
    folded; the O(n cells) relabel is deferred to one final ``_resolve_labels``
    pass instead of being redone every step.  The greedy merge order is unchanged.
    """
    n_boxes = state.sizes.shape[0]
    cluster_exists = state.sizes > 0
    parent = np.arange(n_boxes)

    delta_LL = np.zeros((n_boxes, n_boxes), dtype=np.float64)
    for i in prange(n_boxes):
        if cluster_exists[i]:
            for j in range(i):
                if cluster_exists[j]:
                    delta_LL[i, j] = find_cluster_distance(i, j, state, prior)

    row_best = np.full(n_boxes, -np.inf)
    row_bestj = np.full(n_boxes, -1, dtype=np.int64)
    for i in prange(n_boxes):
        if cluster_exists[i]:
            row_best[i], row_bestj[i] = _row_argmax(delta_LL, cluster_exists, i)

    max_steps = _count_clusters(state) - n_cluster_threshold
    merge_hierarchy = np.empty((max_steps, 2), dtype=np.int64)
    delta_history = np.empty(max_steps, dtype=np.float64)
    n_done = 0

    for _ in range(max_steps):
        D_max = -np.inf
        i_max = -1
        for i in range(n_boxes):
            if cluster_exists[i] and row_best[i] > D_max:
                D_max = row_best[i]
                i_max = i

        if D_max <= LL_threshold or i_max < 0:
            break

        survivor = row_bestj[i_max]  # smaller index -> cluster_new
        removed = i_max              # larger index  -> cluster_old
        merge_hierarchy[n_done, 0] = survivor
        merge_hierarchy[n_done, 1] = removed
        delta_history[n_done] = D_max
        n_done += 1

        _merge_box_aggregates(survivor, removed, D_max, state)
        parent[removed] = survivor
        cluster_exists[removed] = False
        row_best[removed] = -np.inf

        # refresh every entry that pairs with the (merged) survivor: one entry
        # per row k, disjoint, so prange is race-free.  Entries pairing with
        # `removed` are dead (guarded by cluster_exists) and left as-is.
        s = survivor
        r = removed
        for k in prange(n_boxes):
            if cluster_exists[k] and k != s:
                if k < s:
                    delta_LL[s, k] = find_cluster_distance(s, k, state, prior)
                else:
                    delta_LL[k, s] = find_cluster_distance(k, s, state, prior)

        # patch row_best: row s in full; rows i > s only if the fresh
        # delta_LL[i, s] ties/overtakes, or their cached best was s (value
        # moved) or r (gone).  Rows i < s have no changed lower-triangle entry
        # (the {i,s} and {i,r} pairs live in rows s and r), so they are exact.
        row_best[s], row_bestj[s] = _row_argmax(delta_LL, cluster_exists, s)
        for i in range(s + 1, n_boxes):
            if not cluster_exists[i]:
                continue
            d = delta_LL[i, s]
            if d > row_best[i]:
                row_best[i] = d
                row_bestj[i] = s
            elif d == row_best[i] or row_bestj[i] == s or row_bestj[i] == r:
                row_best[i], row_bestj[i] = _row_argmax(delta_LL, cluster_exists, i)

    _resolve_labels(state.labels, parent)
    return merge_hierarchy[:n_done].copy(), delta_history[:n_done].copy()


# --------------------------------------------------------------------------- #
# per-move likelihood deltas (delta = LL_after - loglik[c])
# --------------------------------------------------------------------------- #

@njit(cache=True)
def delta_LL_new(m, c_new, cells, state, prior):
    N_c = state.box_umi_sum[c_new]
    n_m = cells.umi_sum[m]
    delta = math.lgamma(N_c + prior.theta) - math.lgamma(N_c + n_m + prior.theta)
    for k in range(cells.ptr[m], cells.ptr[m + 1]):
        g = cells.gidx[k]
        v = cells.gval[k]
        C_g = state.gene_counts[c_new, g]
        delta += _lg(prior, g, C_g + v) - _lg(prior, g, C_g)
    return delta


@njit(cache=True)
def delta_LL_old(m, c_old, cells, state, prior):
    if state.sizes[c_old] <= 1:
        # cluster becomes empty: LL_c_old = 0, so delta = -loglik[c_old]
        return -state.loglik[c_old]
    N_old = state.box_umi_sum[c_old]
    n_m = cells.umi_sum[m]
    delta = math.lgamma(N_old + prior.theta) - math.lgamma(N_old - n_m + prior.theta)
    for k in range(cells.ptr[m], cells.ptr[m + 1]):
        g = cells.gidx[k]
        v = cells.gval[k]
        C_g = state.gene_counts[c_old, g]
        delta += _lg(prior, g, C_g - v) - _lg(prior, g, C_g)
    return delta


@njit(cache=True)
def apply_move(m, c_new, d_old, d_new, cells, state):
    c_old = state.labels[m]
    if c_old == c_new:
        return
    state.sizes[c_old] -= 1
    state.labels[m] = c_new
    state.sizes[c_new] += 1
    state.loglik[c_old] += d_old          # == LL_c_old (0 if it just emptied)
    state.loglik[c_new] += d_new          # == LL_c_new
    n_m = cells.umi_sum[m]
    state.box_umi_sum[c_old] -= n_m
    state.box_umi_sum[c_new] += n_m
    for k in range(cells.ptr[m], cells.ptr[m + 1]):
        g = cells.gidx[k]
        v = cells.gval[k]
        state.gene_counts[c_old, g] -= v
        state.gene_counts[c_new, g] += v


# --------------------------------------------------------------------------- #
# MCMC
# --------------------------------------------------------------------------- #

@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def biased_mc_moves_chunk(chunk_max_tries, target_successes, min_index,
                          cells, state, prior):
    """The paper's from-singletons Metropolis-Hastings search, one chunk of
    tries: pick a random cell and a random destination box, accept/reject on
    the exact ΔLL (Metropolis).  ``move_bias`` / ``d_nclusters`` implement the
    detailed-balance correction a move that creates or destroys a cluster
    needs (the split/merge proposal isn't symmetric in the number of boxes).
    Runs until ``target_successes`` accepted moves or ``chunk_max_tries``,
    chunked so the caller (``Cluster.biased_monte_carlo_sampling``) can refresh
    the likelihood and check for Ctrl-C between chunks.  Returns (tries,
    successes)."""
    N = state.labels.shape[0]
    n_boxes = state.sizes.shape[0]
    n_rand_max = N - min_index
    tries = 0
    successes = 0
    raw_iter = 0
    raw_cap = 100 * chunk_max_tries + 100_000
    n_clusters = _count_clusters(state)

    while tries < chunk_max_tries and successes < target_successes:
        raw_iter += 1
        if raw_iter > raw_cap:
            break
        m = np.random.randint(0, n_rand_max) + min_index
        c_new = np.random.randint(0, n_boxes)
        c_old = state.labels[m]

        if c_old == c_new:
            continue
        elif state.sizes[c_new] == 0:
            if state.sizes[c_old] == 1:
                continue
            else:
                move_bias = -math.log(n_boxes - n_clusters)
                d_nclusters = 1
        elif state.sizes[c_old] == 1 and state.sizes[c_new] != 0:
            move_bias = math.log(n_boxes - n_clusters + 1.0)
            d_nclusters = -1
        else:
            move_bias = 0.0
            d_nclusters = 0

        tries += 1

        d_new = delta_LL_new(m, c_new, cells, state, prior)
        d_old = delta_LL_old(m, c_old, cells, state, prior)
        move_likelihood = move_bias + d_old + d_new

        if move_likelihood < MIN_EXP_ARG:
            continue
        elif move_likelihood < 0 and np.random.random() > math.exp(move_likelihood):
            continue

        apply_move(m, c_new, d_old, d_new, cells, state)
        n_clusters += d_nclusters
        successes += 1

    return tries, successes


# --------------------------------------------------------------------------- #
# deterministic cell-position optimisation
# --------------------------------------------------------------------------- #

@njit(cache=True)
def best_move_full(m, cells, state, prior):
    c_old = state.labels[m]
    c_best = c_old
    best_delta = -np.inf
    d_old = delta_LL_old(m, c_old, cells, state, prior)
    for c_new in range(state.sizes.shape[0]):
        if c_new == c_old:
            continue
        delta_LL = d_old + delta_LL_new(m, c_new, cells, state, prior)
        if delta_LL > best_delta:
            best_delta = delta_LL
            c_best = c_new
    return c_best, best_delta


@njit(cache=True)
def best_move_knn(m, cells, state, prior):
    """``best_move_full`` with the candidate set restricted to
    ``{clusters of cell m's k nearest neighbours} + {one empty box}``.

    Same exact ΔLL test.  The only move it can miss vs. the full scan is a jump
    to a non-neighbour *occupied* cluster (~0.1 % of accepted sweep moves /
    ~0.2 % of ΔLL at k=100 on a warm-started partition).  The empty box keeps the
    split move available (dropped only if there is no free box).  Neighbour
    clusters are deduped.
    """
    knn_row = state.move_knn[m]
    c_old = state.labels[m]
    c_best = c_old
    best_delta = -np.inf
    d_old = delta_LL_old(m, c_old, cells, state, prior)
    k = knn_row.shape[0]
    for r in range(k):
        c_new = state.labels[knn_row[r]]
        if c_new == c_old:
            continue
        seen = False
        for r2 in range(r):
            if state.labels[knn_row[r2]] == c_new:
                seen = True
                break
        if seen:
            continue
        delta_LL = d_old + delta_LL_new(m, c_new, cells, state, prior)
        if delta_LL > best_delta:
            best_delta = delta_LL
            c_best = c_new

    # split: one empty box (all equivalent: delta depends only on m + prior)
    for c_new in range(state.sizes.shape[0]):
        if state.sizes[c_new] == 0:
            delta_LL = d_old + delta_LL_new(m, c_new, cells, state, prior)
            if delta_LL > best_delta:
                best_delta = delta_LL
                c_best = c_new
            break
    return c_best, best_delta


@njit(cache=True, inline="always")
def _best_move(m, cells, state, prior):
    """The k-NN-pruned scan if ``state.move_knn`` has columns, else a full scan."""
    if state.move_knn.shape[1] > 0:
        return best_move_knn(m, cells, state, prior)
    return best_move_full(m, cells, state, prior)


@njit(cache=True)
def optimize_cells(cell_iter, cells, state, prior):
    for idx in range(cell_iter.shape[0]):
        m = cell_iter[idx]
        c_best, delta_LL = _best_move(m, cells, state, prior)
        if delta_LL > 0.0:
            c_old = state.labels[m]
            d_old = delta_LL_old(m, c_old, cells, state, prior)
            d_new = delta_LL_new(m, c_best, cells, state, prior)
            apply_move(m, c_best, d_old, d_new, cells, state)


@njit(cache=True, parallel=True)
def batch_sweep(cells, state, prior, recheck, prop_c, prop_delta):
    """Jacobi / batch ICM: one pass = score EVERY cell's best move against the
    frozen state (phase 1, parallel), then apply the improving ones ΔLL-
    descending (phase 2, serial).

    recheck: 0 = apply on the frozen ΔLL (fastest, can dip within a pass)
             1 = recompute ΔLL for the proposed destination against the live
                 state, skip if no longer > 0  (O(support) per applied)
             2 = full re-scan of the best destination against the live state
                 (every applied move is the true greedy move at apply time)

    prop_c / prop_delta: caller-owned scratch, length N (= n cells).  Returns
    the number of moves applied.
    """
    N = state.labels.shape[0]
    # ---- phase 1: independent, read-only -> parallel ----
    for m in prange(N):
        c_best, delta = _best_move(m, cells, state, prior)
        prop_c[m] = c_best
        prop_delta[m] = delta

    # ---- phase 2: apply, ΔLL-descending, serial ----
    order = np.argsort(-prop_delta)
    applied = 0
    for oi in range(N):
        m = order[oi]
        if prop_delta[m] <= 0.0:
            break
        c_old = state.labels[m]
        c_new = prop_c[m]
        if c_new == c_old:
            continue
        if recheck == 2:
            c_new, d = _best_move(m, cells, state, prior)
            if d <= 0.0 or c_new == c_old:
                continue
        d_old = delta_LL_old(m, c_old, cells, state, prior)
        d_new = delta_LL_new(m, c_new, cells, state, prior)
        if recheck == 1 and d_old + d_new <= 0.0:
            continue
        apply_move(m, c_new, d_old, d_new, cells, state)
        applied += 1
    return applied
