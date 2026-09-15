"""
numba (nopython) kernels for a faithful port of Breda et al.'s Sanity
(github.com/jmbreda/Sanity, Nat. Biotechnol. 2021): per-gene empirical-Bayes
correction of Poisson sampling noise in UMI counts, and the accompanying
uncertainty-aware cell-cell distance.  Ported line-for-line from the
reference C++ (``src/calc_true_variation_parallel_prior_mu_sigma.cpp``,
``src/FitFrac.cpp``, ``src/Digamma_Trigamma.cpp``,
``src/compute_distance.cpp`` at commit c65d780), including its numerical
quirks (the asymptotic digamma/trigamma series, the large-x Lambert-W
branch), so this should reproduce the reference tool's results, not just its
model.

Model: for gene g, cell c, ``n_gc ~ Poisson(N_c * exp(mu_g + delta_gc))``,
``delta_gc ~ Normal(0, v_g)``.  Per gene: a 1-D grid search over the prior
variance ``v_g`` (log-spaced, ``vmin``..``vmax``, ``numbin`` points); at each
grid point, ``mu_g`` and every ``delta_gc`` are fit *jointly* (not one fixed
then the other) by finding the root of a strictly-monotone concave function
of a single scalar ``q`` via Newton's method, where each term of that
function is a Lambert-W evaluation (``fitfrac``, called ``FitFrac.cpp``
there).  The grid gives a (numerically-integrated) posterior over ``v_g``,
collapsed to a point via one of four estimators (``v_method``); the default
(2, MAP) is Sanity's own default, with an implicit ``1/v`` (Jeffreys) prior.
This is the *only* place a prior on ``v_g`` enters.

Everything here operates on one gene (or one cell pair) at a time inside a
``prange`` loop over genes (or cells): this is not vectorizable across genes
as clean array ops the way ``graph._pca_knn``'s preprocessing is, because
each gene's Newton iteration count and grid-search work are data-dependent
-- the same reason the reference implementation is a straight nested-loop
C++ program, not a matrix library call.
"""

import math

import numpy as np
from numba import njit, prange

# ------------------------------------------------------------------ #
# Lambert W0 (principal branch, real, z >= 0 only -- the only branch this
# model ever needs).  numba has no built-in lambertw, so this is a Halley
# (cubic) iteration from a robust starting guess; verified against
# scipy.special.lambertw elsewhere (see test).  For the large-argument case
# (x = log z > 50) the reference code switches to a closed-form asymptotic
# expansion evaluated directly in x, both to stay accurate and to avoid ever
# computing exp(x) for x that could overflow float64 -- ported verbatim
# below as ``_lambertw0_logarg``.
# ------------------------------------------------------------------ #


@njit(cache=True, inline="always")
def _lambertw0(z):
    """W0(z) for real z >= 0, by Halley's iteration."""
    if z == 0.0:
        return 0.0
    if z < 1.0:
        w = z * (1.0 - z + 1.5 * z * z)
    else:
        lz = math.log(z)
        if lz <= 1.0:
            w = z / (1.0 + z)
        else:
            llz = math.log(lz)
            w = lz - llz + llz / lz
    for _ in range(100):
        ew = math.exp(w)
        f = w * ew - z
        wp1 = w + 1.0
        denom = ew * wp1 - (w + 2.0) * f / (2.0 * wp1)
        if denom == 0.0:
            break
        dw = f / denom
        w -= dw
        if abs(dw) <= 1e-13 * (1.0 + abs(w)):
            break
    return w


@njit(cache=True, inline="always")
def _lambertw0_logarg(x):
    """W0(exp(x)) for real x, any sign -- the quantity this model always
    needs (never a bare W0(z)).  x > 50 uses the closed-form asymptotic
    expansion in x (Sanity's ``LambertW0_approximation``, itself the
    standard large-z expansion of W0 written in terms of L1 = log(z) = x),
    so exp(x) is never evaluated once it would risk overflow."""
    if x > 50.0:
        L1 = x
        L2 = math.log(L1)
        return (
            L1
            - L2
            + L2 / L1
            + L2 * (L2 - 2.0) / (2.0 * L1 * L1)
            + L2 * (6.0 - 9.0 * L2 + 2.0 * L2 * L2) / (6.0 * L1**3)
            + L2 * (-12.0 + 36.0 * L2 - 22.0 * L2 * L2 + 3.0 * L2**3) / (12.0 * L1**4)
        )
    return _lambertw0(math.exp(x))


# ------------------------------------------------------------------ #
# digamma / trigamma: Sanity's own fixed-order asymptotic series (exact
# port, including the x==1 special case), NOT scipy's exact implementation.
# This is a deliberate fidelity choice, not an accuracy one: the series is
# markedly less accurate for small integer gene totals (n of a few counts,
# common for lowly-expressed genes) than scipy.special.digamma/polygamma,
# but matching the reference tool's own numerics -- quirks included -- is
# the point of this module.
# ------------------------------------------------------------------ #


@njit(cache=True, inline="always")
def _digamma(x):
    if x == 1.0:
        return -0.577215664901532
    return (
        math.log(x)
        - 1.0 / (2.0 * x)
        - 1.0 / (12.0 * x**2)
        + 1.0 / (120.0 * x**4)
        - 1.0 / (252.0 * x**6)
        + 1.0 / (240.0 * x**8)
        - 5.0 / (660.0 * x**10)
        + 691.0 / (32760.0 * x**12)
        - 1.0 / (12.0 * x**14)
    )


@njit(cache=True, inline="always")
def _trigamma(x):
    if x == 1.0:
        return 1.644934066848226
    return (
        1.0 / x
        + 1.0 / (2.0 * x**2)
        + 1.0 / (6.0 * x**3)
        - 1.0 / (30.0 * x**5)
        + 1.0 / (42.0 * x**7)
        - 1.0 / (30.0 * x**9)
        + 5.0 / (66.0 * x**11)
        - 691.0 / (2730.0 * x**13)
        + 7.0 / (6.0 * x**15)
    )


# ------------------------------------------------------------------ #
# fitfrac: jointly fits (mu, every delta_c) at a fixed prior variance v, by
# finding the scalar root q of fq(.) = 0.  f[i] = exp(mu + delta_i) is
# recovered from q via Lambert W; by construction of the root (fq(q) = 0
# means beta = sum_i W_i, and f_i = W_i / beta), sum(f) == 1 exactly at
# convergence -- the correctness check used in this module's tests.
# ------------------------------------------------------------------ #


@njit(cache=True, inline="always")
def _fq(Q, beta, q):
    val = beta
    for i in range(Q.shape[0]):
        val -= _lambertw0_logarg(Q[i] - q)
    return val


@njit(cache=True, inline="always")
def _deltaq(Q, beta, q):
    num = beta
    denom = 0.0
    for i in range(Q.shape[0]):
        W = _lambertw0_logarg(Q[i] - q)
        num -= W
        denom += W / (1.0 + W)
    return num / denom


@njit(cache=True)
def _fitfrac(n_c, N_c, n, v, prev_q, f):
    """Solve for q (and fill f[i] = exp(mu + delta_i) in place) at fixed v.
    ``prev_q`` warm-starts from the previous (adjacent) grid point's q, or
    pass 0.0 for a cold start."""
    C = n_c.shape[0]
    beta = n * v
    inv_beta = 1.0 / beta
    logbeta = math.log(beta)
    Q = np.empty(C)
    qsum = 0.0
    for i in range(C):
        Q[i] = math.log(N_c[i]) + n_c[i] * v + logbeta
        qsum += N_c[i]
    q_init = math.log(qsum) + 0.5 * v

    q = q_init
    need_bracket = True
    if prev_q != 0.0:
        fq_prev = _fq(Q, beta, prev_q)
        if fq_prev > 0.0:
            dq0 = _deltaq(Q, beta, prev_q)
            cand = prev_q - dq0
            if _fq(Q, beta, cand) > 0.0:
                need_bracket = True
            else:
                q = cand
                need_bracket = False
        else:
            q = prev_q
            need_bracket = False

    if need_bracket:
        fqq = _fq(Q, beta, q)
        it = 0
        while fqq > 0.0 and it < 10000:
            q -= 0.5 * v
            fqq = _fq(Q, beta, q)
            it += 1

    dq = -1.0
    tol = 1e-7
    it = 0
    while abs(dq) > tol and dq < 0.0 and it < 200:
        dq = _deltaq(Q, beta, q)
        q = q - dq
        it += 1

    for i in range(C):
        f[i] = _lambertw0_logarg(Q[i] - q) * inv_beta

    return q


@njit(cache=True, inline="always")
def _get_epsilon2(d, v, n, f):
    """Posterior variance of delta for a zero-count cell: the Laplace
    (Gaussian) approximation is a poor fit to this heavily-skewed case, so
    Sanity instead finds, by bisection, the epsilon with
    ``KL(true || Laplace-at-epsilon) == 0.5`` (``get_epsilon_2``)."""
    vnf = v * n * f
    e_high = (-(d + vnf) + math.sqrt((d + vnf) ** 2 + v * (1.0 + vnf))) / (1.0 + vnf)
    e_low = 0.0
    tol = 1e-7
    diff = 1.0
    e = 0.0
    it = 0
    while diff > tol and it < 200:
        e = (e_high + e_low) / 2.0
        dL = e * (2.0 * d + e) / (2.0 * v) + n * f * (math.exp(e) - 1.0)
        if dL < 0.5:
            e_low = e
        else:
            e_high = e
        diff = abs(dL - 0.5)
        it += 1
    return e * e


# ------------------------------------------------------------------ #
# Per-gene fit: the v-grid search, one gene at a time.
# ------------------------------------------------------------------ #


@njit(cache=True)
def _fit_gene(n_c, N_c, v_grid, v_method):
    C = n_c.shape[0]
    numbin = v_grid.shape[0]
    n = 0.0
    for i in range(C):
        n += n_c[i]

    delta_v = np.empty((numbin, C))
    sig2_delta_v = np.empty((numbin, C))
    mu_v = np.empty(numbin)
    lik = np.empty(numbin)
    f = np.empty(C)
    sig2_delta_c = np.empty(C)
    sig2_delta_den2 = np.empty(C)

    Lmax = -1e300
    Lmax_ind = 0
    prev_q = 0.0

    for k in range(numbin):
        v = v_grid[k]
        inv_beta_outer = 1.0 / (n * v)
        q = _fitfrac(n_c, N_c, n, v, prev_q, f)
        prev_q = q
        mu_v[k] = _digamma(n) - q

        delsq = 0.0
        L = -0.5 * C * math.log(v)
        for i in range(C):
            d_ki = math.log(f[i]) - math.log(N_c[i]) + q
            delta_v[k, i] = d_ki
            L += n_c[i] * d_ki
            delsq += d_ki * d_ki
        L -= delsq / (2.0 * v)
        L -= n * q

        ldet = 0.0
        for i in range(C):
            ldet += (f[i] * f[i]) / (f[i] + inv_beta_outer)
        ldet = math.log(1.0 - ldet)
        for i in range(C):
            ldet += math.log(f[i] + inv_beta_outer)
        L -= 0.5 * ldet
        lik[k] = L
        if L > Lmax:
            Lmax = L
            Lmax_ind = k

        inv_v = 1.0 / v
        den1 = 1.0
        for i in range(C):
            c_i = n * f[i] * f[i] / (n * f[i] + inv_v)
            sig2_delta_c[i] = c_i
            den1 -= c_i
            sig2_delta_den2[i] = n * f[i] + inv_v
        for i in range(C):
            num_i = den1 + sig2_delta_c[i]
            sig2_delta_v[k, i] = num_i / (den1 * sig2_delta_den2[i])
        for i in range(C):
            if n_c[i] <= 0.5:
                sig2_delta_v[k, i] = _get_epsilon2(delta_v[k, i], v, n, f[i])

    sum_L = 0.0
    for k in range(numbin):
        lik[k] = math.exp(lik[k] - Lmax)
        sum_L += lik[k]
    for k in range(numbin):
        lik[k] /= sum_L

    vindex = 0
    if v_method == 1:  # MLE
        vindex = Lmax_ind
    elif v_method == 2:  # MAP (implicit 1/v Jeffreys prior)
        mapmax = -1e300
        for k in range(numbin):
            val = lik[k] / v_grid[k]
            if val > mapmax:
                mapmax = val
                vindex = k
    elif v_method == 3:  # EAP
        postmean = 0.0
        for k in range(numbin):
            postmean += lik[k] * v_grid[k]
        mindist = 1e300
        for k in range(numbin):
            dist = abs(v_grid[k] - postmean)
            if dist < mindist:
                mindist = dist
                vindex = k
    # v_method == 0 (MARG): vindex unused, everything below marginalizes.

    mu = 0.0
    for k in range(numbin):
        mu += lik[k] * mu_v[k]
    var_mu = _trigamma(n)
    for k in range(numbin):
        var_mu += lik[k] * (mu_v[k] - mu) ** 2

    delta = np.empty(C)
    var_delta = np.empty(C)
    var_gene = 0.0

    if v_method == 0:
        for i in range(C):
            acc = 0.0
            for k in range(numbin):
                acc += lik[k] * delta_v[k, i]
            delta[i] = acc
        for i in range(C):
            acc = 0.0
            for k in range(numbin):
                diff = delta_v[k, i] - delta[i]
                acc += lik[k] * diff * diff + lik[k] * sig2_delta_v[k, i]
            var_delta[i] = acc
        for k in range(numbin):
            var_gene += v_grid[k] * lik[k]
    else:
        var_gene = v_grid[vindex]
        mu = mu_v[vindex]
        var_mu = _trigamma(n)
        for i in range(C):
            delta[i] = delta_v[vindex, i]
            var_delta[i] = sig2_delta_v[vindex, i]

    return mu, var_mu, delta, var_delta, var_gene


@njit(cache=True, parallel=True)
def fit_all_genes(X, N_c, v_grid, v_method):
    """X: (G, N) counts, every row with a nonzero total (filter beforehand).
    N_c: (N,) per-cell library sizes, all > 0.  v_grid: (numbin,) candidate
    prior variances.  v_method: 0=MARG 1=MLE 2=MAP(default) 3=EAP.

    Returns mu (G,), var_mu (G,), delta (G, N), var_delta (G, N),
    var_gene (G,)."""
    G, N = X.shape
    mu = np.empty(G)
    var_mu = np.empty(G)
    delta = np.empty((G, N))
    var_delta = np.empty((G, N))
    var_gene = np.empty(G)
    for g in prange(G):
        mu_g, var_mu_g, delta_g, var_delta_g, var_gene_g = _fit_gene(X[g], N_c, v_grid, v_method)
        mu[g] = mu_g
        var_mu[g] = var_mu_g
        delta[g] = delta_g
        var_delta[g] = var_delta_g
        var_gene[g] = var_gene_g
    return mu, var_mu, delta, var_delta, var_gene


# ------------------------------------------------------------------ #
# Sanity_distance: uncertainty-weighted pairwise cell distance
# (compute_distance.cpp's ``get_distance_errorbar`` / ``get_Di_errorbar``).
# O(N^2 * G_kept * N_BIN): a compiled nested loop in the reference tool for
# the same reason it is one here -- no clean array-op vectorization once
# it's a per-pair marginalization.
# ------------------------------------------------------------------ #

_N_BIN = 401
_DA = 0.005


@njit(cache=True, parallel=True)
def sanity_distance_kernel(delta, eps2, var_gene):
    """delta, eps2: (N, G) posterior mean / variance, already rescaled (see
    ``sanity.sanity_distance``).  var_gene: (G,) per-gene prior variance.
    Returns (N, N) distance matrix."""
    N, G = delta.shape
    alpha_v = np.empty((G, _N_BIN))
    for g in range(G):
        vg = var_gene[g]
        for k in range(_N_BIN):
            alpha_v[g, k] = (k * _DA) * vg

    D = np.zeros((N, N))
    for i in prange(N):
        x2 = np.empty(G)
        eps2_sum = np.empty(G)
        lik = np.empty(_N_BIN)
        for j in range(i + 1, N):
            for g in range(G):
                diff = delta[i, g] - delta[j, g]
                x2[g] = diff * diff
                eps2_sum[g] = eps2[i, g] + eps2[j, g]

            lik_max = -1e300
            for k in range(_N_BIN):
                s = 0.0
                for g in range(G):
                    q = eps2_sum[g] + alpha_v[g, k]
                    s -= 0.5 * x2[g] / q
                    s -= 0.5 * math.log(q)
                lik[k] = s
                if s > lik_max:
                    lik_max = s

            lik_tot = 0.0
            for k in range(_N_BIN):
                lik[k] = math.exp(lik[k] - lik_max)
                lik_tot += lik[k]

            d2 = 0.0
            for k in range(_N_BIN):
                w = lik[k] / lik_tot
                acc = 0.0
                for g in range(G):
                    q = alpha_v[g, k] + eps2_sum[g]
                    fq = alpha_v[g, k] / q
                    acc += fq * fq * x2[g] + fq * eps2_sum[g]
                d2 += w * acc

            dist = math.sqrt(d2)
            D[i, j] = dist
            D[j, i] = dist
    return D


@njit(cache=True, parallel=True)
def euclidean_distance_kernel(delta):
    """Plain Euclidean distance in (unweighted) gene-space, no PCA
    (``get_distance_euclidean``): the ``with_error_bar=False`` option."""
    N, G = delta.shape
    D = np.zeros((N, N))
    for i in prange(N):
        for j in range(i + 1, N):
            acc = 0.0
            for g in range(G):
                diff = delta[i, g] - delta[j, g]
                acc += diff * diff
            dist = math.sqrt(acc)
            D[i, j] = dist
            D[j, i] = dist
    return D
