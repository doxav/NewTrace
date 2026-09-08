import math
import random

def propose(history, bounds, seed):
    rng = random.Random(seed + len(history))
    d = len(bounds)
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    ranges = [highs[i] - lows[i] for i in range(d)]

    if not history:
        return [rng.uniform(lows[i], highs[i]) for i in range(d)]

    # Normalize history to [0,1]^d
    X = []
    y = []
    for h in history:
        x = h['x']
        X.append([(x[i] - lows[i]) / ranges[i] for i in range(d)])
        y.append(h['value'])

    n = len(X)
    if n == 0:
        return [rng.uniform(lows[i], highs[i]) for i in range(d)]

    # Hyperparameters
    if n > 1:
        y_mean = sum(y) / n
        y_var = sum((v - y_mean) ** 2 for v in y) / n
        sigma_f = math.sqrt(y_var) if y_var > 0 else 1.0
    else:
        sigma_f = 1.0
    noise = 1e-6
    prior_mean = sum(y) / n

    def kernel(x1, x2):
        dist2 = sum((x1[i] - x2[i]) ** 2 for i in range(d))
        return sigma_f ** 2 * math.exp(-0.5 * dist2)

    # Build and invert covariance matrix
    def invert_matrix(A):
        m = len(A)
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(m)]
               for i, row in enumerate(A)]
        for col in range(m):
            pivot = max(range(col, m), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot][col]) < 1e-12:
                raise ValueError("Singular")
            if pivot != col:
                aug[col], aug[pivot] = aug[pivot], aug[col]
            pv = aug[col][col]
            for j in range(col, 2 * m):
                aug[col][j] /= pv
            for r in range(m):
                if r != col:
                    factor = aug[r][col]
                    if factor != 0:
                        for j in range(col, 2 * m):
                            aug[r][j] -= factor * aug[col][j]
        return [row[m:] for row in aug]

    K = [[kernel(X[i], X[j]) + (noise if i == j else 0.0)
          for j in range(n)] for i in range(n)]

    try:
        K_inv = invert_matrix(K)
    except ValueError:
        noise = 1e-3
        K = [[kernel(X[i], X[j]) + (noise if i == j else 0.0)
              for j in range(n)] for i in range(n)]
        K_inv = invert_matrix(K)

    y_centered = [v - prior_mean for v in y]
    alpha = [sum(K_inv[i][j] * y_centered[j] for j in range(n))
             for i in range(n)]

    best_y = min(y)
    best_idx = y.index(best_y)
    best_x_norm = X[best_idx]

    # Generate candidate points
    candidates = []
    for _ in range(100):
        candidates.append([rng.uniform(0, 1) for _ in range(d)])
    for _ in range(20):
        cand = [best_x_norm[i] + rng.gauss(0, 0.1) for i in range(d)]
        cand = [min(1.0, max(0.0, c)) for c in cand]
        candidates.append(cand)

    best_ei = -1.0
    best_cand = None

    for cand in candidates:
        k_vec = [kernel(X[i], cand) for i in range(n)]
        mean = prior_mean + sum(k_vec[i] * alpha[i] for i in range(n))
        k_cand = kernel(cand, cand)
        K_inv_k = [sum(K_inv[i][j] * k_vec[j] for j in range(n))
                   for i in range(n)]
        var = k_cand - sum(k_vec[i] * K_inv_k[i] for i in range(n))
        if var < 0:
            var = 0.0
        sigma = math.sqrt(var)
        if sigma < 1e-12:
            ei = 0.0
        else:
            z = (best_y - mean) / sigma
            phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
            Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
            ei = sigma * (z * Phi + phi)
        if ei > best_ei:
            best_ei = ei
            best_cand = cand

    if best_cand is None:
        best_cand = [rng.uniform(0, 1) for _ in range(d)]

    result = [lows[i] + best_cand[i] * ranges[i] for i in range(d)]
    result = [min(highs[i], max(lows[i], result[i])) for i in range(d)]
    return result
