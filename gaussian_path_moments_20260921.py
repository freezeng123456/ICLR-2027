import numpy as np


def path_moment(parameters, grid, power):
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or not (np.diff(grid) < 0).all() or not np.isfinite(grid).all():
        raise ValueError("grid must be finite and strictly decreasing")
    if (grid < 0).any() or not np.isfinite(power) or power <= 0:
        raise ValueError("invalid grid or moment power")
    variance, means = parameters["variance"], parameters["means"]
    if not np.array_equal(means[..., 0], means[..., 1]):
        raise ValueError("This exact path integral requires Gaussian factors")
    size = len(grid)
    rows = []
    for d in range(variance.shape[1]):
        mean = np.zeros(size)
        loading = np.zeros((size, size))
        loading[0, 0] = 1
        quadratic, linear = np.zeros(size), np.zeros(size)
        constant = 0.0
        for k, (u, next_u) in enumerate(zip(grid[:-1], grid[1:])):
            h = u - next_u
            v = 1 - np.exp(-u) + np.exp(-u) * variance[:, d]
            a, b = 1 - 1 / v, np.exp(-u / 2) * means[:, d, 0] / v
            aa, bb = a.sum(), b.sum()
            quadratic[k] = h * (aa ** 2 - np.square(a).sum()) / 2
            linear[k] = h * (aa * bb - (a * b).sum())
            constant += h * (bb ** 2 - np.square(b).sum()) / 2
            multiplier = 1 - h / 2 + h * aa
            mean[k + 1] = multiplier * mean[k] + h * bb
            loading[k + 1] = multiplier * loading[k]
            loading[k + 1, k + 1] = np.sqrt(h)
        matrix = np.eye(size) - 2 * power * loading.T @ (quadratic[:, None] * loading)
        minimum = float(np.linalg.eigvalsh(matrix).min())
        log_moment = None
        if minimum > 0:
            vector = power * loading.T @ (2 * quadratic * mean + linear)
            sign, logdet = np.linalg.slogdet(matrix)
            assert sign == 1
            log_moment = float(power * (constant + np.dot(quadratic, mean ** 2) + np.dot(linear, mean))
                               - logdet / 2 + np.dot(vector, np.linalg.solve(matrix, vector)) / 2)
        rows.append(dict(coordinate=d, minimum_eigenvalue=minimum, finite=minimum > 0, log_moment=log_moment))
    finite = all(row["finite"] for row in rows)
    return dict(power=power, finite=finite, coordinates=rows,
                log_moment=sum(row["log_moment"] for row in rows) if finite else None)
