import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm


GH_ORDER = 30
GH_NODES_1D, GH_WEIGHTS_1D = hermgauss(GH_ORDER)
GH_U, GH_V = np.meshgrid(GH_NODES_1D, GH_NODES_1D, indexing="ij")
GH_W = np.outer(GH_WEIGHTS_1D, GH_WEIGHTS_1D)


def make_spd_matrix(entries):
    matrix = np.array(entries, dtype=float).reshape(2, 2)
    return 0.5 * (matrix + matrix.T)


def parse_constraint_rows(raw_rows):
    constraints = []
    for row in raw_rows:
        a1, a2, nu = [float(token.strip()) for token in row.split(",")]
        constraints.append((np.array([a1, a2], dtype=float), nu))
    return constraints


def scale_constraints(constraints, epsilon_scale):
    scaled = []
    for direction, nu in constraints:
        scaled.append((direction.copy(), nu / epsilon_scale))
    return scaled


def potential(x, precision0, constraints):
    quad_term = 0.5 * np.einsum("...i,ij,...j->...", x, precision0, x)
    barrier = np.zeros_like(quad_term)
    for direction, nu in constraints:
        barrier -= norm.logcdf(np.tensordot(x, direction, axes=([-1], [0])) / nu)
    return quad_term + barrier


def grid_posterior_stats(cov0, constraints, grid_size=351, radius_scale=5.5):
    eigvals = np.linalg.eigvalsh(cov0)
    radius = radius_scale * np.sqrt(eigvals.max())
    x1 = np.linspace(-radius, radius, grid_size)
    x2 = np.linspace(-radius, radius, grid_size)
    dx = x1[1] - x1[0]
    dy = x2[1] - x2[0]
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    grid = np.stack([X1, X2], axis=-1)

    precision0 = np.linalg.inv(cov0)
    log_unnorm = -potential(grid, precision0=precision0, constraints=constraints)
    logZ = logsumexp(log_unnorm.ravel()) + np.log(dx * dy)
    weights = np.exp(log_unnorm - logZ)

    mean = np.zeros(2)
    mean[0] = np.sum(weights * X1) * dx * dy
    mean[1] = np.sum(weights * X2) * dx * dy

    centered = grid - mean
    cov = np.zeros((2, 2))
    cov[0, 0] = np.sum(weights * centered[..., 0] * centered[..., 0]) * dx * dy
    cov[0, 1] = np.sum(weights * centered[..., 0] * centered[..., 1]) * dx * dy
    cov[1, 0] = cov[0, 1]
    cov[1, 1] = np.sum(weights * centered[..., 1] * centered[..., 1]) * dx * dy

    return {
        "x1": x1,
        "x2": x2,
        "X1": X1,
        "X2": X2,
        "grid": grid,
        "log_unnorm": log_unnorm,
        "weights": weights,
        "logZ": logZ,
        "mean_p": mean,
        "cov_p": cov,
        "precision0": precision0,
    }


def params_to_mean_cov(params):
    mean = np.array(params[:2], dtype=float)
    l11 = np.exp(params[2])
    l21 = params[3]
    l22 = np.exp(params[4])
    chol = np.array([[l11, 0.0], [l21, l22]])
    cov = chol @ chol.T
    return mean, chol, cov


def gh_expectation_under_q(mean, chol, func):
    standard_nodes = np.stack([GH_U, GH_V], axis=-1)
    transformed = mean + np.sqrt(2.0) * np.einsum("ij,...j->...i", chol, standard_nodes)
    values = func(transformed)
    return np.sum(GH_W * values) / np.pi


def kl_objective(params, precision0, constraints):
    mean, chol, cov = params_to_mean_cov(params)
    entropy = 0.5 * np.log(np.linalg.det(2.0 * np.pi * np.e * cov))
    expected_potential = gh_expectation_under_q(
        mean,
        chol,
        lambda samples: potential(samples, precision0=precision0, constraints=constraints),
    )
    return expected_potential - entropy


def fit_variational_gaussian(mean_init, cov_init, precision0, constraints):
    chol_init = np.linalg.cholesky(cov_init)
    params0 = np.array(
        [
            mean_init[0],
            mean_init[1],
            np.log(chol_init[0, 0]),
            chol_init[1, 0],
            np.log(chol_init[1, 1]),
        ]
    )
    result = minimize(
        kl_objective,
        params0,
        args=(precision0, constraints),
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (-8.0, 4.0), (None, None), (-8.0, 4.0)],
    )
    mean_q, _, cov_q = params_to_mean_cov(result.x)
    return mean_q, cov_q, result


def format_vector(vec):
    return "[" + ", ".join(f"{value:.6f}" for value in vec) + "]"


def format_matrix(mat):
    rows = []
    for row in mat:
        rows.append("[" + ", ".join(f"{value:.6f}" for value in row) + "]")
    return "[\n  " + ",\n  ".join(rows) + "\n]"


def save_figure_with_png(fig, output_path, dpi=220, bbox_inches="tight"):
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    png_path = None
    if output_path.suffix.lower() == ".pdf":
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    return png_path


def build_summary(cov0, constraints, posterior_stats, mean_q, cov_q, opt_result):
    lines = [
        "2D Gaussian prior + linear probit constraints example",
        "",
        f"Sigma0 = {format_matrix(cov0)}",
        "constraints (a, nu):",
    ]
    for direction, nu in constraints:
        lines.append(f"  a = {format_vector(direction)}, nu = {nu:.6f}")
    lines.extend(
        [
            "",
            f"mean_p = {format_vector(posterior_stats['mean_p'])}",
            f"Sigma_p = {format_matrix(posterior_stats['cov_p'])}",
            "",
            f"mean_q = {format_vector(mean_q)}",
            f"Sigma_q = {format_matrix(cov_q)}",
            "",
            f"Sigma_p - Sigma_q = {format_matrix(posterior_stats['cov_p'] - cov_q)}",
            f"eigenvalues(Sigma_p) = {format_vector(np.linalg.eigvalsh(posterior_stats['cov_p']))}",
            f"eigenvalues(Sigma_q) = {format_vector(np.linalg.eigvalsh(cov_q))}",
            f"KL(q||p) objective (up to additive log Z) = {opt_result.fun:.8f}",
        ]
    )
    return "\n".join(lines)


def build_sweep_summary(cov0, base_constraints, sweep_rows):
    lines = [
        "2D Gaussian prior + linear probit constraints epsilon sweep",
        "",
        f"Sigma0 = {format_matrix(cov0)}",
        "base constraints (a, nu_base):",
    ]
    for direction, nu in base_constraints:
        lines.append(f"  a = {format_vector(direction)}, nu_base = {nu:.6f}")
    lines.extend(["", "rows:"])
    for row in sweep_rows:
        lines.append(
            "  "
            + ", ".join(
                [
                    f"epsilon = {row['epsilon']:.6f}",
                    f"nu_eff = {format_vector(np.array(row['nu_eff']))}",
                    f"eigmin(diff) = {row['eigmin_diff']:.6e}",
                    f"eigmax(diff) = {row['eigmax_diff']:.6e}",
                    f"det(diff) = {row['det_diff']:.6e}",
                    f"pd = {row['is_pd']}",
                ]
            )
        )
        lines.append(f"    Sigma_p = {format_matrix(row['cov_p'])}")
        lines.append(f"    Sigma_q = {format_matrix(row['cov_q'])}")
        lines.append(f"    Sigma_p - Sigma_q = {format_matrix(row['diff'])}")
    return "\n".join(lines)


def build_figure(cov0, constraints, posterior_stats, mean_q, cov_q):
    x1 = posterior_stats["x1"]
    x2 = posterior_stats["x2"]
    X1 = posterior_stats["X1"]
    X2 = posterior_stats["X2"]
    weights = posterior_stats["weights"]
    mean_p = posterior_stats["mean_p"]
    cov_p = posterior_stats["cov_p"]

    p_pdf = weights
    q_pdf = multivariate_normal(mean=mean_q, cov=cov_q).pdf(np.stack([X1, X2], axis=-1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    ax.contour(X1, X2, p_pdf, levels=8, cmap="Blues")
    ax.contour(X1, X2, q_pdf, levels=8, cmap="Oranges")
    ax.scatter([mean_p[0]], [mean_p[1]], color="tab:blue", s=40, label="mean_p")
    ax.scatter([mean_q[0]], [mean_q[1]], color="tab:orange", s=40, label="mean_q")
    ax.set_title("Posterior vs Variational Gaussian")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    im = ax.imshow(
        np.log(np.maximum(q_pdf, 1e-300)) - np.log(np.maximum(p_pdf, 1e-300)),
        origin="lower",
        extent=[x1.min(), x1.max(), x2.min(), x2.max()],
        aspect="auto",
        cmap="coolwarm",
    )
    ax.set_title(r"$\log q(x)-\log p(x)$")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.15)
    fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle("2D Gaussian prior with two linear probit constraints", fontsize=13)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    return fig


def build_sweep_figure(sweep_rows):
    eps = np.array([row["epsilon"] for row in sweep_rows])
    eigmin = np.array([row["eigmin_diff"] for row in sweep_rows])
    eigmax = np.array([row["eigmax_diff"] for row in sweep_rows])
    dets = np.array([row["det_diff"] for row in sweep_rows])
    diff11 = np.array([row["diff"][0, 0] for row in sweep_rows])
    diff12 = np.array([row["diff"][0, 1] for row in sweep_rows])
    diff22 = np.array([row["diff"][1, 1] for row in sweep_rows])
    pd_mask = np.array([1.0 if row["is_pd"] else 0.0 for row in sweep_rows])

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    ax = axes[0]
    ax.plot(eps, diff11, marker="o", lw=2.0, label=r"$(\Sigma_p-\Sigma_q)_{11}$")
    ax.plot(eps, diff12, marker="s", lw=2.0, label=r"$(\Sigma_p-\Sigma_q)_{12}$")
    ax.plot(eps, diff22, marker="^", lw=2.0, label=r"$(\Sigma_p-\Sigma_q)_{22}$")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylabel("Matrix Entries")
    ax.set_title(r"Entries of $\Sigma_p-\Sigma_q$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(eps, eigmin, marker="o", lw=2.0, label=r"$\lambda_{\min}(\Sigma_p-\Sigma_q)$")
    ax.plot(eps, eigmax, marker="s", lw=2.0, label=r"$\lambda_{\max}(\Sigma_p-\Sigma_q)$")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylabel("Eigenvalues")
    ax.set_title(r"Eigenvalues of $\Sigma_p-\Sigma_q$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    ax = axes[2]
    ax.plot(eps, dets, marker="o", lw=2.0, label=r"$\det(\Sigma_p-\Sigma_q)$")
    ax.scatter(eps, pd_mask, c=np.where(pd_mask > 0.5, "tab:green", "tab:red"), s=35, label="PD indicator")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"Common Constraint Scale $\epsilon$")
    ax.set_ylabel("Determinant / PD")
    ax.set_title(r"Determinant and Positive-Definiteness of $\Sigma_p-\Sigma_q$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    tick_values = np.unique(np.geomspace(eps.min(), eps.max(), min(8, len(eps))))
    axes[2].set_xticks(tick_values)
    axes[2].set_xticklabels([f"{tick:.3g}" for tick in tick_values])

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Numerical 2D example for Sigma_p and Sigma_q under multiple linear probit constraints."
    )
    parser.add_argument(
        "--sigma0",
        type=float,
        nargs=4,
        default=[1.0, 0.45, 0.45, 1.4],
        metavar=("s11", "s12", "s21", "s22"),
        help="Entries of the 2x2 prior covariance matrix.",
    )
    parser.add_argument(
        "--constraint",
        action="append",
        default=["1.0,0.7,0.6", "-0.4,1.1,0.8"],
        help="Constraint row 'a1,a2,nu'. Can be passed multiple times.",
    )
    parser.add_argument("--grid-size", type=int, default=351, help="Grid size for posterior quadrature.")
    parser.add_argument("--output", type=str, default="multivariate_probit_2d.pdf", help="Output figure path.")
    parser.add_argument("--summary", type=str, default="multivariate_probit_2d.txt", help="Output text summary path.")
    parser.add_argument("--epsilons", type=str, default=None, help="Comma-separated list of common epsilon scales for a sweep.")
    parser.add_argument("--epsilon-min", type=float, default=None, help="Minimum epsilon for a log-spaced sweep.")
    parser.add_argument("--epsilon-max", type=float, default=None, help="Maximum epsilon for a log-spaced sweep.")
    parser.add_argument("--num-epsilon", type=int, default=12, help="Number of epsilon points in a log-spaced sweep.")
    parser.add_argument("--show", action="store_true", help="Also show the figure interactively.")
    args = parser.parse_args()

    cov0 = make_spd_matrix(args.sigma0)
    base_constraints = parse_constraint_rows(args.constraint)

    epsilon_values = None
    if args.epsilons is not None:
        epsilon_values = [float(token.strip()) for token in args.epsilons.split(",") if token.strip()]
    elif args.epsilon_min is not None and args.epsilon_max is not None:
        epsilon_values = np.geomspace(args.epsilon_min, args.epsilon_max, args.num_epsilon).tolist()

    figure_path = Path(args.output)
    summary_path = Path(args.summary)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if epsilon_values is None:
        constraints = base_constraints
        posterior_stats = grid_posterior_stats(cov0=cov0, constraints=constraints, grid_size=args.grid_size)
        mean_q, cov_q, opt_result = fit_variational_gaussian(
            mean_init=posterior_stats["mean_p"],
            cov_init=posterior_stats["cov_p"],
            precision0=posterior_stats["precision0"],
            constraints=constraints,
        )

        summary_text = build_summary(cov0, constraints, posterior_stats, mean_q, cov_q, opt_result)
        summary_path.write_text(summary_text + "\n", encoding="utf-8")
        print(summary_text)
        print(f"summary saved to: {summary_path.resolve()}")

        fig = build_figure(cov0, constraints, posterior_stats, mean_q, cov_q)
        png_path = save_figure_with_png(fig, figure_path, dpi=220, bbox_inches="tight")
        print(f"figure saved to: {figure_path.resolve()}")
        if png_path is not None:
            print(f"png saved to: {png_path.resolve()}")
    else:
        sweep_rows = []
        for epsilon_scale in epsilon_values:
            constraints = scale_constraints(base_constraints, epsilon_scale=epsilon_scale)
            posterior_stats = grid_posterior_stats(cov0=cov0, constraints=constraints, grid_size=args.grid_size)
            mean_q, cov_q, opt_result = fit_variational_gaussian(
                mean_init=posterior_stats["mean_p"],
                cov_init=posterior_stats["cov_p"],
                precision0=posterior_stats["precision0"],
                constraints=constraints,
            )
            diff = posterior_stats["cov_p"] - cov_q
            eigvals = np.linalg.eigvalsh(diff)
            row = {
                "epsilon": float(epsilon_scale),
                "nu_eff": [nu / epsilon_scale for _, nu in base_constraints],
                "cov_p": posterior_stats["cov_p"],
                "cov_q": cov_q,
                "diff": diff,
                "eigmin_diff": eigvals[0],
                "eigmax_diff": eigvals[1],
                "det_diff": np.linalg.det(diff),
                "is_pd": bool(np.all(eigvals > 1e-10)),
                "kl_objective": opt_result.fun,
            }
            sweep_rows.append(row)
            print(
                f"epsilon={epsilon_scale:.6f}, eigmin(diff)={row['eigmin_diff']:.6e}, "
                f"det(diff)={row['det_diff']:.6e}, pd={row['is_pd']}"
            )

        summary_text = build_sweep_summary(cov0, base_constraints, sweep_rows)
        summary_path.write_text(summary_text + "\n", encoding="utf-8")
        print(f"sweep summary saved to: {summary_path.resolve()}")

        fig = build_sweep_figure(sweep_rows)
        png_path = save_figure_with_png(fig, figure_path, dpi=220, bbox_inches="tight")
        print(f"sweep figure saved to: {figure_path.resolve()}")
        if png_path is not None:
            print(f"sweep png saved to: {png_path.resolve()}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
