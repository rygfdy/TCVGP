import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq, minimize
from scipy.stats import norm


GH_ORDER = 80
GH_NODES, GH_WEIGHTS = hermgauss(GH_ORDER)


def inverse_mills_ratio(t):
    return np.exp(norm.logpdf(t) - norm.logcdf(t))


def V(x, alpha, sigma0):
    return 0.5 * (x / sigma0) ** 2 - norm.logcdf(alpha * x)


def V_prime(x, alpha, sigma0):
    t = alpha * x
    lam = inverse_mills_ratio(t)
    return x / sigma0**2 - alpha * lam


def V_prime2(x, alpha, sigma0):
    t = alpha * x
    lam = inverse_mills_ratio(t)
    return 1.0 / sigma0**2 + alpha**2 * lam * (t + lam)


def V_prime3(x, alpha):
    t = alpha * x
    lam = inverse_mills_ratio(t)
    return alpha**3 * lam * (1.0 - (t + lam) * (t + 2.0 * lam))


def log_unnormalized_p(x, alpha, sigma0):
    return -V(x, alpha=alpha, sigma0=sigma0)


def normalize_p(alpha, sigma0, radius):
    Z, _ = quad(
        lambda z: np.exp(log_unnormalized_p(z, alpha=alpha, sigma0=sigma0)),
        -radius,
        radius,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=200,
    )
    return Z


def p_moments(alpha, sigma0, radius, Z):
    mean, _ = quad(
        lambda z: z * np.exp(log_unnormalized_p(z, alpha=alpha, sigma0=sigma0)) / Z,
        -radius,
        radius,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=200,
    )
    second_moment, _ = quad(
        lambda z: z**2 * np.exp(log_unnormalized_p(z, alpha=alpha, sigma0=sigma0)) / Z,
        -radius,
        radius,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=200,
    )
    return mean, second_moment - mean**2


def gh_expectation(func, mean, std, *args):
    samples = mean + np.sqrt(2.0) * std * GH_NODES
    return np.dot(GH_WEIGHTS, func(samples, *args)) / np.sqrt(np.pi)


def kl_q_to_p(params, alpha, sigma0):
    mean, log_std = params
    std = np.exp(log_std)
    entropy = 0.5 * np.log(2.0 * np.pi * np.e * std**2)
    expected_V = gh_expectation(V, mean, std, alpha, sigma0)
    return expected_V - entropy


def fit_variational_gaussian(alpha, sigma0, mean_init, var_init):
    result = minimize(
        kl_q_to_p,
        x0=np.array([mean_init, 0.5 * np.log(var_init)]),
        args=(alpha, sigma0),
        method="L-BFGS-B",
        bounds=[(None, None), (-8.0, 4.0)],
    )
    mean_q, log_std_q = result.x
    return mean_q, np.exp(log_std_q), result


def find_mode_p(alpha, sigma0, left=-10.0, right=10.0):
    return brentq(lambda z: V_prime(z, alpha=alpha, sigma0=sigma0), left, right)


def find_zero_crossings(x, y, atol=1e-10, rtol=1e-6):
    scale = np.max(np.abs(y))
    threshold = max(atol, rtol * scale)
    y_work = np.array(y, copy=True)
    y_work[np.abs(y_work) < threshold] = 0.0

    roots = []
    nz_idx = np.flatnonzero(y_work != 0.0)
    if len(nz_idx) < 2:
        return roots

    for left_idx, right_idx in zip(nz_idx[:-1], nz_idx[1:]):
        y_left = y_work[left_idx]
        y_right = y_work[right_idx]
        if y_left * y_right > 0.0:
            continue

        x_left = x[left_idx]
        x_right = x[right_idx]
        y_left_raw = y[left_idx]
        y_right_raw = y[right_idx]
        root = x_left - y_left_raw * (x_right - x_left) / (y_right_raw - y_left_raw)
        roots.append(root)

    unique_roots = []
    for root in roots:
        if not unique_roots or abs(root - unique_roots[-1]) > 1e-3:
            unique_roots.append(root)
    return unique_roots


def interp_value(x, y, x_star):
    return np.interp(x_star, x, y)


def format_roots(roots):
    if not roots:
        return "none"
    return ", ".join(f"{root:.4f}" for root in roots)


def sign_label(value, tol=1e-12):
    if value > tol:
        return "+"
    if value < -tol:
        return "-"
    return "0"


def summarize_sign_intervals(x, y, roots):
    boundaries = [x[0], *roots, x[-1]]
    intervals = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        mask = (x >= left) & (x <= right)
        x_segment = x[mask]
        y_segment = y[mask]
        if len(y_segment) == 0:
            label = "0"
        else:
            rep_idx = int(np.argmax(np.abs(y_segment)))
            label = sign_label(y_segment[rep_idx])
        intervals.append(f"({left:.3f}, {right:.3f}): {label}")
    return " | ".join(intervals)


def parse_float_list(raw_value):
    if raw_value is None:
        return None
    values = []
    for token in raw_value.split(","):
        stripped = token.strip()
        if stripped:
            values.append(float(stripped))
    return values


def build_log_grid(min_value, max_value, num_points):
    if min_value <= 0.0 or max_value <= 0.0:
        raise ValueError("Log-grid endpoints must be positive.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    return np.geomspace(min_value, max_value, num_points).tolist()


def sanitize_float_tag(value):
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def choose_log_ticks(values, max_ticks=10):
    unique_values = np.unique(np.array(values, dtype=float))
    if len(unique_values) <= max_ticks:
        return unique_values
    idx = np.linspace(0, len(unique_values) - 1, max_ticks, dtype=int)
    return unique_values[idx]


def save_figure_with_png(fig, output_path, dpi=220, bbox_inches="tight"):
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    png_path = None
    if output_path.suffix.lower() == ".pdf":
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    return png_path


def analyze_case(nu, sigma0, grid_size=4001):
    alpha = 1.0 / nu
    radius = 12.0 * sigma0
    epsilon = alpha * sigma0

    Z = normalize_p(alpha=alpha, sigma0=sigma0, radius=radius)
    mean_p, var_p = p_moments(alpha=alpha, sigma0=sigma0, radius=radius, Z=Z)
    mean_q, std_q, opt_result = fit_variational_gaussian(
        alpha=alpha, sigma0=sigma0, mean_init=mean_p, var_init=var_p
    )
    mode_p = find_mode_p(alpha=alpha, sigma0=sigma0, left=-radius, right=radius)
    mode_q = mean_q

    left = min(-radius, mean_p - 8.0 * np.sqrt(var_p), mean_q - 8.0 * std_q)
    right = max(radius, mean_p + 8.0 * np.sqrt(var_p), mean_q + 8.0 * std_q)
    x = np.linspace(left, right, grid_size)

    V_x = V(x, alpha=alpha, sigma0=sigma0)
    V1_x = V_prime(x, alpha=alpha, sigma0=sigma0)
    V2_x = V_prime2(x, alpha=alpha, sigma0=sigma0)
    V3_x = V_prime3(x, alpha=alpha)

    p_x = np.exp(log_unnormalized_p(x, alpha=alpha, sigma0=sigma0)) / Z
    q_x = norm.pdf(x, loc=mean_q, scale=std_q)
    h_x = q_x - p_x

    P_x = cumulative_trapezoid(p_x, x, initial=0.0)
    P_x /= P_x[-1]
    Q_x = norm.cdf(x, loc=mean_q, scale=std_q)
    H_x = Q_x - P_x

    v3_roots = find_zero_crossings(x, V3_x)
    h_roots = find_zero_crossings(x, h_x)
    H_roots = find_zero_crossings(x, H_x)

    h_root_extrema = [(root, interp_value(x, H_x, root)) for root in h_roots]
    max_cdf_gap_idx = int(np.argmax(np.abs(H_x)))
    max_cdf_gap_x = x[max_cdf_gap_idx]
    max_cdf_gap = H_x[max_cdf_gap_idx]

    return {
        "alpha": alpha,
        "nu": nu,
        "sigma0": sigma0,
        "epsilon": epsilon,
        "radius": radius,
        "Z": Z,
        "mean_p": mean_p,
        "var_p": var_p,
        "std_p": np.sqrt(var_p),
        "mean_q": mean_q,
        "std_q": std_q,
        "var_q": std_q**2,
        "mode_p": mode_p,
        "mode_q": mode_q,
        "kl_objective": opt_result.fun,
        "x": x,
        "V_x": V_x,
        "V1_x": V1_x,
        "V2_x": V2_x,
        "V3_x": V3_x,
        "p_x": p_x,
        "q_x": q_x,
        "h_x": h_x,
        "P_x": P_x,
        "Q_x": Q_x,
        "H_x": H_x,
        "v3_roots": v3_roots,
        "h_roots": h_roots,
        "H_roots": H_roots,
        "h_root_extrema": h_root_extrema,
        "max_cdf_gap_x": max_cdf_gap_x,
        "max_cdf_gap": max_cdf_gap,
        "sigma_gap": np.sqrt(var_p) - std_q,
        "var_gap": var_p - std_q**2,
    }


def build_figure(results):
    alpha = results["alpha"]
    nu = results["nu"]
    sigma0 = results["sigma0"]
    x = results["x"]
    V_x = results["V_x"]
    V1_x = results["V1_x"]
    V2_x = results["V2_x"]
    V3_x = results["V3_x"]
    p_x = results["p_x"]
    q_x = results["q_x"]
    h_x = results["h_x"]
    H_x = results["H_x"]
    v3_roots = results["v3_roots"]
    h_roots = results["h_roots"]
    H_roots = results["H_roots"]
    mean_p = results["mean_p"]
    mean_q = results["mean_q"]
    mode_p = results["mode_p"]
    mode_q = results["mode_q"]
    max_cdf_gap_x = results["max_cdf_gap_x"]
    max_cdf_gap = results["max_cdf_gap"]

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))

    ax = axes[0, 0]
    ax.plot(x, V_x, label="$V(x)$", color="black", lw=2.0)
    ax.plot(x, V1_x, label="$V'(x)$", lw=1.8)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_title("Potential and First Derivative")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, V2_x, label="$V''(x)$", color="tab:blue", lw=2.0)
    ax.plot(x, V3_x, label="$V'''(x)$", color="tab:red", lw=1.8)
    ax.axhline(0.0, color="gray", lw=0.8)
    for root in v3_roots:
        ax.axvline(root, color="tab:red", ls="--", lw=0.8, alpha=0.5)
    ax.set_title("Curvature and Curvature Slope")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, p_x, label="$p(x)$", color="tab:blue", lw=2.5)
    ax.plot(x, q_x, label="$q(x)$", color="tab:orange", lw=2.0)
    ax.axvline(mean_p, color="tab:blue", ls="--", lw=1.2, alpha=0.9, label=r"$\mathbb{E}_p[X]$")
    ax.axvline(mean_q, color="tab:orange", ls="--", lw=1.2, alpha=0.9, label=r"$\mathbb{E}_q[X]$")
    ax.axvline(mode_p, color="tab:blue", ls=":", lw=1.2, alpha=0.9, label=r"$\mathrm{mode}(p)$")
    ax.axvline(mode_q, color="tab:orange", ls=":", lw=1.2, alpha=0.9, label=r"$\mathrm{mode}(q)$")
    ax.set_title("Density Comparison")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    log_ratio = np.log(np.maximum(q_x, 1e-300)) - np.log(np.maximum(p_x, 1e-300))
    ax.plot(x, log_ratio, color="tab:brown", lw=1.8)
    ax.axhline(0.0, color="gray", lw=0.8)
    for root in h_roots:
        ax.axvline(root, color="tab:purple", ls="--", lw=0.8, alpha=0.5)
    ax.set_title(r"$\log q(x)-\log p(x)$")
    ax.grid(alpha=0.25)

    ax = axes[2, 0]
    ax.plot(x, h_x, label="$q(x)-p(x)$", color="tab:purple", lw=2.0)
    ax.axhline(0.0, color="gray", lw=0.8)
    for root in h_roots:
        ax.axvline(root, color="tab:purple", ls="--", lw=0.8, alpha=0.5)
    ax.set_title(r"Density Gap $h(x)=q(x)-p(x)$")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(x, H_x, label="$Q(x)-P(x)$", color="tab:green", lw=2.0)
    ax.axhline(0.0, color="gray", lw=0.8)
    for root in H_roots:
        ax.axvline(root, color="tab:green", ls="--", lw=0.8, alpha=0.5)
    for root in h_roots:
        ax.axvline(root, color="tab:purple", ls=":", lw=0.8, alpha=0.4)
    ax.scatter([max_cdf_gap_x], [max_cdf_gap], color="tab:green", zorder=3)
    ax.set_title(r"CDF Gap $H(x)=Q(x)-P(x)$ with $H'(x)=h(x)$")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle(
        rf"Gaussian prior + probit constraint, $\nu={nu:.3f}$, $\alpha={alpha:.3f}$, $\epsilon={results['epsilon']:.3f}$, $\sigma_0={sigma0:.2f}$",
        fontsize=14,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    return fig


def build_text_summary(results):
    lines = [
        f"nu = {results['nu']:.6f}, alpha = {results['alpha']:.6f}, epsilon = {results['epsilon']:.6f}, sigma0 = {results['sigma0']:.6f}",
        f"Z = {results['Z']:.8f}",
        f"mean_p = {results['mean_p']:.6f}, var_p = {results['var_p']:.6f}, std_p = {results['std_p']:.6f}",
        f"mean_q = {results['mean_q']:.6f}, var_q = {results['var_q']:.6f}, std_q = {results['std_q']:.6f}",
        f"mode_p = {results['mode_p']:.6f}, mode_q = {results['mode_q']:.6f}",
        f"sigma_p - sigma_q = {results['sigma_gap']:.6e}",
        f"var_p - var_q = {results['var_gap']:.6e}",
        f"KL(q||p) objective (up to additive log Z) = {results['kl_objective']:.8f}",
        f"V''' zero crossings: {format_roots(results['v3_roots'])}",
        f"sign of V''' on intervals: {summarize_sign_intervals(results['x'], results['V3_x'], results['v3_roots'])}",
        f"q - p zero crossings: {format_roots(results['h_roots'])}",
        f"sign of q-p on intervals: {summarize_sign_intervals(results['x'], results['h_x'], results['h_roots'])}",
        f"Q - P zero crossings: {format_roots(results['H_roots'])}",
        f"sign of Q-P on intervals: {summarize_sign_intervals(results['x'], results['H_x'], results['H_roots'])}",
        f"max |Q-P| = {abs(results['max_cdf_gap']):.6e} at x = {results['max_cdf_gap_x']:.4f}",
    ]
    if results["h_root_extrema"]:
        lines.append("Q-P evaluated at roots of q-p (these are extrema of Q-P):")
        for root, value in results["h_root_extrema"]:
            lines.append(f"  x = {root:.4f}, Q-P = {value:.6e}")
    return "\n".join(lines)


def save_case_outputs(results, figure_path, text_path):
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(results)
    png_path = save_figure_with_png(fig, figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    summary_text = build_text_summary(results)
    text_path.write_text(summary_text + "\n", encoding="utf-8")
    return summary_text, png_path


def write_batch_summary(results_list, summary_path):
    header = [
        "nu",
        "alpha",
        "sigma0",
        "epsilon",
        "mean_p",
        "std_p",
        "mean_q",
        "std_q",
        "sigma_gap",
        "var_gap",
        "kl_objective",
        "max_abs_cdf_gap",
        "max_cdf_gap_x",
        "h_roots",
        "H_roots",
    ]
    lines = ["\t".join(header)]
    for results in results_list:
        row = [
            f"{results['nu']:.6g}",
            f"{results['alpha']:.6g}",
            f"{results['sigma0']:.6g}",
            f"{results['epsilon']:.6g}",
            f"{results['mean_p']:.8f}",
            f"{results['std_p']:.8f}",
            f"{results['mean_q']:.8f}",
            f"{results['std_q']:.8f}",
            f"{results['sigma_gap']:.8e}",
            f"{results['var_gap']:.8e}",
            f"{results['kl_objective']:.8f}",
            f"{abs(results['max_cdf_gap']):.8e}",
            f"{results['max_cdf_gap_x']:.8f}",
            format_roots(results["h_roots"]),
            format_roots(results["H_roots"]),
        ]
        lines.append("\t".join(row))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_epsilon_curve_plot(results_list, output_path):
    grouped_results = {}
    for results in results_list:
        grouped_results.setdefault(results["sigma0"], []).append(results)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    epsilon_values = sorted({results["epsilon"] for results in results_list})
    tick_values = choose_log_ticks(epsilon_values)

    for sigma0 in sorted(grouped_results):
        group = sorted(grouped_results[sigma0], key=lambda item: item["epsilon"])
        eps = np.array([item["epsilon"] for item in group])
        std_p = np.array([item["std_p"] for item in group])
        std_q = np.array([item["std_q"] for item in group])

        axes[0].plot(
            eps,
            std_p,
            marker="o",
            lw=2.0,
            label=rf"$\sigma_p$, $\sigma_0={sigma0:.3g}$",
        )
        axes[0].plot(
            eps,
            std_q,
            marker="s",
            lw=2.0,
            ls="--",
            label=rf"$\sigma_q$, $\sigma_0={sigma0:.3g}$",
        )

        axes[1].plot(
            eps,
            std_p / sigma0,
            marker="o",
            lw=2.0,
            label=rf"$\sigma_p/\sigma_0$, $\sigma_0={sigma0:.3g}$",
        )
        axes[1].plot(
            eps,
            std_q / sigma0,
            marker="s",
            lw=2.0,
            ls="--",
            label=rf"$\sigma_q/\sigma_0$, $\sigma_0={sigma0:.3g}$",
        )

    axes[0].set_xscale("log")
    axes[0].set_ylabel("Standard Deviation")
    axes[0].set_title(r"$\sigma_p$ and $\sigma_q$ vs $\epsilon=\alpha\sigma_0$")
    axes[0].set_xticks(tick_values)
    axes[0].set_xticklabels([f"{tick:.3g}" for tick in tick_values])
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(ncol=2, fontsize=9)

    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\epsilon=\alpha\sigma_0$")
    axes[1].set_ylabel(r"Normalized Std ($/\sigma_0$)")
    axes[1].set_title(r"Normalized $\sigma_p$ and $\sigma_q$ vs $\epsilon$")
    axes[1].set_xticks(tick_values)
    axes[1].set_xticklabels([f"{tick:.3g}" for tick in tick_values], rotation=0)
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(ncol=2, fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = save_figure_with_png(fig, output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze V, V', V'', V''', q-p and Q-P for a Gaussian prior with probit constraint."
    )
    parser.add_argument("--nu", type=float, default=1.0 / 3.0, help="Constraint softness nu = 1 / alpha.")
    parser.add_argument("--sigma0", type=float, default=1.0, help="Prior std of the Gaussian term.")
    parser.add_argument("--alpha", type=float, default=None, help="Legacy alias for alpha. Overrides --nu when provided.")
    parser.add_argument(
        "--nus",
        type=str,
        default=None,
        help="Comma-separated list of nu values for batch runs.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Legacy alias for alpha lists. Converted internally to nu = 1 / alpha.",
    )
    parser.add_argument(
        "--sigma0s",
        type=str,
        default=None,
        help="Comma-separated list of sigma0 values for batch runs.",
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default=None,
        help="Comma-separated list of epsilon = alpha * sigma0 values for dense curve sweeps.",
    )
    parser.add_argument("--epsilon-min", type=float, default=None, help="Minimum epsilon for an automatic log-spaced epsilon grid.")
    parser.add_argument("--epsilon-max", type=float, default=None, help="Maximum epsilon for an automatic log-spaced epsilon grid.")
    parser.add_argument("--num-epsilon", type=int, default=25, help="Number of epsilon points in automatic log-spaced sweeps.")
    parser.add_argument(
        "--output",
        type=str,
        default="nonconvex_p_q_analysis.pdf",
        help="Path to the saved figure for single-case mode.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="nonconvex_p_q_outputs",
        help="Directory for batch outputs.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="batch_summary.txt",
        help="Filename for the batch summary table.",
    )
    parser.add_argument(
        "--curve-name",
        type=str,
        default="sigma_vs_epsilon.pdf",
        help="Filename for the batch epsilon-curve summary plot.",
    )
    parser.add_argument("--show", action="store_true", help="Also show the figure interactively in single-case mode.")
    args = parser.parse_args()

    sigma0_list = parse_float_list(args.sigma0s) or [args.sigma0]
    epsilon_list = parse_float_list(args.epsilons)
    if epsilon_list is None and args.epsilon_min is not None and args.epsilon_max is not None:
        epsilon_list = build_log_grid(args.epsilon_min, args.epsilon_max, args.num_epsilon)

    if epsilon_list is not None:
        case_list = []
        for sigma0, epsilon in itertools.product(sigma0_list, epsilon_list):
            nu = sigma0 / epsilon
            case_list.append((nu, sigma0))
    else:
        if args.alphas is not None:
            alpha_list = parse_float_list(args.alphas)
            nu_list = [1.0 / alpha for alpha in alpha_list]
        else:
            nu_list = parse_float_list(args.nus)
            if nu_list is None:
                nu_value = 1.0 / args.alpha if args.alpha is not None else args.nu
                nu_list = [nu_value]
        case_list = list(itertools.product(nu_list, sigma0_list))

    is_batch = len(case_list) > 1 or args.nus is not None or args.alphas is not None or args.sigma0s is not None or epsilon_list is not None

    if is_batch:
        outdir = Path(args.outdir)
        results_list = []
        for nu, sigma0 in case_list:
            results = analyze_case(nu=nu, sigma0=sigma0)
            stem = (
                f"nu_{sanitize_float_tag(results['nu'])}"
                f"__sigma0_{sanitize_float_tag(sigma0)}"
                f"__eps_{sanitize_float_tag(results['epsilon'])}"
            )
            figure_path = outdir / f"{stem}.pdf"
            text_path = outdir / f"{stem}.txt"
            _, png_path = save_case_outputs(results, figure_path=figure_path, text_path=text_path)
            results_list.append(results)
            print(
                f"[saved] nu={results['nu']:.4f}, alpha={results['alpha']:.4f}, "
                f"epsilon={results['epsilon']:.4f}, sigma0={sigma0:.4f}"
            )
            print(f"  figure: {figure_path.resolve()}")
            if png_path is not None:
                print(f"  png:    {png_path.resolve()}")
            print(f"  text:   {text_path.resolve()}")
        summary_path = outdir / args.summary_name
        write_batch_summary(results_list, summary_path)
        curve_path = outdir / args.curve_name
        curve_png_path = write_epsilon_curve_plot(results_list, curve_path)
        print(f"batch summary saved to: {summary_path.resolve()}")
        print(f"epsilon curve plot saved to: {curve_path.resolve()}")
        if curve_png_path is not None:
            print(f"epsilon curve png saved to: {curve_png_path.resolve()}")
        return

    nu_value, sigma0_value = case_list[0]
    results = analyze_case(nu=nu_value, sigma0=sigma0_value)
    output_path = Path(args.output)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".pdf")
    text_path = output_path.with_suffix(".txt")
    summary_text, png_path = save_case_outputs(results, figure_path=output_path, text_path=text_path)
    print(summary_text)
    print(f"figure saved to: {output_path.resolve()}")
    if png_path is not None:
        print(f"png saved to: {png_path.resolve()}")
    print(f"text summary saved to: {text_path.resolve()}")

    if args.show:
        fig = build_figure(results)
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
