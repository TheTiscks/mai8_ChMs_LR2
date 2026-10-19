import math
import os
import re
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAFE_MATH = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log, # натуральный
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
    "pi": math.pi,
    "e": math.e,
}


def normalize_equation(eq: str) -> str:
    eq = re.sub(r"(\d),(\d)", r"\1.\2", eq)
    eq = eq.replace("^", "**")
    eq = re.sub(r"\bln\b", "log", eq)
    eq = re.sub(r"\blg\b", "log10", eq)
    eq = re.sub(r"\btg\b", "tan", eq)
    eq = re.sub(
        r"\b(sin|cos|tan|log|log10|sqrt|exp)\s+([a-zA-Z0-9_]+)",
        r"\1(\2)",
        eq,
    )
    eq = eq.replace(" ", "")
    eq = re.sub(r"^\.", "", eq)
    eq = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", eq)
    return eq


def make_func(expr: str):
    code = compile(expr, "<expr>", "eval")
    def f(x):
        try:
            return eval(code, {"__builtins__": {}}, {**SAFE_MATH, "x": x})
        except (ValueError, ZeroDivisionError, OverflowError):
            return float("nan")
    return f


def numerical_derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def find_positive_root_interval(f, start=1e-6, end=100.0, step=0.1):
    #ищеи интервал [a, b] где f меняет знак - начало с x>0
    a = start
    fa = f(a)
    while a < end:
        b = a + step
        fb = f(b)
        if not math.isnan(fa) and not math.isnan(fb):
            if fa * fb <= 0:
                return a, b
        a = b
        fa = fb
    raise ValueError("no positive root found on [1e-6, 100]")


def max_abs_derivative(f, a, b, n=100): # максимум |f'(x)| на [a, b]
    max_val = 0.0
    for i in range(n + 1):
        x = a + (b - a) * i / n
        df = numerical_derivative(f, x)
        if not math.isnan(df):
            max_val = max(max_val, abs(df))
    return max_val


def simple_iteration(f, a, b, eps, max_iter=100):
    x0 = (a + b) / 2.0
    max_df = max_abs_derivative(f, a, b)
    lam = 1.0 / max_df if max_df > 1e-12 else 1.0
    x_prev = x0
    rows = []
    for k in range(1, max_iter + 1):
        x = x_prev - lam * f(x_prev)
        if math.isnan(x):
            break
        delta = abs(x - x_prev)
        fx = f(x)
        rows.append({"iter": k, "x": x, "f": fx, "delta": delta})
        if delta < eps or abs(fx) < eps:
            break
        x_prev = x
    return rows, lam


def newton_method(f, a, b, eps, max_iter=100): # с числ производной
    x_prev = (a + b) / 2.0
    rows = []
    for k in range(1, max_iter + 1):
        fx = f(x_prev)
        dfx = numerical_derivative(f, x_prev)
        if abs(dfx) < 1e-15:
            break
        x = x_prev - fx / dfx
        if math.isnan(x):
            break
        delta = abs(x - x_prev)
        fx_new = f(x)
        rows.append({"iter": k, "x": x, "f": fx_new, "delta": delta})
        if delta < eps or abs(fx_new) < eps:
            break
        x_prev = x
    return rows


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["iter", "x", "f", "delta"])
        writer.writeheader()
        writer.writerows(rows)


def save_plot(f, a, b, simple_rows, newton_rows, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # графики
    left = a - 0.5
    right = b + 0.5
    xs = [left + (right - left) * i / 300 for i in range(301)]
    ys = [f(x) for x in xs]
    valid = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
    if valid:
        xs_v, ys_v = zip(*valid)
        ax1.plot(xs_v, ys_v, label="f(x)")
    ax1.axhline(0, color="black", linewidth=0.8)
    if newton_rows:
        root = newton_rows[-1]["x"]
        ax1.plot([root], [f(root)], "ro", label=f"корень ≈ {root:.8g}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("График f(x)")
    ax1.grid(True)
    ax1.legend()
    # Зависимость погрешности от числа итераций
    iter_s = [r["iter"] for r in simple_rows]
    delta_s = [max(r["delta"], 1e-16) for r in simple_rows]
    iter_n = [r["iter"] for r in newton_rows]
    delta_n = [max(r["delta"], 1e-16) for r in newton_rows]
    ax2.semilogy(iter_s, delta_s, "o-", label="простая итерация")
    ax2.semilogy(iter_n, delta_n, "s-", label="Ньютон")
    ax2.set_xlabel("номер итерации")
    ax2.set_ylabel("|x_n − x_{n−1}|")
    ax2.set_title("Зависимость погрешности от количества итераций")
    ax2.grid(True, which="both")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def main():
    input_file = "input.txt"
    if not os.path.exists(input_file):
        with open(input_file, "w", encoding="utf-8") as file:
            file.write("2^x - x^2 - 0.5 = 0; 1e-6\n")
            file.write("ln(x+2) - x^2 = 0; 1e-6\n")
            file.write("sqrt(1-x^2) - e^x + 0.1 = 0; 1e-6\n")
            file.write("x^3 + x^2 - x - 0.5 = 0; 1e-6\n")
            file.write("cos x + 0.25x - 0.5 = 0; 1e-6\n")
        print(f"Создан пример файла {input_file}. Отредактируйте его и запустите снова.")
        return

    with open(input_file, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip() and not line.startswith("#")]
    for idx, line in enumerate(lines, start=1):
        if ";" not in line:
            print(f"Строка {idx} пропущена: нет ';'")
            continue
        eq_str, eps_str = line.split(";", 1)
        eq_str = eq_str.strip()
        eps = float(eps_str.strip())
        print(f"\nУравнение: {eq_str}")
        print(f"Точность: {eps}")
        try:
            norm_eq = normalize_equation(eq_str)
            if "=" not in norm_eq:
                print("Ошибка: уравнение должно содержать '='")
                continue
            left, right = norm_eq.split("=", 1)
            f_expr = f"({left}) - ({right})"
            f = make_func(f_expr)
        except Exception as e:
            print(f"Ошибка разбора уравнения: {e}")
            continue
        try:
            a, b = find_positive_root_interval(f)
            print(f"Интервал положительного корня: [{a:.4f}, {b:.4f}]")
        except Exception as e:
            print(f"Не удалось найти положительный корень: {e}")
            continue
        try:
            simple_rows, lam = simple_iteration(f, a, b, eps)
            if simple_rows:
                root_s = simple_rows[-1]["x"]
                print(f"Простая итерация: корень ≈ {root_s:.10f}, итераций: {len(simple_rows)}")
            else:
                print("Простая итерация: не сошлась")
        except Exception as e:
            print(f"Ошибка в методе простой итерации: {e}")
            simple_rows = []
        try:
            newton_rows = newton_method(f, a, b, eps)
            if newton_rows:
                root_n = newton_rows[-1]["x"]
                print(f"Ньютон: корень ≈ {root_n:.10f}, итераций: {len(newton_rows)}")
            else:
                print("Ньютон: не сошёлся")
        except Exception as e:
            print(f"Ошибка в методе Ньютона: {e}")
            newton_rows = []
        outdir = f"results/eq_{idx}"
        os.makedirs(outdir, exist_ok=True)
        if simple_rows:
            save_csv(os.path.join(outdir, "simple_iteration.csv"), simple_rows)
        if newton_rows:
            save_csv(os.path.join(outdir, "newton.csv"), newton_rows)
        with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f_sum:
            f_sum.write(f"Уравнение: {eq_str}\n")
            f_sum.write(f"Точность: {eps}\n")
            f_sum.write(f"Интервал корня: [{a}, {b}]\n\n")
            f_sum.write("Метод простой итерации:\n")
            for r in simple_rows:
                f_sum.write(f"{r['iter']:4d}  x = {r['x']:.15g}  "
                    f"f(x) = {r['f']:.6e}  delta = {r['delta']:.6e}\n")
            f_sum.write("\nМетод Ньютона:\n")
            for r in newton_rows:
                f_sum.write(f"{r['iter']:4d}  x = {r['x']:.15g}  "
                    f"f(x) = {r['f']:.6e}  delta = {r['delta']:.6e}\n")
        save_plot(f, a, b, simple_rows, newton_rows, os.path.join(outdir, "convergence.png"))
        print(f"Результаты сохранены в {outdir}")

if __name__ == "__main__":
    main()