import json
import math
import os
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAFE_MATH = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "log10": math.log10,
    "sqrt": math.sqrt, "abs": abs, "pi": math.pi, "e": math.e,
    "ln": math.log, "lg": math.log10,
}


def normalize_expr(expr: str) -> str:
    return expr.replace("^", "**")


def make_func(expr: str):
    #функция из строки
    code = compile(normalize_expr(expr), "<expr>", "eval")
    def f(x, a):
        try:
            local_vars = {**SAFE_MATH, "x": x, "a": a}
            for i, val in enumerate(x):
                local_vars[f"x{i + 1}"] = val
            return eval(code, {"__builtins__": {}}, local_vars)
        except Exception:
            return float("nan")
    return f


def solve_linear_system(A, b):
    # решаем A*delta = b
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        # ищем ведущий элемент
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[max_row][i]):
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]
        if abs(M[i][i]) < 1e-15:
            raise ValueError("Матрица вырождена или близка к вырожденной")
        for k in range(i + 1, n):
            factor = M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


def numerical_jacobian(F, x, a, h=1e-6): # матр Якоби для системы размера n
    n = len(x)
    J = [[0.0] * n for _ in range(n)]
    for j in range(n):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[j] += h
        x_minus[j] -= h
        F_plus = F(x_plus, a)
        F_minus = F(x_minus, a)
        for i in range(n):
            if math.isnan(F_plus[i]) or math.isnan(F_minus[i]):
                J[i][j] = 0.0
            else:
                J[i][j] = (F_plus[i] - F_minus[i]) / (2.0 * h)
    return J


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def simple_iteration(G, x0, a, eps, max_iter=100):
    x = x0[:]
    rows = []
    for k in range(1, max_iter + 1):
        x_new = G(x, a)
        if any(math.isnan(v) or math.isinf(v) for v in x_new):
            break
        delta = norm([x_new[i] - x[i] for i in range(len(x))])
        rows.append({"iter": k, "x": x_new[:], "delta": delta})
        if delta < eps:
            break
        x = x_new
    return rows


def newton_method(F, x0, a, eps, max_iter=100):
    x = x0[:]
    rows = []
    for k in range(1, max_iter + 1):
        Fx = F(x, a)
        if any(math.isnan(v) for v in Fx):
            break
        J = numerical_jacobian(F, x, a)
        b = [-val for val in Fx]
        try:
            delta = solve_linear_system(J, b)
        except ValueError:
            break
        x_new = [x[i] + delta[i] for i in range(len(x))]
        F_new = F(x_new, a)
        norm_delta = norm(delta)
        rows.append({"iter": k, "x": x_new[:], "f": F_new[:], "delta": norm_delta})
        if norm_delta < eps or norm(F_new) < eps:
            break
        x = x_new
    return rows


def save_csv(path, rows):
    if not rows: return
    n = len(rows[0]["x"])
    keys = ["iter"] + [f"x{i + 1}" for i in range(n)] + ["delta"]
    if "f" in rows[0]:
        keys += [f"f{i + 1}" for i in range(n)]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            row_dict = {"iter": r["iter"], "delta": r["delta"]}
            for i in range(n):
                row_dict[f"x{i + 1}"] = r["x"][i]
                if "f" in r:
                    row_dict[f"f{i + 1}"] = r["f"][i]
            writer.writerow(row_dict)


def save_summary(path, cfg, simple_rows, newton_rows, g_desc):
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"Система f: {cfg['f']}\n")
        file.write(f"a = {cfg['a']}, eps = {cfg['eps']}\n")
        file.write(f"x0 = {cfg['x0']}\n\n")
        file.write(f"Функция перехода G: {g_desc}\n\n")
        file.write("Метод простой итерации:\n")
        for r in simple_rows:
            file.write(f"{r['iter']:4d}  x = {['%.10f' % v for v in r['x']]}  delta = {r['delta']:.6e}\n")
        file.write("\nМетод Ньютона:\n")
        for r in newton_rows:
            file.write(f"{r['iter']:4d}  x = {['%.10f' % v for v in r['x']]}  delta = {r['delta']:.6e}\n")


def plot_convergence(path, simple_rows, newton_rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    if simple_rows:
        it_s = [r["iter"] for r in simple_rows]
        d_s = [max(r["delta"], 1e-16) for r in simple_rows]
        ax.semilogy(it_s, d_s, "o-", label="Простая итерация")
    if newton_rows:
        it_n = [r["iter"] for r in newton_rows]
        d_n = [max(r["delta"], 1e-16) for r in newton_rows]
        ax.semilogy(it_n, d_n, "s-", label="Ньютон")
    ax.set_xlabel("Номер итерации")
    ax.set_ylabel("||x_n - x_{n-1}||")
    ax.set_title("Зависимость погрешности от количества итераций")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    config_file = "input_system.json"
    if not os.path.exists(config_file):
        example = [
            {
                "f": ["(x1^2 + a^2)*x2 - a^3", "(x1 - a/2)^2 + (x2 - a/2)^2 - a^2"],
                "phi": ["a/2 + sqrt(a^2 - (x2 - a/2)^2)", "a^3 / (x1^2 + a^2)"],
                "a": 2.0,
                "eps": 1e-6,
                "x0": [1.0, 1.0]
            }
        ]
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(example, f, indent=2, ensure_ascii=False)
        print(f"Создан пример файла {config_file}. Отредактируйте его и запустите снова.")
        return
    with open(config_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    for idx, cfg in enumerate(configs, start=1):
        print(f"\n- Решение системы {idx} -")
        f_exprs = cfg["f"]
        phi_exprs = cfg.get("phi", [])
        a = float(cfg.get("a", 0.0))
        eps = float(cfg["eps"])
        x0 = [float(v) for v in cfg["x0"]]
        # Формируем F(x, a)
        f_funcs = [make_func(expr) for expr in f_exprs]
        def F(x, a_val):
            return [fn(x, a_val) for fn in f_funcs]
        # Формируем G(x, a)
        if phi_exprs and len(phi_exprs) == len(f_exprs):
            g_funcs = [make_func(expr) for expr in phi_exprs]
            def G(x, a_val):
                return [fn(x, a_val) for fn in g_funcs]
            g_desc = "Задана пользователем"
        else:
            # Если phi не задана, строим G(x) = x - J(x0)^(-1) * F(x)
            J0 = numerical_jacobian(F, x0, a)
            def G(x, a_val):
                Fx = F(x, a_val)
                try:
                    delta = solve_linear_system(J0, Fx)
                    return [x[i] - delta[i] for i in range(len(x))]
                except ValueError:
                    return [float('nan')] * len(x)
            g_desc = "Автоматически (x - J(x0)^(-1) * F(x))"
        try: # Простая итерация
            simple_rows = simple_iteration(G, x0, a, eps)
            if simple_rows:
                print(
                    f"  Простая итерация: корень = {['%.8f' % v for v in simple_rows[-1]['x']]}, итераций: {len(simple_rows)}")
            else:
                print("  Простая итерация: не сошлась")
        except Exception as e:
            print(f"  Ошибка в простой итерации: {e}")
            simple_rows = []
        try: # Ньютон
            newton_rows = newton_method(F, x0, a, eps)
            if newton_rows:
                print(f"  Ньютон: корень = {['%.8f' % v for v in newton_rows[-1]['x']]}, итераций: {len(newton_rows)}")
            else:
                print("  Ньютон: не сошёлся")
        except Exception as e:
            print(f"  Ошибка в методе Ньютона: {e}")
            newton_rows = []
        outdir = f"results/sys_{idx}" # Сохр результатов
        os.makedirs(outdir, exist_ok=True)
        save_csv(os.path.join(outdir, "simple_iteration.csv"), simple_rows)
        save_csv(os.path.join(outdir, "newton.csv"), newton_rows)
        save_summary(os.path.join(outdir, "summary.txt"), cfg, simple_rows, newton_rows, g_desc)
        plot_convergence(os.path.join(outdir, "convergence.png"), simple_rows, newton_rows)
        print(f"  Результаты сохранены в {outdir}")


if __name__ == "__main__":
    main()