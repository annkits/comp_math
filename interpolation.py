from math import factorial
import matplotlib.pyplot as plt
import numpy as np

import math


def read_data(filename="input_interpolation.txt"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

        if len(data_lines) < 5:
            raise ValueError("Недостаточно данных в файле")

        n        = int(data_lines[0])
        x0       = float(data_lines[1])
        h        = float(data_lines[2])
        x_target = float(data_lines[3])
        func_str = data_lines[4]

        x_nodes = [x0 + i * h for i in range(n + 1)]

        def target_func(x):
            safe_dict = {
                "math": math,
                "sin": math.sin, "cos": math.cos,
                "exp": math.exp, "sqrt": math.sqrt,
                "x": x, "pi": math.pi, "e": math.e,
                "log": math.log, "log10": math.log10,
            }
            expr = func_str.replace('^', '**')
            return eval(expr, {"__builtins__": None}, safe_dict)

        y_nodes = [target_func(xi) for xi in x_nodes]

        return n, x_nodes, y_nodes, x_target, target_func, func_str

    except Exception as e:
        print(f"Ошибка чтения/вычисления: {e}")
        n_def = 10
        x0_def, h_def = 1.0, 1.0
        x_nodes = [x0_def + i * h_def for i in range(n_def + 1)]
        y_nodes = [math.sqrt(x) for x in x_nodes]
        return n_def, x_nodes, y_nodes, 2.5, lambda x: math.sqrt(x), "sqrt(x)"

def get_differences(y_nodes, verbose=False):
    n = len(y_nodes)
    diffs = [list(y_nodes)]

    if verbose:
        print("\n--- Таблица конечных разностей ---")
        print(f"Δ⁰y: {y_nodes}")

    for k in range(1, n):
        prev_layer = diffs[k - 1]
        current_layer = [prev_layer[i + 1] - prev_layer[i] for i in range(len(prev_layer) - 1)]
        diffs.append(current_layer)

        if verbose:
            print(f"Δ^{k}y: {[round(d, 6) for d in current_layer]}")
    return diffs

def newton_interpolation1(x_nodes, y_nodes, x, verbose=False):
    n = len(x_nodes)
    if n < 2: return y_nodes[0]

    h = x_nodes[1] - x_nodes[0]
    q = (x - x_nodes[0]) / h

    if verbose:
        print(f"\n--- Расчет по 1-й формуле Ньютона (q = {q:.4f}) ---")

    diff_table = get_differences(y_nodes, verbose=False)

    result = y_nodes[0]
    if verbose:
        print(f"Шаг 0: P = {result:.6f}")

    q_product = 1.0

    for k in range(1, n):
        q_product *= (q - (k - 1))
        term = (diff_table[k][0] / factorial(k)) * q_product
        result += term

        if verbose:
            print(f"Шаг {k}: добавлено {term:.6f} | Текущее P = {result:.6f}")

    return result

def newton_interpolation2(x_nodes, y_nodes, x, verbose=False):
    n = len(x_nodes)
    if n < 2: return y_nodes[0]

    h = x_nodes[1] - x_nodes[0]
    q = (x - x_nodes[-1]) / h

    if verbose:
        print(f"\n--- Расчет по 2-й формуле Ньютона (q = {q:.4f}) ---")

    diff_table = get_differences(y_nodes, verbose=False)

    result = y_nodes[-1]
    if verbose:
        print(f"Шаг 0: P = {result:.6f}")

    q_product = 1.0

    for k in range(1, n):
        q_product *= (q + (k - 1))
        term = (diff_table[k][-1] / factorial(k)) * q_product
        result += term

        if verbose:
            print(f"Шаг {k}: добавлено {term:.6f} | Текущее P = {result:.6f}")

    return result

def lagrange_interpolation(x_nodes, y_nodes, x, verbose=False):
    n = len(x_nodes)
    result = 0.0

    if verbose:
        print(f"\n--- Расчет Лагранжа для x = {x} ---")

    for i in range(n):
        g_i = 1.0
        for j in range(n):
            if i != j:
                g_i *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])

        term = y_nodes[i] * g_i
        result += term

        if verbose:
            print(f"  Шаг {i}: g_{i} = {g_i:.4f}, Слагаемое = {term:.4f}")

    return result


def aitken_interpolation(x_nodes, y_nodes, x):
    n = len(x_nodes)
    p = [y_nodes[i] for i in range(n)]

    print("\n=== Таблица схемы Эйткена ===")
    for j in range(1, n):
        for i in range(n - j):
            chisl = p[i] * (x - x_nodes[i + j]) - p[i + 1] * (x - x_nodes[i])
            znam = x_nodes[i]- x_nodes[i + j]
            p[i] = chisl / znam
            print(f"P_{i},{i + j} = {p[i]:.6f}")

    return p[0]


def plot_interpolation(x_nodes, y_nodes, x_target, res_l, res_n1, res_n2, target_func):
    x_min, x_max = min(x_nodes), max(x_nodes)
    data_range = x_max - x_min

    ext_margin = data_range
    # x_fine = np.linspace(x_min - ext_margin, x_max + ext_margin, 500)
    x_fine = np.linspace(0.1, x_max + 1.5, 1000)

    y_true = [target_func(xi) for xi in x_fine]
    y_poly = [lagrange_interpolation(x_nodes, y_nodes, xi, verbose=False) for xi in x_fine]

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.plot(x_fine, y_true, 'g--', label='f(x)', alpha=0.4)
    plt.plot(x_fine, y_poly, 'b-', label='Интерполянт', linewidth=2)

    plt.scatter(x_nodes, y_nodes, color='red', zorder=5, label='Узлы')
    plt.scatter([x_target], [res_l], color='blue', marker='o', s=100,
                zorder=6, label=f'L(x): {res_l:.4f}')
    plt.scatter([x_target], [res_n1], color='orange', marker='s', s=80,
                zorder=7, label=f'N1(x): {res_n1:.4f}')
    plt.scatter([x_target], [res_n2], color='purple', marker='^', s=80,
                zorder=7, label=f'N2(x): {res_n2:.4f}')

    plot_margin = data_range * 0.15
    plt.xlim(x_min - plot_margin, x_max + plot_margin)

    y_min, y_max = min(y_nodes), max(y_nodes)
    y_range = y_max - y_min if y_max != y_min else 1.0
    plt.ylim(y_min - y_range * 0.15, y_max + y_range * 0.15)

    plt.title('График интерполяции')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def main():
    n, x_nodes, y_nodes, x_target, target_func, func_name = read_data()

    print(f"Функция: f(x) = {func_name}")
    print(f"Узлы интерполяции: {list(zip(x_nodes, y_nodes))}")
    print(f"Точка для расчета: x = {x_target}")

    res_lagrange = lagrange_interpolation(x_nodes, y_nodes, x_target, verbose=True)
    print(f"\nРезультат (Лагранж): {res_lagrange:.6f}")

    res_aitken = aitken_interpolation(x_nodes, y_nodes, x_target)
    print(f"\nРезультат (Эйткен): {res_aitken:.6f}")

    get_differences(y_nodes, verbose=True)

    res_newton1 = newton_interpolation1(x_nodes, y_nodes, x_target, verbose=True)
    print(f"\nРезультат (Ньютон, 1-я формула): {res_newton1:.6f}")

    res_newton2 = newton_interpolation2(x_nodes, y_nodes, x_target, verbose=True)
    print(f"\nРезультат (Ньютон, 2-я формула): {res_newton2:.6f}")

    y_true_val = target_func(x_target)
    print(f"\nИстинное значение f({x_target}) = {y_true_val:.6f}")

    plot_interpolation(x_nodes, y_nodes, x_target, res_lagrange, res_newton1, res_newton2, target_func)


if __name__ == "__main__":
    main()
