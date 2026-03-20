from math import factorial
import matplotlib.pyplot as plt
import numpy as np

import math


def read_data(filename="input_interpolation.txt"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            n = int(f.readline().strip())
            x_nodes = list(map(float, f.readline().strip().split()))
            y_nodes = list(map(float, f.readline().strip().split()))
            x_target = float(f.readline().strip())
            func_str = f.readline().strip()

        def target_func(x):
            safe_dict = {
                "math": math,
                "sin": math.sin,
                "cos": math.cos,
                "exp": math.exp,
                "sqrt": math.sqrt,
                "x": x
            }
            return eval(func_str.replace('^', '**'), {"__builtins__": None}, safe_dict)

        return n, x_nodes, y_nodes, x_target, target_func, func_str

    except Exception as e:
        print(f"Ошибка чтения: {e}")
        return 4, [1.0, 2.0, 3.0, 4.0], [1.0, 0.5, 0.33, 0.25], 2.5, lambda x: 1 / x

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


def plot_interpolation(x_nodes, y_nodes, x_target, res_lagrange, res_newton1, res_newton2):
    x_fine = np.linspace(min(x_nodes) - 0.5, max(x_nodes) + 0.5, 500)

    y_true = 1 / x_fine
    y_lagrange = [lagrange_interpolation(x_nodes, y_nodes, xi, verbose=False) for xi in x_fine]
    y_newton = [newton_interpolation1(x_nodes, y_nodes, xi, verbose=False) for xi in x_fine]

    plt.figure(figsize=(12, 7))
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.plot(x_fine, y_true, 'g--', label='Истинная функция $f(x)=1/x$', alpha=0.3, linewidth=1)
    plt.plot(x_fine, y_lagrange, 'b-', label='Многочлен Лагранжа', linewidth=2, alpha=0.6)
    plt.plot(x_fine, y_newton, 'r:', label='Многочлен Ньютона (1-я форма)', linewidth=2)

    plt.vlines(x_nodes, 0, y_nodes, colors='gray', linestyles='dashed', alpha=0.3)
    plt.hlines(y_nodes, min(x_nodes) - 0.5, x_nodes, colors='gray', linestyles='dashed', alpha=0.3)

    plt.scatter(x_nodes, y_nodes, color='red', s=50, zorder=5, label='Узлы (данные)')

    plt.scatter([x_target], [res_lagrange], color='blue', marker='o', s=100,
                zorder=6, label=f'L(x): {res_lagrange:.4f}')
    plt.scatter([x_target], [res_newton1], color='orange', marker='s', s=80,
                zorder=7, label=f'N1(x): {res_newton1:.4f}')
    plt.scatter([x_target], [res_newton2], color='purple', marker='^', s=80,
                zorder=7, label=f'N2(x): {res_newton2:.4f}')

    margin = 0.5
    plt.xlim(min(x_nodes) - margin, max(x_nodes) + margin)
    plt.ylim(min(y_nodes) - margin, max(y_nodes) + margin)
    plt.axhline(0, color='black', linewidth=0.8)

    plt.title('Сравнение методов интерполяции (Лагранж vs Ньютон)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best', fontsize='small')

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

    plot_interpolation(x_nodes, y_nodes, x_target, res_lagrange, res_newton1, res_newton2)


if __name__ == "__main__":
    main()