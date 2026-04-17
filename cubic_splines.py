import math
import matplotlib.pyplot as plt


def read_data(filename="input_spline.txt"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

        if len(data_lines) < 5:
            raise ValueError("Недостаточно данных в файле (нужно: n, x0, h, targets, func)")

        n_intervals = int(data_lines[0])
        x0 = float(data_lines[1])
        h_step = float(data_lines[2])
        x_targets = list(map(float, data_lines[3].split()))
        func_str = data_lines[4]

        def target_func(x):
            safe_dict = {
                "math": math,
                "sin": math.sin, "cos": math.cos,
                "exp": math.exp, "sqrt": math.sqrt,
                "x": x, "pi": math.pi, "e": math.e,
                "log": math.log,
            }
            expr = func_str.replace('^', '**')
            return eval(expr, {"__builtins__": None}, safe_dict)

        x_nodes = [x0 + i * h_step for i in range(n_intervals + 1)]
        y_nodes = [target_func(xi) for xi in x_nodes]

        return x_nodes, y_nodes, x_targets, target_func, func_str

    except Exception as e:
        print(f"Ошибка чтения: {e}")
        x_nodes = [0.0, 1.57, 3.14, 4.71, 6.28]
        y_nodes = [0.0, 1.0, 0.0, -1.0, 0.0]
        return x_nodes, y_nodes, [1.0, 2.0], math.sin, "sin(x)"


def solve_gauss(A, b):
    n = len(b)
    M = [A[i] + [b[i]] for i in range(n)]
    for i in range(n):
        max_el = abs(M[i][i]);
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_el:
                max_el = abs(M[k][i]);
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]
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


def get_spline_moments(x, y):
    n = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(n)]
    size = n - 1
    A = [[0.0] * size for _ in range(size)]
    B = [0.0] * size
    for i in range(size):
        A[i][i] = (h[i] + h[i + 1]) / 3
        if i > 0: A[i][i - 1] = h[i] / 6
        if i < size - 1: A[i][i + 1] = h[i + 1] / 6
        B[i] = (y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i]
    m_internal = solve_gauss(A, B)
    return [0.0] + m_internal + [0.0]


def spline_calc(x_nodes, y_nodes, M, x):
    n = len(x_nodes)
    if x <= x_nodes[0]: return y_nodes[0]
    if x >= x_nodes[-1]: return y_nodes[-1]
    i = 1
    while i < n and x > x_nodes[i]: i += 1
    h = x_nodes[i] - x_nodes[i - 1]
    t1 = M[i - 1] * ((x_nodes[i] - x) ** 3) / (6 * h)
    t2 = M[i] * ((x - x_nodes[i - 1]) ** 3) / (6 * h)
    t3 = (y_nodes[i - 1] - M[i - 1] * h ** 2 / 6) * (x_nodes[i] - x) / h
    t4 = (y_nodes[i] - M[i] * h ** 2 / 6) * (x - x_nodes[i - 1]) / h
    return t1 + t2 + t3 + t4


def main():
    x_nodes, y_nodes, x_targets, target_func, func_name = read_data()

    M = get_spline_moments(x_nodes, y_nodes)

    print(f"=== Сплайн-интерполяция функции: f(x) = {func_name} ===")
    print(f"Узлы интерполяции: {list(zip([round(x, 2) for x in x_nodes], [round(y, 4) for y in y_nodes]))}\n")

    results_y = []
    print("Результаты расчетов:")
    for x in x_targets:
        y_spline = spline_calc(x_nodes, y_nodes, M, x)
        y_exact = target_func(x)
        results_y.append(y_spline)
        error = abs(y_exact - y_spline)
        print(f"x = {x:6.2f} | S(x) = {y_spline:10.6f} | f(x) = {y_exact:10.6f} | Δ = {error:.2e}")

    steps = 200
    x_plot = [x_nodes[0] + i * (x_nodes[-1] - x_nodes[0]) / steps for i in range(steps + 1)]
    y_spline_plot = [spline_calc(x_nodes, y_nodes, M, xi) for xi in x_plot]
    y_exact_plot = [target_func(xi) for xi in x_plot]

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_exact_plot, 'g--', label=f'Исходная функция {func_name}', alpha=0.5)
    plt.plot(x_plot, y_spline_plot, 'b-', label='Кубический сплайн', linewidth=1.5)
    plt.scatter(x_nodes, y_nodes, color='red', label='Узлы сетки', zorder=5)
    plt.scatter(x_targets, results_y, color='orange', marker='D', label='Рассчитанные точки', zorder=6)

    plt.title(f'Интерполяция функции {func_name} сплайнами')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()