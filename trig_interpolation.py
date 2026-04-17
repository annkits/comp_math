import math
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename="input_trig_interpolation.txt"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

        if len(data_lines) < 5:
            raise ValueError("Недостаточно данных в файле")

        n = int(data_lines[0])
        x0 = float(data_lines[1])
        h = float(data_lines[2])
        x_target = float(data_lines[3])
        func_str = data_lines[4]

        x_nodes = [x0 + i * h for i in range(n)]

        def target_func(x):
            safe_dict = {
                "math": math, "sin": math.sin, "cos": math.cos,
                "exp": math.exp, "sqrt": math.sqrt,
                "x": x, "pi": math.pi, "e": math.e,
                "log": math.log, "log10": math.log10,
            }
            expr = func_str.replace('^', '**')
            return eval(expr, {"__builtins__": None}, safe_dict)

        y_nodes = [target_func(xi) for xi in x_nodes]

        return n, x_nodes, y_nodes, x_target, target_func, func_str

    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        print("Используются значения по умолчанию: y = [1, 0, 1, 4] на x = [1, 3, 5, 7]\n")

        n_def = 4
        x_nodes_def = [1.0, 3.0, 5.0, 7.0]
        y_nodes_def = [1.0, 0.0, 1.0, 4.0]
        x_target_def = 2.0
        func_str_def = "example (y = [1,0,1,4])"

        def target_func_def(x):
            return 0

        return n_def, x_nodes_def, y_nodes_def, x_target_def, target_func_def, func_str_def


def trigonometric_interpolation(x_nodes, y_nodes, verbose=True):
    n = len(x_nodes)

    x0 = x_nodes[0]
    h = x_nodes[1] - x_nodes[0]
    T = n * h
    omega = 2 * math.pi / T

    if verbose:
        print(f"\n=== Тригонометрическая интерполяция ===")
        print(f"Узлов: {n}, x0 = {x0}, h = {h}, T = {T}\n")

    A_real = [0.0] * n
    A_imag = [0.0] * n

    for j in range(n):
        sum_re = 0.0
        sum_im = 0.0
        for k in range(n):
            theta = -2 * math.pi * k * j / n
            sum_re += y_nodes[k] * math.cos(theta)
            sum_im += y_nodes[k] * math.sin(theta)

        A_real[j] = sum_re / n
        A_imag[j] = sum_im / n

        j_print = j if j <= n // 2 else j - n
        if verbose:
            print(f"A_{j_print:2d} = {A_real[j]:12.8f} + {A_imag[j]:12.8f}i")

    def y_at(x):
        phi = omega * (x - x0)
        y_re = 0.0
        y_im = 0.0

        for j_idx in range(n):
            jj = j_idx if j_idx <= n // 2 else j_idx - n
            arg = jj * phi
            c = math.cos(arg)
            s = math.sin(arg)

            y_re += A_real[j_idx] * c - A_imag[j_idx] * s
            y_im += A_real[j_idx] * s + A_imag[j_idx] * c

        return y_re, y_im

    return y_at


def plot(x_nodes, y_nodes, y_trig, target_func):
    x_min = min(x_nodes)
    x_max = max(x_nodes)

    left_margin = 0.5
    right_margin = 1.5

    x_fine = np.linspace(x_min - left_margin, x_max + right_margin, 1000)

    y_true = [target_func(xi) for xi in x_fine]
    y_re_list = []
    y_im_list = []

    for xi in x_fine:
        re, im = y_trig(xi)
        y_re_list.append(re)
        y_im_list.append(im)

    plt.figure(figsize=(12, 7))
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.plot(x_fine, y_true, 'g--', linewidth=2, label='Исходная функция f(x) = 1/x')
    plt.plot(x_fine, y_re_list, 'b-', linewidth=2.5, label='Re y(x) — вещественная часть')
    plt.plot(x_fine, y_im_list, 'r-', linewidth=2, label='Im y(x) — мнимая часть')

    plt.scatter(x_nodes, y_nodes, color='red', s=80, zorder=5, label='Узлы')

    plt.xlim(x_min - left_margin, x_max + right_margin)

    all_y = y_re_list + y_im_list + y_true
    y_min = min(all_y)
    y_max = max(all_y)
    y_range = y_max - y_min if y_max != y_min else 1.0
    plt.ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

    plt.title('Тригонометрическая интерполяция\n(вещественная и мнимая части)', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    n, x_nodes, y_nodes, x_target, target_func, func_str = read_data()

    print(f"Функция: f(x) = {func_str}")
    print(f"Узлы интерполяции ({n} точек):")
    for x, y in zip(x_nodes, y_nodes):
        print(f"  x = {x:6.3f}   y = {y:10.6f}")

    y_trig = trigonometric_interpolation(x_nodes, y_nodes, verbose=True)

    print("\n=== Значения тригонометрического интерполянта ===")
    for x in range(1, 8):
        re, im = y_trig(x)
        print(f"x = {x:2d} → Re(y) = {re:12.6f}   Im(y) = {im:12.6f}")

    plot(x_nodes, y_nodes, y_trig, target_func)


if __name__ == "__main__":
    main()