import math
import sys

class Logger(object):
    def __init__(self, filename="output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def read_data(filename="input_diffs.txt"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]

        m = int(lines[0])
        x0 = float(lines[1])
        x_end = float(lines[2])
        h0 = float(lines[3])
        eps = float(lines[4])

        y0 = list(map(float, lines[5].split()))

        f_str = lines[6]

        if len(y0) != m:
            raise ValueError("Количество начальных условий не совпадает с порядком уравнения")

        return m, x0, x_end, h0, eps, y0, f_str

    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        print("Используются значения по умолчанию: y'' + y = 0, y(0)=1, y'(0)=0, [0, 10]")

        def f_default(x, y):
            return [y[1], -y[0]]  # y'' = -y

        return 2, 0.0, 10.0, 0.5, 1e-5, [1.0, 0.0], "y[1], -y[0]"


def eval_f(x, y, f_str):
    safe_dict = {
        "math": math, "sin": math.sin, "cos": math.cos, "exp": math.exp,
        "sqrt": math.sqrt, "log": math.log, "pi": math.pi, "e": math.e,
        "x": x, "y": y
    }
    for i in range(len(y)):
        safe_dict[f"y{i}"] = y[i]

    expr = f_str.replace('^', '**')
    parts = [p.strip() for p in expr.split(',')]

    f_values = []
    for part in parts:
        val = eval(part, {"__builtins__": {}}, safe_dict)
        f_values.append(float(val))
    return f_values

def euler_step(x, y, h, f_str):
    f = eval_f(x, y, f_str)
    y_new = [yi + h * fi for yi, fi in zip(y, f)]
    return y_new

def euler_method(m, x0, x_end, h, y0, f_str, eps, verbose=True):
    x = x0
    y = y0[:]
    n = 0

    while x < x_end - eps:
        y = euler_step(x, y, h, f_str)
        x += h
        n += 1

        if verbose:
            print(f"x = {x:8.4f}, y = {[round(val, 6) for val in y]}")

    return x, y, n

def runge_kutta_2nd_der_step(x, y, h, f_str):
    # k1 = f(x_i, y_i)
    k1 = eval_f(x, y, f_str)

    y_aster = [yi + h * k1_i for yi, k1_i in zip(y, k1)]

    # k2 = f(x_(i+1), y_aster)
    k2 = eval_f(x + h, y_aster, f_str)

    y_new = [yi + (h / 2.0) * (k1_i + k2_i) for yi, k1_i, k2_i in zip(y, k1, k2)]

    return y_new

def runge_kutta_2nd_der(m, x0, x_end, h, y0, f_str, eps, verbose=True):
    x = x0
    y = y0[:]
    n = 0

    if verbose:
        print(f"{'x':>8} | {'y':>12} | {'y' :>12} | {'h':>8}")
        print("-" * 45)
        print(f"{x:8.4f} | {y[0]:12.8f} | {y[1]:12.8f} | {h:8.4f}")

    while x < x_end - eps:
        y = runge_kutta_2nd_der_step(x, y, h, f_str)
        x += h
        n += 1

        if verbose:
            print(f"{x:8.4f} | {y[0]:12.8f} | {y[1]:12.8f} | {h:8.4f}")

    return x, y, n

def runge_kutta_2nd_time_step(x, y, h, f_str):
    # k1 = f(x_i, y_i)
    k1 = eval_f(x, y, f_str)

    x_mid = x + h / 2.0
    y_mid = [yi + (h / 2.0) * k1_i for yi, k1_i in zip(y, k1)]

    k2 = eval_f(x_mid, y_mid, f_str)

    y_new = [yi + h * k2_i for yi, k2_i in zip(y, k2)]

    return y_new

def runge_kutta_2nd_time(m, x0, x_end, h, y0, f_str, eps, verbose=True):
    x = x0
    y = y0[:]
    n = 0

    if verbose:
        print(f"{'x':>8} | {'y':>12} | {'y' :>12} | {'h':>8}")
        print("-" * 45)
        print(f"{x:8.4f} | {y[0]:12.8f} | {y[1]:12.8f} | {h:8.4f}")

    while x < x_end - eps:
        y = runge_kutta_2nd_time_step(x, y, h, f_str)
        x += h
        n += 1

        if verbose:
            print(f"{x:8.4f} | {y[0]:12.8f} | {y[1]:12.8f} | {h:8.4f}")

    return x, y, n

def runge_kutta_4nd_step(x, y, h, f_str):
    k1 = eval_f(x, y, f_str)

    y_k2 = [yi + 0.5 * h * k1_i for yi, k1_i in zip(y, k1)]
    k2 = eval_f(x + 0.5 * h, y_k2, f_str)

    y_k3 = [yi + 0.5 * h * k2_i for yi, k2_i in zip(y, k2)]
    k3 = eval_f(x + 0.5 * h, y_k3, f_str)

    y_k4 = [yi + h * k3_i for yi, k3_i in zip(y, k3)]
    k4 = eval_f(x + h, y_k4, f_str)

    y_new = [
        yi + (h / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        for yi, k1_i, k2_i, k3_i, k4_i in zip(y, k1, k2, k3, k4)
    ]

    return y_new

def runge_kutta_4nd(m, x0, x_end, h, y0, f_str, eps, verbose=True):
    x = x0
    y = y0[:]
    n = 0

    if verbose:
        print(f"{'x':>8} | {'y':>12} | {'y' :>12} | {'h':>8}")
        print("-" * 45)
        print(f"{x:8.4f} | {y[0]:12.8f} | {y[1]:12.8f} | {h:8.4f}")

    while x < x_end - eps:
        y = runge_kutta_4nd_step(x, y, h, f_str)
        x += h
        n += 1

        if verbose:
            print(f"{x:8.4f} | {y[0]:12.8f} | {y[1]:12.8f} | {h:8.4f}")

    return x, y, n

def double_recalculation(m, x0, x_end, h0, eps, y0, f_str, method="euler", max_iter = 30):
    if method == "euler":
        method_name = "Эйлера"
    elif method == "rk2der":
        method_name = "Рунге-Кутта 2го порядка с уср по производной"
    elif method == "rk2time":
        method_name = "Рунге-Кутта 2го порядка с уср по времени"
    elif method == "rk4":
        method_name = "Рунге-Кутта 4го порядка"

    print(f"\n=== Метод {method_name} ===")

    h = h0
    I_prev = None
    I_h = None
    n = 0

    for k in range(1, max_iter + 1):
        # is_verbose = (k == 1)
        is_verbose = True
        if method == "euler":
            last_x, last_y, n = euler_method(m, x0, x_end, h, y0, f_str, eps, verbose=is_verbose)
            I_h = last_y[0]
        elif method == "rk2der":
            last_x, last_y, n = runge_kutta_2nd_der(m, x0, x_end, h, y0, f_str, eps, verbose=is_verbose)
            I_h = last_y[0]
        elif method == "rk2time":
            last_x, last_y, n = runge_kutta_2nd_time(m, x0, x_end, h, y0, f_str, eps, verbose=is_verbose)
            I_h = last_y[0]
        elif method == "rk4":
            last_x, last_y, n = runge_kutta_4nd(m, x0, x_end, h, y0, f_str, eps, verbose=is_verbose)
            I_h = last_y[0]

        if I_prev is not None:
            delta_run = abs(I_h - I_prev)

            print(f"Итерация {k:2d}: h={h:.8f}  n={n:3d}  I={I_h:.12f}  Δ ≈ {delta_run:.2e}")

            if delta_run <= eps:
                print(f"Сошлось за {k} итераций\n")
                return I_h, delta_run
        else:
            print(f"Итерация {k:2d}: h={h:.8f}  n={n:3d}  I={I_h:.12f}")

        I_prev = I_h
        h /= 2.0

    delta_run = abs(I_h - I_prev)
    print(f"Не достигнута требуемая точность за {max_iter} итераций.\n")
    return I_h, delta_run

def main():
    sys.stdout = Logger("output.txt")

    try:
        m, x0, x_end, h0, eps, y0, f_str = read_data()

        print(f"Порядок уравнения: m = {m}")
        print(f"Интервал: [{x0}, {x_end}], начальный шаг h0 = {h0}, ε = {eps}")
        print(f"Начальные условия: y(0) = {y0}")
        print(f"Правая часть системы: [{f_str}]")

        # I_euler, delta_euler = double_recalculation(m, x0, x_end, h0, eps, y0, f_str, "euler")
        I_rk2der, delta_rk2der = double_recalculation(m, x0, x_end, h0, eps, y0, f_str, "rk2der")
        I_rk2time, delta_rk2time = double_recalculation(m, x0, x_end, h0, eps, y0, f_str, "rk2time")
        I_rk4, delta_rk4 = double_recalculation(m, x0, x_end, h0, eps, y0, f_str, "rk4")

        # print(f"Метод Эйлера: {I_euler}")
        print(f"Метод Рунге-Кутта 2го с уср по производной: {I_rk2der}")
        print(f"Метод Рунге-Кутта 2го с уср по времени: {I_rk2time}")
        print(f"Метод Рунге-Кутта 4го порядка: {I_rk4}")

    finally:
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()