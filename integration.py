import math
import numpy as np


def read_data(filename="input_integration.txt"):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

        a = float(data[0])
        b = float(data[1])
        h = float(data[2])
        eps = float(data[3])
        func_str = data[4]

        def f(x):
            safe_dict = {"math": math, "sin": math.sin, "cos": math.cos, "exp": math.exp,
                         "sqrt": math.sqrt, "log": math.log, "x": x, "pi": math.pi, "e": math.e}
            expr = func_str.replace('^', '**')
            return eval(expr, {"__builtins__": None}, safe_dict)

        return a, b, h, eps, f, func_str

    except Exception:
        print("Используются значения по умолчанию: ∫[1, 2] 1/x dx")

        def f_def(x):
            return 1/x

        return 1, 2, 0.1, 1e-4, f_def, "1/x"


def trapezoidal_rule(f, a, b, h):
    n = int((b - a) / h)
    if n < 1: n = 1; h = (b - a)
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h, n


def simpson_rule(f, a, b, h):
    n = int((b - a) / h)
    if n % 2 == 1: n += 1; h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        s += 4 * f(x) if i % 2 == 1 else 2 * f(x)
    return s * h / 3.0, n


def estimate_max_derivative(f, a, b, order=2, points=2000):
    x = np.linspace(a, b, points)
    y = np.array([f(xi) for xi in x])

    if order == 2:
        h = x[1] - x[0]
        d2 = (y[2:] - 2 * y[1:-1] + y[:-2]) / (h ** 2)
        return np.max(np.abs(d2))

    elif order == 4:
        h = x[1] - x[0]
        d4 = (y[4:] - 4 * y[3:-1] + 6 * y[2:-2] - 4 * y[1:-3] + y[:-4]) / (h ** 4)
        return np.max(np.abs(d4))

    else:
        dy = np.diff(y) / np.diff(x)
        return np.max(np.abs(dy)) * 10


def double_refinement_with_errors(f, a, b, h0, eps, method="simpson", max_iter=30):
    method_name = "Симпсона" if method == "simpson" else "трапеций"
    p = 4 if method == "simpson" else 2
    divider = 2**p - 1

    print(f"\n=== Метод {method_name} ===")

    M = estimate_max_derivative(f, a, b, order=4 if method == "simpson" else 2)
    eta = np.finfo(float).eps

    print(f"Оценка M{'₂' if method == 'trapezoidal' else '₄'} ≈ {M:.2e}")

    h = h0
    I_prev = None
    I_h = None
    n = 0

    for k in range(1, max_iter + 1):
        if method == "simpson":
            I_h, n = simpson_rule(f, a, b, h)
        else:
            I_h, n = trapezoidal_rule(f, a, b, h)

        if method == "trapezoidal":
            eps_usec = (h ** 2 * (b - a) * M) / 12
        else:
            eps_usec = (h ** 4 * (b - a) * M) / 180

        eps_okr = (b - a) * eta

        if I_prev is not None:
            delta_run = abs(I_h - I_prev) / divider

            print(f"Итерация {k:2d}: h={h:.6f}  n={n:3d}  I={I_h:.12f}   ε_усеч ≈ {eps_usec:.2e}   ε_окр ≈ {eps_okr:.2e}   Δ ≈ {delta_run:.2e}")

            if delta_run <= eps:
                print(f"Сошлось за {k} итераций\n")
                return I_h, eps_usec, eps_okr, delta_run
        else:
            print(f"Итерация {k:2d}: h={h:.6f}  n={n:3d}  I={I_h:.12f}   ε_усеч ≈ {eps_usec:.2e}   ε_окр ≈ {eps_okr:.2e}")

        I_prev = I_h
        h /= 2.0

    delta_run = abs(I_h - I_prev) / divider
    print(f"Не достигнута требуемая точность за {max_iter} итераций.\n")
    return I_h, eps_usec, eps_okr, delta_run


def main():
    a, b, h, eps, f, func_str = read_data()

    print("=" * 80)
    print(f"Интеграл ∫({func_str})dx по интервалу [{a};{b}] ")
    print(f"Начальный шаг h = {h}, требуемая точность ε = {eps}")
    print("=" * 80)

    try:
        from scipy.integrate import quad
        exact, _ = quad(f, a, b)
        print(f"Точное значение (scipy): {exact:.15f}\n")
    except:
        exact = None

    I_trap, eps_usec_trap, eps_okr_trap, delta_trap = double_refinement_with_errors(f, a, b, h, eps, "trapezoidal")
    I_simp, eps_usec_simp, eps_okr_simp, delta_simp = double_refinement_with_errors(f, a, b, h, eps, "simpson")


    print("=" * 80)
    print(f"{'Метод':<18} {'Значение':<20} {'ε_усеч (теор)':<18} {'ε_окр':<15} {'Δ':<15}")
    print(f"Трапеций          {I_trap:.12f}        {eps_usec_trap:.2e}           {eps_okr_trap:.2e}        {delta_trap:.2e}")
    print(f"Симпсона          {I_simp:.12f}        {eps_usec_simp:.2e}           {eps_okr_simp:.2e}        {delta_simp:.2e}")
    if exact is not None:
        print(f"Точное значение   {exact:.12f}")
    print("=" * 80)


if __name__ == "__main__":
    main()