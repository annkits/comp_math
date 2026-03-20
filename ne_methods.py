import math

def read_from_file(filename="input2.txt"):
    with open(filename, 'r', encoding='utf-8') as f:
        a = float(f.readline().strip())
        b = float(f.readline().strip())
        func_str = f.readline().strip()
        def f(x):
            expr = func_str.replace('^', '**')
            return eval(expr, {"__builtins__": {}}, {"x": x, "sin": __import__("math").sin,
                                                     "cos": __import__("math").cos,
                                                     "exp": __import__("math").exp,
                                                     "sqrt": __import__("math").sqrt})
    return a, b, f

def bisection(f, a, b, eps):
    if f(a) * f(b) >= 0:
        raise ValueError("Функция не меняет знак на интервале")
    steps = 0
    while (b - a) / 2 > eps:
        c = (a + b) / 2
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        steps += 1
    return (a + b) / 2, steps

def chord_method(f, a, b, eps):
    if f(a) * f(b) >= 0:
        raise ValueError("Функция не меняет знак на интервале")
    old_c = a
    steps = 0
    for i in range(1, 1000):
        c = (f(b) * a - f(a) * b) / (f(b) - f(a))
        if abs(c - old_c) < eps:
            return c, i
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

        old_c = c
        steps = i
    return (a + b) / 2, steps

def newton(f, a, b, eps):
    if f(a) * f(b) >= 0:
        raise ValueError("Функция не меняет знак на интервале")

    h = (b - a) / 100.0
    fa = f(a)
    fah = f(a + h)
    fa2h = f(a + 2 * h)
    delta2_fa = fa2h - 2 * fah + fa
    sign_fa = math.copysign(1, fa)
    sign_delta2 = math.copysign(1, delta2_fa)

    if sign_fa == sign_delta2:
        x = a
    else:
        x = b

    steps = 0
    max_iter = 100
    while steps < max_iter:
        fx = f(x)
        if abs(fx) < eps:
            return x, steps
        dfx = (f(x + h) - f(x - h)) / (2 * h)
        if abs(dfx) < 1e-10:
            raise ValueError("Производная близка к нулю")
        x_new = x - fx / dfx
        steps += 1
        if abs(x_new - x) < eps:
            return x_new, steps
        x = x_new
    raise ValueError("Метод не сошелся за макс кол-во итераций")

if __name__ == "__main__":
    eps = float(input("Введите точность eps: "))

    a, b, f = read_from_file()

    solutionBM, stepsBM = bisection(f, a, b, eps)
    solutionCHM, stepsCHM = chord_method(f, a, b, eps)
    solutionNewton, stepsNewton = newton(f, a, b, eps)

    print("Приближенный корень, метод половинного деления: ", solutionBM)
    print("Количество шагов, метод половинного деления: ", stepsBM)
    print("Приближенный корень, метод хорд: ", solutionCHM)
    print("Количество шагов, метод хорд: ", stepsCHM)
    print("Приближенный корень, метод хорд: ", solutionNewton)
    print("Количество шагов, метод хорд: ", stepsNewton)