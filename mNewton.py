import math

def read_from_file(filename="input_mNewton.txt"):
    with open(filename, 'r', encoding='utf-8') as f:
        n = int(f.readline().strip())
        x0 = list(map(float, f.readline().strip().split()))
        func_strs = [f.readline().strip() for _ in range(n)]

    def F(x):
        safe_globals = {
            "__builtins__": {},
            "sin": math.sin,
            "cos": math.cos,
            "exp": math.exp,
            "sqrt": math.sqrt,
        }
        return [
            eval(fs.replace('^', '**'), safe_globals, {"x": x})
            for fs in func_strs
        ]

    return n, x0[:], F

def jacoby(F, x, h = 1e-8):
    n = len(x)
    J = [[0.0] * n for _ in range(n)]
    x_base = x[:]
    for j in range(n):
        x_plus = x_base[:]
        x_plus[j] += h
        f_plus = F(x_plus)

        x_minus = x_base[:]
        x_minus[j] -= h
        f_minus = F(x_minus)

        for i in range(n):
            J[i][j] = (f_plus[i] - f_minus[i]) / (2 * h)

    return J

def print_matrix(M, label=""):
    if label:
        print(f"\n{label}:")
    for row in M:
        print("  " + " ".join(f"{x:10.5f}" for x in row))
    print()

def gauss_pivot(A, b, eps=1e-10):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    print_matrix(M, "Исходная расширенная матрица (Гаусс с выбором)")

    for i in range(n):
        max_row = i
        max_val = abs(M[i][i])

        for k in range(i + 1, n):
            if abs(M[k][i]) > max_val:
                max_val = abs(M[k][i])
                max_row = k

        if max_val < eps:
            return None, "Матрица вырожденная или почти вырожденная"

        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
            print(f"  → Перестановка строк {i + 1} ↔ {max_row + 1}")

        for k in range(i + 1, n):
            factor = M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]

    print_matrix(M, "После прямого хода (Гаусс с выбором)")

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]

    return x

def substraction(A, B):
    n_size = len(B)
    result = [0.0] * n_size
    for i in range(n_size):
        result[i] = A[i] - B[i]
    return result

def multiply(A, B):
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += A[i][j] * B[j]
    return result

def matrix_inverse(J, eps=1e-10):
    n = len(J)
    inv = [[0.0] * n for _ in range(n)]
    for col in range(n):
        e = [1.0 if i == col else 0.0 for i in range(n)]
        col_vec = gauss_pivot(J, e, eps)
        for row in range(n):
            inv[row][col] = col_vec[row]
    return inv

def newtonSLAU(F, x0, eps, max_iter = 30):
    x = x0[:]
    n = len(x)

    print("=== Многомерный метод Ньютона (через СЛАУ) ===")
    for i in range(1, max_iter+1):
        Fx = F(x)

        if max(abs(fi) for fi in Fx) < eps:
            print(f"Сошлось по ||F|| < eps за {i} итераций")
            return x, i

        J = jacoby(F, x)
        Y = gauss_pivot(J, Fx, eps)
        x_new = substraction(x, Y)

        max_diff = 0
        for j in range(len(x)):
            diff = abs(x_new[j] - x[j])
            if diff > max_diff:
                max_diff = diff

        print(f"Итерация {i}:")
        print("  x =", [f"{v:.8f}" for v in x_new])
        print(f"  ||Δx|| = {max_diff:.2e}")

        if max_diff < eps:
            print(f"Сошлось за {i} итераций (по ||Δx|| < eps)")
            return x_new, i

        x = x_new

    print(f"Не сошлось за {max_iter} итераций")
    return x, max_iter

def newtonINVERSE(F, x0, eps, max_iter = 30):
    x = x0[:]
    n = len(x)

    print("\n=== Многомерный метод Ньютона (через обратную матрицу) ===")
    for i in range(1, max_iter+1):
        Fx = F(x)

        if max(abs(fi) for fi in Fx) < eps:
            print(f"Сошлось по ||F|| < eps за {i} итераций")
            return x, i

        J = jacoby(F, x)
        J_inv = matrix_inverse(J)
        Y = multiply(J_inv, Fx)
        x_new = substraction(x, Y)

        max_diff = 0
        for j in range(len(x)):
            diff = abs(x_new[j] - x[j])
            if diff > max_diff:
                max_diff = diff

        print(f"Итерация {i}:")
        print("  x =", [f"{v:.8f}" for v in x_new])
        print(f"  ||Δx|| = {max_diff:.2e}")

        if max_diff < eps:
            print(f"Сошлось за {i} итераций (по ||Δx|| < eps)")
            return x_new, i

        x = x_new

    print(f"Не сошлось за {max_iter} итераций")
    return x, max_iter


if __name__ == "__main__":
    eps = float(input("Введите точность eps: "))

    n, x0, F = read_from_file()

    solutionSLAU, stepsSLAU = newtonSLAU(F, x0, eps)
    solutionINV, stepsINV = newtonINVERSE(F, x0, eps)
    print("Приближенный корень, метод с исп. СЛАУ, кол-во шагов: ", solutionSLAU, stepsSLAU)
    print("Приближенный корень, метод с исп. обр. м-цы, кол-во шагов: ", solutionINV, stepsINV)