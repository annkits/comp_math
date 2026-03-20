import math

def read_from_file(filename="input.txt"):
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        A = []
        for _ in range(n):
            row = list(map(float, f.readline().strip().split()))
            A.append(row)
        b = list(map(float, f.readline().strip().split()))
    return n, A, b

def print_matrix(M, label=""):
    if label:
        print(f"\n{label}:")
    if isinstance(M[0], (list, tuple)):
        for row in M:
            print("  " + " ".join(f"{x:10.5f}" for x in row))
    else:
        print("  " + " ".join(f"{x:10.5f}" for x in M))
    print()

def transform_to_C(A, b):
    n = len(b)
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    b_norm = [0.0] * n
    for i in range(n):
        if abs(A[i][i]) < 1e-12:
            raise ValueError("Диагональный элемент не должен быть около нулевым")
        b_norm[i] = b[i] / A[i][i]
        for j in range(n):
            if i != j:
                C[i][j] = -A[i][j] / A[i][i]
    return C, b_norm

def multiply(C, x):
    n = len(C)
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += C[i][j] * x[j]
    return result

def substraction(Cx, b_norm):
    n = len(b_norm)
    result = [0.0] * n
    for i in range(n):
        result[i] = b_norm[i] + Cx[i]
    return result

def simple_iteration(A, b, eps, max_iter=100):
    n = len(b)
    x = [0.0] * n
    C, b_norm = transform_to_C(A, b)
    calc_N(C, b_norm, eps, n)

    for i in range(1, max_iter + 1):
        Cx = multiply(C, x)
        new_x = substraction(Cx, b_norm)

        if len(new_x) != n:
            print("Ошибка: new_x имеет длину", len(new_x), "вместо", n)
            return None

        print(f"Итерация {i} (простая итерация):")
        print_matrix(new_x)

        max_diff = 0.0
        for j in range(n):
            diff = abs(new_x[j] - x[j])
            if max_diff < diff:
                max_diff = diff
        if max_diff < eps:
            print(f"Сошлось за {i} итераций")
            return new_x
        x = new_x
    print(f"Не сошлось за {max_iter} итераций")
    return x

def seidel(A, b, eps, max_iter=100):
    n = len(b)
    x = [0.0] * n
    C, b_norm = transform_to_C(A, b)
    calc_N(C, b_norm, eps, n)

    for k in range(1, max_iter + 1):
        x_old = x[:]

        for i in range(n):
            s = b_norm[i]
            for j in range(n):
                if j < i:
                    s += C[i][j] * x[j]
                elif j > i:
                    s += C[i][j] * x_old[j]
            x[i] = s

        print(f"Итерация {k} (Зейдель):")
        print_matrix(x)

        max_diff = 0.0
        for j in range(n):
            diff = abs(x[j] - x_old[j])
            if max_diff < diff:
                max_diff = diff
        if max_diff < eps:
            print(f"Сошлось за {k} итераций")
            return x

    print(f"Не сошлось за {max_iter} итераций")
    return x

def calc_N(C, b_norm, eps, n):
    norm_C = 0.0
# for i in range (n-1, -1, -1)
    for i in range(n):
        row_sum = sum(abs(C[i][j]) for j in range(n))
        norm_C = max(norm_C, row_sum)
    norm_b = max(abs(bi) for bi in b_norm)
    if norm_C >= 1:
        return "Метод не сходится (||C|| >= 1)"
    argument = (1 - norm_C) * eps / norm_b
    log_term = math.log(argument) / math.log(norm_C)
    N = math.ceil(log_term)
    print("Оценочное число итераций N: ", N)
    return N

if __name__ == "__main__":
    eps = float(input("Введите точность eps: "))

    n, A, b = read_from_file()

    solutionSI = simple_iteration(A, b, eps)
    solutionS = seidel(A, b, eps)
    print_matrix(solutionSI)
    print_matrix(solutionS)