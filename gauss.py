def print_matrix(M, label=""):
    if label:
        print(f"\n{label}:")
    for row in M:
        print("  " + " ".join(f"{x:10.5f}" for x in row))
    print()


def input_matrix(n, name="A"):
    print(f"\nВведите матрицу {name} ({n}×{n}):")
    matrix = []
    for i in range(n):
        while True:
            try:
                row_str = input(f"  Строка {i + 1}: ").strip()
                row = [float(x) for x in row_str.split()]
                if len(row) != n:
                    print(f"Ошибка: нужно ровно {n} чисел")
                    continue
                matrix.append(row)
                break
            except ValueError:
                print("Ошибка: вводите числа через пробел")
    return matrix


def input_vector(n, name="b"):
    print(f"\nВведите вектор {name} (размер {n}):")
    while True:
        try:
            vec_str = input("  → ").strip()
            vec = [float(x) for x in vec_str.split()]
            if len(vec) != n:
                print(f"Ошибка: нужно ровно {n} чисел")
                continue
            return vec
        except ValueError:
            print("Ошибка: вводите числа через пробел")


def gauss_classic(A, b, eps=1e-10):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    print_matrix(M, "Исходная расширенная матрица (обычный Гаусс)")

    for i in range(n):
        if abs(M[i][i]) < eps:
            return None, f"Нулевой ведущий элемент на позиции ({i + 1},{i + 1})"

        for k in range(i + 1, n):
            if abs(M[i][i]) < eps:
                return None, "Деление на очень малый элемент"
            factor = M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]

    print_matrix(M, "После прямого хода (обычный Гаусс)")

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        if abs(M[i][i]) < eps:
            return None, "Деление на ноль в обратном ходе"
        x[i] = s / M[i][i]

    return x, None


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

    return x, None



if __name__ == "__main__":
    print("Программа: Метод Гаусса (обычный и с выбором ведущего элемента)")
    print("-------------------------------------------------------------\n")

    while True:
        try:
            n = int(input("Введите размер системы (n): "))
            if n < 1:
                print("n должно быть положительным числом")
                continue
            break
        except ValueError:
            print("Введите целое число")

    A = input_matrix(n, "A")
    b = input_vector(n, "b")

    print("\n" + "=" * 60 + "\n")

    print("=== 1. Обычный метод Гаусса ===")
    x1, err1 = gauss_classic(A, b)
    if err1:
        print("Ошибка:", err1)
    else:
        print("Решение:", [round(v, 6) for v in x1])

    print("\n" + "-" * 60 + "\n")

    print("=== 2. Метод Гаусса с выбором ведущего элемента ===")
    x2, err2 = gauss_pivot(A, b)
    if err2:
        print("Ошибка:", err2)
    else:
        print("Решение:", [round(v, 6) for v in x2])

    print("\n" + "=" * 60)
    input("\nНажмите Enter для выхода...")