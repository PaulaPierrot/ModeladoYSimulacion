import sympy as sp

def rk4_detallado(f, t0, y0, b, h):
    N = int(round((b - t0) / h))  # número de pasos
    t, y = t0, y0
    resultados = []

    for i in range(N):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h, y + h*k3)

        y_next = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        resultados.append((i, t, y, k1, k2, k3, k4, y_next))

        # avanzar
        t += h
        y = y_next

    # fila final solo con el resultado en t=b
    resultados.append(("Final", t, y, None, None, None, None, y))

    return resultados


def main():
    # Entrada de datos
    expr_str = input("Ingrese f(t,y): ")   # ejemplo: t*y
    t0 = float(input("Ingrese t0: "))
    y0 = float(input("Ingrese y0: "))
    a = float(input("Ingrese a (inicio del intervalo): "))
    b = float(input("Ingrese b (fin del intervalo): "))
    h = float(input("Ingrese h (paso): "))

    t, y = sp.symbols('t y')
    f_expr = sp.sympify(expr_str)
    f = sp.lambdify((t, y), f_expr, 'numpy')

    resultados = rk4_detallado(f, t0, y0, b, h)

    # Imprimir tabla
    print("\nIteraciones con RK4 detallado:\n")
    print(f"{'i':<7}{'t':<10}{'y':<12}{'k1':<12}{'k2':<12}{'k3':<12}{'k4':<12}{'y_next':<12}")
    print("-"*87)
    for fila in resultados:
        i, t, y, k1, k2, k3, k4, y_next = fila
        if i == "Final":
            print(f"{i:<7}{t:<10.4f}{y:<12.6f}{'-':<12}{'-':<12}{'-':<12}{'-':<12}{y_next:<12.6f}")
        else:
            print(f"{i:<7}{t:<10.4f}{y:<12.6f}{k1:<12.6f}{k2:<12.6f}{k3:<12.6f}{k4:<12.6f}{y_next:<12.6f}")


if __name__ == "__main__":
    main()
