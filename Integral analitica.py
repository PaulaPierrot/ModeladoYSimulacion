import sympy as sp

def resolver_integral():
    x = sp.symbols('x')

    # 1. Ingreso de función
    expr_str = input("Ingresá la función en términos de x: ")
    expr = sp.sympify(expr_str)

    print("\n=== Función ingresada ===")
    sp.pprint(expr)

    print("\n=== Integral indefinida paso a paso ===")
    print("1) Planteamos la integral:")
    sp.pprint(sp.Integral(expr, x))

    # 2. Integral indefinida
    indef = sp.integrate(expr, x)
    print("\n2) Resolvemos la integral indefinida:")
    sp.pprint(indef + sp.Symbol("C"))

    # 3. Preguntar si quiere evaluar
    opcion = input("\n¿Querés evaluarla en algún punto o intervalo? (n = no, p = punto, d = definida) [n/p/d]: ").strip().lower()

    if opcion == "p":
        valor = float(input("Ingresá el valor de x: "))
        resultado = indef.subs(x, valor)
        print(f"\nIntegral indefinida evaluada en x = {valor}:")
        sp.pprint(resultado)

    elif opcion == "d":
        a = float(input("Límite inferior a = "))
        b = float(input("Límite superior b = "))

        print("\n=== Integral definida paso a paso ===")
        print("1) Planteamos la integral definida:")
        sp.pprint(sp.Integral(expr, (x, a, b)))

        print("\n2) Tomamos la integral indefinida encontrada antes:")
        sp.pprint(indef)

        parte_superior = indef.subs(x, b)
        parte_inferior = indef.subs(x, a)

        print(f"\n3) Evaluamos en los límites: F({b}) - F({a})")
        sp.pprint(parte_superior - parte_inferior)

        print("\n4) Resultado final exacto:")
        resultado_exact = sp.integrate(expr, (x, a, b))
        sp.pprint(resultado_exact)

        print("\n5) Resultado aproximado (decimal):")
        print((resultado_exact))

    else:
        print("\nListo. Te dejo la integral indefinida con constante:")
        sp.pprint(indef + sp.Symbol("C"))


# -----------------------------
if __name__ == "__main__":
    resolver_integral()
