import sympy as sp

def derivar_funcion():
    # Definimos la variable simbólica
    x = sp.Symbol('x')
    
    # Pedimos la función
    expresion = input("Ingresá la función en x: ")
    
    # Convertimos la cadena en una expresión simbólica
    funcion = sp.sympify(expresion)
    
    # Calculamos la derivada
    derivada = sp.diff(funcion, x)
    
    print(f"\nFunción: {funcion}")
    print(f"Derivada: {derivada}")
    derivada_simplificada = sp.simplify(derivada)
    print(derivada_simplificada)
    
    # Preguntar si quiere evaluarla en un punto
    resp = input("\n¿Querés evaluarla en un punto? (s/n): ").lower()
    if resp == 's':
        valor = float(input("Ingresá el valor de x: "))
        resultado = derivada.subs(x, valor)
        print(f"Derivada en x={valor} → {resultado}")

# Ejemplo de uso
derivar_funcion()


