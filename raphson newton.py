def funcion(x):
    return (x**3 - 2*x-5)

def derivada(x):
    return (3*(x**2)-2)

def mostrar_iteracion(anterior,fun,der):
    print("X= ", anterior)
    print("F(x) = ", fun)
    print("F'(x) = ",der)
    print()
    return


def newton_raphson(x):
    error_absoluto = 10000
    while (error_absoluto > 0.001): #aca podes modificar la tolerancia si el profe pide más o menos.
        anterior = x
        fun= funcion(anterior)
        der =derivada(anterior)
        mostrar_iteracion(anterior,fun,der)
        x = anterior - fun/der
        error_absoluto = abs(x-anterior)
        #error_relativo = abs((x-anterior)/anterior *100)
    return x
    
resultado = newton_raphson(0.5)
print("Raíz encontrada en ",resultado)
