import math
def ecuacion(x):
    #funcion sobre la que iterar
    return (math.cos(x)+x)

def puntoFijo(puntoInicio,tolerancia):
    errorAbs=1
    x = ecuacion(puntoInicio)
    i=0
    print("Iteracion", i)
    print("f("+ str(puntoInicio)+") = "+str(x))
    print()
    while (errorAbs>tolerancia):
        x1 = ecuacion(x)
        i+=1
        print("Iteracion", i)
        print("f("+ str(x)+") = "+str(x1))
        print()
        errorAbs = abs(abs(x)-abs(x1))
        x = x1
    return x


puntoInicio = 1
tolerancia = 0.00001
print("Se halló una posible raíz en x=",puntoFijo(puntoInicio, tolerancia)," con una tolerancia de",tolerancia)
