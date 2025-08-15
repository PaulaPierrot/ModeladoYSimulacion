import math
def funcion(x):
#     Aca ir cambiando la función
    return (math.cos(x))


def acelerar(x0,x1,x2):
    #esto no cambiarlo nunca!!
    return (x0- (x1-x0)**2/(x2-2*x1+x0))

def actualizarTolerancia(x,y):
    return abs(y-x)

def aiken(x0):
    x1 = funcion(x0)
    x2 = funcion(x1)
    tolerancia = 100
    n = 0
    print("iteracion 0")
    print("x0: ",x0)
    print("x1: ",x1)
    print("x2: ",x2)
    x_acelerado = x0
    #ahora que tenemos los 3 valores aceleramos
    while (tolerancia > 0.001):
        n+=1
        x_anterior = x_acelerado 
        x_acelerado = acelerar(x_acelerado,x1,x2)
        print("iteracion: ",n)
        print("x acelerado: ",x_acelerado)
        x1 = funcion(x_acelerado)
        print("x1: ",x1)
        x2 = funcion(x1)
        print("x2: ",x2)
        tolerancia = actualizarTolerancia(x1,x2)


aiken(0.5)
        
