import math
import matplotlib.pyplot as plt

def graficar(a,b)

def ecuacion(x):
    return (math.exp(x)-2-x)


def busquedaBinariaRaices(a,b,tolerancia):
    #precondicion = f(a) y f(b) deben tener distinto signo.
    c = (a+b)/2
    if (abs(ecuacion(c))<=tolerancia):
        return c
    elif ((ecuacion(a)*ecuacion(c))<0):
        return busquedaBinariaRaices(a,c,tolerancia)
    elif ((ecuacion(c)* ecuacion(b))<0):
        return busquedaBinariaRaices(c,b,tolerancia)
    
print("Raíz encontrada! x=",busquedaBinariaRaices(-2,-1,0.00001))





