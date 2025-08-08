#búsqueda binaria de raices. Aka bisección
#tolerancia 10 -3
import math


def f (x):
    return math.exp(x) - (2 + x)

def bbraices (a,b):
    #caso base
    c = (a+b)/2
    if 10**-3 > abs(f(c)):
        return c
    #comparamos con a y b para ver si buscamos a derecha o izquierda
    if f(a)*f(c)<0:
        return bbraices(a,c)
    if f(b)*f(c)<0:
        return bbraices(b,c)
    
print(bbraices(-2,-1))

    
    
    
