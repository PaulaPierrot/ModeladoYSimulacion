import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# --- Métodos numéricos ---
def euler(f, t0, y0, h, N):
    t_vals, y_vals = [t0], [y0]
    for _ in range(N):
        y0 = y0 + h * f(t0, y0)
        t0 += h
        t_vals.append(t0)
        y_vals.append(y0)
    return np.array(t_vals), np.array(y_vals)

def heun(f, t0, y0, h, N):
    t_vals, y_vals = [t0], [y0]
    for _ in range(N):
        k1 = f(t0, y0)
        k2 = f(t0 + h, y0 + h * k1)
        y0 = y0 + (h/2) * (k1 + k2)
        t0 += h
        t_vals.append(t0)
        y_vals.append(y0)
    return np.array(t_vals), np.array(y_vals)

def rk4(f, t0, y0, h, N):
    t_vals, y_vals = [t0], [y0]
    for _ in range(N):
        k1 = f(t0, y0)
        k2 = f(t0 + h/2, y0 + h * k1/2)
        k3 = f(t0 + h/2, y0 + h * k2/2)
        k4 = f(t0 + h, y0 + h * k3)
        y0 = y0 + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t0 += h
        t_vals.append(t0)
        y_vals.append(y0)
    return np.array(t_vals), np.array(y_vals)

# --- Programa principal ---
def main():
    # Ingreso de datos
    expr_str = input("Ingrese f(t,y): ")   # ejemplo: y - t**2 + 1
    t0 = float(sp.sympify(input("Ingrese t0: ")))
    y0 = float(sp.sympify(input("Ingrese y0: ")))
    a  = float(sp.sympify(input("Ingrese a (inicio del intervalo): ")))
    b  = float(sp.sympify(input("Ingrese b (fin del intervalo): ")))
    h  = float(sp.sympify(input("Ingrese h (paso): ")))

    # Definir símbolos y función
    t, y = sp.symbols('t y')
    f_expr = sp.sympify(expr_str)
    f = sp.lambdify((t, y), f_expr, 'numpy')

    # Resolver EDO de forma exacta (si se puede)
    Y = sp.Function('Y')
    edo = sp.Eq(sp.diff(Y(t), t), f_expr.subs(y, Y(t)))
    exacta_ok = True
    try:
        sol_exacta = sp.dsolve(edo, ics={Y(t0): y0})
        y_exacta = sp.lambdify(t, sol_exacta.rhs, 'numpy')
    except Exception:
        exacta_ok = False

    # Número de pasos
    N = int(np.ceil((b - a) / h))

    # Calcular métodos
    t_euler, y_euler = euler(f, t0, y0, h, N)
    t_heun, y_heun   = heun(f, t0, y0, h, N)
    t_rk4, y_rk4     = rk4(f, t0, y0, h, N)

    # Solución exacta en mismos puntos (si existe)
    if exacta_ok:
        try:
            y_ex = [float(val) for val in y_exacta(t_euler)]
        except Exception:
            y_ex = [np.nan] * len(t_euler)
    else:
        y_ex = [np.nan] * len(t_euler)

    # --- Tabla con errores ---
    print("\nTabla comparativa con errores:\n")
    header = (f"{'i':<3}{'t':<10}{'Exacta':<12}"
              f"{'Euler':<12}{'ErrAbs(E)':<12}{'Err%(E)':<12}"
              f"{'Heun':<12}{'ErrAbs(H)':<12}{'Err%(H)':<12}"
              f"{'RK4':<12}{'ErrAbs(RK4)':<12}{'Err%(RK4)':<12}")
    print(header)

    for i in range(len(t_euler)):
        exact_val = y_ex[i]

        if not (exact_val is None or np.isnan(exact_val)):
            err_e_abs   = abs(y_euler[i] - exact_val)
            err_e_pct   = abs(err_e_abs / exact_val) * 100 if exact_val != 0 else 0
            err_h_abs   = abs(y_heun[i] - exact_val)
            err_h_pct   = abs(err_h_abs / exact_val) * 100 if exact_val != 0 else 0
            err_rk4_abs = abs(y_rk4[i] - exact_val)
            err_rk4_pct = abs(err_rk4_abs / exact_val) * 100 if exact_val != 0 else 0

            print(f"{i:<3}{t_euler[i]:<10.6f}"
                  f"{exact_val:<12.6f}"
                  f"{y_euler[i]:<12.6f}{err_e_abs:<12.6f}{err_e_pct:<12.2f}"
                  f"{y_heun[i]:<12.6f}{err_h_abs:<12.6f}{err_h_pct:<12.2f}"
                  f"{y_rk4[i]:<12.6f}{err_rk4_abs:<12.6f}{err_rk4_pct:<12.2f}")
        else:
            print(f"{i:<3}{t_euler[i]:<10.6f}"
                  f"{'N/A':<12}"
                  f"{y_euler[i]:<12.6f}{'N/A':<12}{'N/A':<12}"
                  f"{y_heun[i]:<12.6f}{'N/A':<12}{'N/A':<12}"
                  f"{y_rk4[i]:<12.6f}{'N/A':<12}{'N/A':<12}")

    # --- Gráfico ---
    plt.figure(figsize=(7,6))
    if exacta_ok:
        plt.plot(t_euler, y_ex, 'k--', linewidth=2, label='Exacta')
    plt.plot(t_euler, y_euler, 'ro-', linewidth=1.5, markersize=6, label='Euler')
    plt.plot(t_heun, y_heun, 'g^-', linewidth=1.5, markersize=6, label='Heun')
    plt.plot(t_rk4, y_rk4, 'bs-', linewidth=1.5, markersize=6, label='RK4')
    plt.title("Comparación: Euler, Heun y RK4")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
