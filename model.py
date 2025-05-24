import numpy as np
import cvxpy as cp

# Параметры модели
t0 # постоянное время следования
tau0 # постоянная времени двигателя
Cf # коэффициенты жесткости задних шин
Cr # коэффициенты жесткости задних шин
m # масса автомобиля
Iz # момент инерции
lf # расстояние переднего колеса до центра масс автомобиля
lr # расстояние заднего колеса до центра масс автомобиля
Td # шаг дискретизации

# Диапазон скоростей
v_min# ограничение по скорости снизу м/с
v_max # ограничение по скорости сверху м/с

# Непрерывные матрицы состояния и управления
def A_tilde(v):
    return np.array([
        [0, 1, -t0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, -1 / tau0, 0, 0],
        [0, 0, 0, -(2*Cf + 2*Cr)/(m*v), ((2*lr*Cr - 2*lf*Cf)/(m*v*v)) - 1],
        [0, 0, 0, -(2*lf*Cf - 2*lr*Cr)/Iz, -(2*lf**2*Cf + 2*lr**2*Cr)/(Iz*v)]
    ])

def B_d_tilde(v):
    return np.array([
        [0, 0],
        [0, 0],
        [1 / tau0, 0],
        [0, 1 / m],
        [0, lf / Iz]
    ])

def B_u_tilde():
    return np.array([
        [0],
        [0],
        [1 / tau0],
        [0],
        [0]
    ])

# Дискретизация по Эйлеру
I = np.eye(5)
A1 = I + Td * A_tilde(v_min)
A2 = I + Td * A_tilde(v_max)
Bd1 = Td * B_d_tilde(v_min)
Bd2 = Td * B_d_tilde(v_max)
Bu = Td * B_u_tilde()

# Интерполяция TS-модели
def get_rho(v2):
    num = (2 * v_min * v_max / (v_min + v_max)) * (2 * v_min * v_max / (v_max - v_min)) / v2
    den = 2 * v_min * v_max / (v_min + v_max)
    return (num - (2 * v_min * v_max / (v_max - v_min))) / den

def interpolate_matrices(v2):
    rho = get_rho(v2)
    eta1 = 0.5 * (1 + rho)
    eta2 = 0.5 * (1 - rho)
    A_eta = eta1 * A1 + eta2 * A2
    Bd_eta = eta1 * Bd1 + eta2 * Bd2
    return A_eta, Bd_eta

# Формулировка задачи MPC
def solve_mpc(x0, A_eta, Bd_eta, d_seq, Np=10):
    n, m = A_eta.shape[0], 1  # размерность состояния и управления
    Q = np.diag([10, 5, 0, 0, 0])
    R = 0.1
    tau = 0.1
    P = Q
    u_max = 2.5

    x = cp.Variable((n, Np + 1))
    u = cp.Variable((m, Np))

    cost = 0
    constraints = [x[:, 0] == x0]

    for k in range(Np):
        dk = d_seq[k]  # возмущение [a1, 0]
        cost += cp.quad_form(x[:, k], Q) + R * cp.square(u[:, k]) - tau * dk[0]**2
        constraints += [
            x[:, k+1] == A_eta @ x[:, k] + Bu @ u[:, k] + Bd_eta @ dk,
            cp.abs(u[:, k]) <= u_max
        ]
    cost += cp.quad_form(x[:, Np], P)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)
    return u[:, 0].value.item() if u[:, 0].value is not None else 0.0
