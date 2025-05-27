import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Параметры модели
t0 = 0.2      # постоянное время следования
tau0 = 0.25   # постоянная времени двигателя
Cf = Cr = 80000  # коэффициенты жесткости шин
m = 1100      # масса автомобиля
Iz = 1343.1   # момент инерции
lf = 1.2      # расстояние переднего колеса до центра масс автомобиля
lr = 1.6      # расстояние заднего колеса до центра масс автомобиля
v_min = 20 / 3.6 # ограничение по скорости снизу м/с
v_max = 160 / 3.6 # ограничение по скорости сверху м/с

s1 = 10 # начальные координаты отслеживаемого автомобиля м
s2 = 0 # начальные координаты EGO автомобиля м
v1 = 30 / 3.6 # начальная скорость отслеживаемого автомобиля м/с
v2 = 21 / 3.6 # начальная скорость EGO автомобиля м/с


# Инициализация состояния модели
# Управляющие входы
Mz = 0 # Момент поворота
delta = 0 # актуальное расстояние между автомобилями
d_des0 = 10 # целевое расстояние между автомобилями при нулевой скорости
x = np.zeros((5, 1)) # вектор состояния

# Параметры симуляции
Td = 0.1 # шаг дискретизации с
T_total = 20 # длительность симулирования с
N = int(T_total / Td) # кол-во шагов симуляции
# История для графиков
hist_d  = np.zeros(N)
hist_v1 = np.zeros(N)
hist_v2 = np.zeros(N)
hist_a1 = np.zeros(N)
hist_a2 = np.zeros(N)

# --- Функции матриц ---
def A_tilde(v):
    return np.array([
        [0, 1, -t0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, -1/tau0, 0, 0],
        [0, 0, 0, -(2*Cf + 2*Cr)/(m*v), ((2*lr*Cr - 2*lf*Cf)/(m*v*v)) - 1],
        [0, 0, 0, -(2*lf*Cf - 2*lr*Cr)/(Iz), -(2*lf**2*Cf + 2*lr**2*Cr)/(Iz*v)]
    ])

def B_d_tilde(v):
    return np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 2*Cf/(m*v**2)],
        [0, 2*lf*Cf/Iz]
    ])

def B_u_tilde():
    return np.array([
        [0, 0],
        [0, 0],
        [1/tau0, 0],
        [0, 0],
        [0, 1/Iz]
    ])

def get_rho(v2):
    v0_bar = 2 * v_min * v_max / (v_max + v_min)
    v1_bar = 2 * v_min * v_max / (v_max - v_min)
    return (v0_bar * v1_bar / v2 - v1_bar) / v0_bar

def interpolate_matrices(v2):
    rho = get_rho(v2)
    eta1 = (1 + rho) / 2
    eta2 = (1 - rho) / 2
    A_eta = eta1 * A1 + eta2 * A2
    Bd_eta = eta1 * Bd1 + eta2 * Bd2
    return A_eta, Bd_eta

def sim_prec_vehicle(sim_step):
    t = sim_step * Td

    # --- Режимы движения ведущего автомобиля ---

    # # Режим 1: Плавное движение по синусоиде
    A = 1
    f = 0.5
    omega = 2 * np.pi * f
    a1 = A * np.sin(omega * t)

    # Режим 2: Постоянное ускорение
    # a1 = 0.5

    # # Режим 3: Постоянное замедление
    # a1 = -0.5

    # # Режим 4: Резкое торможение после 10 секунды
    # a1 = -4 if t > 10 else 0

    # # Режим 5: Резкое ускорение после 10 секунды
    # a1 = 3 if t > 10 else 0

    # # Режим 6: Постоянная скорость
    # a1 = 0

    return a1

def solve_mpc(x_now, a1_now, A_eta, Bd_eta, Bu_col):
    Np = 10
    Q = np.diag([10, 5, 0, 0, 0])
    R = 0.1
    tau_penalty = 0.1
    umin = -5
    umax = 2.5

    x = cp.Variable((5, Np+1))
    u = cp.Variable((1, Np))

    constraints = [x[:, 0] == x_now.flatten()]
    objective = 0
    d = np.tile(np.array([[a1_now], [0.0]]), (1, Np))

    for k in range(Np):
        objective += cp.quad_form(x[:, k], Q) + R * cp.sum_squares(u[:, k]) - tau_penalty * np.sum(d[:, k]**2)
        constraints += [x[:, k+1] == A_eta @ x[:, k] + Bu_col @ u[:, k] + Bd_eta @ d[:, k]]
        constraints += [u[:, k] <= umax]
        constraints += [u[:, k] >= umin]

    P = Q
    objective += cp.quad_form(x[:, Np], P)

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    return u.value[0, 0]



# --- Предварительные матрицы ---
A1  = np.eye(5) + Td * A_tilde(v_min)
A2  = np.eye(5) + Td * A_tilde(v_max)
Bd1 = Td * B_d_tilde(v_min)
Bd2 = Td * B_d_tilde(v_max)
Bu  = Td * B_u_tilde()

# --- Основной цикл ---
for k in range(N):
    # Вычисление состояния отслеживаемого автомобиля
    a1 = sim_prec_vehicle(k) # имитация получения информации
                             # об отслеживаемом автомобиле
    v1 += a1 * Td
    s1 += v1 * Td

    # Обновление состояния модели
    v2 += x[2, 0] * Td
    d_des = d_des0 + t0 * v2
    delta_d = (s1 - s2) - d_des
    delta_v = v1 - v2
    x0 = np.array([[delta_d], [delta_v], [x[2, 0]], [x[3, 0]], [x[4, 0]]])
    A_eta, Bd_eta  = interpolate_matrices(v2)

    # Вычисление целевого ускорения
    a_des = solve_mpc(x0, a1, A_eta, Bd_eta, Bu[:, [0]])

    # Обновление состояние автомобиля на основе целевого ускорения
    u = np.array([[a_des], [Mz]])
    d = np.array([[a1], [delta]])
    x = A_eta @ x0 + Bu[:, [0]] @ np.array([[a_des]]) + Bd_eta @ d
    s2 += v2 * Td

    hist_d[k]  = s1 - s2
    hist_v1[k] = v1
    hist_v2[k] = v2
    hist_a1[k] = a1
    hist_a2[k] = x[2, 0]

# --- Построение графиков ---
t = np.arange(0, T_total, Td)
plt.figure(figsize=(10, 8))

plt.subplot(3,1,1)
plt.plot(t, hist_d, linewidth=2)
plt.ylabel('Расстояние (м)')
plt.title('Расстояние между автомобилями')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t, hist_v1, 'r', label='Отслеживаемый v1')
plt.plot(t, hist_v2, 'b', label='EGO v2')
plt.ylabel('Скорость (м/с)')
plt.title('Скорости автомобилей')
plt.legend()
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(t, hist_a1, 'r--', label='Отслеживаемый a1')
plt.plot(t, hist_a2, 'b', label='EGO a2')
plt.xlabel('Время (с)')
plt.ylabel('Ускорение (м/с²)')
plt.title('Ускорения автомобилей')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
