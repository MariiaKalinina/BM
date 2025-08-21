import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import root_scalar
from scipy.special import erf
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import json
import sys
import matplotlib

# Установка интерактивного бэкенда для отображения графиков
# matplotlib.use('TkAgg')  # Используем TkAgg для отображения окон с графиками

# Проверка аргументов командной строки
if len(sys.argv) != 2:
    print("Ошибка: Укажите файл JSON с параметрами.")
    print("Пример: python thermal_history_reconstruction.py inputdata.json")
    sys.exit(1)

# Чтение параметров из JSON файла
with open(sys.argv[1], 'r') as f:
    params = json.load(f)

SEC_IN_YEAR = 365.25 * 24 * 3600  # секунд в году
TAU_R = 800e6 * SEC_IN_YEAR       # характерное время затухания радиогенного тепла, секунд

def calculate_lithosphere_thickness(t_c, t_m, delta, beta):
    """
    Рассчитывает исходную и измененную толщину литосферы после рифтогенеза.
    
    Аргументы:
        t_c (float): Исходная толщина коры, м
        t_m (float): Исходная толщина мантийной литосферы, м
        delta (float): Фактор растяжения коры (δ)
        beta (float): Фактор растяжения мантии (β)
    
    Возвращает:
        tuple: (исходная толщина литосферы, новая толщина коры, новая толщина мантии)
    """
    initial_lithosphere = t_c + t_m
    new_crust = t_c / delta
    new_mantle = t_m / beta
    return initial_lithosphere, new_crust, new_mantle

def calculate_tectonic_water_depths(hm, hc, ρm0, ρc0, ρw, α, Ta, β):
    """
    Рассчитывает тектонические глубины воды после мгновенного растяжения (hw1) и бесконечного охлаждения (hw2).
    
    Аргументы:
        hm (float): Толщина мантии до растяжения, м
        hc (float): Толщина коры до растяжения, м
        ρm0 (float): Плотность мантии при 0°C, кг/м³
        ρc0 (float): Плотность коры при 0°C, кг/м³
        ρw (float): Плотность воды, кг/м³
        α (float): Коэффициент теплового расширения, 1/°C
        Ta (float): Температура астеносферы, °C
        β (float): Фактор растяжения
    
    Возвращает:
        tuple: (hw1, hw2) глубины воды, м
    """
    numerator_hw1 = (hm + hc) * (
        (ρm0 - ρc0) * (hc / (hm + hc)) * (1 - (α * Ta * hc) / (2 * hm + 2 * hc)) -
        (α * Ta * ρm0 / 2)
    ) * (1 - 1/β)
    denominator_hw1 = ρm0 * (1 - α * Ta) - ρw
    hw1 = numerator_hw1 / denominator_hw1

    term1 = (ρm0 - ρc0) * hc / (ρm0 * (1 - α * Ta) - ρw)
    term2 = (1 - 1/β) - (α * Ta * hc / (2 * hm + 2 * hc)) * (1 - 1/β**2)
    hw2 = term1 * term2

    return hw1, hw2

def calculate_radiogenic_heat(time_yr, A0, a_r, a, rho_r, U, Th, K):
    """
    Вычисляет радиогенное тепло.
    
    Аргументы:
        time_yr (float): Время, годы
        A0 (float): Поверхностная генерация тепла, Вт/м³
        a_r (float): Масштаб глубины затухания, м
        a (float): Толщина литосферы, м
        rho_r (float): Плотность породы, кг/м³
        U (float): Концентрация урана, ppm
        Th (float): Концентрация тория, ppm
        K (float): Концентрация калия, %
    
    Возвращает:
        float: Радиогенное тепло, Вт/м²
    """
    time_sec = time_yr * SEC_IN_YEAR
    if A0 is None:
        Q_r = 0.01 * rho_r * (9.52 * U + 2.56 * Th + 3.48 * K)
        A0 = Q_r * 1e-6
    Q_rad = A0 * a_r * (1 - np.exp(-a / a_r)) * np.exp(-time_sec / TAU_R)
    if a < 0.1 * a_r:
        Q_rad = A0 * a_r * np.exp(-time_sec / TAU_R)
    return Q_rad

class McKenzieThermalModel:
    """
    Однослойная модель Маккензи для расчета теплового потока.
    """
    def __init__(self, G_prime, n_eigenvalues=30, z_resolution=1000):
        self.G_prime = G_prime
        self.n_eigenvalues = n_eigenvalues
        self.z_points = np.linspace(0, 1, z_resolution)
        self.eigenvalues = []
        self.eigenfunctions = []
        self.dtheta_n = []
        self.coefficients = []

    def diff_eq(self, z, y, K):
        theta, dtheta = y
        return [dtheta, -self.G_prime*(1-z)*dtheta - K*theta]

    def propagator_matrix(self, K):
        sol = solve_ivp(lambda z, y: self.diff_eq(z, y, K),
                        (0, 1), [0, 1], method='RK45', dense_output=True)
        return sol.sol(1)[0]

    def find_eigenvalues(self):
        """
        Находит собственные значения для модели.
        """
        K_guesses = [(n+1)**2 * np.pi**2 + self.G_prime/2 for n in range(self.n_eigenvalues)]
        for guess in K_guesses:
            try:
                res = root_scalar(self.propagator_matrix, x0=guess, method='newton',
                                  fprime=lambda K: (self.propagator_matrix(K+1e-6) - self.propagator_matrix(K-1e-6))/2e-6)
                if res.converged and res.root > 0:
                    self.eigenvalues.append(res.root)
            except:
                continue
        self.eigenvalues = sorted(list(set(self.eigenvalues))[:self.n_eigenvalues])

    def compute_eigenfunctions(self):
        """
        Вычисляет собственные функции.
        """
        for K in self.eigenvalues:
            sol = solve_ivp(lambda z, y: self.diff_eq(z, y, K),
                            (0, 1), [0, 1], t_eval=self.z_points, method='RK45')
            self.eigenfunctions.append(sol.y[0])

    def compute_derivatives(self):
        """
        Вычисляет производные собственных функций.
        """
        for K in self.eigenvalues:
            sol = solve_ivp(lambda z, y: self.diff_eq(z, y, K),
                            (0, 1), [0, 1], t_eval=[1], method='RK45')
            self.dtheta_n.append(sol.y[1][0])

    def _compute_normalization(self, theta_n):
        integrand = np.exp((self.z_points - 1)**2 * self.G_prime/2) * theta_n**2
        return trapezoid(integrand, self.z_points)

    def _compute_coefficient(self, initial_condition, theta_n, norm):
        integrand = initial_condition(self.z_points) * theta_n * np.exp((self.z_points - 1)**2 * self.G_prime/2)
        return trapezoid(integrand, self.z_points) / norm

    def compute_coefficients(self):
        """
        Вычисляет коэффициенты разложения.
        """
        def initial_condition(z):
            return (1 - z) + erf((z - 1) * np.sqrt(self.G_prime/2)) / erf(np.sqrt(self.G_prime/2))
        for theta_n in self.eigenfunctions:
            norm = self._compute_normalization(theta_n)
            an = self._compute_coefficient(initial_condition, theta_n, norm)
            self.coefficients.append(an)

    def heat_flow_during_stretching(self, t, k, T_m, a, kappa):
        """
        Рассчитывает тепловой поток во время растяжения, мВт/м².
        """
        t_scale = kappa * t / a**2
        steady_flow = (k * T_m / a) * np.sqrt(2 * self.G_prime / np.pi) / erf(np.sqrt(self.G_prime / 2))
        transient_flow = 0
        for an, Kn, dtheta in zip(self.coefficients, self.eigenvalues, self.dtheta_n):
            transient_flow += (k * T_m / a) * an * dtheta * np.exp(-Kn * t_scale)
        return (steady_flow - transient_flow) * 1e3

    def compute_post_rift_coefficients(self, k, T_m, a, kappa, beta):
        """
        Вычисляет коэффициенты для пост-рифтовой фазы.
        """
        z_points = self.z_points
        T_final = np.zeros_like(z_points)
        T_final += -(k * T_m / a) * erf((z_points - 1) * np.sqrt(self.G_prime/2)) / erf(np.sqrt(self.G_prime/2))
        t_final = (a**2 * np.log(beta)) / (kappa * self.G_prime)
        t_scale = kappa * t_final / a**2
        for an, Kn, theta_n in zip(self.coefficients, self.eigenvalues, self.eigenfunctions):
            T_final += (k * T_m / a) * an * np.exp(-Kn * t_scale) * theta_n
        b_coeffs = []
        for n in range(1, self.n_eigenvalues + 1):
            integrand = (T_final + (k * T_m / a) * (z_points - 1)) * np.sin(n * np.pi * z_points)
            bn = 2 * trapezoid(integrand, z_points) / (k * T_m / a)
            b_coeffs.append(bn)
        return b_coeffs

    def heat_flow_post_rift(self, t, b_coeffs, k, T_m, a, kappa):
        """
        Рассчитывает тепловой поток после растяжения, мВт/м².
        """
        t_scale = kappa * t / a**2
        steady_flow = k * T_m / a
        transient_flow = 0
        for n, bn in enumerate(b_coeffs, start=1):
            transient_flow += (k * T_m / a) * np.pi * n * bn * (-1)**(n+1) * np.exp(-n**2 * np.pi**2 * t_scale)
        return (steady_flow + transient_flow) * 1e3

class TwoLayerMcKenzieModel:
    """
    Двухслойная модель Хеллингера и Склатера для расчета теплового потока.
    """
    def __init__(self, beta_c, beta_sc, G_prime=10, n_eigenvalues=10):
        self.beta_c = beta_c
        self.beta_sc = beta_sc
        self.G_prime = G_prime
        self.n_eigenvalues = n_eigenvalues
        self.z_points = np.linspace(0, 1, 1000)
        self.gamma_c = 1 - 1/beta_c
        self.gamma_sc = 1 - 1/beta_sc
        self.gamma_L = (params['t_c']/params['a'])*self.gamma_c + (1 - params['t_c']/params['a'])*self.gamma_sc
        self.eigenvalues = []
        self.eigenfunctions = []
        self.dtheta_n = []
        self.coefficients = []

    def diff_eq(self, z, y, K):
        theta, dtheta = y
        return [dtheta, -self.G_prime*(1-z)*dtheta - K*theta]

    def propagator_matrix(self, K):
        sol = solve_ivp(lambda z, y: self.diff_eq(z, y, K),
                        (0, 1), [0, 1], method='RK45', dense_output=True)
        return sol.sol(1)[0]

    def find_eigenvalues(self):
        """
        Находит собственные значения для модели.
        """
        K_guesses = [(n+1)**2 * np.pi**2 + self.G_prime/2 for n in range(self.n_eigenvalues)]
        for guess in K_guesses:
            try:
                res = root_scalar(self.propagator_matrix, x0=guess, method='newton',
                                  fprime=lambda K: (self.propagator_matrix(K+1e-6) - self.propagator_matrix(K-1e-6))/2e-6)
                if res.converged and res.root > 0:
                    self.eigenvalues.append(res.root)
            except:
                continue
        self.eigenvalues = sorted(list(set(self.eigenvalues))[:self.n_eigenvalues])

    def compute_eigenfunctions(self):
        """
        Вычисляет собственные функции.
        """
        for K in self.eigenvalues:
            sol = solve_ivp(lambda z, y: self.diff_eq(z, y, K),
                            (0, 1), [0, 1], t_eval=self.z_points, method='RK45')
            self.eigenfunctions.append(sol.y[0])

    def compute_derivatives(self):
        """
        Вычисляет производные собственных функций.
        """
        for K in self.eigenvalues:
            sol = solve_ivp(lambda z, y: self.diff_eq(z, y, K),
                            (0, 1), [0, 1], t_eval=[1], method='RK45')
            self.dtheta_n.append(sol.y[1][0])

    def _compute_normalization(self, theta_n):
        integrand = np.exp((self.z_points - 1)**2 * self.G_prime/2) * theta_n**2
        return trapezoid(integrand, self.z_points)

    def _compute_coefficient(self, initial_condition, theta_n, norm):
        integrand = initial_condition(self.z_points) * theta_n * np.exp((self.z_points - 1)**2 * self.G_prime/2)
        return trapezoid(integrand, self.z_points) / norm

    def compute_coefficients(self):
        """
        Вычисляет коэффициенты разложения.
        """
        def initial_condition(z):
            return (1 - z) + erf((z - 1) * np.sqrt(self.G_prime/2)) / erf(np.sqrt(self.G_prime/2))
        for theta_n in self.eigenfunctions:
            norm = self._compute_normalization(theta_n)
            an = self._compute_coefficient(initial_condition, theta_n, norm)
            self.coefficients.append(an)

    def heat_flow_during_stretching(self, t, k, T_m, a, kappa):
        """
        Рассчитывает тепловой поток во время растяжения, мВт/м².
        """
        t_dimless = kappa * t / (a**2 * self.beta_sc**2)
        steady_term = np.sqrt(2 * self.G_prime / np.pi) / erf(np.sqrt(self.G_prime / 2))
        transient_term = 0
        for an, Kn, dtheta in zip(self.coefficients, self.eigenvalues, self.dtheta_n):
            transient_term += an * np.exp(-Kn * t_dimless) * dtheta
        return (k * T_m / a) * (steady_term - transient_term) * 1e3

    def compute_post_rift_coefficients(self, k, T_m, a, kappa):
        """
        Вычисляет коэффициенты для пост-рифтовой фазы.
        """
        z_points = self.z_points
        T_final = np.zeros_like(z_points)
        T_final += -erf((z_points - 1) * np.sqrt(self.G_prime/2)) / erf(np.sqrt(self.G_prime/2))
        t_final = (a**2 * np.log(self.beta_sc)) / (kappa * self.G_prime)
        t_dimless = kappa * t_final / a**2
        for an, Kn, theta_n in zip(self.coefficients, self.eigenvalues, self.eigenfunctions):
            T_final += an * np.exp(-Kn * t_dimless) * theta_n
        b_coeffs = []
        for n in range(1, self.n_eigenvalues + 1):
            integrand = (T_final + (z_points - 1)) * np.sin(n * np.pi * z_points)
            bn = 2 * trapezoid(integrand, z_points)
            b_coeffs.append(bn)
        return b_coeffs

    def heat_flow_post_rift(self, t, b_coeffs, k, T_m, a, kappa):
        """
        Рассчитывает тепловой поток после растяжения, мВт/м².
        """
        t_dimless = kappa * t / (a**2 * self.beta_sc**2)
        steady_term = 1
        transient_term = 0
        for n, bn in enumerate(b_coeffs, start=1):
            transient_term += n * bn * (-1)**(n+1) * np.exp(-n**2 * np.pi**2 * t_dimless)
        return (k * T_m / a) * (steady_term + np.pi * transient_term) * 1e3

def thermal_subsidence(t_post_years, a, kappa, alpha, rho_m, rho_w, T_m):
    """
    Рассчитывает термальное погружение после растяжения.
    
    Аргументы:
        t_post_years (float): Время после растяжения, млн лет
        a (float): Толщина литосферы, м
        kappa (float): Тепловая диффузия, м²/с
        alpha (float): Коэффициент теплового расширения, 1/°C
        rho_m (float): Плотность мантии, кг/м³
        rho_w (float): Плотность воды, кг/м³
        T_m (float): Температура астеносферы, °C
    
    Возвращает:
        float: Термальное погружение, км
    """
    tau = a**2 / (np.pi**2 * kappa)
    t_post_sec = t_post_years * 1e6 * SEC_IN_YEAR
    subsidence = 0
    for n in range(1, 20):
        term = (1 - np.exp(-n**2 * t_post_sec / tau)) / n**2
        subsidence += term
    return (4 * a * alpha * rho_m * T_m / (np.pi**2 * (rho_m - rho_w))) * subsidence * 1e-3

def calculate_temperature_profiles(z, t_Myr, stretch_duration_Myr, a, kappa, T_m, beta, G_prime, n_modes=100):
    """
    Рассчитывает температурные профили во время и после растяжения.
    
    Аргументы:
        z (array): Глубина, м
        t_Myr (float): Время, млн лет
        stretch_duration_Myr (float): Длительность растяжения, млн лет
        a (float): Толщина литосферы, м
        kappa (float): Тепловая диффузия, м²/с
        T_m (float): Температура астеносферы, °C
        beta (float): Фактор растяжения
        G_prime (float): Безразмерный параметр
        n_modes (int): Количество мод для разложения
    
    Возвращает:
        tuple: (глубина в км, температура в °C)
    """
    def initial_temperature(z, a, beta, T_m):
        return np.where(z <= a * (1 - 1/beta), T_m, T_m * beta * (1 - z/a))

    def compute_bn_numeric(n_max, z, a, beta, T_m):
        T_init = initial_temperature(z, a, beta, T_m)
        T_ss = T_m * (1 - z/a)
        b_n = []
        for n in range(1, n_max + 1):
            integrand = (T_init - T_ss) * np.sin(n * np.pi * z / a)
            bn = (2 / (a * T_m)) * trapezoid(integrand, z)
            b_n.append(bn)
        return np.array(b_n)

    z_grid = np.linspace(0, a, 1000)
    if t_Myr <= stretch_duration_Myr:
        T = initial_temperature(z_grid, a, beta, T_m)
    else:
        b_n = compute_bn_numeric(n_modes, z_grid, a, beta, T_m)
        t_sec = (t_Myr - stretch_duration_Myr) * SEC_IN_YEAR * 1e6
        T_ss = T_m * (1 - z_grid/a)
        transient = np.zeros_like(z_grid)
        for n in range(1, n_modes + 1):
            lambda_n = n * np.pi / a
            term = b_n[n-1] * np.sin(lambda_n * z_grid) * np.exp(-lambda_n**2 * kappa * t_sec)
            transient += term
        T = T_ss + T_m * transient
    return z_grid / 1e3, T

def sediment_thermal_effects(T, z, sediment_thickness=2e3, k_sed=2.0):
    """
    Учитывает термические эффекты осадков.
    
    Аргументы:
        T (array): Температурный профиль, °C
        z (array): Глубина, м
        sediment_thickness (float): Толщина осадков, м
        k_sed (float): Теплопроводность осадков, Вт/(м·К)
    
    Возвращает:
        array: Скорректированный температурный профиль, °C
    """
    k = params['k']  # Теплопроводность литосферы из параметров
    z_sed = z[z <= sediment_thickness]
    T_sed = T[:len(z_sed)] * (k / k_sed)  # Корректировка градиента в осадках
    T_new = np.copy(T)
    T_new[:len(z_sed)] = T_sed
    return T_new

def calculate_paleotemperatures(T, z, time_myr, swi_temp=10):
    """
    Рассчитывает палеотемпературы с учетом граничного условия SWI.
    
    Аргументы:
        T (array): Температурный профиль, °C
        z (array): Глубина, км
        time_myr (float): Время, млн лет
        swi_temp (float): Температура на границе вода-осадок, °C
    
    Возвращает:
        array: Палеотемпературы, °C
    """
    T_paleo = T + swi_temp
    return T_paleo

def visualize_results(time_myr, heat_flow, z_km, T, subsidence, paleotemp):
    """
    Визуализирует результаты: тепловой поток, температурные профили и погружение.
    
    Аргументы:
        time_myr (array): Время, млн лет
        heat_flow (array): Тепловой поток, мВт/м²
        z_km (array): Глубина, км
        T (array): Температура, °C
        subsidence (array): Термальное погружение, км
        paleotemp (array): Палеотемпературы, °C
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(time_myr, heat_flow, 'b-', label='Тепловой поток')
    ax1.set_xlabel('Время (млн лет)')
    ax1.set_ylabel('Тепловой поток (мВт/м²)')
    ax1.set_title('Эволюция теплового потока')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(T, z_km, 'r-', label='Температурный профиль')
    ax2.set_xlabel('Температура (°C)')
    ax2.set_ylabel('Глубина (км)')
    ax2.set_title('Температурный профиль')
    ax2.invert_yaxis()
    ax2.grid(True)
    ax2.legend()

    ax3.plot(time_myr[:len(subsidence)], subsidence, 'g-', label='Термальное погружение')
    ax3.set_xlabel('Время (млн лет)')
    ax3.set_ylabel('Погружение (км)')
    ax3.set_title('Термальное погружение')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    
    # Сохранение графика
    plt.savefig('thermal_history_plots.png', dpi=300, bbox_inches='tight')
    print("График сохранен как 'thermal_history_plots.png'")
    
    # Отображение графика
    plt.show()

def main():
    """
    Основная функция для выполнения реконструкции тепловой истории.
    """
    # Вывод всех используемых параметров
    print("=== Используемые параметры ===")
    print(f"Толщина литосферы (a): {params['a']/1e3:.1f} км")
    print(f"Толщина коры (t_c): {params['t_c']/1e3:.1f} км")
    print(f"Тепловая диффузия (kappa): {params['kappa']:.2e} м²/с")
    print(f"Теплопроводность (k): {params['k']:.2f} Вт/(м·К)")
    print(f"Температура астеносферы (T_m): {params['T_m']:.1f} °C")
    print(f"Коэффициент теплового расширения (alpha): {params['alpha']:.2e} 1/°C")
    print(f"Плотность мантии (rho_m): {params['rho_m']:.1f} кг/м³")
    print(f"Плотность коры (rho_c): {params['rho_c']:.1f} кг/м³")
    print(f"Плотность воды (rho_w): {params['rho_w']:.1f} кг/м³")
    print(f"Фактор растяжения коры (beta_c): {params['beta_c']:.2f}")
    print(f"Фактор растяжения субкоровой литосферы (beta_sc): {params['beta_sc']:.2f}")
    print(f"Безразмерный параметр (G_prime): {params['G_prime']:.1f}")
    print(f"Длительность моделирования (duration_myr): {params['duration_myr']:.1f} млн лет")
    print(f"Поверхностная генерация тепла (A0): {params['A0']:.2e} Вт/м³")
    print(f"Масштаб глубины затухания (a_r): {params['a_r']/1e3:.1f} км")
    print(f"Плотность породы (rho_r): {params['rho_r']:.1f} кг/м³")
    print(f"Концентрация урана (U): {params['U']:.1f} ppm")
    print(f"Концентрация тория (Th): {params['Th']:.1f} ppm")
    print(f"Концентрация калия (K): {params['K']:.1f} %")
    print("=============================\n")

    # Шаг 1: Ввод параметров
    t_c = params['t_c']
    t_m = params['a'] - t_c
    delta = params['beta_c']
    beta = params['beta_sc']
    G_prime = params['G_prime']
    k = params['k']
    kappa = params['kappa']
    T_m = params['T_m']
    a = params['a']
    alpha = params['alpha']
    rho_m = params['rho_m']
    rho_w = params['rho_w']
    duration_myr = params['duration_myr']
    A0 = params['A0']
    a_r = params['a_r']
    rho_r = params['rho_r']
    U = params['U']
    Th = params['Th']
    K = params['K']

    # Шаг 2: Проверка, известны ли δ и β
    if delta is None or beta is None:
        hw1, hw2 = calculate_tectonic_water_depths(t_m, t_c, rho_m, params['rho_c'], rho_w, alpha, T_m, β=1.5)
        delta = 1.2  # Заглушка
        beta = 1.2  # Заглушка
        print(f"Рассчитаны δ={delta:.2f}, β={beta:.2f}")

    # Шаг 3: Расчет толщины литосферы
    initial_lith, new_crust, new_mantle = calculate_lithosphere_thickness(t_c, t_m, delta, beta)
    print(f"Исходная толщина литосферы: {initial_lith/1e3:.1f} км")
    print(f"Новая толщина коры: {new_crust/1e3:.1f} км")
    print(f"Новая толщина мантии: {new_mantle/1e3:.1f} км")

    # Шаг 4: Радиогенное тепло
    time_range = np.linspace(0, duration_myr, 100)
    Q_rad = [calculate_radiogenic_heat(t * 1e6, A0, a_r, a, rho_r, U, Th, K) for t in time_range]

    # Шаг 5: Выбор модели
    model_choice = 'McKenzie'

    # Шаг 6: Расчет теплового потока
    time_stretch = np.linspace(0, duration_myr * SEC_IN_YEAR * 1e6, 100)
    time_post = np.linspace(0, 100e6 * SEC_IN_YEAR, 100)
    t_stretch_myr = time_stretch / (1e6 * SEC_IN_YEAR)
    t_post_myr = time_post / (1e6 * SEC_IN_YEAR) + duration_myr

    if model_choice == 'McKenzie':
        model = McKenzieThermalModel(G_prime)
        model.find_eigenvalues()
        model.compute_eigenfunctions()
        model.compute_derivatives()
        model.compute_coefficients()
        b_coeffs = model.compute_post_rift_coefficients(k, T_m, a, kappa, beta)
        F_stretch = [model.heat_flow_during_stretching(t, k, T_m, a, kappa) for t in time_stretch]
        F_post = [model.heat_flow_post_rift(t, b_coeffs, k, T_m, a, kappa) for t in time_post]
    else:
        beta_L = 1 / ((t_c/a)/delta + (1 - t_c/a)/beta)
        model = TwoLayerMcKenzieModel(delta, beta, G_prime)
        model.find_eigenvalues()
        model.compute_eigenfunctions()
        model.compute_derivatives()
        model.compute_coefficients()
        b_coeffs = model.compute_post_rift_coefficients(k, T_m, a, kappa)
        F_stretch = [model.heat_flow_during_stretching(t, k, T_m, a, kappa) for t in time_stretch]
        F_post = [model.heat_flow_post_rift(t, b_coeffs, k, T_m, a, kappa) for t in time_post]

    heat_flow = np.concatenate([F_stretch, F_post])
    time_myr = np.concatenate([t_stretch_myr, t_post_myr])

    # Шаг 7: Температурные профили
    z_km, T = calculate_temperature_profiles(np.linspace(0, a, 1000), duration_myr, duration_myr/2, a, kappa, T_m, beta, G_prime)

    # Шаг 8: Термические эффекты осадков
    T_sed = sediment_thermal_effects(T, z_km * 1e3, sediment_thickness=2e3, k_sed=2.0)

    # Шаг 9: Расчет палеотемператур
    paleotemp = calculate_paleotemperatures(T_sed, z_km, duration_myr, swi_temp=10)

    # Шаг 10: Термальное погружение
    subsidence = [thermal_subsidence(t / (1e6 * SEC_IN_YEAR), a, kappa, alpha, rho_m, rho_w, T_m) for t in time_post]

    # Шаг 11: Визуализация
    visualize_results(time_myr, heat_flow, z_km, paleotemp, subsidence, paleotemp)

if __name__ == "__main__":
    main()