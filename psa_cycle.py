# Created on 27 Mar 2024 by Zihao Wang, zwang@mpi-magdeburg.mpg.de
# Simulation of Pressure Swing Adsorption
# PDE system is transformed to ODE using finite volume method
# Source: https://doi.org/10.1021/acs.iecr.5b03122, https://doi.org/10.1039/D0ME00060D, https://github.com/PEESEgroup/PSA


import numpy as np
from scipy.integrate import solve_ivp

# ==================
# Process Parameters
# ==================
P_0 = 10e5  # Adsorption pressure [Pa]
v_feed = 0.18  # Inlet velocity [m/s]

N = 10  # 1: number of finite volumes
deltaU_1 = -33000  # 2: enthalpy of adsorption of Component#1
deltaU_2 = -12000  # 3: enthalpy of adsorption of Component#2
rho_s = 1659  # 4: density of the adsorbent [kg/m^3]
T_0 = 313.15  # 5: Feed temperature of flue gas [K]
epsilon = 0.37  # 6: Void fraction
r_p = 1e-3  # 7: Radius of the pellets [m]
mu = 1.72e-5  # 8: Viscosity of gas [Pa*s]
R = 8.314  # 9: Universal gas constant [J/mol/K, Pa*m^3/mol/K]
ndot_0 = P_0 * v_feed / R / T_0  # Inlet molar flux [mol/s/m^2]
Ctot_0 = P_0 / R / T_0  # Inlet total concentration [mol/m^3])
v_0 = ndot_0 / Ctot_0  # 10: Inlet velocity and scaling parameter [m/s]

q_s = 5.84  # Molar loading scaling factor [mol/kg]
q_s0 = q_s * rho_s  # 11: Molar loading scaling factor [mol/m^3]
C_pg = 30.7  # 12: Specific heat of gas [J/mol/k]
C_pa = 30.7  # 13: Specific heat of adsorbed phase [J/mol/k]
C_ps = 1070  # 14: Specific heat capacity of the adsorbent [J/kg/K]
D_m = 1.2995e-5  # 15: Molecular diffusivity [m^2/s]
K_z = 0.09  # 16: Thermal conduction in gas phase [W/m/k]
L = 1.01  # 18: Length of the column [m]
MW_CO2 = 0.04402  # 19: Molecular weight of CO2 [kg/mol]
MW_N2 = 0.02802  # 20: Molecular weight of N2 [kg/mol]

k_1_LDF = 0.1631  # 21: Mass transfer coefficient for CO2 [1/s]
k_2_LDF = 0.2044  # 22: Mass transfer coefficient for N2 [1/s]
y_0 = 0.15  # 23: Inlet gas CO2 mole fraction[-]
tau = 0.5  # 24: Parameter used for determining speed of pressure change

# ===================
# Isotherm Parameters
# ===================
iso_params = np.array([5., 3., 9.46e-11, 6.15e-16, -33000., -48000.,
                       12.7, 0., 4.29e-10, 0., -12300., 0.])


def isotherm(y1, pres, temp, isotherm_params):
    """
    calculate gas loading using Dual-site Langmuir isotherm
    """

    q_b1, q_d1, b1, d1, U_b1, U_d1, q_b2, q_d2, b2, d2, U_b2, U_d2 = isotherm_params

    B1 = b1 * np.exp(-U_b1 / (R * temp))
    B2 = b2 * np.exp(-U_b2 / (R * temp))
    D1 = d1 * np.exp(-U_d1 / (R * temp))
    D2 = d2 * np.exp(-U_d2 / (R * temp))

    P1, P2 = pres * y1, pres * (1 - y1)
    sum_Bi_Pi = B1 * P1 + B2 * P2
    sum_Di_Pi = D1 * P1 + D2 * P2

    q1 = (q_b1 * B1 * P1) / (1 + sum_Bi_Pi) + (q_d1 * D1 * P1) / (1 + sum_Di_Pi)
    q2 = (q_b2 * B2 * P2) / (1 + sum_Bi_Pi) + (q_d2 * D2 * P2) / (1 + sum_Di_Pi)

    return [q1, q2]


def WENO(flux_c, flow_dir):
    """
    Weighted Essentially NonOscillatory scheme
    """
    oo = 10 ** -10
    N = len(flux_c) - 2
    flux_w = np.zeros(N + 1)
    alpha0 = np.zeros(N + 2)
    alpha1 = np.zeros(N + 2)

    flux_w[0] = flux_c[0]
    flux_w[N] = flux_c[N + 1]

    if flow_dir == "backward":
        alpha0[1:N] = (2 / 3) / ((flux_c[2:N + 1] - flux_c[1:N] + oo) ** 4)
        alpha1[2:N] = (1 / 3) / ((flux_c[2:N] - flux_c[1:N - 1] + oo) ** 4)
        alpha1[1] = (1 / 3) / ((2 * (flux_c[1] - flux_c[0]) + oo) ** 4)
        flux_w[2:N] = (alpha0[2:N] / (alpha0[2:N] + alpha1[2:N])) * \
                      (0.5 * (flux_c[2:N] + flux_c[3:N + 1])) + \
                      (alpha1[2:N] / (alpha0[2:N] + alpha1[2:N])) * \
                      (1.5 * flux_c[2:N] - 0.5 * flux_c[1:N - 1])
        flux_w[1] = (alpha0[1] / (alpha0[1] + alpha1[1])) * (0.5 * (flux_c[1] + flux_c[2])) + \
                    (alpha1[1] / (alpha0[1] + alpha1[1])) * (2 * flux_c[1] - flux_c[0])

    elif flow_dir == "forward":
        alpha0[1:N] = (2 / 3) / ((flux_c[1:N] - flux_c[2:N + 1] + oo) ** 4)
        alpha1[1:N - 1] = (1 / 3) / ((flux_c[2:N] - flux_c[3:N + 1] + oo) ** 4)
        alpha1[N - 1] = (1 / 3) / ((2 * (flux_c[N] - flux_c[N + 1]) + oo) ** 4)

        flux_w[1:N - 1] = (alpha0[1:N - 1] / (alpha0[1:N - 1] + alpha1[1:N - 1])) * \
                          (0.5 * (flux_c[1:N - 1] + flux_c[2:N])) + \
                          (alpha1[1:N - 1] / (alpha0[1:N - 1] + alpha1[1:N - 1])) * \
                          (1.5 * flux_c[2:N] - 0.5 * flux_c[3:N + 1])
        flux_w[N - 1] = (alpha0[N - 1] / (alpha0[N - 1] + alpha1[N - 1])) * (0.5 * (flux_c[N - 1] + flux_c[N])) + \
                        (alpha1[N - 1] / (alpha0[N - 1] + alpha1[N - 1])) * (2 * flux_c[N] - flux_c[N + 1])

    return flux_w


def PressurizationModel(t, x):
    print(t)

    # ================
    # Process Variable
    # ================
    P = x[:N + 2]
    y = x[N + 2:2 * (N + 2)]
    x1 = x[2 * (N + 2):3 * (N + 2)]
    x2 = x[3 * (N + 2):4 * (N + 2)]
    T = x[4 * (N + 2):5 * (N + 2)]

    # =====================
    # Derivative Definition
    # =====================
    dPdt = np.zeros(N + 2)
    dydt = np.zeros(N + 2)
    dx1dt = np.zeros(N + 2)
    dx2dt = np.zeros(N + 2)
    dTdt = np.zeros(N + 2)

    dPdz = np.zeros(N + 2)
    dPdzh = np.zeros(N + 1)
    dydz = np.zeros(N + 2)
    d2ydz2 = np.zeros(N + 2)
    dTdz = np.zeros(N + 2)
    d2Tdz2 = np.zeros(N + 2)

    derivatives = np.zeros(5 * (N + 2))

    # =====================
    # Intermediate Variable
    # =====================
    dz = 1 / N
    D_l = 0.7 * D_m + v_0 * r_p
    Pe = v_0 * L / D_l
    phi = R * T_0 * q_s0 * (1 - epsilon) / epsilon / P_0
    rho_g = P * P_0 / R / T / T_0

    # ==================
    # Boundary Condition
    # ==================
    y[0] = y_0
    T[0] = 1
    if P[1] > P[0]:
        P[0] = P[1]

    y[N + 1] = y[N]
    T[N + 1] = T[N]
    P[N + 1] = P[N]

    # ================
    # First Derivative
    # ================
    dP = P[1:N + 2] - P[0:N + 1]
    idx_f = np.where(dP <= 0)
    idx_b = np.where(dP > 0)
    Ph = np.zeros(N + 1)
    Ph_f = WENO(P, "backward")
    Ph_b = WENO(P, "forward")
    Ph[idx_f] = Ph_f[idx_f]
    Ph[idx_b] = Ph_b[idx_b]
    Ph[0] = P[0]
    Ph[N] = P[N + 1]
    dPdz[1: N + 1] = (Ph[1:N + 1] - Ph[0: N]) / dz
    dPdzh[1: N] = (P[2:N + 1] - P[1: N]) / dz
    dPdzh[0] = 2 * (P[1] - P[0]) / dz
    dPdzh[N] = 2 * (P[N + 1] - P[N]) / dz

    yh = np.zeros(N + 1)
    yh_f = WENO(y, "backward")
    yh_b = WENO(y, "forward")
    yh[idx_f] = yh_f[idx_f]
    yh[idx_b] = yh_b[idx_b]
    if P[0] > P[1]:
        yh[0] = y[0]
    else:
        yh[0] = y[1]
    yh[N] = y[N + 1]
    dydz[1:N + 1] = (yh[1:N + 1] - yh[0: N]) / dz

    Th = np.zeros(N + 1)
    Th_f = WENO(T, "backward")
    Th_b = WENO(T, "forward")
    Th[idx_f] = Th_f[idx_f]
    Th[idx_b] = Th_b[idx_b]
    if P[0] > P[1]:
        Th[0] = T[0]
    else:
        Th[0] = T[1]
    Th[N] = T[N + 1]
    dTdz[1: N + 1] = (Th[1:N + 1] - Th[0: N]) / dz

    # =================
    # Second Derivative
    # =================
    d2ydz2[2:N] = (y[3:N + 1] + y[1:N - 1] - 2 * y[2:N]) / dz / dz
    d2ydz2[1] = (y[2] - y[1]) / dz / dz
    d2ydz2[N] = (y[N - 1] - y[N]) / dz / dz

    d2Tdz2[2:N] = (T[3:N + 1] + T[1:N - 1] - 2 * T[2:N]) / dz / dz
    d2Tdz2[1] = 4 * (Th[1] + T[0] - 2 * T[1]) / dz / dz
    d2Tdz2[N] = 4 * (Th[N - 1] + T[N + 1] - 2 * T[N]) / dz / dz

    # ==============
    # Ergun Equation
    # ==============
    rho_gh = (P_0 / R / T_0) * Ph[0:N + 1] / Th[0: N + 1]
    viscous_term = 150 * mu * (1 - epsilon) ** 2 / 4 / r_p ** 2 / epsilon ** 2
    kinetic_term_h = (rho_gh * (MW_N2 + (MW_CO2 - MW_N2) * yh)) * (1.75 * (1 - epsilon) / 2 / r_p / epsilon)
    vh = (-viscous_term + (np.abs(viscous_term ** 2 + 4 * kinetic_term_h * np.abs(dPdzh) * P_0 / L)) ** 0.5) / \
         (2. * kinetic_term_h * v_0) * (- np.sign(dPdzh))

    # ====================
    # Linear Driving Force
    # ====================
    q_1, q_2 = isotherm(y, P * P_0, T * T_0, iso_params)
    k_1 = k_1_LDF * L / v_0
    k_2 = k_2_LDF * L / v_0
    dx1dt[1:N + 1] = k_1 * (q_1[1:N + 1] / q_s - x1[1:N + 1])
    dx2dt[1:N + 1] = k_2 * (q_2[1:N + 1] / q_s - x2[1:N + 1])

    # ==============
    # Energy Balance
    # ==============
    sink_term = (1 - epsilon) * (rho_s * C_ps + q_s0 * C_pa) + (epsilon * rho_g[1:N + 1] * C_pg)
    transfer_term = K_z / v_0 / L
    dTdt1 = transfer_term * d2Tdz2[1: N + 1] / sink_term
    PvT = Ph[0:N + 1] * vh[0: N + 1] / Th[0: N + 1]
    Pv = Ph[0:N + 1] * vh[0: N + 1]
    dTdt2 = -epsilon * C_pg * (P_0 / R / T_0) * (
            (Pv[1:N + 1] - Pv[0:N]) - T[1: N + 1] * (PvT[1:N + 1] - PvT[0: N])) / dz / sink_term
    generation_term_1 = (1 - epsilon) * q_s0 * (-(deltaU_1 - R * T[1:N + 1] * T_0)) / T_0
    generation_term_2 = (1 - epsilon) * q_s0 * (-(deltaU_2 - R * T[1:N + 1] * T_0)) / T_0
    dTdt3 = (generation_term_1 * dx1dt[1:N + 1] + generation_term_2 * dx2dt[1: N + 1]) / sink_term
    dTdt[1:N + 1] = dTdt1 + dTdt2 + dTdt3

    # ====================
    # Overall Mass Balance
    # ====================
    dPdt1 = -T[1: N + 1] * (PvT[1:N + 1] - PvT[0: N]) / dz
    dPdt2 = -phi * T[1: N + 1] * (dx1dt[1:N + 1] + dx2dt[1: N + 1])
    dPdt3 = P[1: N + 1] * dTdt[1: N + 1] / T[1: N + 1]
    dPdt[1: N + 1] = dPdt1 + dPdt2 + dPdt3

    # ======================
    # Component Mass Balance
    # ======================
    dydt1 = (1 / Pe) * (d2ydz2[1:N + 1] + (dydz[1:N + 1] * dPdz[1:N + 1] / P[1:N + 1]) - (
            dydz[1:N + 1] * dTdz[1:N + 1] / T[1:N + 1]))
    ypvt = yh[0:N + 1] * Ph[0: N + 1] * vh[0: N + 1] / Th[0: N + 1]
    dydt2 = -(T[1:N + 1] / P[1: N + 1]) * ((ypvt[1:N + 1] - ypvt[0:N]) - y[1: N + 1] * (PvT[1:N + 1] - PvT[0: N])) / dz
    dydt3 = (phi * T[1:N + 1] / P[1: N + 1]) * ((y[1:N + 1] - 1) * dx1dt[1: N + 1] + y[1: N + 1] * dx2dt[1: N + 1])
    dydt[1: N + 1] = dydt1 + dydt2 + dydt3

    # ===================
    # Boundary Derivative
    # ===================
    dPdt[0] = tau * L / v_0 * (1 - P[0])
    dPdt[N + 1] = dPdt[N]
    dydt[0] = 0
    dydt[N + 1] = dydt[N]
    dx1dt[0] = 0
    dx2dt[0] = 0
    dx1dt[N + 1] = 0
    dx2dt[N + 1] = 0
    dTdt[0] = 0
    dTdt[N + 1] = dTdt[N]

    derivatives[:N + 2] = dPdt
    derivatives[N + 2:2 * (N + 2)] = dydt
    derivatives[2 * (N + 2):3 * (N + 2)] = dx1dt
    derivatives[3 * (N + 2):4 * (N + 2)] = dx2dt
    derivatives[4 * (N + 2):5 * (N + 2)] = dTdt
    return derivatives


if __name__ == "__main__":
    np.random.seed(42)

    t_end = 20 * v_0 / L

    x0 = [1.11359645e-02, 1.11359645e-02, 1.29246367e-02, 1.43794013e-02,
          1.55469306e-02, 1.64533882e-02, 1.71376783e-02, 1.76346880e-02,
          1.79736251e-02, 1.81791229e-02, 1.82724382e-02, 1.82724382e-02,
          1.50000000e-01, 9.99765295e-01, 9.99770798e-01, 9.99764837e-01,
          9.99761283e-01, 9.99758566e-01, 9.99755977e-01, 9.99753158e-01,
          9.99750090e-01, 9.99746988e-01, 9.99744596e-01, 9.99744596e-01,
          5.02951269e-01, 5.02951269e-01, 4.27106470e-01, 2.82264781e-01,
          2.06652289e-01, 1.84537968e-01, 1.79441235e-01, 1.79435643e-01,
          1.80501295e-01, 1.81312259e-01, 1.83980882e-01, 1.83980882e-01,
          1.25821323e-06, 1.25821323e-06, 1.54687514e-06, 1.87781262e-06,
          2.01261998e-06, 2.08717325e-06, 2.15544792e-06, 2.22097837e-06,
          2.28438187e-06, 2.34653996e-06, 2.42263475e-06, 2.42263475e-06,
          1.00000000e+00, 8.99989932e-01, 9.34500232e-01, 9.94866238e-01,
          1.03760968e+00, 1.05569801e+00, 1.06266561e+00, 1.06529254e+00,
          1.06633907e+00, 1.06684204e+00, 1.06558632e+00, 1.06558632e+00]

    sol = solve_ivp(lambda t, x: PressurizationModel(t, x),
                    [0, t_end], x0, method="BDF", rtol=1e-6)
