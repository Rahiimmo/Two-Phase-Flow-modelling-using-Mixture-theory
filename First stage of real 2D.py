#============================================ Importing libraries ======================================================
import matplotlib.pyplot as plt
import numpy as np
import os, math, secrets
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.patches import Rectangle, Ellipse
import imageio.v2 as iio                                                                            #GIF writer (v2 API)
import time
RUN_START = time.perf_counter()

# Changing the font of plots to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

#-----------------------------------------------------------------------------------------------------------------------
#====================================== Parameters for Well 23 (SI units) ==============================================
#-----------------------------------------------------------------------------------------------------------------------
# Length of sections
L_g = 0.0                                        # Initial length of gas cap at wellhead                                (m)
L_w = 1                                          # Initial mud column length                                            (m)
L_c = 0                                          # Length of the cemented interval                                      (m)
L_k = L_c + L_w + L_g                            # Gas leak depth                                                       (m)

# Material properties
miu_g  = 2E-5                                    # Gas shear viscosity [0.02 cP]                                        (Pa·s)
rho_wr = 1198                                    # Reference mud density                                                (kg/m^3)
C_w    = 1.4E06                                  # Inverse of mud equivalent compressibility                            (m^2/s^2)
C_g    = 3.5E05                                  # Inverse of gas equivalent compressibility                            (m^2/s^2)
miu_w  = 1.0E-2                                  # Mud shear viscosity [10 cP]                                          (Pa·s)
kappa_g = 0.0                                    # Gas bulk viscosity                                                   (Pa.s)
kappa_w = 0.0                                    # Mud bulk viscosity                                                   (Pa.s)

# Interaction coefficients
I_w_a = 1E08                                     # Mud–annulus
I_g_a = 1E08                                     # Gas–annulus
I_a   = 6E04                                     # Mud–gas in annulus                                                   (Pa·s)^(-1)
I_w_c = 20                                       # Mud–cement
I_g_c = 1                                        # Gas–cement
I_c   = 500                                      # Mud–gas in cement                                                    (Pa·s)^(-1)

# Corey/ capillary/ viscous coefficients
alpha_1 = 1.0                                    # Corey exponent for mud in annulus
beta_1  = 0.6                                    # Corey exponent for gas in annulus
alpha_2 = 1.0                                    # Corey exponent for mud in cement
beta_2  = 0.6                                    # Corey exponent for gas in cement
epsilon_w = 0.0                                  # Coefficient for mud viscous terms                                    (m^2/s)
epsilon_g = 0.0                                  # Coefficient for gas viscous terms                                    (m^2/s)
P_star_c1 = 10000.0                              # Constant for capillary function                                      (Pa)
delta_1 = 0.08                                   # Constant for capillary function
a_1 = 2.0                                        # Constant for capillary function

# Gravity/ gridding/ cementing parameters
g = -9.81                                       # Gravitational acceleration                                            (m/s^2)
N_x = 10                                        # Number of grid cells in the x direction
N_y = 100                                       # Number of grid cells in the y direction
w = 0.02                                        # Width of the annulus                                                  (m)
phi_c = 0.01                                    # Cement-specific porosity
P_f = 54_000_000                                # Matched source pressure [540 bar]                                     (Pa)
K_c = 0.18 * 9.86923E-16                        # Matched cement permeability [0.18 mD]                                 (m^2)

#-----------------------------------------------------------------------------------------------------------------------
#=================================================== Time step =========================================================
#-----------------------------------------------------------------------------------------------------------------------
dt = 0.001                                      # Delta t                                                               (s)
total_time = 5                                  # Total simulation time                                                 (s)
M = int(2 * total_time / dt)                    # Number of time steps
print(f"Running simulation with {M / 2} time steps. (dt = {dt:.3f} seconds)")
print(f"Simulation will cover: {M * dt /2:.1f} seconds.")

#-----------------------------------------------------------------------------------------------------------------------
#=================================================== Gridding ==========================================================
#-----------------------------------------------------------------------------------------------------------------------
dx = w / N_x                                    # Delta x                                                               (m)
dy = L_k / N_y                                  # Delta y                                                               (m)
y_list = (np.arange(N_y) + 0.5) * dy            # Center of each row

#-----------------------------------------------------------------------------------------------------------------------
#================================================= List of variables ===================================================
#-----------------------------------------------------------------------------------------------------------------------
P_w_list = np.zeros((M + 1, N_y, N_x))          # Water Pressure                         (Rows for N_x, columns for N_y, and Clusters for time steps)
P_g_list = np.zeros((M + 1, N_y, N_x))          # Gas Pressure                           (Rows for N_x, columns for N_y, and Clusters for time steps)
S_w_list = np.zeros((M + 1, N_y, N_x))          # Water Saturation                       (Rows for N_x, columns for N_y, and Clusters for time steps)
S_g_list = np.zeros((M + 1, N_y, N_x))          # Gas Saturation                         (Rows for N_x, columns for N_y, and Clusters for time steps)
u_w_list = np.zeros((M + 1, N_y, N_x + 1))      # Water horizontal velocity              (Rows for N_x, columns for N_y, and Clusters for time steps)
u_g_list = np.zeros((M + 1, N_y, N_x + 1))      # Gas horizontal velocity                (Rows for N_x, columns for N_y, and Clusters for time steps)
v_w_list = np.zeros((M + 1, N_y + 1, N_x))      # Water vertical velocity                (Rows for N_x, columns for N_y, and Clusters for time steps)
v_g_list = np.zeros((M + 1, N_y + 1, N_x))      # Gas vertical velocity                  (Rows for N_x, columns for N_y, and Clusters for time steps)
Pc_list = np.zeros((M + 1, N_y, N_x))           # Capillary Pressure                     (Rows for N_x, columns for N_y, and Clusters for time steps)
rho_g_list = np.zeros((M + 1, N_y, N_x))        # Gas Density                            (Rows for N_x, columns for N_y, and Clusters for time steps)
rho_w_list = np.zeros((M + 1, N_y, N_x))        # Water Density                          (Rows for N_x, columns for N_y, and Clusters for time steps)
m_list = np.zeros((M + 1, N_y, N_x))            # Mass of water (rho_w * s_w)            (Rows for N_x, columns for N_y, and Clusters for time steps)
n_list = np.zeros((M + 1, N_y, N_x))            # Mass of gas (rho_g * s_g)              (Rows for N_x, columns for N_y, and Clusters for time steps)
eta_tilde = np.zeros((M + 1, N_y, N_x))         # Compressibility coefficient            (Rows for N_x, columns for N_y, and Clusters for time steps)
rho_g_tilde = np.zeros((M + 1, N_y, N_x))       # Modified gas density (Eq18)            (Rows for N_x, columns for N_y, and Clusters for time steps)

#-----------------------------------------------------------------------------------------------------------------------
#================================================= List of outputs =====================================================
#-----------------------------------------------------------------------------------------------------------------------
volume_gas = np.zeros(M + 1)                    # Gas volume fraction
max_sg = np.zeros(M + 1)                        # Max of gas volume fraction in each timestep
outlet_pressure_gas = np.zeros(M + 1)           # Gas Pressure at the outlet
outlet_pressure_water = np.zeros(M + 1)         # Water Pressure at the outlet
inlet_pressure_gas = np.zeros(M + 1)            # Gas Pressure at the inlet
inlet_pressure_water = np.zeros(M + 1)          # Water Pressure at the inlet
outlet_gas_velocity = np.zeros(M + 1)           # Gas vertical velocity at the outlet
outlet_water_velocity = np.zeros(M + 1)         # Water vertical velocity at the outlet
inlet_gas_velocity = np.zeros(M + 1)            # Gas vertical velocity at the inlet
inlet_water_velocity = np.zeros(M + 1)          # Water vertical velocity at the inlet
max_ug = np.zeros(M + 1)                        # Max of gas horizontal velocity in each timestep

#-----------------------------------------------------------------------------------------------------------------------
#================================== Numerical tolerances (phase absence & diagonals) ===================================
#-----------------------------------------------------------------------------------------------------------------------
SAT_TOL = 1E-6                                  # Interface saturation below this → phase treated as absent.
CFL_TARGET = 0.25                               # Stability target for explicit transport.
CFL_CAP = 400                                   # Max sub-cycles to avoid pathological dt_local ~ 1E-7 s.

#-----------------------------------------------------------------------------------------------------------------------
def cement_check(Y):
    """
    Check if the cell is in the cemented region.
    """
    if L_g + L_w < Y <= L_k:
        print("Cell is in the cemented region!!!!!")
        return True
    else:
        return False

#-----------------------------------------------------------------------------------------------------------------------
#================================================ Initial Conditions ===================================================
#-----------------------------------------------------------------------------------------------------------------------
# Initial gas saturation
for j in range(N_y):
    if 0 <= y_list[j] <= L_g:
        for i in range(N_x):
            S_g_list[0, j, i] = 0.0

    elif L_g < y_list[j] <= L_g + L_w:
        for i in range(N_x):
            S_g_list[0, j, i] = 0.0

    elif L_g + L_w < y_list[j] <= L_k:
        for i in range(N_x):
            S_g_list[0, j, i] = 0.0

S_w_list[0, :, :] = 1.0 - S_g_list[0, :, :]

# Initial capillary pressure
for j in range(N_y):
    if cement_check(y_list[j]):                 # Only in the cemented region.
        for i in range(N_x):
            Pc_list[0, j, i] = -P_star_c1 * np.log(delta_1 + (S_w_list[0, j, i] / a_1))

    else:
        for i in range(N_x):
            Pc_list[0, j, i] = 0.0              # No capillary effects in the uncemented region.

# Initial pressures (hydrostatic at t = 0)
P_top = 101325                                  # Wellhead pressure.                                                    (Pa)
for j in range(N_y):
    y = y_list[j]
    if y <= L_g:                                # Gas cap near the top (isothermal ideal-gas hydrostatic)
        for i in range(N_x):
            P_g_list[0, j, i] = np.exp((g / C_g) * (y - L_k)) * P_top
            P_w_list[0, j, i] = P_g_list[0, j, i]                              # Pc = 0 in this region.

    elif y <= L_g + L_w:                        # Mud column starts at y = L_g; enforce continuity with gas there.
        for i in range(N_x):
            P_w_list[0, j, i] = rho_wr * g * (y - L_k) + P_top
            P_g_list[0, j, i] = P_w_list[0, j, i]                               # Pc = 0 in this region.

    else:                                       # Cemented interval below y_ref = L_g + L_w.
        y_ref = L_g + L_w
        # Water and gas pressures at the top of cement.
        P_g_at_Lg = np.exp((g / C_g) * (L_g - 0.0)) * P_top
        P_w_at_Lg = P_g_at_Lg
        P_w_ref = P_w_at_Lg + rho_wr * g * L_w
        P_g_ref = P_w_ref                       # Same phase in the uncemented region.
        # Gas hydrostatic below y_ref; water = gas - Pc in cement.
        for i in range(N_x):
            P_g_list[0, j, i] = np.exp((g / C_g) * (y - y_ref)) * P_g_ref
            P_w_list[0, j, i] = P_g_list[0, j, i] - Pc_list[0, j, i]

# Initial velocities
u_w_list[0, :, :] = 0.0
u_g_list[0, :, :] = 0.0
v_w_list[0, :, :] = 0.0
v_g_list[0, :, :] = 0.0

#-----------------------------------------------------------------------------------------------------------------------
#=============================================== Boundary conditions ===================================================
#-----------------------------------------------------------------------------------------------------------------------
u_g_in = 0.05                                   # Constant gas injection velocity                                       (m/s)
v_g_list[:, 0, :] = u_g_in                      # Constant injection rate
v_w_list[:, 0, :] = 0.0                         # No water inflow
u_w_list[:, 0, :] = 0.0                         # No water inflow
u_g_list[:, :, 0] = 0.0                         # No gas horizontal velocity at the left wall.
u_g_list[:, :, -1] = 0.0                        # No gas horizontal velocity at the right wall.
u_w_list[:, :, 0] = 0.0                         # No water horizontal velocity at the left wall.
u_w_list[:, :, -1] = 0.0                        # No water horizontal velocity at the right wall.
v_g_list[:, :, 0] = 0.0                         # No gas vertical velocity at the left wall.
v_g_list[:, :, -1] = 0.0                        # No gas vertical velocity at the right wall.
v_w_list[:, :, 0] = 0.0                         # No water vertical velocity at the left wall.
v_w_list[:, :, -1] = 0.0                        # No water vertical velocity at the right wall.
P_g_list[:, -1, :] = P_top                      # Constant pressure at the outlet
P_w_list[:, -1, :] = P_top                      # Constant pressure at the outlet
P_outlet = P_top                                # Outlet static pressure
P_backflow = P_top                              # Pressure used when flow reverses into the domain
Sw_backflow = 1.0                               # Backflow water composition
Sg_backflow = 0.0                               # Backflow gas composition
rho_w_backflow = P_backflow / C_w + rho_wr      # Backflow density

#-----------------------------------------------------------------------------------------------------------------------
#=============================================== Initial Densities =====================================================
#-----------------------------------------------------------------------------------------------------------------------
rho_g_list[0, :, :] = P_g_list[0, :, :] / C_g
rho_w_list[0, :, :] = P_w_list[0, :, :] / C_w + rho_wr
rho_g_in = rho_g_list[0, 0, 0]

#-----------------------------------------------------------------------------------------------------------------------
#=============================================== Initial n and m =======================================================
#-----------------------------------------------------------------------------------------------------------------------
n_list[0, :, :] = S_g_list[0, :, :] * rho_g_list[0, :, :]
m_list[0, :, :] = S_w_list[0, :, :] * rho_w_list[0, :, :]

#-----------------------------------------------------------------------------------------------------------------------
#================================================ Functions ============================================================
#-----------------------------------------------------------------------------------------------------------------------
def phi(J, I = None):
    """
    Assign porosity to the whole wellbore.
    Cemented region has a porosity of 0.01, and we assume porosity to be 1 in the uncemented region.
    :param J: Index of the node.
    :param I: Index of the node.

    Returns:
    Porosity
    """
    if I is None:
        print('Please enter the horizontal index of the cell.')

    if cement_check(y_list[J]):
        return phi_c
    else:
        return 1.0


def C_at_interface(C_left, C_right):
    """
    Calculate C_(i+1/2, j) or C_(i, j + 1/2)
    Parameters:
    C_left: C at the grid point i (C_(i, j)
    C_right: C at the grid point i + 1 (C_(i + 1, j))
    or
    C_bottom: C at the grid point j (C_(i, j))
    C_top: C at the grid point j + 1 (C_(i, j + 1))

    Returns:
    C at the interface (i + 1/2): (C_(i + 1/2, j) or at the (j + 1/2): C_(i, j + 1/2))
    """
    return max(C_left, C_right)


def phi_at_interface(phi_left, phi_right):
    """
    Calculate phi_(i + 1/2, j) or phi_(i, j + 1/2)
    Parameters:
    phi_left: Porosity at the grid point i (phi_(i, j))
    phi_right: Porosity at the grid point i + 1 (phi_(i + 1, j))
    or
    phi_bottom: Porosity at the grid point j (phi_(i, j))
    phi_top: Porosity at the grid point j + 1 (phi_(i, j + 1))

    Returns:
    Phi at the interface (i + 1/2): phi_(i + 1/2, j) or at the (j + 1/2): phi_(i, j + 1/2)
    """
    return min(phi_left, phi_right)


def calculate_horizontal_interface_flux(phi_left, phi_right, n_left, n_right, u_g):
    """
    Calculate [φ*n*u_g]_(i + 1/2, j)^k at interface using an upwind scheme (Equation A2).
    This corresponds to the flux terms in equation (A1).

    Upwind rule from equation (A2):
    -If u_g_(i + 1/2, j)^k ≥ 0: Use left value → φ_(i + 1/2, j) * n_(i, j)^k * u_g_(i + 1/2, j)^k
    -If u_g_(i + 1/2, j)^k < 0: Use right value → φ_(i + 1/2, j) * n_(i + 1, j)^k * u_g_(i + 1/2, j)^k

    Parameters:
    phi_left: Porosity at left cell (φ_(i, j))
    phi_right: Porosity at right cell (φ_(i + 1, j))
    n_left: Mass flow at left cell at time k (n_(i, j)^k)
    n_right: Mass flow at right cell at time k (n_(i + 1, j)^k)
    u_g: Horizontal velocity at interface at time k (u_g_(i + 1/2, j)^k)

    Returns:
    Flux: [φ*n*u_g]_(i + 1/2, j)^k at the interface.

    Note: This function is used for both gas (n, u_g) and water (m, u_w) phases.
    """
    phi_average = phi_at_interface(phi_left, phi_right)
    if u_g >= 0:
        return phi_average * n_left * u_g

    else:
        return phi_average * n_right * u_g


def calculate_vertical_interface_flux(phi_bottom, phi_top, n_bottom, n_top, v_g):
    """
    Calculate [φ*n*v_g]_(i, j + 1/2)^k at interface using an upwind scheme (Equation A2).
    This corresponds to the flux terms in equation (A1).

    Upwind rule from equation (A2):
    -If v_g_(i, j + 1/2)^k ≥ 0: Use bottom value → φ_(i, j + 1/2) * n_(i, j)^k * v_g_(i, j + 1/2)^k
    -If v_g_(i, j + 1/2)^k < 0: Use top value → φ_(i, j + 1/2) * n_(i, j + 1)^k * v_g_(i, j + 1/2)^k

    Parameters:
    phi_bottom: Porosity at bottom cell (φ_(i, j))
    phi_top: Porosity at top cell (φ_(i, j + 1))
    n_bottom: Mass flow at bottom cell at time k (n_(i, j)^k)
    n_top: Mass flow at top cell at time k (n_(i, j + 1)^k)
    v_g: Vertical velocity at interface at time k (v_g_(i, j + 1/2)^k)

    Returns:
    Flux: [φ*n*v_g]_(i, j + 1/2)^k at the interface.

    Note: This function is used for both gas (n, v_g) and water (m, v_w) phases.
    """
    phi_average = phi_at_interface(phi_bottom, phi_top)
    if v_g >= 0:
        return phi_average * n_bottom * v_g

    else:
        return phi_average * n_top * v_g


def calculate_n_next_timestep(n_current, n_right, n_left, n_top, n_bottom, phi_current, phi_right, phi_left,
                              phi_top, phi_bottom, u_g_right, u_g_left, v_g_top, v_g_bottom, dt_eff=None):
    """
    Calculate n_(i, j)^(k + 1) based on equation (A1) using an explicit scheme with a
    possibly shorter local time step dt_eff (sub-cycling).
    From equation (A1):
    (φ_(i, j) * n_(i, j)^(k + 1) - φ_(i, j) * n_(i, j)^k) / Δt + (1/Δx) * ([φ*n*u_g]_(i + 1/2, j)^k
     - [φ*n*u_g]_(i - 1/2, j)^k) + (1/Δy) * ([φ*n*v_g]_(i, j + 1/2)^k - [φ*n*v_g]_(i, j - 1/2)^k) = 0

    Rearranging to solve for n_(i, j)^(k + 1):
    n_(i, j)^(k + 1) = n_(i, j)^k - (Δt/(φ_(i, j)) * [([φ*n*u_g]_(i + 1/2, j)^k - [φ*n*u_g]_(i - 1/2, j)^k) / Δx
    + ([φ*n*v_g]_(i, j + 1/2)^k - [φ*n*v_g]_(i, j - 1/2)^k) / Δy]

    Parameters:
    n_current: Mass flow at grid point (i, j) at time k (n_(i, j)^k)
    n_right: Mass flow at grid point (i + 1, j) at time k (n_(i + 1, j)^k)
    n_left: Mass flow at grid point (i - 1, j) at time k (n_(i - 1, j)^k)
    n_top: Mass flow at grid point (i, j + 1) at time k (n_(i, j + 1)^k)
    n_bottom: Mass flow at grid point (i, j - 1) at time k (n_(i, j - 1)^k)
    phi_current: Porosity at grid point (i, j): φ_(i, j)
    phi_right: Porosity at grid point (i + 1, j): φ_(i + 1, j)
    phi_left: Porosity at grid point (i - 1, j): φ_(i - 1, j)
    phi_top: Porosity at grid point (i, j + 1): φ_(i, j + 1)
    phi_bottom: Porosity at grid point (i, j - 1): φ_(i, j - 1)
    u_g_right: Horizontal velocity at (i + 1/2, j) interface at time k (u_g_(i + 1/2, j)^k)
    u_g_left: Horizontal velocity at (i - 1/2, j) interface at time k (u_g_(i - 1/2, j)^k)
    v_g_top: Vertical velocity at (i, j + 1/2) interface at time k (v_g_(i, j + 1/2)^k)
    v_g_bottom: Vertical velocity at (i, j - 1/2) interface at time k (v_g_(i, j - 1/2)^k)

    Returns:
    n_(i, j)^(k + 1): Mass flow at the next time step.

    Note: This function is used for both gas (n, u_g, v_g) and water (m, u_w, v_w) phases.
    """
    left_flux = calculate_horizontal_interface_flux(phi_left, phi_current, n_left, n_current, u_g_left)
    right_flux = calculate_horizontal_interface_flux(phi_current, phi_right, n_current, n_right, u_g_right)
    bottom_flux = calculate_vertical_interface_flux(phi_bottom, phi_current, n_bottom, n_current, v_g_bottom)
    top_flux = calculate_vertical_interface_flux(phi_current, phi_top, n_current, n_top, v_g_top)

    dt_loc = dt if dt_eff is None else dt_eff

    return n_current - (dt_loc / phi_current) * ((right_flux - left_flux) / dx + (top_flux - bottom_flux) / dy)


def C_1(J):
    """
    Calculate C_1 based on equation (A12).
    """
    if cement_check(y_list[J]):
        return I_w_c * miu_w / K_c

    else:
        return I_w_a * miu_w


def F_1(I, J, K, Q):
    """
    Calculate F_1 based on equation (A13).
    """
    if cement_check(y_list[J]):
        return S_w_list[K + Q, J, I] ** alpha_2

    else:
        return S_w_list[K + Q, J, I] ** alpha_1


def F_2(I, J, K, Q):
    """
    Calculate F_2 based on equation (A13) for gas.
    """
    if cement_check(y_list[J]):
        return S_g_list[K + Q, J, I] ** beta_2

    else:
        return S_g_list[K + Q, J, I] ** beta_1


def C(J):
    """
    Calculate C based on equation (A15).
    """
    if cement_check(y_list[J]):
        return I_c * miu_w * miu_g / K_c

    else:
        return I_a * miu_w * miu_g


def C_2(J):
    """
    Calculate C_2 based on equation (A12) for gas.
    """
    if cement_check(y_list[J]):
        return I_g_c * miu_g / K_c

    else:
        return I_g_a * miu_g


def F(I, J, K, Q):
    """
    Calculate F based on equation (A16).
    """
    return S_w_list[K + Q, J, I] * S_g_list[K + Q, J, I]


def calculate_interaction_coefficient_water(I, J, K, Q, direction):
    """
    Calculate interaction coefficient for water k_w based on equation (A11).

    Returns:
    k_w: Interaction coefficient for water.
    """
    # Calculate F1 based on the velocity direction.
    if direction == 'horizontal':
        velocity = u_w_list[K, J, I + 1]                # Water horizontal velocity at (i + 1/2, j)
        if velocity > 0:
            F1 = F_1(I, J, K, Q)

        elif velocity == 0:
            F1 = (F_1(I, J, K, Q) + F_1(I + 1, J, K, Q)) / 2

        else:
            F1 = F_1(I + 1, J, K, Q)

        return C_at_interface(C_1(J), C_1(J)) * phi_at_interface(phi(J, I), phi(J, I + 1)) * F1

    elif direction == 'vertical':
        velocity = v_w_list[K, J + 1, I]                # Water vertical velocity at (i, j + 1/2)
        if velocity > 0:
            F1 = F_1(I, J, K, Q)

        elif velocity == 0:
            F1 = (F_1(I, J, K, Q) + F_1(I, J + 1, K, Q)) / 2

        else:
            F1 = F_1(I, J + 1, K, Q)

        return C_at_interface(C_1(J), C_1(J + 1)) * phi_at_interface(phi(J, I), phi(J + 1, I)) * F1

    else:
        print('Invalid direction')
        return None


def calculate_interaction_coefficient_gas(I, J, K, Q, direction):
    """
    Calculate interaction coefficient for gas k_g based on equation (A11).

    Returns:
    k_w: Interaction coefficient for water.
    """
    # Calculate F2 based on the velocity direction.
    if direction == 'horizontal':
        velocity = u_g_list[K, J, I + 1]
        if velocity > 0:
            F2 = F_2(I, J, K, Q)

        elif velocity == 0:
            F2 = (F_2(I, J, K, Q) + F_2(I + 1, J, K, Q)) / 2

        else:
            F2 = F_2(I + 1, J, K, Q)

        return C_at_interface(C_2(J), C_2(J)) * phi_at_interface(phi(J, I), phi(J, I + 1)) * F2

    elif direction == 'vertical':
        velocity = v_g_list[K, J + 1, I]
        if velocity > 0:
            F2 = F_2(I, J, K, Q)

        elif velocity == 0:
            F2 = (F_2(I, J, K, Q) + F_2(I, J + 1, K, Q)) / 2

        else:
            F2 = F_2(I, J + 1, K, Q)

        return C_at_interface(C_2(J), C_2(J + 1)) * phi_at_interface(phi(J, I), phi(J + 1, I)) * F2

    else:
        print('Invalid direction')
        return None


def calculate_interaction_coefficient_water_gas(I, J, K, Q, direction):
    """
    Calculate interaction coefficient for water-gas k based on equation (A14).

    Returns:
    k_hat: Interaction coefficient for water-gas.
    """
    # Calculate FF based on the velocity direction.
    if direction == 'horizontal':
        gas_velocity = u_g_list[K, J, I + 1]
        water_velocity = u_w_list[K, J, I + 1]
        if (water_velocity > 0) and (gas_velocity > 0):
            FF = F(I, J, K, Q)

        elif (water_velocity < 0) and (gas_velocity < 0):
            FF = F(I + 1, J, K, Q)

        else:
            FF = (F(I, J, K, Q) + F(I + 1, J, K, Q)) / 2

        return C_at_interface(C(J), C(J)) * phi_at_interface(phi(J, I), phi(J, I + 1)) * FF

    elif direction == 'vertical':
        gas_velocity = v_g_list[K, J + 1, I]
        water_velocity = v_w_list[K, J + 1, I]
        if (water_velocity > 0) and (gas_velocity > 0):
            FF = F(I, J, K, Q)

        elif (water_velocity < 0) and (gas_velocity < 0):
            FF = F(I, J + 1, K, Q)

        else:
            FF = (F(I, J + 1, K, Q) + F(I, J, K, Q)) / 2

        return C_at_interface(C(J), C(J + 1)) * phi_at_interface(phi(J, I), phi(J + 1, I)) * FF

    else:
        print('Invalid direction')
        return None


def calculate_upwind_scheme(S_left, S_right, velocity):
    """
    Calculate upwind saturation based on the velocity direction. (equation A10)
    Parameters:
    S_left: Saturation at left cell (S_(i, j)^(k + 1/2))
    S_right: Saturation at right cell (S_(i + 1, j)^(k + 1/2))
    S_bottom: Saturation at bottom cell (S_(i, j - 1)^(k + 1/2)) ~ S_left for the vertical direction.
    S_top: Saturation at top cell (S_(i, j + 1)^(k + 1/2)) ~ S_right for the vertical direction.
    velocity: Velocity at interface (u_(i + 1/2, j)^k) or (v_(i, j + 1/2)^k)
    This function is used for both water and gas phases. (m, S_w) and (n, S_g)

    Returns:
    Upwind saturation S_(i + 1/2, j)^(k + 1/2) or S_(i, j + 1/2)^(k + 1/2)
    """
    if velocity > 0:
        return S_left

    elif velocity == 0:
        return (S_left + S_right) / 2

    else:
        return S_right


def upwind_then_average(phi_bottom_left, phi_top_left, phi_bottom_right, phi_top_right, n_bottom_left, n_top_left,
                        n_bottom_right, n_top_right, v_left, v_right, direction = None):
    """
    Calculate φ(j + 1/2, i + 1/2) * n_(j + 1/2, i + 1/2)^(k + 1) using the upwind-then-average method.
    Parameters:
        phi_bottom_left = φ(j, i)
        phi_top_left = φ(j + 1, i)
        phi_bottom_right = φ(j, i + 1)
        phi_top_right = φ(j + 1, i + 1)
        n_bottom_left = n(j, i)
        n_top_left = n(j + 1, i)
        n_bottom_right = n(j + 1, i + 1)
        n_top_right = n(j + 1, i + 1)
        v_left = v(j + 1/2, i) or v[j + 1, i]
        v_right = v(j + 1/2, i + 1) or v[j + 1, i + 1]
        direction = vertical or horizontal
    This function can be used for both water and gas phases. (m, v_w, u_w) and (n, v_g, u_g)
    When the direction is horizontal, v_left and v_right will be u_down and u_up respectively.
    So: u_down = u(j, i + 1/2) or u[j, i + 1], and u_up = u(j + 1, i + 1/2) or u[j + 1, i + 1].
    """
    if direction == 'vertical':
        phi_left_face = phi_at_interface(phi_bottom_left, phi_top_left)
        n_left_face = calculate_upwind_scheme(n_bottom_left, n_top_left, v_left)
        phi_right_face = phi_at_interface(phi_bottom_right, phi_top_right)
        n_right_face = calculate_upwind_scheme(n_bottom_right, n_top_right, v_right)
        return (phi_left_face * n_left_face + phi_right_face * n_right_face) / 2

    elif direction == 'horizontal':
        phi_bottom_face = phi_at_interface(phi_bottom_left, phi_bottom_right)
        n_bottom_face = calculate_upwind_scheme(n_bottom_left, n_bottom_right, v_left)
        phi_top_face = phi_at_interface(phi_top_left, phi_top_right)
        n_top_face = calculate_upwind_scheme(n_top_left, n_top_right, v_right)
        return (phi_bottom_face * n_bottom_face + phi_top_face * n_top_face) / 2

    else:
        print('Invalid direction.')
        return 0


def validate_saturations(S_w, S_g, time_step):
    """
    Validate that saturations sum to 1 and are within physical bounds.
    """
    total_s = S_w + S_g
    min_sat = np.min(total_s)
    max_sat = np.max(total_s)

    if abs(max_sat - 1.0) > 1E-10 or abs(min_sat - 1.0) > 1E-10:
        print(f"⚠ WARNING at time step {time_step}: Saturation sum not equal to 1.")
        print(f"Min sum: {min_sat:.10f}, Max sum: {max_sat:.10f}.")

    if np.any(S_w < 0) or np.any(S_g < 0):
        print(f"⚠ WARNING at time step {time_step}: Negative saturations detected.")

    if np.any(S_w > 1) or np.any(S_g > 1):
        print(f"⚠ WARNING at time step {time_step}: Saturations > 1 detected.")

#-----------------------------------------------------------------------------------------------------------------------
#============================================ Explicit mass transport ==================================================
#-----------------------------------------------------------------------------------------------------------------------
# k = current time step, k + 1 = next time step.
for k in range(0, M, 2):                        # k goes from 0 to M - 1.
    k_next = k + 2
    q = 1                                       # (k + 1/2) time step ~ k + q
    if k % 200 == 0:
        print(f"Time step {k_next / 2}.")

    # Check stability condition using velocities at time k.
    max_horizontal_gas_velocity = np.max(np.abs(u_g_list[k, :, :]))
    max_horizontal_water_velocity = np.max(np.abs(u_w_list[k, :, :]))
    max_vertical_gas_velocity = np.max(np.abs(v_g_list[k, :, :]))
    max_vertical_water_velocity = np.max(np.abs(v_w_list[k, :, :]))
    max_u = max(max_horizontal_gas_velocity, max_horizontal_water_velocity)
    max_v = max(max_vertical_gas_velocity, max_vertical_water_velocity)
    max_velocity = max(max_u, max_v)
    cfl_x = (max_u * dt / dx) if max_u > 0 else 0.0
    cfl_y = (max_v * dt / dy) if max_v > 0 else 0.0

    if cfl_x > cfl_y:
        cfl = cfl_x
        max_cfl_velocity = max_u
        direction_of_cfl = 'Horizontal'
        if max_cfl_velocity == max_horizontal_gas_velocity:
            phase_with_high_cfl = 'Gas'

        else:
            phase_with_high_cfl = 'Water'

    else:
        cfl = cfl_y
        max_cfl_velocity = max_v
        direction_of_cfl = 'Vertical'
        if max_cfl_velocity == max_vertical_gas_velocity:
            phase_with_high_cfl = 'Gas'

        else:
            phase_with_high_cfl = 'Water'

    if max_velocity == max_vertical_gas_velocity:
        direction_with_high_velocity = 'Vertical'
        phase_with_high_velocity = 'Gas'

    elif max_velocity == max_horizontal_gas_velocity:
        direction_with_high_velocity = 'Horizontal'
        phase_with_high_velocity = 'Gas'

    elif max_velocity == max_vertical_water_velocity:
        direction_with_high_velocity = 'Vertical'
        phase_with_high_velocity = 'Water'

    else:
        direction_with_high_velocity = 'Horizontal'
        phase_with_high_velocity = 'Water'

    if cfl > CFL_TARGET:
        if k % 200 == 0:
            print("⚠ WARNING: Stability condition violated!")
            print(f"CFL number: {cfl:.4f}.")
            if direction_of_cfl == 'Horizontal':
                print(f"Maximum CFL Velocity: {max_cfl_velocity:.6f} m/s.")
                print(f"Direction of maximum CFL velocity: {direction_of_cfl}, Phase with high CFL velocity: {phase_with_high_cfl}.")

                if max_velocity > max_cfl_velocity:
                    print(f"Maximum velocity: {max_velocity:.6f} m/s.")
                    print(f"Direction of maximum velocity: {direction_with_high_velocity}, Phase with high velocity: {phase_with_high_velocity}.")

            elif direction_of_cfl == 'Vertical':
                print(f"Maximum velocity: {max_v:.6f} m/s.")
                print(f"Direction of maximum CFL velocity: {direction_of_cfl}, Phase with high CFL velocity: {phase_with_high_cfl}.")

                if max_velocity > max_cfl_velocity:
                    print(f"Maximum velocity: {max_velocity:.6f} m/s.")
                    print(f"Direction of maximum velocity: {direction_with_high_velocity}, Phase with high velocity: {phase_with_high_velocity}.")

    # Step 1: From equation (A1): Calculate n_(i, j)^(k + 1) and m_(i, j)^(k + 1)
    n_sub = max(1, int(np.ceil(cfl / CFL_TARGET)))
    n_sub = min(n_sub, CFL_CAP)                         # Hard cap to avoid astronomical sub-cycling.
    dt_local = dt / n_sub
    if n_sub > 1:
        if k % 200 == 0:
            print(f"CFL={cfl:.2f} > {CFL_TARGET} → Sub-cycling {n_sub}× with dt_local={dt_local:.5e}s.")

    # Temporary arrays that start from time level k.
    n_tmp = n_list[k, :, :].copy()
    m_tmp = m_list[k, :, :].copy()

    for _ in range(n_sub):
        n_new = n_tmp.copy()
        m_new = m_tmp.copy()
        for j in range(N_y):
            for i in range (N_x):
                if j == 0:                                  # Bottom boundary (no bottom neighbor in the scheme.)
                    # There isn't any cell with j - 1 index. We assume there is a gas tank at the bottom of the domain.
                    if i == 0:                                  # Bottom left corner.
                        # For the left boundary (i - 1), we assume phi, n, and m = 0.
                        n_new[j, i] = calculate_n_next_timestep(n_tmp[j, i], n_tmp[j, i + 1], 0.0,
                                                                n_tmp[j + 1, i], rho_g_in, phi(j, i),
                                                                phi(j, i + 1), 0.0, phi(j + 1, i),1.0,
                                                                u_g_list[k, j, i + 1], u_g_list[k, j, i],
                                                                v_g_list[k, j + 1, i], v_g_list[k, j, i],
                                                                dt_eff = dt_local)

                        m_new[j, i] = calculate_n_next_timestep(m_tmp[j, i], m_tmp[j, i + 1], 0.0,
                                                                m_tmp[j + 1, i], 0.0, phi(j, i), phi(j, i + 1),
                                                                0.0, phi(j + 1, i),1.0,
                                                                u_w_list[k, j, i + 1], u_w_list[k, j, i],
                                                                v_w_list[k, j + 1, i], v_w_list[k, j, i],
                                                                dt_eff = dt_local)

                    elif i == N_x - 1:                          # Bottom right corner.
                        # For the right boundary (i + 1), we assume phi, n, and m = 0.
                        n_new[j, i] = calculate_n_next_timestep(n_tmp[j, i], 0.0, n_tmp[j, i - 1],
                                                                n_tmp[j + 1, i], rho_g_in, phi(j, i),0.0,
                                                                phi(j, i - 1), phi(j + 1, i),1.0,
                                                                u_g_list[k, j, i + 1], u_g_list[k, j, i],
                                                                v_g_list[k, j + 1, i], v_g_list[k, j, i],
                                                                dt_eff = dt_local)

                        m_new[j, i] = calculate_n_next_timestep(m_tmp[j, i], 0.0, m_tmp[j, i - 1],
                                                                m_tmp[j + 1, i], 0.0, phi(j, i), 0.0,
                                                                phi(j, i - 1), phi(j + 1, i),1.0,
                                                                u_w_list[k, j, i + 1], u_w_list[k, j, i],
                                                                v_w_list[k, j + 1, i], v_w_list[k, j, i],
                                                                dt_eff = dt_local)

                    else:                                       # Middle cells of the bottom row.
                        n_new[j, i] = calculate_n_next_timestep(n_tmp[j, i], n_tmp[j, i + 1], n_tmp[j, i - 1],
                                                                n_tmp[j + 1, i], rho_g_in, phi(j, i),
                                                                phi(j, i + 1), phi(j, i - 1), phi(j + 1, i),
                                                                1.0, u_g_list[k, j, i + 1], u_g_list[k, j, i],
                                                                v_g_list[k, j + 1, i], v_g_list[k, j, i],
                                                                dt_eff = dt_local)

                        m_new[j, i] = calculate_n_next_timestep(m_tmp[j, i], m_tmp[j, i + 1], m_tmp[j, i - 1],
                                                                m_tmp[j + 1, i], 0.0, phi(j, i), phi(j, i + 1),
                                                                phi(j, i - 1), phi(j + 1, i),1.0,
                                                                u_w_list[k, j, i + 1], u_w_list[k, j, i],
                                                                v_w_list[k, j + 1, i], v_w_list[k, j, i],
                                                                dt_eff = dt_local)

                elif j == N_y - 1:                          # Top boundary (no top neighbor in the scheme.)
                    # There isn't any cell with j + 1 index. We assume there is a water tank at the top of the domain.
                    n_top_ghost = n_tmp[j, i]
                    m_top_ghost = m_tmp[j, i]
                    phi_top_ghost = phi(j, i)
                    vg_top = v_g_list[k, j + 1, i]
                    vw_top = v_w_list[k, j + 1, i]
                    # If flow reverses (velocity < 0), inject the backflow into the upwind flux.
                    if vg_top < 0.0:
                        if cement_check(y_list[j]):
                            Pc_bf = -P_star_c1 * np.log(delta_1 + (Sw_backflow / a_1))

                        else:
                            Pc_bf = 0.0

                        rho_g_backflow = (P_backflow + Pc_bf) / C_g
                        n_top_ghost = Sg_backflow * rho_g_backflow

                    if vw_top < 0.0:
                        m_top_ghost = Sw_backflow * rho_w_backflow

                    if i == 0.0:                                # Top left corner.
                        # For the left boundary (i - 1), we assume phi, n, and m = 0.
                        n_new[j, i] = calculate_n_next_timestep(n_tmp[j, i], n_tmp[j, i + 1], 0.0, n_top_ghost,
                                                                n_tmp[j - 1, i], phi(j, i), phi(j, i + 1), 0.0,
                                                                phi_top_ghost, phi(j - 1, i),
                                                                u_g_list[k, j, i + 1], u_g_list[k, j, i],
                                                                vg_top, v_g_list[k, j, i], dt_eff = dt_local)

                        m_new[j, i] = calculate_n_next_timestep(m_tmp[j, i], m_tmp[j, i + 1], 0.0, m_top_ghost,
                                                                m_tmp[j - 1, i], phi(j, i), phi(j, i + 1), 0.0,
                                                                phi_top_ghost, phi(j - 1, i),
                                                                u_w_list[k, j, i + 1], u_w_list[k, j, i],
                                                                vw_top, v_w_list[k, j, i], dt_eff = dt_local)

                    elif i == N_x - 1:                          # Top right corner.
                        # For the right boundary (i + 1), we assume phi, n, and m = 0.
                        n_new[j, i] = calculate_n_next_timestep(n_tmp[j, i], 0.0, n_tmp[j, i - 1],n_top_ghost,
                                                                n_tmp[j - 1, i], phi(j, i),0.0, phi(j, i - 1),
                                                                phi_top_ghost, phi(j - 1, i),
                                                                u_g_list[k, j, i + 1], u_g_list[k, j, i],
                                                                vg_top, v_g_list[k, j, i], dt_eff = dt_local)

                        m_new[j, i] = calculate_n_next_timestep(m_tmp[j, i],0.0, m_tmp[j, i - 1], m_top_ghost,
                                                                m_tmp[j - 1, i], phi(j, i),0.0, phi(j, i - 1),
                                                                phi_top_ghost, phi(j - 1, i),
                                                                u_w_list[k, j, i + 1], u_w_list[k, j, i],
                                                                vw_top, v_w_list[k, j, i], dt_eff = dt_local)

                    else:                                       # Middle cells of the top row.
                        n_new[j, i] = calculate_n_next_timestep(n_tmp[j, i], n_tmp[j, i + 1], n_tmp[j, i - 1],
                                                                n_top_ghost, n_tmp[j - 1, i], phi(j, i), phi(j, i + 1),
                                                                phi(j, i - 1),phi_top_ghost, phi(j - 1, i),
                                                                u_g_list[k, j, i + 1], u_g_list[k, j, i],
                                                                vg_top, v_g_list[k, j, i], dt_eff = dt_local)

                        m_new[j, i] = calculate_n_next_timestep(m_tmp[j, i], m_tmp[j, i + 1], m_tmp[j, i - 1],
                                                                m_top_ghost, m_tmp[j - 1, i], phi(j, i), phi(j, i + 1),
                                                                phi(j, i - 1),phi_top_ghost, phi(j - 1, i),
                                                                u_w_list[k, j, i + 1], u_w_list[k, j, i],
                                                                vw_top, v_w_list[k, j, i], dt_eff = dt_local)

                else:                                       # Middle rows
                    if i == 0:                              # Left boundary of the middle row.
                        # For the left boundary (i - 1), we assume phi, n, and m = 0.
                        n_new[j, i] = calculate_n_next_timestep(n_tmp[j, i], n_tmp[j, i + 1], 0.0,
                                                                n_tmp[j + 1, i], n_tmp[j - 1, i], phi(j, i),
                                                                phi(j, i + 1), 0.0, phi(j + 1, i), phi(j - 1, i),
                                                                u_g_list[k, j, i + 1], u_g_list[k, j, i],
                                                                v_g_list[k, j + 1, i], v_g_list[k, j, i],
                                                                dt_eff = dt_local)

                        m_new[j, i] = calculate_n_next_timestep(m_tmp[j, i], m_tmp[j, i + 1],0.0,
                                                                m_tmp[j + 1, i], m_tmp[j - 1, i], phi(j, i),
                                                                phi(j, i + 1),0.0, phi(j + 1, i), phi(j - 1, i),
                                                                u_w_list[k, j, i + 1], u_w_list[k, j, i],
                                                                v_w_list[k, j + 1, i], v_w_list[k, j, i],
                                                                dt_eff = dt_local)

                    elif i == N_x - 1:                          # Right boundary of the middle row
                        # For the right boundary (i + 1), we assume phi, n, and m = 0.
                        n_new[j, i] = calculate_n_next_timestep(n_tmp[j, i],0.0, n_tmp[j, i - 1],
                                                                n_tmp[j + 1, i], n_tmp[j - 1, i], phi(j, i),0.0,
                                                                phi(j, i - 1), phi(j + 1, i), phi(j - 1, i),
                                                                u_g_list[k, j, i + 1], u_g_list[k, j, i],
                                                                v_g_list[k, j + 1, i], v_g_list[k, j, i],
                                                                dt_eff = dt_local)

                        m_new[j, i] = calculate_n_next_timestep(m_tmp[j, i],0.0, m_tmp[j, i - 1],
                                                                m_tmp[j + 1, i], m_tmp[j - 1, i], phi(j, i),0.0,
                                                                phi(j, i - 1), phi(j + 1, i), phi(j - 1, i),
                                                                u_w_list[k, j, i + 1], u_w_list[k, j, i],
                                                                v_w_list[k, j + 1, i], v_w_list[k, j, i],
                                                                dt_eff = dt_local)

                    else:                                       # Middle cells of the middle row.
                        n_new[j, i] = calculate_n_next_timestep(n_tmp[j, i], n_tmp[j, i + 1], n_tmp[j, i - 1],
                                                                n_tmp[j + 1, i], n_tmp[j - 1, i],
                                                                phi(j, i), phi(j, i + 1), phi(j, i - 1),
                                                                phi(j + 1, i), phi(j - 1, i),
                                                                u_g_list[k, j, i + 1], u_g_list[k, j, i],
                                                                v_g_list[k, j + 1, i], v_g_list[k, j, i],
                                                                dt_eff = dt_local)

                        m_new[j, i] = calculate_n_next_timestep(m_tmp[j, i], m_tmp[j, i + 1], m_tmp[j, i - 1],
                                                                m_tmp[j + 1, i], m_tmp[j - 1, i],
                                                                phi(j, i), phi(j, i + 1), phi(j, i - 1),
                                                                phi(j + 1, i), phi(j - 1, i),
                                                                u_w_list[k, j, i + 1], u_w_list[k, j, i],
                                                                v_w_list[k, j + 1, i], v_w_list[k, j, i],
                                                                dt_eff = dt_local)

        # Positivity clamp for mass after each sub-step
        n_tmp[:, :] = np.maximum(n_new, 0.0)
        m_tmp[:, :] = np.maximum(m_new, 0.0)

    # Write the sub-cycled result to k_next.
    n_list[k_next, :, :] = n_tmp[:, :]
    m_list[k_next, :, :] = m_tmp[:, :]

    #--- Compute S*^(k + 1/2) after the sub-cycles, using densities at time k (A3,A6) ---
    for j in range(N_y):
        for i in range(N_x):
            rho_g_k = rho_g_list[k, j, i]
            rho_w_k = rho_w_list[k, j, i]
            if (not np.isfinite(rho_g_k)) or (rho_g_k <= 0) or (not np.isfinite(rho_w_k)) or (rho_w_k <= 0):
                raise RuntimeError(f"Bad densities at k = {k}, i = {i}, j = {j}: "
                                   f"rho_g = {rho_g_k}, rho_w = { rho_w_k}.")

            s_g_star = n_list[k_next, j, i] / rho_g_k
            s_w_star = m_list[k_next, j, i] / rho_w_k
            total_sat = s_w_star + s_g_star
            if (total_sat <= 0) or (not np.isfinite(total_sat)):
                raise RuntimeError(f"Non-physical saturation values at k = {k}, i = {i}, j = {j}.")

            # Normalized half-step saturations (A7)
            S_w_list[k + q, j, i] = s_w_star / total_sat
            S_g_list[k + q, j, i] = 1.0 - S_w_list[k + q, j, i]

            # Capillary pressure and P_g at half-step for Step-2 coefficients.
            if cement_check(y_list[j]):
                Pc_list[k + q, j, i] = -P_star_c1 * np.log(delta_1 + (S_w_list[k + q, j, i] / a_1))

            else:
                Pc_list[k + q, j, i] = 0.0

            P_g_list[k + q, j, i] = P_w_list[k, j, i] + Pc_list[k + q, j, i]

#-----------------------------------------------------------------------------------------------------------------------
#================================= Implicit computation of velocities and pressures ====================================
#-----------------------------------------------------------------------------------------------------------------------
    # Gas and water velocities at the time step k + 1: u_g_(i + 1/2, j)^(k + 1), v_g_(i, j + 1/2)^(k + 1),
    # u_w_(i + 1/2, j)^(k + 1), and v_w_(i, j + 1/2)^(k + 1); and water pressure at the time step k + 1:
    # P_w_(i, j)^(k + 1) are implicitly and simultaneously calculated, through solving the system of linear equations
    # AX = B. (A8) and (A9).
    #The system of linear equations is solved iteratively for convergence.

    #---- Iterate to convergence ----
    max_iter = 20
    tol_pw = 1                                                                                                          #Pa
    tol_u = 1E-6                                                                                                        #m/s

    for it in range(max_iter):
        q = 1 if it == 0 else 2                     # The first pass uses k + 1/2; later passes use k + 1.
        if it > 0:
            Pw_prev = P_w_list[k_next, : N_y - 1, : ].copy()
            vg_prev = v_g_list[k_next, 1:, :].copy()
            vw_prev = v_w_list[k_next, 1:, :].copy()
            ug_prev = u_g_list[k_next, :, 1: N_x].copy()
            uw_prev = u_w_list[k_next, :, 1: N_x].copy()

        # q = 1: Build coefficients from the half‑step (k + 1/2).
        # q = 2: Rebuild with the full step (k + 1) = k_next, repeat A8 and A9 solve.
        # Solve pressure evolution equation (A8) for interior cells.
        for j in range(N_y):
            for i in range(N_x):
                # Capillary pressure derivative
                P_c_prime = (-P_star_c1 / (a_1 * delta_1 + S_w_list[k + q, j, i])) if cement_check(y_list[j]) else 0.0
                rho_g_list[k + q, j, i] = P_g_list[k + q, j, i] / C_g
                # Calculate rho_g_tilde from equation 18.
                rho_g_tilde[k + q, j, i] = rho_g_list[k + q, j, i] - (S_g_list[k + q, j, i] * P_c_prime) / C_g
                # Calculate η̃ from equation 18.
                denominator = (S_g_list[k + q, j, i] * rho_w_list[k, j, i] * C_w +
                               S_w_list[k + q, j, i] * rho_g_tilde[k + q, j, i] * C_g)
                eta_tilde[k + q, j, i] = (C_w * C_g / phi(j, i)) / denominator

        #===== Early Active Set Identification =====
        # Initialize activity arrays for all vertical interfaces: all rows and all columns except walls and the top row.
        active_uw = np.zeros((N_y - 1, N_x - 1), dtype = bool)    # Water is active at the interface i + 1/2.
        active_ug = np.zeros((N_y - 1, N_x - 1), dtype = bool)    # Gas is active at the interface i + 1/2.
        # Initialize activity arrays for all horizontal interfaces: all columns and all rows except j = 0, and N_y - 1.
        active_vw = np.zeros((N_y - 1, N_x), dtype = bool)    # Water is active at the interface j + 1/2.
        active_vg = np.zeros((N_y - 1, N_x), dtype = bool)    # Gas is active at the interface j + 1/2.

        # Determine interface activity based on the upwind saturations.
        for j in range(N_y - 1):
            # We assume horizontal velocities are 0 at both walls. There will be (N_x - 1) column of interfaces.
            for i_ifc in range(N_x - 1):               #i_ifc = 0, 1, 2, ..., N-2 (interface i_ifc + 1/2)
                # Upwind interface saturation using velocities at time k.
                S_w_ifc = calculate_upwind_scheme(S_w_list[k + q, j, i_ifc], S_w_list[k + q, j, i_ifc + 1],
                                                  u_w_list[k, j, i_ifc + 1])
                S_g_ifc = calculate_upwind_scheme(S_g_list[k + q, j, i_ifc], S_g_list[k + q, j, i_ifc + 1],
                                                  u_g_list[k, j, i_ifc + 1])

                active_uw[j, i_ifc] = (S_w_ifc > SAT_TOL)
                active_ug[j, i_ifc] = (S_g_ifc > SAT_TOL)
                # (j, i + 1/2) interface is between (j, i) and (j, i + 1) cells, and we will go from 0 to N_x - 2.

        for i in range(N_x):
            for j_ifc in range(N_y - 1):
                S_w_ifc = calculate_upwind_scheme(S_w_list[k + q, j_ifc, i], S_w_list[k + q, j_ifc + 1, i],
                                                  v_w_list[k, j_ifc + 1, i])
                S_g_ifc = calculate_upwind_scheme(S_g_list[k + q, j_ifc, i], S_g_list[k + q, j_ifc + 1, i],
                                                  v_g_list[k, j_ifc + 1, i])

                active_vw[j_ifc, i] = (S_w_ifc > SAT_TOL)
                active_vg[j_ifc, i] = (S_g_ifc > SAT_TOL)
                # (j + 1/2, i) interface is between (j, i) and (j + 1, i) cells, and we will go from 0 to N_y - 2.

        # Count active interfaces.
        n_active_uw = np.sum(active_uw)
        n_active_ug = np.sum(active_ug)
        n_active_vw = np.sum(active_vw)
        n_active_vg = np.sum(active_vg)

        if (k % 200 == 0) and (q == 2):
            print(f"[ActiveSet] Active water vertical interfaces: {n_active_uw}/{(N_y - 1) * (N_x - 1)}.")
            print(f"[ActiveSet] Active gas vertical interfaces: {n_active_ug}/{(N_y - 1) * (N_x - 1)}.")
            print(f"[ActiveSet] Active water horizontal interfaces: {n_active_vw}/{(N_y - 1) * N_x}.")
            print(f"[ActiveSet] Active gas horizontal interfaces: {n_active_vg}/{(N_y - 1) * N_x}.")

        #===== Create Reduced Variable Mapping =====
        # All pressure unknowns are always active (one per cell).
        # We know the values of pressure at the last row (Pressure outlet).
        n_pressure_vars = (N_y - 1) * N_x                                # P_w[k, j, i]

        # Create active interface index arrays.
        row1, col1 = np.where(active_uw)                                    # Global indices where water is active.
        active_uw_indices = np.column_stack((row1, col1))                   # Pair up row and col indices.
        row2, col2 = np.where(active_ug)                                    # Global indices where gas is active.
        active_ug_indices = np.column_stack((row2, col2))                   # Pair up row and col indices.
        row3, col3 = np.where(active_vw)                                    # Global indices where water is active.
        active_vw_indices = np.column_stack((row3, col3))                   # Pair up row and col indices.
        row4, col4 = np.where(active_vg)                                    # Global indices where gas is active.
        active_vg_indices = np.column_stack((row4, col4))                   # Pair up row and col indices.
        uw_pos_map = -np.ones_like(active_uw, dtype = np.int32)
        uw_pos_map[active_uw] = np.arange(active_uw_indices.shape[0], dtype = np.int32)
        ug_pos_map = -np.ones_like(active_ug, dtype = np.int32)
        ug_pos_map[active_ug] = np.arange(active_ug_indices.shape[0], dtype = np.int32)
        vw_pos_map = -np.ones_like(active_vw, dtype = np.int32)
        vw_pos_map[active_vw] = np.arange(active_vw_indices.shape[0], dtype = np.int32)
        vg_pos_map = -np.ones_like(active_vg, dtype = np.int32)
        vg_pos_map[active_vg] = np.arange(active_vg_indices.shape[0], dtype = np.int32)

        # Total reduced system size.
        n_reduced_vars = n_pressure_vars + n_active_vg + n_active_vw + n_active_ug + n_active_uw
        # n_total = [N_x * (N_y - 1)] + [N_x * (N_y - 1)] + [N_x * (N_y - 1)] +
        # [(N_x - 1) * (N_y - 1)] + [(N_x - 1) * (N_y - 1)] = 2 * [(N_x - 1) * (N_y - 1)] + 3 * [N_x * (N_y - 1)]
        # n_total = (5N_x - 2) * (N_y - 1) = 5 * N_x * N_y - 5 * N_x - 2 * N_y + 2
        if (k % 200 == 0) and (q == 2):
            print(f"[ActiveSet] Reduced system size: {n_reduced_vars} "
                  f"(vs full size: {5 * N_x * N_y - 5 * N_x - 2 * N_y + 2}).")

        # Variable layout in the reduced system:
        # [0: n_pressure_vars]                                                                                          = P_w[i, j]
        # [n_pressure_vars: n_pressure_vars + n_active_vg)                                                              = v_g active interfaces.
        # [n_pressure_vars + n_active_vg: n_pressure_vars + n_active_vg + n_active_vw)                                  = v_w active interfaces.
        # [n_pressure_vars + n_active_vg + n_active_vw: n_pressure_vars + n_active_vg + n_active_vw + n_active_ug]      = u_g active interfaces.
        # [n_pressure_vars + n_active_vg + n_active_vw + n_active_ug: n_reduced_vars]                                   = u_w active interfaces.
        p_start_id = 0
        vg_start_id = p_start_id + n_pressure_vars
        vw_start_id = vg_start_id + n_active_vg
        ug_start_id = vw_start_id + n_active_vw
        uw_start_id = ug_start_id + n_active_ug

        #===== Build reduced system matrix =====
        A_reduced = lil_matrix((n_reduced_vars, n_reduced_vars), dtype = np.float64)
        b_reduced = np.zeros(n_reduced_vars)

        #==================================== Pressure Evolution Equation (A8) =========================================
        # Build equation A8 for the interior cells: j = 0 to N_y - 2, and i = 0 to N_x - 1
        for j in range(N_y - 1):
            for i in range(N_x):
                eq_idx = (j * N_x) + i                          #Equation index: 0 to (N_y - 2) * N_x
                var_idx = (j * N_x) + i                         #Variable index for P_w_(j, i)^(k_next)

                # Equation A8
                # Coefficient for P_w(j, i)^(k_next)
                A_reduced[eq_idx, var_idx] += 1.0 / dt

                # Right hand side: [P_w(j, i)^k] / dt
                b_reduced[eq_idx] += P_w_list[k, j, i] / dt

                # Mass flux terms from equation A8:       [φ * n * u_g], [φ * n * v_g], [φ * m * u_w], and [φ * m * v_w]
                # Coefficient for v_g(j + 1/2, i)^(k_next) - Only if gas is active at this interface.
                # Find the position of this gas interface in the reduced system.
                vg_pos_in_active = vg_pos_map[j, i]                           # Position in the active list.
                if vg_pos_in_active >= 0:                                     # Gas is active at interface (j + 1/2, i).
                    vg_idx = vg_start_id + vg_pos_in_active                   # Position in the reduced system.
                    v_g_sig = v_g_list[k, j + 1, i]
                    phi_avg_right = phi_at_interface(phi(j, i), phi(j + 1, i))
                    n_upwind = calculate_upwind_scheme(n_list[k_next, j, i], n_list[k_next, j + 1, i], v_g_sig)
                    A_reduced[eq_idx, vg_idx] += (eta_tilde[k + q, j, i] * rho_w_list[k, j, i]
                                                  * phi_avg_right * n_upwind / dy)

                # Coefficient for v_g(j - 1/2, i)^(k_next) - Only if gas is active at this interface.
                if j > 0:
                    # Find the position of this gas interface in the reduced system.
                    vg_pos_in_active = vg_pos_map[j - 1, i]                   # Position in the active list.
                    if vg_pos_in_active >= 0:                                 # Gas is active at interface (j - 1/2, i).
                        vg_idx = vg_start_id + vg_pos_in_active               # Position in the reduced system.
                        phi_avg_left = phi_at_interface(phi(j - 1, i), phi(j, i))
                        v_g_sigL = v_g_list[k, j, i]
                        n_upwind = calculate_upwind_scheme(n_list[k_next, j - 1, i], n_list[k_next, j, i], v_g_sigL)
                        A_reduced[eq_idx, vg_idx] -= (eta_tilde[k + q, j, i] * rho_w_list[k, j, i]
                                                      * phi_avg_left * n_upwind / dy)

                elif j == 0:
                    phi_avg_left = phi_at_interface(1.0, phi(j, i))
                    v_g_sigL = v_g_list[k, j, i]
                    n_upwind = calculate_upwind_scheme(rho_g_in, n_list[k_next, j, i], v_g_sigL)
                    b_reduced[eq_idx] += (eta_tilde[k + q, j, i] * rho_w_list[k, j, i]
                                          * phi_avg_left * n_upwind * v_g_sigL / dy)

                # Coefficient for v_w(j + 1/2, i)^(k_next) - Only if water is active at this interface.
                # Find the position of this water interface in the reduced system.
                vw_pos_in_active = vw_pos_map[j, i]                         # Position in the active list.
                if vw_pos_in_active >= 0:                                   # Water is active at interface (j + 1/2, i).
                    vw_idx = vw_start_id + vw_pos_in_active                 # Position in the reduced system.
                    v_w_sig = v_w_list[k, j + 1, i]
                    phi_avg_right = phi_at_interface(phi(j, i), phi(j + 1, i))
                    m_upwind = calculate_upwind_scheme(m_list[k_next, j, i], m_list[k_next, j + 1, i], v_w_sig)
                    A_reduced[eq_idx, vw_idx] += (eta_tilde[k + q, j, i] * rho_g_tilde[k + q, j, i]
                                                  * phi_avg_right * m_upwind / dy)

                # Coefficient for v_w(j - 1/2, i)^(k_next) - Only if water is active at this interface.
                if j > 0:
                    # Find the position of this water interface in the reduced system.
                    vw_pos_in_active = vw_pos_map[j - 1, i]                 # Position in the active list.
                    if vw_pos_in_active >= 0:                               # Water is active at interface (j - 1/2, i).
                        vw_idx = vw_start_id + vw_pos_in_active             # Position in the reduced system.
                        phi_avg_left = phi_at_interface(phi(j - 1, i), phi(j, i))
                        v_w_sigL = v_w_list[k, j, i]
                        m_upwind = calculate_upwind_scheme(m_list[k_next, j - 1, i], m_list[k_next, j, i], v_w_sigL)
                        A_reduced[eq_idx, vw_idx] -= (eta_tilde[k + q, j, i] * rho_g_tilde[k + q, j, i]
                                                      * phi_avg_left * m_upwind / dy)

                # Since water vertical velocity at row j = 0 is zero. So we don't need to build another block for that.

                # Coefficient for u_g(j, i + 1/2)^(k_next) - Only if gas is active at this interface.
                if i < N_x - 1:
                    # Since when i = N_x - 1, its corresponding interface will be the right wall, and we know its
                    # horizontal velocity is zero. so we don't need to write its term in the equation.
                    # Find the position of this gas interface in the reduced system.
                    ug_pos_in_active = ug_pos_map[j, i]                       # Position in the active list.
                    if ug_pos_in_active >= 0:                                 # Gas is active at interface (j, i + 1/2).
                        ug_idx = ug_start_id + ug_pos_in_active               # Position in the reduced system.
                        phi_avg_right = phi_at_interface(phi(j, i), phi(j, i + 1))
                        u_g_sig = u_g_list[k, j, i + 1]
                        n_upwind = calculate_upwind_scheme(n_list[k_next, j, i], n_list[k_next, j, i + 1], u_g_sig)
                        A_reduced[eq_idx, ug_idx] += (eta_tilde[k + q, j, i] * rho_w_list[k, j, i]
                                                      * phi_avg_right * n_upwind / dx)

                # Coefficient for u_g(j, i - 1/2)^(k_next) - Only if gas is active at this interface.
                if i > 0:
                    # Since when i = 0, its corresponding interface will be the left wall, and we know its
                    # horizontal velocity is zero. so we don't need to write its term in the equation.
                    # Find the position of this gas interface in the reduced system.
                    ug_pos_in_active = ug_pos_map[j, i - 1]                   # Position in the active list.
                    if ug_pos_in_active >= 0:                                 # Gas is active at interface (j, i - 1/2).
                        ug_idx = ug_start_id + ug_pos_in_active               # Position in the reduced system.
                        phi_avg_left = phi_at_interface(phi(j, i - 1), phi(j, i))
                        u_g_sigL = u_g_list[k, j, i]
                        n_upwind = calculate_upwind_scheme(n_list[k_next, j, i - 1], n_list[k_next, j, i], u_g_sigL)
                        A_reduced[eq_idx, ug_idx] -= (eta_tilde[k + q, j, i] * rho_w_list[k, j, i]
                                                      * phi_avg_left * n_upwind / dx)

                # Coefficient for u_w(j, i + 1/2)^(k_next) - Only if water is active at this interface.
                if i < N_x - 1:
                    # Since when i = N_x - 1, its corresponding interface will be the right wall, and we know its
                    # horizontal velocity is zero. so we don't need to write its term in the equation.
                    # Find the position of this water interface in the reduced system.
                    uw_pos_in_active = uw_pos_map[j, i]                     # Position in the active list.
                    if uw_pos_in_active >= 0:                               # Water is active at interface (j, i + 1/2).
                        uw_idx = uw_start_id + uw_pos_in_active             # Position in the reduced system.
                        phi_avg_right = phi_at_interface(phi(j, i), phi(j, i + 1))
                        u_w_sig = u_w_list[k, j, i + 1]
                        m_upwind = calculate_upwind_scheme(m_list[k_next, j, i], m_list[k_next, j, i + 1], u_w_sig)
                        A_reduced[eq_idx, uw_idx] += (eta_tilde[k + q, j, i] * rho_g_tilde[k + q, j, i]
                                                      * phi_avg_right * m_upwind / dx)

                # Coefficient for u_w(j, i - 1/2)^(k_next) - Only if water is active at this interface.
                if i > 0:
                    # Since when i = 0, its corresponding interface will be the left wall, and we know its
                    # horizontal velocity is zero. so we don't need to write its term in the equation.
                    # Find the position of this water interface in the reduced system.
                    uw_pos_in_active = uw_pos_map[j, i - 1]                 # Position in the active list.
                    if uw_pos_in_active >= 0:                               # Water is active at interface (j, i - 1/2).
                        uw_idx = uw_start_id + uw_pos_in_active             # Position in the reduced system.
                        phi_avg_left = phi_at_interface(phi(j, i - 1), phi(j, i))
                        u_w_sigL = u_w_list[k, j, i]
                        m_upwind = calculate_upwind_scheme(m_list[k_next, j, i - 1], m_list[k_next, j, i], u_w_sigL)
                        A_reduced[eq_idx, uw_idx] -= (eta_tilde[k + q, j, i] * rho_g_tilde[k + q, j, i]
                                                      * phi_avg_left * m_upwind / dx)

        #========================= Gas Momentum Equation (A9 first part) in the Y direction ============================
        # Build equations A9 (Momentum balance) for interfaces (Only for active gas interfaces).
        # When the gas phase is absent at an interface, we don't build the momentum equation,
        # and velocity automatically stays 0.
        for idx, indices in enumerate(active_vg_indices):                       # indices is the global interface index.
            j = indices[0]
            i = indices[1]
            # Equation index for gas momentum in the Y direction in the reduced system.
            eq_idx_gas_y = vg_start_id + idx

            # Calculate K̂ coefficients at interface (j + 1/2, i).
            v_w_sig = v_w_list[k, j + 1, i]
            v_g_sig = v_g_list[k, j + 1, i]
            S_w_interface = calculate_upwind_scheme(S_w_list[k + q, j, i], S_w_list[k + q, j + 1, i], v_w_sig)
            S_g_interface = calculate_upwind_scheme(S_g_list[k + q, j, i], S_g_list[k + q, j + 1, i], v_g_sig)

            k_hat_g = calculate_interaction_coefficient_gas(i, j, k, q, direction = 'vertical')
            k_hat = calculate_interaction_coefficient_water_gas(i, j, k, q, direction = 'vertical')

            var_idx = (j * N_x) + i                                               # Variable index for P_w(j, i)^(k + 1)
            vg_idx = vg_start_id + idx                                       # v_g variable index in the reduced system.

            # Time derivative upwind quantity of gas.
            n_upwind = calculate_upwind_scheme(n_list[k_next, j, i], n_list[k_next, j + 1, i], v_g_list[k, j + 1, i])   # n_(j + 1/2, i)^(k + 1)
            n_upwind_k = calculate_upwind_scheme(n_list[k, j, i], n_list[k, j + 1, i], v_g_list[k, j + 1, i])           # n_(j + 1/2, i)^k

            # Interaction terms
            # Coupling with water velocity (if water is also active at this interface).
            vw_pos_in_active = vw_pos_map[j, i]                        # Position in the active list.
            if vw_pos_in_active >= 0:                                  # water is also active at interface (j + 1/2, i).
                vw_idx = vw_start_id + vw_pos_in_active                # Position in the reduced system.
                A_reduced[eq_idx_gas_y, vw_idx] -= k_hat                              # k_hat * v_w_(j + 1/2, i)^(k + 1)

            A_reduced[eq_idx_gas_y, vg_idx] += k_hat + k_hat_g            # (k_hat + k_hat_g) * v_g_(j + 1/2, i)^(k + 1)

            # Time-derivative term:                             (n_(j + 1/2, i)^(k + 1) / dt) * v_g_(j + 1/2, i)^(k + 1)
            A_reduced[eq_idx_gas_y, vg_idx] += n_upwind / dt

            # Convection terms (LHS of A9 first equation) - only when the gas phase is present.
            # Coefficient for v_g(j + 1/2, i)^(k + 1)
            # The first term at dy parenthesis * v_g(j + 1/2, i)^(k + 1)
            A_reduced[eq_idx_gas_y, vg_idx] += (((phi(j + 1, i) * n_list[k_next, j + 1, i]) *
                                                (v_g_list[k, j + 1, i] + v_g_list[k, j + 2, i])) /
                                                (4 * dy * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # The second term at dy parenthesis * v_g(j + 1/2, i)^(k + 1)
            A_reduced[eq_idx_gas_y, vg_idx] -= (phi(j, i) * n_list[k_next, j, i] *
                                                (v_g_list[k, j, i] + v_g_list[k, j + 1, i]) /
                                                (4 * dy * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # The first term at dx parenthesis * v_g(j + 1/2, i)^(k + 1)
            if i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1), phi(j + 1, i + 1),
                                                           n_list[k_next, j, i], n_list[k_next, j + 1, i],
                                                           n_list[k_next, j, i + 1], n_list[k_next, j + 1, i + 1],
                                                           v_g_list[k, j + 1, i], v_g_list[k, j + 1, i + 1],
                                                           direction = 'vertical')
                A_reduced[eq_idx_gas_y, vg_idx] += (phi_n_upwind_average * (u_g_list[k, j, i + 1] +
                                                                            u_g_list[k, j + 1, i + 1]) /
                                                    (4 * dx * phi_at_interface(phi(j, i), phi(j + 1, i))))

            else:
                # When we are at the rightmost column, we assume there are a ghost column next to it, where n = 0,
                # phi = 0, and velocity is zero.
                phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), 0, 0,
                                                           n_list[k_next, j, i], n_list[k_next, j + 1, i],
                                                           0, 0,
                                                           v_g_list[k, j + 1, i], 0,
                                                           direction = 'vertical')
                A_reduced[eq_idx_gas_y, vg_idx] += (phi_n_upwind_average * (u_g_list[k, j, i + 1] +
                                                                            u_g_list[k, j + 1, i + 1]) /
                                                    (4 * dx * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # The second term at dx parenthesis * v_g(j + 1/2, i)^(k + 1)
            if i > 0:
                phi_n_upwind_average = upwind_then_average(phi(j , i - 1), phi(j + 1, i - 1), phi(j, i), phi(j + 1, i),
                                                           n_list[k_next, j, i - 1], n_list[k_next, j + 1, i - 1],
                                                           n_list[k_next, j, i], n_list[k_next, j + 1, i],
                                                           v_g_list[k, j + 1, i - 1], v_g_list[k, j + 1, i],
                                                           direction = 'vertical')
                A_reduced[eq_idx_gas_y, vg_idx] -= (phi_n_upwind_average * (u_g_list[k, j, i] +
                                                                            u_g_list[k, j + 1, i]) /
                                                    (4 * dx * phi_at_interface(phi(j, i), phi(j + 1, i))))

            else:
                # When we are at the leftmost column, we assume there are a ghost column next to it, where n = 0,
                # phi = 0, and velocity is zero.
                phi_n_upwind_average = upwind_then_average(0, 0, phi(j, i), phi(j + 1, i),
                                                           0, 0, n_list[k_next, j, i],
                                                           n_list[k_next, j + 1, i], 0, v_g_list[k, j + 1, i],
                                                           direction = 'vertical')
                A_reduced[eq_idx_gas_y, vg_idx] -= (phi_n_upwind_average * (u_g_list[k, j, i] +
                                                                            u_g_list[k, j + 1, i]) /
                                                    (4 * dx * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_g(j + 3/2, i)^(k + 1) - only if that interface is also active.
            if j < N_y - 2:
                vg_next_pos = vg_pos_map[j + 1, i]                        # Position in the active list.
                if vg_next_pos >= 0:                                      # Check if the (j + 1, i) interface is active.
                    vg_next_idx = vg_start_id + vg_next_pos               # Position in the reduced system.
                    A_reduced[eq_idx_gas_y, vg_next_idx] += (phi(j + 1, i) * n_list[k_next, j + 1, i] *
                                                             (v_g_list[k, j + 1, i] + v_g_list[k, j + 2, i]) /
                                                             (4 * dy * phi_at_interface(phi(j, i), phi(j + 1, i))))

            elif j == N_y - 2:      # When we are at the j = N_y - 2 row, the velocity of the next interface equals to
                                    # velocity of this interface, So will write it for this interface.
                A_reduced[eq_idx_gas_y, vg_idx] += (phi(j + 1, i) * n_list[k_next, j + 1, i] *
                                                             (v_g_list[k, j + 1, i] + v_g_list[k, j + 2, i]) /
                                                             (4 * dy * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_g(j - 1/2, i)^(k + 1) - only if that interface is also active.
            if j > 0:
                vg_prev_pos = vg_pos_map[j - 1, i]                        # Position in the active list.
                if vg_prev_pos >= 0:                                      # Check if the (j - 1, i) interface is active.
                    vg_prev_idx = vg_start_id + vg_prev_pos               # Position in the reduced system.
                    A_reduced[eq_idx_gas_y, vg_prev_idx] -= (phi(j, i) * n_list[k_next, j, i] *
                                                             (v_g_list[k, j, i] + v_g_list[k, j + 1, i]) /
                                                             (4 * dy * phi_at_interface(phi(j, i), phi(j + 1, i))))

            elif j == 0:     # With j = 0, the previous interface is the inlet, so its velocity (here j - 1) is constant.
                b_reduced[eq_idx_gas_y] += (phi(j, i) * n_list[k_next, j, i] *
                                            (v_g_list[k, j, i] + v_g_list[k, j + 1, i]) /
                                            (4 * dy * phi_at_interface(phi(j, i), phi(j + 1, i)))) * v_g_list[k, j, i]

            # Coefficient for v_g(j + 1/2, i + 1)^(k + 1) - only if that interface is also active.
            if i < N_x - 1:
                vg_next_pos = vg_pos_map[j, i + 1]                             # Position in the active list.
                if vg_next_pos >= 0:                                           # Check if the i + 1 interface is active.
                    vg_next_idx = vg_start_id + vg_next_pos                    # Position in the reduced system.
                    phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                               phi(j + 1, i + 1), n_list[k_next, j, i],
                                                               n_list[k_next, j + 1, i], n_list[k_next, j, i + 1],
                                                               n_list[k_next, j + 1, i + 1], v_g_list[k, j + 1, i],
                                                               v_g_list[k, j + 1, i + 1], direction = 'vertical')
                    A_reduced[eq_idx_gas_y, vg_next_idx] += (phi_n_upwind_average * (u_g_list[k, j, i + 1] +
                                                                                u_g_list[k, j + 1, i + 1]) /
                                                        (4 * dx * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_g(j + 1/2, i - 1)^(k + 1) - only if that interface is also active.
            if i > 0:
                vg_prev_pos = vg_pos_map[j, i - 1]                             # Position in the active list.
                if vg_prev_pos >= 0:                                           # Check if the i - 1 interface is active.
                    vg_prev_idx = vg_start_id + vg_prev_pos                    # Position in the reduced system.
                    phi_n_upwind_average = upwind_then_average(phi(j, i - 1), phi(j + 1, i - 1), phi(j, i),
                                                               phi(j + 1, i), n_list[k_next, j, i - 1],
                                                               n_list[k_next, j + 1, i - 1], n_list[k_next, j, i],
                                                               n_list[k_next, j + 1, i], v_g_list[k, j + 1, i - 1],
                                                               v_g_list[k, j + 1, i], direction = 'vertical')
                    A_reduced[eq_idx_gas_y, vg_prev_idx] -= (phi_n_upwind_average * (u_g_list[k, j, i] +
                                                                                     u_g_list[k, j + 1, i]) /
                                                        (4 * dx * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Pressure gradient terms (LHS of A9 first equation)
            # Coefficient for P_w(j + 1, i)^(k + 1)
            if j + 1 < N_y - 1:                                                                  # Row j + 1 is unknown.
                A_reduced[eq_idx_gas_y, var_idx + N_x] += S_g_interface / dy

            else:                                                 # Row j + 1 is a boundary condition. (j + 1 = N_y - 1)
                b_reduced[eq_idx_gas_y] -= S_g_interface * P_top / dy

            # Coefficient for P_w(j, i)^(k + 1)
            A_reduced[eq_idx_gas_y, var_idx] -= S_g_interface / dy

            # Capillary pressure and gravity terms (RHS of A9 first equation)
            # Coefficient for P_c_(j, i)^(k + 1)
            b_reduced[eq_idx_gas_y] += S_g_interface * Pc_list[k + q, j, i] / dy

            # Coefficient for P_c_(j + 1, i)^(k + 1)
            b_reduced[eq_idx_gas_y] -= S_g_interface * Pc_list[k + q, j + 1, i] / dy

            # Gravity
            b_reduced[eq_idx_gas_y] += n_upwind * g

            # Coefficient for v_g(j + 1/2, i)^k
            b_reduced[eq_idx_gas_y] += n_upwind_k * v_g_list[k, j + 1, i] / dt

            # Viscous-stress terms (The last term of RHS)
            # Coefficient for v_g(j + 1/2, i + 1)^(k + 1) - only if that interface is also active.
            if i < N_x - 1:
                vg_next_pos = vg_pos_map[j, i + 1]                             # Position in the active list.
                if vg_next_pos >= 0:                                           # Check if the i + 1 interface is active.
                    vg_next_idx = vg_start_id + vg_next_pos                    # Position in the reduced system.
                    phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                               phi(j + 1, i + 1), n_list[k_next, j, i],
                                                               n_list[k_next, j + 1, i], n_list[k_next, j, i + 1],
                                                               n_list[k_next, j + 1, i + 1], v_g_list[k, j + 1, i],
                                                               v_g_list[k, j + 1, i + 1], direction = 'vertical')
                    A_reduced[eq_idx_gas_y, vg_next_idx] -= (phi_n_upwind_average * miu_g / (dx * dx *
                                                             phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_g(j + 1/2, i)^(k + 1) at first dx parenthesis.
            if i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), n_list[k_next, j, i],
                                                           n_list[k_next, j + 1, i], n_list[k_next, j, i + 1],
                                                           n_list[k_next, j + 1, i + 1], v_g_list[k, j + 1, i],
                                                           v_g_list[k, j + 1, i + 1], direction = 'vertical')
                A_reduced[eq_idx_gas_y, vg_idx] += (phi_n_upwind_average * miu_g / (dx * dx *
                                                    phi_at_interface(phi(j, i), phi(j + 1, i))))

            # When we are at the rightmost column, the velocity on the wall is zero.

            # Coefficient for u_g(j + 1, i + 1/2)^k at first dx parenthesis.
            if i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), n_list[k_next, j, i],
                                                           n_list[k_next, j + 1, i], n_list[k_next, j, i + 1],
                                                           n_list[k_next, j + 1, i + 1], v_g_list[k, j + 1, i],
                                                           v_g_list[k, j + 1, i + 1], direction = 'vertical')
                b_reduced[eq_idx_gas_y] += (phi_n_upwind_average * miu_g * u_g_list[k, j + 1, i + 1] / (dx * dy *
                                                             phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for u_g(j, i + 1/2)^k at first dx parenthesis.
            if i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), n_list[k_next, j, i],
                                                           n_list[k_next, j + 1, i], n_list[k_next, j, i + 1],
                                                           n_list[k_next, j + 1, i + 1], v_g_list[k, j + 1, i],
                                                           v_g_list[k, j + 1, i + 1], direction = 'vertical')
                b_reduced[eq_idx_gas_y] -= (phi_n_upwind_average * miu_g * u_g_list[k, j, i + 1] / (dx * dy *
                                            phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_g(j + 1/2, i)^(k + 1) at the second dx parenthesis.
            if i > 0:
                phi_n_upwind_average = upwind_then_average(phi(j, i - 1), phi(j + 1, i - 1), phi(j, i), phi(j + 1, i),
                                                           n_list[k_next, j, i - 1], n_list[k_next, j + 1, i - 1],
                                                           n_list[k_next, j, i], n_list[k_next, j + 1, i],
                                                           v_g_list[k, j + 1, i - 1], v_g_list[k, j + 1, i],
                                                           direction = 'vertical')
                A_reduced[eq_idx_gas_y, vg_idx] += (phi_n_upwind_average * miu_g / (dx * dx *
                                                    phi_at_interface(phi(j, i), phi(j + 1, i))))

            # When we at the leftmost column, the velocity on the wall is zero.

            # Coefficient for v_g(j + 1/2, i - 1)^(k + 1) at second dx parenthesis - only if that interface is active.
            if i > 0:
                vg_prev_pos = vg_pos_map[j, i - 1]                             # Position in the active list.
                if vg_prev_pos >= 0:                                           # Check if the i - 1 interface is active.
                    vg_prev_idx = vg_start_id + vg_prev_pos                    # Position in the reduced system.
                    phi_n_upwind_average = upwind_then_average(phi(j, i - 1), phi(j + 1, i - 1), phi(j, i),
                                                               phi(j + 1, i), n_list[k_next, j, i - 1],
                                                               n_list[k_next, j + 1, i - 1], n_list[k_next, j, i],
                                                               n_list[k_next, j + 1, i], v_g_list[k, j + 1, i - 1],
                                                               v_g_list[k, j + 1, i], direction = 'vertical')
                    A_reduced[eq_idx_gas_y, vg_prev_idx] -= (phi_n_upwind_average * miu_g / (dx * dx *
                                                             phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for u_g(j + 1, i - 1/2)^k at second dx parenthesis.
            if i > 0:
                phi_n_upwind_average = upwind_then_average(phi(j, i - 1), phi(j + 1, i - 1), phi(j, i),
                                                           phi(j + 1, i), n_list[k_next, j, i - 1],
                                                           n_list[k_next, j + 1, i - 1], n_list[k_next, j, i],
                                                           n_list[k_next, j + 1, i], v_g_list[k, j + 1, i - 1],
                                                           v_g_list[k, j + 1, i], direction = 'vertical')
                b_reduced[eq_idx_gas_y] -= (phi_n_upwind_average * miu_g * u_g_list[k, j + 1, i] / (dx * dy *
                                                             phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for u_g(j, i - 1/2)^k  at second dx parenthesis.
            if i > 0:
                 phi_n_upwind_average = upwind_then_average(phi(j, i - 1), phi(j + 1, i - 1), phi(j, i),
                                                            phi(j + 1, i), n_list[k_next, j, i - 1],
                                                            n_list[k_next, j + 1, i - 1], n_list[k_next, j, i],
                                                            n_list[k_next, j + 1, i], v_g_list[k, j + 1, i - 1],
                                                            v_g_list[k, j + 1, i], direction = 'vertical')
                 b_reduced[eq_idx_gas_y] += (phi_n_upwind_average * miu_g * u_g_list[k, j, i] / (dx * dy *
                                                              phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_g(j + 3/2, i)^(k + 1) at the first dy parenthesis - only if that interface is active.
            if j < N_y - 2:
                vg_next_pos = vg_pos_map[j + 1, i]                        # Position in the active list.
                if vg_next_pos >= 0:                                      # Check if the (j + 1, i) interface is active.
                    vg_next_idx = vg_start_id + vg_next_pos               # Position in the reduced system.
                    A_reduced[eq_idx_gas_y, vg_next_idx] -= ((2 * miu_g + kappa_g) * phi(j + 1, i) / (dy * dy *
                                                             phi_at_interface(phi(j, i), phi(j + 1, i))))

            elif j == N_y - 2:      # When we are at the j = N_y - 2 row, the velocity of the next interface equals to
                                    # velocity of this interface, So we will write it for this interface.
                A_reduced[eq_idx_gas_y, vg_idx] -= ((2 * miu_g + kappa_g) * phi(j + 1, i) / (dy * dy *
                                                    phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_g(j + 1/2, i)^(k + 1) at the first dy parenthesis.
            A_reduced[eq_idx_gas_y, vg_idx] += ((2 * miu_g + kappa_g) * phi(j + 1, i) / (dy * dy *
                                                 phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for u_g(j + 1, i + 1/2)^k at the first dy parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_gas_y] += (kappa_g * phi(j + 1, i) * u_g_list[k, j + 1, i + 1] / (dy * dx *
                                            phi_at_interface(phi(j, i), phi(j + 1, i))))

                # When we are the rightmost column, the velocity on the wall is zero.

            # Coefficient for u_g(j + 1, i - 1/2)^k at the first dy parenthesis.
            if i > 0:
                b_reduced[eq_idx_gas_y] -= (kappa_g * phi(j + 1, i) * u_g_list[k, j + 1, i] / (dy * dx *
                                            phi_at_interface(phi(j, i), phi(j + 1, i))))

                # When we are the leftmost column, the velocity on the wall is zero.

            # Coefficient for v_g(j + 1/2, i)^(k + 1) at the second dy parenthesis.
            A_reduced[eq_idx_gas_y, vg_idx] += ((2 * miu_g + kappa_g) * phi(j, i) / (dy * dy *
                                                phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_g(j - 1/2, i)^(k + 1) at the second dy parenthesis - only if that interface is active.
            if j > 0:
                vg_prev_pos = vg_pos_map[j - 1, i]                        # Position in the active list.
                if vg_prev_pos >= 0:                                      # Check if the (j - 1, i) interface is active.
                    vg_prev_idx = vg_start_id + vg_prev_pos               # Position in the reduced system.
                    A_reduced[eq_idx_gas_y, vg_prev_idx] -= ((2 * miu_g + kappa_g) * phi(j, i) / (dy * dy *
                                                             phi_at_interface(phi(j, i), phi(j + 1, i))))

            elif j == 0:    # With j = 0, the previous interface is the inlet, so its velocity (here j - 1) is constant.
                b_reduced[eq_idx_gas_y] += ((2 * miu_g + kappa_g) * phi(j, i) / (dy * dy *
                                            phi_at_interface(phi(j, i), phi(j + 1, i)))) * v_g_list[k, j, i]

            # Coefficient for u_g(j, i + 1/2)^k at the second dy parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_gas_y] -= (kappa_g * phi(j, i) * u_g_list[k, j, i + 1] / (dy * dx *
                                            phi_at_interface(phi(j, i), phi(j + 1, i))))

                # When we are the rightmost column, the velocity on the wall is zero.

            # Coefficient for u_g(j, i - 1/2)^k at the second dy parenthesis.
            if i > 0:
                b_reduced[eq_idx_gas_y] += (kappa_g * phi(j, i) * u_g_list[k, j, i] / (dy * dx *
                                            phi_at_interface(phi(j, i), phi(j + 1, i))))

                # When we are the leftmost column, the velocity on the wall is zero.

        #========================= Water Momentum Equation (A9 second part) in the Y direction =========================
        # Build equations A9 (Momentum balance) for interfaces (Only for active water interfaces).
        # When the water phase is absent at an interface, we don't build the momentum equation,
        # and velocity automatically stays 0.
        for idx, indices in enumerate(active_vw_indices):                       # indices is the global interface index.
            j = indices[0]
            i = indices[1]
            # Equation index for water momentum in the Y direction in the reduced system.
            eq_idx_water_y = vw_start_id + idx

            # Calculate K̂ coefficients at interface (j + 1/2, i).
            v_w_sig = v_w_list[k, j + 1, i]
            v_g_sig = v_g_list[k, j + 1, i]
            S_w_interface = calculate_upwind_scheme(S_w_list[k + q, j, i], S_w_list[k + q, j + 1, i], v_w_sig)
            S_g_interface = calculate_upwind_scheme(S_g_list[k + q, j, i], S_g_list[k + q, j + 1, i], v_g_sig)

            k_hat_w = calculate_interaction_coefficient_water(i, j, k, q, direction = 'vertical')
            k_hat = calculate_interaction_coefficient_water_gas(i, j, k, q, direction = 'vertical')

            var_idx = (j * N_x) + i                                               # Variable index for P_w(j, i)^(k + 1)
            vw_idx = vw_start_id + idx                                       # v_w variable index in the reduced system.

            # Time derivative upwind quantity of water.
            m_upwind = calculate_upwind_scheme(m_list[k_next, j, i], m_list[k_next, j + 1, i], v_w_list[k, j + 1, i])   # m_(j + 1/2, i)^(k + 1)
            m_upwind_k = calculate_upwind_scheme(m_list[k, j, i], m_list[k, j + 1 , i], v_w_list[k, j + 1, i])          # m_(j + 1/2, i)^k

            # Interaction terms
            # Coupling with gas velocity (if gas is also active at this interface).
            vg_pos_in_active = vg_pos_map[j, i]                          # Position in the active list.
            if vg_pos_in_active >= 0:                                         # gas is also active at interface (j + 1/2, i).
                vg_idx = vg_start_id + vg_pos_in_active                  # Position in the reduced system.
                A_reduced[eq_idx_water_y, vg_idx] -= k_hat                            # k_hat * v_g_(j + 1/2, i)^(k + 1)

            A_reduced[eq_idx_water_y, vw_idx] += k_hat + k_hat_w          # (k_hat + k_hat_w) * v_w_(j + 1/2, i)^(k + 1)

            # Time-derivative term:                             (m_(j + 1/2, i)^(k + 1) / dt) * v_w_(j + 1/2, i)^(k + 1)
            A_reduced[eq_idx_water_y, vw_idx] += m_upwind / dt

            # Convection terms (LHS of A9 second equation) - only when the water phase is present.
            # Coefficient for v_w(j + 1/2, i)^(k + 1)
            # The first term at dy parenthesis * v_w(j + 1/2, i)^(k + 1)
            A_reduced[eq_idx_water_y, vw_idx] += (((phi(j + 1, i) * m_list[k_next, j + 1, i]) *
                                                   (v_w_list[k, j + 1, i] + v_w_list[k, j + 2, i])) /
                                                  (4 * dy * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # The second term at dy parenthesis * v_w(j + 1/2, i)^(k + 1)
            A_reduced[eq_idx_water_y, vw_idx] -= (phi(j, i) * m_list[k_next, j, i] *
                                                  (v_w_list[k, j, i] + v_w_list[k, j + 1, i]) /
                                                  (4 * dy * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # The first term at dx parenthesis * v_w(j + 1/2, i)^(k + 1)
            if i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1), phi(j + 1, i + 1),
                                                           m_list[k_next, j, i], m_list[k_next, j + 1, i],
                                                           m_list[k_next, j, i + 1], m_list[k_next, j + 1, i + 1],
                                                           v_w_list[k, j + 1, i], v_w_list[k, j + 1, i + 1],
                                                           direction = 'vertical')
                A_reduced[eq_idx_water_y, vw_idx] += (phi_m_upwind_average * (u_w_list[k, j, i + 1] +
                                                                              u_w_list[k, j + 1, i + 1]) /
                                                      (4 * dx * phi_at_interface(phi(j, i), phi(j + 1, i))))

            else:
                # When we are at the rightmost column, we assume there are a ghost column next to it, where m = 0,
                # phi = 0, and velocity is zero.
                phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), 0, 0,
                                                           m_list[k_next, j, i], m_list[k_next, j + 1, i],
                                                           0, 0, v_w_list[k, j + 1, i], 0,
                                                           direction = 'vertical')
                A_reduced[eq_idx_water_y, vw_idx] += (phi_m_upwind_average * (u_w_list[k, j, i + 1] +
                                                                              u_w_list[k, j + 1, i + 1]) /
                                                      (4 * dx * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # The second term at dx parenthesis * v_w(j + 1/2, i)^(k + 1)
            if i > 0:
                phi_m_upwind_average = upwind_then_average(phi(j, i - 1), phi(j + 1, i - 1), phi(j, i), phi(j + 1, i),
                                                           m_list[k_next, j, i - 1], m_list[k_next, j + 1, i - 1],
                                                           m_list[k_next, j, i], m_list[k_next, j + 1, i],
                                                           v_w_list[k, j + 1, i - 1], v_w_list[k, j + 1, i],
                                                           direction = 'vertical')
                A_reduced[eq_idx_water_y, vw_idx] -= (phi_m_upwind_average * (u_w_list[k, j, i] +
                                                                              u_w_list[k, j + 1, i]) /
                                                      (4 * dx * phi_at_interface(phi(j, i), phi(j + 1, i))))

            else:
                # When we are at the leftmost column, we assume there are a ghost column next to it, where m = 0,
                # phi = 0, and velocity is zero.
                phi_m_upwind_average = upwind_then_average(0, 0, phi(j, i), phi(j + 1, i),
                                                           0, 0, m_list[k_next, j, i],
                                                           m_list[k_next, j + 1, i], 0, v_w_list[k, j + 1, i],
                                                           direction = 'vertical')
                A_reduced[eq_idx_water_y, vw_idx] -= (phi_m_upwind_average * (u_w_list[k, j, i] +
                                                                              u_w_list[k, j + 1, i]) /
                                                      (4 * dx * phi_at_interface(phi(j , i), phi(j + 1, i))))

            # Coefficient for v_w(j + 3/2, i)^(k + 1) - only if that interface is also active.
            if j < N_y - 2:
                vw_next_pos = vw_pos_map[j + 1, i]                            # Position in the active list.
                if vw_next_pos >= 0:                                          # Check if (j + 1, i) interface is active.
                    vw_next_idx = vw_start_id + vw_next_pos                   # Position in the reduced system.
                    A_reduced[eq_idx_water_y, vw_next_idx] += (phi(j + 1, i) * m_list[k_next, j + 1, i] *
                                                               (v_w_list[k, j + 1, i] + v_w_list[k, j + 2, i]) /
                                                               (4 * dy * phi_at_interface(phi(j, i), phi(j + 1, i))))

            elif j == N_y - 2:      # When we are at the j = N_y - 2 row, the velocity of the next interface equals to
                                    # velocity of this interface, So will write it for this interface.
                A_reduced[eq_idx_water_y, vw_idx] += (phi(j + 1, i) * m_list[k_next, j + 1, i] *
                                                               (v_w_list[k, j + 1, i] + v_w_list[k, j + 2, i]) /
                                                               (4 * dy * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_w(j - 1/2, i)^(k + 1) - only if that interface is also active.
            if j > 0:
                vw_prev_pos = vw_pos_map[j - 1, i]                            # Position in the active list.
                if vw_prev_pos >= 0:                                          # Check if (j - 1, i) interface is active.
                    vw_prev_idx = vw_start_id + vw_prev_pos                   # Position in the reduced system.
                    A_reduced[eq_idx_water_y, vw_prev_idx] -= (phi(j, i) * m_list[k_next, j, i] *
                                                               (v_w_list[k, j, i] + v_w_list[k, j + 1, i]) /
                                                               (4 * dy * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # With j = 0, the previous interface is the inlet, and we know its water velocity is zero.

            # Coefficient for v_w(j + 1/2, i + 1)^(k + 1) - only if that interface is also active.
            if i < N_x - 1:
                vw_next_pos = vw_pos_map[j, i + 1]                            # Position in the active list.
                if vw_next_pos >= 0:                                          # Check if (j, i + 1) interface is active.
                    vw_next_idx = vw_start_id + vw_next_pos                   # Position in the reduced system.
                    phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                               phi(j + 1, i + 1), m_list[k_next, j, i],
                                                               m_list[k_next, j + 1, i], m_list[k_next, j, i + 1],
                                                               m_list[k_next, j + 1, i + 1], v_w_list[k, j + 1, i],
                                                               v_w_list[k, j + 1, i + 1], direction = 'vertical')
                    A_reduced[eq_idx_water_y, vw_next_idx] += (phi_m_upwind_average * (u_w_list[k, j, i + 1] +
                                                                                       u_w_list[k, j + 1, i + 1]) /
                                                               (4 * dx * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_w(j + 1/2, i - 1)^(k + 1) - only if that interface is also active.
            if i > 0:
                vw_prev_pos = vw_pos_map[j, i - 1]                            # Position in the active list.
                if vw_prev_pos >= 0:                                          # Check if (j, i - 1) interface is active.
                    vw_prev_idx = vw_start_id + vw_prev_pos                   # Position in the reduced system.
                    phi_m_upwind_average = upwind_then_average(phi(j, i - 1), phi(j + 1, i - 1), phi(j, i),
                                                               phi(j + 1, i), m_list[k_next, j, i - 1],
                                                               m_list[k_next, j + 1, i - 1], m_list[k_next, j, i],
                                                               m_list[k_next, j + 1, i], v_w_list[k, j + 1, i - 1],
                                                               v_w_list[k, j + 1, i], direction = 'vertical')
                    A_reduced[eq_idx_water_y, vw_prev_idx] -= (phi_m_upwind_average * (u_w_list[k, j, i] +
                                                                                       u_w_list[k, j + 1, i]) /
                                                               (4 * dx * phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Pressure gradient terms (LHS of A9 second equation)
            # Coefficient for P_w(j + 1, i)^(k + 1)
            if j + 1 < N_y - 1:                                                                  # Row j + 1 is unknown.
                A_reduced[eq_idx_water_y, var_idx + N_x] += S_w_interface / dy

            else:                                                 # Row j + 1 is a boundary condition. (j + 1 = N_y - 1)
                b_reduced[eq_idx_water_y] -= S_w_interface * P_top / dy

            # Coefficient for P_w(j, i)^(k + 1)
            A_reduced[eq_idx_water_y, var_idx] -= S_w_interface / dy

            # Gravity term (RHS of A9 second equation)
            b_reduced[eq_idx_water_y] += m_upwind * g

            # Coefficient for v_w(j + 1/2, i)^k
            b_reduced[eq_idx_water_y] += m_upwind_k * v_w_list[k, j + 1, i] / dt

            # Viscous-stress terms (The last term of RHS)
            # Coefficient for v_w(j + 1/2, i + 1)^(k + 1) - only if that interface is also active.
            if i < N_x - 1:
                vw_next_pos = vw_pos_map[j, i + 1]                             # Position in the active list.
                if vw_next_pos >= 0:                                           # Check if the i + 1 interface is active.
                    vw_next_idx = vw_start_id + vw_next_pos                    # Position in the reduced system.
                    phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                               phi(j + 1, i + 1), m_list[k_next, j, i],
                                                               m_list[k_next, j + 1, i], m_list[k_next, j, i + 1],
                                                               m_list[k_next, j + 1, i + 1], v_w_list[k, j + 1, i],
                                                               v_w_list[k, j + 1, i + 1], direction = 'vertical')
                    A_reduced[eq_idx_water_y, vw_next_idx] -= (phi_m_upwind_average * miu_w / (dx * dx *
                                                               phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_w(j + 1/2, i)^(k + 1) at first dx parenthesis.
            if i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), m_list[k_next, j, i],
                                                           m_list[k_next, j + 1, i], m_list[k_next, j, i + 1],
                                                           m_list[k_next, j + 1, i + 1], v_w_list[k, j + 1, i],
                                                           v_w_list[k, j + 1, i + 1], direction = 'vertical')
                A_reduced[eq_idx_water_y, vw_idx] += (phi_m_upwind_average * miu_w / (dx * dx *
                                                      phi_at_interface(phi(j, i), phi(j + 1, i))))

            # When we are at the rightmost column, the velocity on the wall is zero.

            # Coefficient for u_w(j + 1, i + 1/2)^k at first dx parenthesis.
            if i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), m_list[k_next, j, i],
                                                           m_list[k_next, j + 1, i], m_list[k_next, j, i + 1],
                                                           m_list[k_next, j + 1, i + 1], v_w_list[k, j + 1, i],
                                                           v_w_list[k, j + 1, i + 1], direction = 'vertical')
                b_reduced[eq_idx_water_y] += (phi_m_upwind_average * miu_w * u_w_list[k, j + 1, i + 1] / (dx * dy *
                                              phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for u_w(j, i + 1/2)^k at first dx parenthesis.
            if i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), m_list[k_next, j, i],
                                                           m_list[k_next, j + 1, i], m_list[k_next, j, i + 1],
                                                           m_list[k_next, j + 1, i + 1], v_w_list[k, j + 1, i],
                                                           v_w_list[k, j + 1, i + 1], direction = 'vertical')
                b_reduced[eq_idx_water_y] -= (phi_m_upwind_average * miu_w * u_w_list[k, j, i + 1] / (dx * dy *
                                              phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_w(j + 1/2, i)^(k + 1) at the second dx parenthesis.
            if i > 0:
                phi_m_upwind_average = upwind_then_average(phi(j, i - 1), phi(j + 1, i - 1), phi(j, i), phi(j + 1, i),
                                                           m_list[k_next, j, i - 1], m_list[k_next, j + 1, i - 1],
                                                           m_list[k_next, j, i], m_list[k_next, j + 1, i],
                                                           v_w_list[k, j + 1, i - 1], v_w_list[k, j + 1, i],
                                                           direction = 'vertical')
                A_reduced[eq_idx_water_y, vw_idx] += (phi_m_upwind_average * miu_w / (dx * dx *
                                                      phi_at_interface(phi(j, i), phi(j + 1, i))))

            # When we at the leftmost column, the velocity on the wall is zero.

            # Coefficient for v_w(j + 1/2, i - 1)^(k + 1) at second dx parenthesis - only if that interface is active.
            if i > 0:
                vw_prev_pos = vw_pos_map[j, i - 1]                             # Position in the active list.
                if vw_prev_pos >= 0:                                           # Check if the i - 1 interface is active.
                    vw_prev_idx = vw_start_id + vw_prev_pos                    # Position in the reduced system.
                    phi_m_upwind_average = upwind_then_average(phi(j, i - 1), phi(j + 1, i - 1), phi(j, i),
                                                               phi(j + 1, i), m_list[k_next, j, i - 1],
                                                               m_list[k_next, j + 1, i - 1], m_list[k_next, j, i],
                                                               m_list[k_next, j + 1, i], v_w_list[k, j + 1, i - 1],
                                                               v_w_list[k, j + 1, i], direction = 'vertical')
                    A_reduced[eq_idx_water_y, vw_prev_idx] -= (phi_m_upwind_average * miu_w / (dx * dx *
                                                               phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for u_w(j + 1, i - 1/2)^k at second dx parenthesis.
            if i > 0:
                phi_m_upwind_average = upwind_then_average(phi(j, i - 1), phi(j + 1, i - 1), phi(j, i),
                                                           phi(j + 1, i), m_list[k_next, j, i - 1],
                                                           m_list[k_next, j + 1, i - 1], m_list[k_next, j, i],
                                                           m_list[k_next, j + 1, i], v_w_list[k, j + 1, i - 1],
                                                           v_w_list[k, j + 1, i], direction = 'vertical')
                b_reduced[eq_idx_water_y] -= (phi_m_upwind_average * miu_w * u_w_list[k, j + 1, i] / (dx * dy *
                                              phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for u_w(j, i - 1/2)^k  at second dx parenthesis.
            if i > 0:
                phi_m_upwind_average = upwind_then_average(phi(j, i - 1), phi(j + 1, i - 1), phi(j, i),
                                                           phi(j + 1, i), m_list[k_next, j, i - 1],
                                                           m_list[k_next, j + 1, i - 1], m_list[k_next, j, i],
                                                           m_list[k_next, j + 1, i], v_w_list[k, j + 1, i - 1],
                                                           v_w_list[k, j + 1, i], direction = 'vertical')
                b_reduced[eq_idx_water_y] += (phi_m_upwind_average * miu_w * u_w_list[k, j, i] / (dx * dy *
                                              phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_w(j + 3/2, i)^(k + 1) at the first dy parenthesis - only if that interface is active.
            if j < N_y - 2:
                vw_next_pos = vw_pos_map[j + 1, i]                        # Position in the active list.
                if vw_next_pos >= 0:                                      # Check if the (j + 1, i) interface is active.
                    vw_next_idx = vw_start_id + vw_next_pos               # Position in the reduced system.
                    A_reduced[eq_idx_water_y, vw_next_idx] -= ((2 * miu_w + kappa_w) * phi(j + 1, i) / (dy * dy *
                                                               phi_at_interface(phi(j, i), phi(j + 1, i))))

            elif j == N_y - 2:  # When we are at the j = N_y - 2 row, the velocity of the next interface equals to the
                # velocity of this interface, So we will write it for this interface.
                A_reduced[eq_idx_water_y, vw_idx] -= ((2 * miu_w + kappa_w) * phi(j + 1, i) / (dy * dy *
                                                      phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_w(j + 1/2, i)^(k + 1) at the first dy parenthesis.
            A_reduced[eq_idx_water_y, vw_idx] += ((2 * miu_w + kappa_w) * phi(j + 1, i) / (dy * dy *
                                                  phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for u_w(j + 1, i + 1/2)^k at the first dy parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_water_y] += (kappa_w * phi(j + 1, i) * u_w_list[k, j + 1, i + 1] / (dy * dx *
                                              phi_at_interface(phi(j, i), phi(j + 1, i))))

                # When we are the rightmost column, the velocity on the wall is zero.

            # Coefficient for u_w(j + 1, i - 1/2)^k at the first dy parenthesis.
            if i > 0:
                b_reduced[eq_idx_water_y] -= (kappa_w * phi(j + 1, i) * u_w_list[k, j + 1, i] / (dy * dx *
                                              phi_at_interface(phi(j, i), phi(j + 1, i))))

                # When we are the leftmost column, the velocity on the wall is zero.

            # Coefficient for v_w(j + 1/2, i)^(k + 1) at the second dy parenthesis.
            A_reduced[eq_idx_water_y, vw_idx] += ((2 * miu_w + kappa_w) * phi(j, i) / (dy * dy *
                                                  phi_at_interface(phi(j, i), phi(j + 1, i))))

            # Coefficient for v_w(j - 1/2, i)^(k + 1) at the second dy parenthesis - only if that interface is active.
            if j > 0:
                vw_prev_pos = vw_pos_map[j - 1, i]                        # Position in the active list.
                if vw_prev_pos >= 0:                                      # Check if the (j - 1, i) interface is active.
                    vw_prev_idx = vw_start_id + vw_prev_pos               # Position in the reduced system.
                    A_reduced[eq_idx_water_y, vw_prev_idx] -= ((2 * miu_w + kappa_w) * phi(j, i) / (dy * dy *
                                                               phi_at_interface(phi(j, i), phi(j + 1, i))))

            # With j = 0, the previous interface is the inlet, so its velocity (here j - 1) is zero.

            # Coefficient for u_w(j, i + 1/2)^k at the second dy parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_water_y] -= (kappa_w * phi(j, i) * u_w_list[k, j, i + 1] / (dy * dx *
                                              phi_at_interface(phi(j, i), phi(j + 1, i))))

                # When we are the rightmost column, the velocity on the wall is zero.

            # Coefficient for u_w(j, i - 1/2)^k at the second dy parenthesis.
            if i > 0:
                b_reduced[eq_idx_water_y] += (kappa_w * phi(j, i) * u_w_list[k, j, i] / (dy * dx *
                                              phi_at_interface(phi(j, i), phi(j + 1, i))))

                # When we are the leftmost column, the velocity on the wall is zero.

        #========================= Gas Momentum Equation (A9 first part) in the X direction ============================
        # Build equations A9 (Momentum balance) for interfaces (Only for active gas interfaces).
        # When the gas phase is absent at an interface, we don't build the momentum equation,
        # and velocity automatically stays 0.
        for idx, indices in enumerate(active_ug_indices):                       # indices is the global interface index.
            j = indices[0]
            i = indices[1]
            # Equation index for gas momentum in the X direction in the reduced system.
            eq_idx_gas_x = ug_start_id + idx

            # Calculate K̂ coefficients at interface (j, i + 1/2).
            u_w_sig = u_w_list[k, j, i + 1]
            u_g_sig = u_g_list[k, j, i + 1]
            S_w_interface = calculate_upwind_scheme(S_w_list[k + q, j, i], S_w_list[k + q, j, i + 1], u_w_sig)
            S_g_interface = calculate_upwind_scheme(S_g_list[k + q, j, i], S_g_list[k + q, j, i + 1], u_g_sig)

            k_hat_g = calculate_interaction_coefficient_gas(i, j , k, q, direction = 'horizontal')
            k_hat = calculate_interaction_coefficient_water_gas(i, j , k, q, direction = 'horizontal')

            var_idx = (j * N_x) + i                                               # Variable index for P_w(j, i)^(k + 1)
            ug_idx = ug_start_id + idx                                       # u_g variable index in the reduced system.

            # Time derivative upwind quantity of gas.
            n_upwind = calculate_upwind_scheme(n_list[k_next, j, i], n_list[k_next, j, i + 1], u_g_list[k, j, i + 1])   # n_(j, i + 1/2)^(k + 1)
            n_upwind_k = calculate_upwind_scheme(n_list[k, j, i], n_list[k, j, i + 1], u_g_list[k, j, i + 1])           # n_(j, i + 1/2)^k

            # Interaction terms
            # Coupling with water velocity (if water is also active at this interface).
            uw_pos_in_active = uw_pos_map[j, i]                        # Position in the active list.
            if uw_pos_in_active >= 0:                                  # Water is also active at (j, i + 1/2) interface.
                uw_idx = uw_start_id + uw_pos_in_active                # Position in the reduced system.
                A_reduced[eq_idx_gas_x, uw_idx] -= k_hat                              # k_hat * u_w_(j, i + 1/2)^(k + 1)

            A_reduced[eq_idx_gas_x, ug_idx] += k_hat + k_hat_g            # (k_hat + k_hat_g) * u_g_(j, i + 1/2)^(k + 1)

            # Time derivative term:                             (n_(j, i + 1/2)^(k + 1) / dt) * u_g_(j, i + 1/2)^(k + 1)
            A_reduced[eq_idx_gas_x, ug_idx] += n_upwind / dt

            # Convection terms (LHS of A9 first equation) - only when the gas phase is present.
            # Coefficient for u_g_(j, i + 1/2)^(k + 1)
            # The first term at dx parenthesis * u_g_(j, i + 1/2)^(k + 1)
            if i < N_x - 1:                                                           # Check if i + 2 is within bounds.
                A_reduced[eq_idx_gas_x, ug_idx] += (((phi(j, i + 1) * n_list[k_next, j, i + 1]) *
                                                     (u_g_list[k, j, i + 1] + u_g_list[k, j, i + 2])) /
                                                    (4 * dx * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # If i + 2 is out of bounds (we are at the rightmost column), its velocity and phi (j, i + 1) will be zero.
            # So we don't need to add anything to the A_reduced matrix.

            # The second term at dx parenthesis * u_g_(j, i + 1/2)^(k + 1)
            if i < N_x - 1:                                                           # Check if i + 1 is within bounds.
                A_reduced[eq_idx_gas_x, ug_idx] -= (phi(j, i) * n_list[k_next, j, i] *
                                                    (u_g_list[k, j, i] + u_g_list[k, j, i + 1]) /
                                                    (4 * dx * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # If i + 1 is out of bounds (we are at the rightmost column), phi (j, i + 1) will be zero.
            # So we don't need to add anything to the A_reduced matrix.

            # The first term at dy parenthesis * u_g_(j, i + 1/2)^(k + 1)
            if i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1), phi(j + 1, i + 1),
                                                           n_list[k_next, j, i], n_list[k_next, j + 1, i],
                                                           n_list[k_next, j, i + 1], n_list[k_next, j + 1, i + 1],
                                                           u_g_list[k, j, i + 1], u_g_list[k, j + 1, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_gas_x, ug_idx] += (phi_n_upwind_average * (v_g_list[k, j + 1, i] +
                                                                            v_g_list[k, j + 1, i + 1]) /
                                                    (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # When we are at the rightmost column, we assume there are a ghost column next to it, where n = 0,
            # phi = 0, and velocity is zero. Since we use min for phi at interfaces, and phi of this ghost column is
            # zero; the whole term will equal zero, and we don't need to add anything to the A_reduced matrix.

            # The second term at dy parenthesis * u_g_(j, i + 1/2)^(k + 1)
            if j > 0 and i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1), phi(j, i + 1),
                                                           n_list[k_next, j - 1, i], n_list[k_next, j, i],
                                                           n_list[k_next, j - 1, i + 1], n_list[k_next, j, i + 1],
                                                           u_g_list[k, j - 1, i + 1], u_g_list[k, j, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_gas_x, ug_idx] -= (phi_n_upwind_average * (v_g_list[k, j, i] + v_g_list[k, j, i + 1])
                                                    / (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == 0 and i < N_x - 1:
                # When we are at the bottom row, we assume there are a ghost row below that, where n = rho_g_in, phi = 1,
                # and u_g is 0. (Full of gas, and inlet velocity is upward)
                phi_n_upwind_average = upwind_then_average(1, phi(j, i), 1, phi(j, i + 1),
                                                           rho_g_in, n_list[k_next, j, i], rho_g_in,
                                                           n_list[k_next, j, i + 1], 0, u_g_list[k, j, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_gas_x, ug_idx] -= (phi_n_upwind_average * (v_g_list[k, j, i] + v_g_list[k, j, i + 1])
                                                    / (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # When we are at the rightmost column, we assume there are a ghost column next to it, where n = 0,
            # phi = 0, and velocity is zero. Since we use min for phi at interfaces, and phi of this ghost column is
            # zero; the whole term will equal zero, and we don't need to add anything to the A_reduced matrix.


            # Coefficient for u_g_(j, i + 3/2)^(k + 1) - only if that interface is also active.
            if i < N_x - 2:
                ug_next_pos = ug_pos_map[j, i + 1]                        # Position in the active list.
                if ug_next_pos >= 0:                                      # Check if the (j, i + 1) interface is active.
                    ug_next_idx = ug_start_id + ug_next_pos               # Position in the reduced system.
                    A_reduced[eq_idx_gas_x, ug_next_idx] += (phi(j, i + 1) * n_list[k_next, j, i + 1] *
                                                             (u_g_list[k, j, i + 1] + u_g_list[k, j, i + 2]) /
                                                             (4 * dx * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for u_g_(j, i - 1/2)^(k + 1) - only if that interface is also active.
            if 0 < i < N_x - 1:
                ug_prev_pos = ug_pos_map[j, i - 1]                        # Position in the active list.
                if ug_prev_pos >= 0:                                      # Check if the (j, i - 1) interface is active.
                    ug_prev_idx = ug_start_id + ug_prev_pos               # Position in the reduced system.
                    A_reduced[eq_idx_gas_x, ug_prev_idx] -= (phi(j, i) * n_list[k_next, j, i] *
                                                             (u_g_list[k, j, i] + u_g_list[k, j, i + 1]) /
                                                             (4 * dx * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for u_g_(j + 1, i + 1/2)^(k + 1) - only if that interface is also active.
            if j < N_y - 2 and i < N_x - 1:
                ug_next_pos = ug_pos_map[j + 1, i]                        # Position in the active list.
                if ug_next_pos >= 0:                                      # Check if the (j + 1, i) interface is active.
                    ug_next_idx = ug_start_id + ug_next_pos               # Position in the reduced system.
                    phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                               phi(j + 1, i + 1), n_list[k_next, j, i],
                                                               n_list[k_next, j + 1, i], n_list[k_next, j, i + 1],
                                                               n_list[k_next, j + 1, i + 1], u_g_list[k, j, i + 1],
                                                               u_g_list[k, j + 1, i + 1], direction = 'horizontal')
                    A_reduced[eq_idx_gas_x, ug_next_idx] += (phi_n_upwind_average * (v_g_list[k, j + 1, i] +
                                                                                v_g_list[k, j + 1, i + 1]) /
                                                        (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == N_y - 2 and i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1), phi(j + 1, i + 1),
                                                           n_list[k_next, j, i], n_list[k_next, j + 1, i],
                                                           n_list[k_next, j, i + 1], n_list[k_next, j + 1, i + 1],
                                                           u_g_list[k, j, i + 1], u_g_list[k, j + 1, i + 1],
                                                           direction ='horizontal')
                A_reduced[eq_idx_gas_x, ug_idx] += (phi_n_upwind_average * (v_g_list[k, j + 1, i] +
                                                                                v_g_list[k, j + 1, i + 1]) /
                                                        (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are at the j = N_y - 2 row, the velocity of the next interface equals to
                # velocity of this interface, So will write it for this interface.

            # When we are at the rightmost column, we assume there are a ghost column next to it, where n = 0,
            # phi = 0, and velocity is zero. Since we use min for phi at interfaces, and phi of this ghost column is
            # zero; the whole term will equal zero, and we don't need to add anything to the A_reduced matrix.

            # Coefficient for u_g_(j - 1, i + 1/2)^(k + 1) - only if that interface is also active.
            if j > 0 and i < N_x - 1:
                ug_prev_pos = ug_pos_map[j - 1, i]                        # Position in the active list.
                if ug_prev_pos >= 0:                                      # Check if the (j - 1, i) interface is active.
                    ug_prev_idx = ug_start_id + ug_prev_pos               # Position in the reduced system.
                    phi_n_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1),
                                                               phi(j, i + 1), n_list[k_next, j - 1, i],
                                                               n_list[k_next, j, i], n_list[k_next, j - 1, i + 1],
                                                               n_list[k_next, j, i + 1], u_g_list[k, j - 1, i + 1],
                                                               u_g_list[k, j, i + 1], direction = 'horizontal')
                    A_reduced[eq_idx_gas_x, ug_prev_idx] -= (phi_n_upwind_average * (v_g_list[k, j, i] +
                                                                                     v_g_list[k, j, i + 1]) /
                                                             (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # When we are at the bottom row, we assume there are a ghost row below that, where n = rho_g_in, phi = 1,
            # and u_g is 0. (Full of gas, and inlet velocity is upward)
            # So we don't need to add anything to the A_reduced matrix.

            # Pressure gradient terms (LHS of A9 first equation)
            # Coefficient for P_w(j, i + 1)^(k + 1)
            if i + 1 < N_x:                                                             # Column i + 1 is within bounds.
                A_reduced[eq_idx_gas_x, var_idx + 1] += S_g_interface / dx

            # If i + 1 is out of bounds (we are at the rightmost column), the pressure gradient term will be zero.

            # Coefficient for P_w(j, i)^(k + 1)
            A_reduced[eq_idx_gas_x, var_idx] -= S_g_interface / dx

            # Capillary pressure terms (RHS of A9 first equation)
            # Coefficient for P_c_(j, i + 1)^(k + 1)
            if i + 1 < N_x:                                                             # Column i + 1 is within bounds.
                b_reduced[eq_idx_gas_x] -= S_g_interface * Pc_list[k + q, j, i + 1] / dx

            # Coefficient for P_c_(j, i)^(k + 1)
            b_reduced[eq_idx_gas_x] += S_g_interface * Pc_list[k + q, j, i] / dx

            # Gravitational acceleration in the X direction is zero.

            # Coefficient for u_g_(j, i + 1/2)^k
            b_reduced[eq_idx_gas_x] += n_upwind_k * u_g_list[k, j, i + 1] / dt

            # Viscous-stress terms (The last term of RHS)
            # Coefficient for u_g(j + 1, i + 1/2)^(k + 1) - only if that interface is also active.
            if j < N_y - 2 and i < N_x - 1:
                ug_next_pos = ug_pos_map[j + 1, i]                             # Position in the active list.
                if ug_next_pos >= 0:                                           # Check if the j + 1 interface is active.
                    ug_next_idx = ug_start_id + ug_next_pos                    # Position in the reduced system.
                    phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                               phi(j + 1, i + 1), n_list[k_next, j, i],
                                                               n_list[k_next, j + 1, i], n_list[k_next, j, i + 1],
                                                               n_list[k_next, j + 1, i + 1], u_g_list[k, j, i + 1],
                                                               u_g_list[k, j + 1, i + 1], direction = 'horizontal')
                    A_reduced[eq_idx_gas_x, ug_next_idx] -= (phi_n_upwind_average * miu_g / (dy * dy *
                                                             phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == N_y - 2 and i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1), phi(j + 1, i + 1),
                                                           n_list[k_next, j, i], n_list[k_next, j + 1, i],
                                                           n_list[k_next, j, i + 1], n_list[k_next, j + 1, i + 1],
                                                           u_g_list[k, j, i + 1], u_g_list[k, j + 1, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_gas_x, ug_idx] -= (phi_n_upwind_average * miu_g / (dy * dy *
                                                             phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are at the j = N_y - 2 row, the velocity of the next interface equals to velocity of this
                # interface, So will write it for this interface.

            # Coefficient for U_g(j, i + 1/2)^(k + 1) at first dy parenthesis.
            if i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), n_list[k_next, j, i],
                                                           n_list[k_next, j + 1, i], n_list[k_next, j, i + 1],
                                                           n_list[k_next, j + 1, i + 1], u_g_list[k, j, i + 1],
                                                           u_g_list[k, j + 1, i + 1], direction = 'horizontal')
                A_reduced[eq_idx_gas_x, ug_idx] += (phi_n_upwind_average * miu_g / (dy * dy *
                                                    phi_at_interface(phi(j, i), phi(j, i + 1))))

            # When we are at the rightmost column, the velocity on the wall is zero.

            # Coefficient for v_g(j + 1/2, i + 1)^k at first dy parenthesis.
            if 0 < i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), n_list[k_next, j, i],
                                                           n_list[k_next, j + 1, i], n_list[k_next, j, i + 1],
                                                           n_list[k_next, j + 1, i + 1], u_g_list[k, j, i + 1],
                                                           u_g_list[k, j + 1, i + 1], direction = 'horizontal')
                b_reduced[eq_idx_gas_x] += (phi_n_upwind_average * miu_g * v_g_list[k, j + 1, i + 1] / (dx * dy *
                                            phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for v_g(j + 1/2, i)^k at the first dy parenthesis.
            if 0 < i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), n_list[k_next, j, i],
                                                           n_list[k_next, j + 1, i], n_list[k_next, j, i + 1],
                                                           n_list[k_next, j + 1, i + 1], u_g_list[k, j, i + 1],
                                                           u_g_list[k, j + 1, i + 1], direction = 'horizontal')
                b_reduced[eq_idx_gas_x] -= (phi_n_upwind_average * miu_g * v_g_list[k, j + 1, i] / (dx * dy *
                                            phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for u_g(j, i + 1/2)^(k + 1) at the second dy parenthesis.
            if j > 0 and i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1), phi(j, i + 1),
                                                           n_list[k_next, j - 1, i], n_list[k_next, j, i],
                                                           n_list[k_next, j - 1, i + 1], n_list[k_next, j, i + 1],
                                                           u_g_list[k, j - 1, i + 1], u_g_list[k, j, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_gas_x, ug_idx] += (phi_n_upwind_average * miu_g / (dy * dy *
                                                    phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == 0 and i < N_x - 1:
                # When we are at the bottom row, we assume there are a ghost row below that, where n = rho_g_in, phi = 1,
                # and u_g is 0. (Full of gas, and inlet velocity is upward)
                phi_n_upwind_average = upwind_then_average(1, phi(j, i), 1, phi(j, i + 1),
                                                           rho_g_in, n_list[k_next, j, i], rho_g_in,
                                                           n_list[k_next, j, i + 1], 0, u_g_list[k, j, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_gas_x, ug_idx] += (phi_n_upwind_average * miu_g / (dy * dy *
                                                    phi_at_interface(phi(j, i), phi(j, i + 1))))

            # When we at the rightmost column, the velocity on the wall is zero.

            # Coefficient for u_g(j - 1, i + 1/2)^(k + 1) at second dy parenthesis - only if that interface is active.
            if j > 0 and i < N_x - 1:
                ug_prev_pos = ug_pos_map[j - 1, i]                             # Position in the active list.
                if ug_prev_pos >= 0:                                           # Check if the j - 1 interface is active.
                    ug_prev_idx = ug_start_id + ug_prev_pos                    # Position in the reduced system.
                    phi_n_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1),
                                                               phi(j, i + 1), n_list[k_next, j - 1, i],
                                                               n_list[k_next, j, i], n_list[k_next, j - 1, i + 1],
                                                               n_list[k_next, j, i + 1], u_g_list[k, j - 1, i + 1],
                                                               u_g_list[k, j, i + 1], direction = 'horizontal')
                    A_reduced[eq_idx_gas_x, ug_prev_idx] -= (phi_n_upwind_average * miu_g / (dy * dy *
                                                             phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are at the bottom row, we assume there are a ghost row below that, where n = rho_g_in, phi = 1,
                # and u_g is 0. (Full of gas, and inlet velocity is upward)
                # So we don't need to add anything to the A_reduced matrix.

            # Coefficient for v_g(j - 1/2, i + 1)^k at the second dy parenthesis.
            if j > 0 and i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1),
                                                               phi(j, i + 1), n_list[k_next, j - 1, i],
                                                               n_list[k_next, j, i], n_list[k_next, j - 1, i + 1],
                                                               n_list[k_next, j, i + 1], u_g_list[k, j - 1, i + 1],
                                                               u_g_list[k, j, i + 1], direction = 'horizontal')
                b_reduced[eq_idx_gas_x] -= (phi_n_upwind_average * miu_g * v_g_list[k, j, i + 1] / (dx * dy *
                                            phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == 0 and i < N_x - 1:
                # When we are at the bottom row, we assume there are a ghost row below that, where n = rho_g_in, phi = 1,
                # and u_g is 0. (Full of gas, and inlet velocity is upward)
                phi_n_upwind_average = upwind_then_average(1, phi(j, i), 1, phi(j, i + 1),
                                                           rho_g_in, n_list[k_next, j, i], rho_g_in,
                                                           n_list[k_next, j, i + 1], 0, u_g_list[k, j, i + 1],
                                                           direction = 'horizontal')
                b_reduced[eq_idx_gas_x] -= (phi_n_upwind_average * miu_g * v_g_list[k, j, i + 1] / (dx * dy *
                                            phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for v_g(j - 1/2, i)^k  at the second dy parenthesis.
            if j > 0 and i < N_x - 1:
                phi_n_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1),
                                                               phi(j, i + 1), n_list[k_next, j - 1, i],
                                                               n_list[k_next, j, i], n_list[k_next, j - 1, i + 1],
                                                               n_list[k_next, j, i + 1], u_g_list[k, j - 1, i + 1],
                                                               u_g_list[k, j, i + 1], direction = 'horizontal')
                b_reduced[eq_idx_gas_x] += (phi_n_upwind_average * miu_g * v_g_list[k, j, i] / (dx * dy *
                                            phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == 0 and i < N_x - 1:
                # When we are at the bottom row, we assume there are a ghost row below that, where n = rho_g_in, phi = 1,
                # and u_g is 0. (Full of gas, and inlet velocity is upward)
                phi_n_upwind_average = upwind_then_average(1, phi(j, i), 1, phi(j, i + 1),
                                                           rho_g_in, n_list[k_next, j, i], rho_g_in,
                                                           n_list[k_next, j, i + 1], 0, u_g_list[k, j, i + 1],
                                                           direction = 'horizontal')
                b_reduced[eq_idx_gas_x] += (phi_n_upwind_average * miu_g * v_g_list[k, j, i] / (dx * dy *
                                            phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for u_g(j, i + 3/2)^(k + 1) at the first dx parenthesis - only if that interface is active.
            if i < N_x - 2:
                ug_next_pos = ug_pos_map[j, i + 1]                        # Position in the active list.
                if ug_next_pos >= 0:                                      # Check if the (j, i + 1) interface is active.
                    ug_next_idx = ug_start_id + ug_next_pos               # Position in the reduced system.
                    A_reduced[eq_idx_gas_x, ug_next_idx] -= ((2 * miu_g + kappa_g) * phi(j, i + 1) / (dx * dx *
                                                              phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are at the i = N_x - 2 column, the velocity of the next interface is velocity on the wall,
                # which is zero. So we don't need to add anything to the A_reduced matrix.

            # Coefficient for u_g(j, i + 1/2)^(k + 1) at the first dx parenthesis.
            if i < N_x - 1:
                A_reduced[eq_idx_gas_x, ug_idx] += ((2 * miu_g + kappa_g) * phi(j, i + 1) / (dx * dx *
                                                     phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for v_g(j + 1/2, i + 1)^k at the first dx parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_gas_x] += (kappa_g * phi(j, i + 1) * v_g_list[k, j + 1, i + 1] / (dy * dx *
                                            phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are the rightmost column, the velocity on the wall is zero.

            # Coefficient for v_g(j - 1 /2, i + 1)^k at the first dx parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_gas_x] -= (kappa_g * phi(j, i + 1) * v_g_list[k, j, i + 1] / (dy * dx *
                                            phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are the rightmost column, the velocity on the wall is zero.

            # Coefficient for u_g(j, i + 1/2)^(k + 1) at the second dx parenthesis.
            if i < N_x - 1:
                A_reduced[eq_idx_gas_x, ug_idx] += ((2 * miu_g + kappa_g) * phi(j, i) / (dx * dx *
                                                     phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for u_g(j, i - 1/2)^(k + 1) at the second dx parenthesis - only if that interface is active.
            if 0 < i < N_x - 1:
                ug_prev_pos = ug_pos_map[j, i - 1]                        # Position in the active list.
                if ug_prev_pos >= 0:                                      # Check if the (j, i - 1) interface is active.
                    ug_prev_idx = ug_start_id + ug_prev_pos               # Position in the reduced system.
                    A_reduced[eq_idx_gas_x, ug_prev_idx] -= ((2 * miu_g + kappa_g) * phi(j, i) / (dx * dx *
                                                              phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for v_g(j + 1/2, i)^k at the second dx parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_gas_x] -= (kappa_g * phi(j, i) * v_g_list[k, j + 1, i] / (dy * dx *
                                            phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for v_g(j - 1/2, i)^k at the second dx parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_gas_x] += (kappa_g * phi(j, i) * v_g_list[k, j, i] / (dy * dx *
                                            phi_at_interface(phi(j, i), phi(j, i + 1))))

        # ========================= Water Momentum Equation (A9 Second part) in the X direction ============================
        # Build equations A9 (Momentum balance) for interfaces (Only for active water interfaces).
        # When the water phase is absent at an interface, we don't build the momentum equation,
        # and velocity automatically stays 0.
        for idx, indices in enumerate(active_uw_indices):                       # indices is the global interface index.
            j = indices[0]
            i = indices[1]
            # Equation index for water momentum in the X direction in the reduced system.
            eq_idx_water_x = uw_start_id + idx

            # Calculate K̂ coefficients at interface (j, i + 1/2).
            u_w_sig = u_w_list[k, j, i + 1]
            u_g_sig = u_g_list[k, j, i + 1]
            S_w_interface = calculate_upwind_scheme(S_w_list[k + q, j, i], S_w_list[k + q, j, i + 1], u_w_sig)
            S_g_interface = calculate_upwind_scheme(S_g_list[k + q, j, i], S_g_list[k + q, j, i + 1], u_g_sig)

            k_hat_w = calculate_interaction_coefficient_water(i, j, k, q, direction = 'horizontal')
            k_hat = calculate_interaction_coefficient_water_gas(i, j, k, q, direction = 'horizontal')

            var_idx = (j * N_x) + i                                               # Variable index for P_w(j, i)^(k + 1)
            uw_idx = uw_start_id + idx                                       # u_w variable index in the reduced system.

            # Time derivative upwind quantity of water.
            m_upwind = calculate_upwind_scheme(m_list[k_next, j, i], m_list[k_next, j, i + 1], u_w_list[k, j, i + 1])   # m_(j, i + 1/2)^(k + 1)
            m_upwind_k = calculate_upwind_scheme(m_list[k, j, i], m_list[k, j, i + 1], u_w_list[k, j, i + 1])           # m_(j, i + 1/2)^k

            # Interaction terms
            # Coupling with gas velocity (if gas is also active at this interface).
            ug_pos_in_active = ug_pos_map[j, i]                          # Position in the active list.
            if ug_pos_in_active >= 0:                                    # Gas is also active at (j, i + 1/2) interface.
                ug_idx = ug_start_id + ug_pos_in_active                  # Position in the reduced system.
                A_reduced[eq_idx_water_x, ug_idx] -= k_hat                            # k_hat * u_g_(j, i + 1/2)^(k + 1)

            A_reduced[eq_idx_water_x, uw_idx] += k_hat + k_hat_w          # (k_hat + k_hat_w) * u_w_(j, i + 1/2)^(k + 1)

            # Time derivative term:                             (m_(j, i + 1/2)^(k + 1) / dt) * u_w_(j, i + 1/2)^(k + 1)
            A_reduced[eq_idx_water_x, uw_idx] += m_upwind / dt

            # Convection terms (LHS of A9 first equation) - only when the water phase is present.
            # Coefficient for u_w_(j, i + 1/2)^(k + 1)
            # The first term at dx parenthesis * u_w_(j, i + 1/2)^(k + 1)
            if i < N_x - 1:                                                           # Check if i + 2 is within bounds.
                A_reduced[eq_idx_water_x, uw_idx] += (((phi(j, i + 1) * m_list[k_next, j, i + 1]) *
                                                     (u_w_list[k, j, i + 1] + u_w_list[k, j, i + 2])) /
                                                    (4 * dx * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # If i + 2 is out of bounds (we are at the rightmost column), its velocity and phi (j, i + 1) will be zero.
            # So we don't need to add anything to the A_reduced matrix.

            # The second term at dx parenthesis * u_w_(j, i + 1/2)^(k + 1)
            if i < N_x - 1:                                                           # Check if i + 1 is within bounds.
                A_reduced[eq_idx_water_x, uw_idx] -= (phi(j, i) * m_list[k_next, j, i] *
                                                    (u_w_list[k, j, i] + u_w_list[k, j, i + 1]) /
                                                    (4 * dx * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # If i + 1 is out of bounds (we are at the rightmost column), phi (j, i + 1) will be zero.
            # So we don't need to add anything to the A_reduced matrix.

            # The first term at dy parenthesis * u_w_(j, i + 1/2)^(k + 1)
            if i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1), phi(j + 1, i + 1),
                                                           m_list[k_next, j, i], m_list[k_next, j + 1, i],
                                                           m_list[k_next, j, i + 1], m_list[k_next, j + 1, i + 1],
                                                           u_w_list[k, j, i + 1], u_w_list[k, j + 1, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_water_x, uw_idx] += (phi_m_upwind_average * (v_w_list[k, j + 1, i] +
                                                                            v_w_list[k, j + 1, i + 1]) /
                                                    (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # When we are at the rightmost column, we assume there are a ghost column next to it, where m = rho_wr,
            # phi = 0, and velocity is zero. Since we use min for phi at interfaces, and phi of this ghost column is
            # zero; the whole term will equal zero, and we don't need to add anything to the A_reduced matrix.

            # The second term at dy parenthesis * u_g_(j, i + 1/2)^(k + 1)
            if j > 0 and i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1), phi(j, i + 1),
                                                           m_list[k_next, j - 1, i], m_list[k_next, j, i],
                                                           m_list[k_next, j - 1, i + 1], m_list[k_next, j, i + 1],
                                                           u_w_list[k, j - 1, i + 1], u_w_list[k, j, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_water_x, uw_idx] -= (phi_m_upwind_average * (v_w_list[k, j, i] + v_w_list[k, j, i + 1])
                                                    / (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == 0 and i < N_x - 1:
                # When we are at the bottom row, we assume there are a ghost row below that, where m = 0, phi = 1,
                # and u_w is 0. (Full of gas, and inlet velocity is upward)
                phi_m_upwind_average = upwind_then_average(1, phi(j, i), 1, phi(j, i + 1),
                                                           0, m_list[k_next, j, i], 0,
                                                           m_list[k_next, j, i + 1], 0, u_w_list[k, j, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_water_x, uw_idx] -= (phi_m_upwind_average * (v_w_list[k, j, i] + v_w_list[k, j, i + 1])
                                                    / (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # When we are at the rightmost column, we assume there are a ghost column next to it, where m = 0,
            # phi = 0, and velocity is zero. Since we use min for phi at interfaces, and phi of this ghost column is
            # zero; the whole term will equal zero, and we don't need to add anything to the A_reduced matrix.

            # Coefficient for u_w_(j, i + 3/2)^(k + 1) - only if that interface is also active.
            if i < N_x - 2:
                uw_next_pos = uw_pos_map[j, i + 1]                        # Position in the active list.
                if uw_next_pos >= 0:                                      # Check if the (j, i + 1) interface is active.
                    uw_next_idx = uw_start_id + uw_next_pos               # Position in the reduced system.
                    A_reduced[eq_idx_water_x, uw_next_idx] += (phi(j, i + 1) * m_list[k_next, j, i + 1] *
                                                             (u_w_list[k, j, i + 1] + u_w_list[k, j, i + 2]) /
                                                             (4 * dx * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for u_w_(j, i - 1/2)^(k + 1) - only if that interface is also active.
            if 0 < i < N_x - 1:
                uw_prev_pos = uw_pos_map[j, i - 1]                        # Position in the active list.
                if uw_prev_pos >= 0:                                      # Check if the (j, i - 1) interface is active.
                    uw_prev_idx = uw_start_id + uw_prev_pos               # Position in the reduced system.
                    A_reduced[eq_idx_water_x, uw_prev_idx] -= (phi(j, i) * m_list[k_next, j, i] *
                                                             (u_w_list[k, j, i] + u_w_list[k, j, i + 1]) /
                                                             (4 * dx * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for u_w_(j + 1, i + 1/2)^(k + 1) - only if that interface is also active.
            if j < N_y - 2 and i < N_x - 1:
                uw_next_pos = uw_pos_map[j + 1, i]                        # Position in the active list.
                if uw_next_pos >= 0:                                      # Check if the (j + 1, i) interface is active.
                    uw_next_idx = uw_start_id + uw_next_pos               # Position in the reduced system.
                    phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                               phi(j + 1, i + 1), m_list[k_next, j, i],
                                                               m_list[k_next, j + 1, i], m_list[k_next, j, i + 1],
                                                               m_list[k_next, j + 1, i + 1], u_w_list[k, j, i + 1],
                                                               u_w_list[k, j + 1, i + 1], direction = 'horizontal')
                    A_reduced[eq_idx_water_x, uw_next_idx] += (phi_m_upwind_average * (v_w_list[k, j + 1, i] +
                                                                                     v_w_list[k, j + 1, i + 1]) /
                                                             (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == N_y - 2 and i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1), phi(j + 1, i + 1),
                                                           m_list[k_next, j, i], m_list[k_next, j + 1, i],
                                                           m_list[k_next, j, i + 1], m_list[k_next, j + 1, i + 1],
                                                           u_w_list[k, j, i + 1], u_w_list[k, j + 1, i + 1],
                                                           direction ='horizontal')
                A_reduced[eq_idx_water_x, uw_idx] += (phi_m_upwind_average * (v_w_list[k, j + 1, i] +
                                                                                     v_w_list[k, j + 1, i + 1]) /
                                                             (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are at the j = N_y - 2 row, the velocity of the next interface equals to
                # velocity of this interface, So will write it for this interface.

            # When we are at the rightmost column, we assume there are a ghost column next to it, where m = 0,
            # phi = 0, and velocity is zero. Since we use min for phi at interfaces, and phi of this ghost column is
            # zero; the whole term will equal zero, and we don't need to add anything to the A_reduced matrix.

            # Coefficient for u_w_(j - 1, i + 1/2)^(k + 1) - only if that interface is also active.
            if j > 0 and i < N_x - 1:
                uw_prev_pos = uw_pos_map[j - 1, i]                        # Position in the active list.
                if uw_prev_pos >= 0:                                      # Check if the (j - 1, i) interface is active.
                    uw_prev_idx = uw_start_id + uw_prev_pos               # Position in the reduced system.
                    phi_m_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1),
                                                               phi(j, i + 1), m_list[k_next, j - 1, i],
                                                               m_list[k_next, j, i], m_list[k_next, j - 1, i + 1],
                                                               m_list[k_next, j, i + 1], u_w_list[k, j - 1, i + 1],
                                                               u_w_list[k, j, i + 1], direction = 'horizontal')
                    A_reduced[eq_idx_water_x, uw_prev_idx] -= (phi_m_upwind_average * (v_w_list[k, j, i] +
                                                                                     v_w_list[k, j, i + 1]) /
                                                             (4 * dy * phi_at_interface(phi(j, i), phi(j, i + 1))))

            # When we are at the bottom row, we assume there are a ghost row below that, where m = 0, phi = 1,
            # and u_w is 0. (Full of gas, and inlet velocity is upward)
            # So we don't need to add anything to the A_reduced matrix.

            # Pressure gradient terms (LHS of A9 first equation)
            # Coefficient for P_w(j, i + 1)^(k + 1)
            if i + 1 < N_x:                                                             # Column i + 1 is within bounds.
                A_reduced[eq_idx_water_x, var_idx + 1] += S_w_interface / dx

            # If i + 1 is out of bounds (we are at the rightmost column), the pressure gradient term will be zero.

            # Coefficient for P_w(j, i)^(k + 1)
            A_reduced[eq_idx_water_x, var_idx] -= S_w_interface / dx

            # Gravitational acceleration in the X direction is zero.

            # Coefficient for u_w_(j, i + 1/2)^k
            b_reduced[eq_idx_water_x] += m_upwind_k * u_w_list[k, j, i + 1] / dt

            # Viscous-stress terms (The last term of RHS)
            # Coefficient for u_w(j + 1, i + 1/2)^(k + 1) - only if that interface is also active.
            if j < N_y - 2 and i < N_x - 1:
                uw_next_pos = uw_pos_map[j + 1, i]                             # Position in the active list.
                if uw_next_pos >= 0:                                           # Check if the j + 1 interface is active.
                    uw_next_idx = uw_start_id + uw_next_pos                    # Position in the reduced system.
                    phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                               phi(j + 1, i + 1), m_list[k_next, j, i],
                                                               m_list[k_next, j + 1, i], m_list[k_next, j, i + 1],
                                                               m_list[k_next, j + 1, i + 1], u_w_list[k, j, i + 1],
                                                               u_w_list[k, j + 1, i + 1], direction = 'horizontal')
                    A_reduced[eq_idx_water_x, uw_next_idx] -= (phi_m_upwind_average * miu_w / (dy * dy *
                                                               phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == N_y - 2 and i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1), phi(j + 1, i + 1),
                                                           m_list[k_next, j, i], m_list[k_next, j + 1, i],
                                                           m_list[k_next, j, i + 1], m_list[k_next, j + 1, i + 1],
                                                           u_w_list[k, j, i + 1], u_w_list[k, j + 1, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_water_x, uw_idx] -= (phi_m_upwind_average * miu_w / (dy * dy *
                                                      phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are at the j = N_y - 2 row, the velocity of the next interface equals to velocity of this
                # interface, So will write it for this interface.

            # Coefficient for U_w(j, i + 1/2)^(k + 1) at first dy parenthesis.
            if i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), m_list[k_next, j, i],
                                                           m_list[k_next, j + 1, i], m_list[k_next, j, i + 1],
                                                           m_list[k_next, j + 1, i + 1], u_w_list[k, j, i + 1],
                                                           u_w_list[k, j + 1, i + 1], direction = 'horizontal')
                A_reduced[eq_idx_water_x, uw_idx] += (phi_m_upwind_average * miu_w / (dy * dy *
                                                      phi_at_interface(phi(j, i), phi(j, i + 1))))

            # When we are at the rightmost column, the velocity on the wall is zero.

            # Coefficient for v_w(j + 1/2, i + 1)^k at first dy parenthesis.
            if i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), m_list[k_next, j, i],
                                                           m_list[k_next, j + 1, i], m_list[k_next, j, i + 1],
                                                           m_list[k_next, j + 1, i + 1], u_w_list[k, j, i + 1],
                                                           u_w_list[k, j + 1, i + 1], direction = 'horizontal')
                b_reduced[eq_idx_water_x] += (phi_m_upwind_average * miu_w * v_w_list[k, j + 1, i + 1] / (dx * dy *
                                              phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for v_w(j + 1/2, i)^k at the first dy parenthesis.
            if 0 < i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j, i), phi(j + 1, i), phi(j, i + 1),
                                                           phi(j + 1, i + 1), m_list[k_next, j, i],
                                                           m_list[k_next, j + 1, i], m_list[k_next, j, i + 1],
                                                           m_list[k_next, j + 1, i + 1], u_w_list[k, j, i + 1],
                                                           u_w_list[k, j + 1, i + 1], direction = 'horizontal')
                b_reduced[eq_idx_water_x] -= (phi_m_upwind_average * miu_w * v_g_list[k, j + 1, i] / (dx * dy *
                                               phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for u_w(j, i + 1/2)^(k + 1) at the second dy parenthesis.
            if j > 0 and i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1), phi(j, i + 1),
                                                           m_list[k_next, j - 1, i], m_list[k_next, j, i],
                                                           m_list[k_next, j - 1, i + 1], m_list[k_next, j, i + 1],
                                                           u_w_list[k, j - 1, i + 1], u_w_list[k, j, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_water_x, uw_idx] += (phi_m_upwind_average * miu_w / (dy * dy *
                                                      phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == 0 and i < N_x - 1:
                # When we are at the bottom row, we assume there are a ghost row below that, where m = 0, phi = 1,
                # and u_w is 0. (Full of gas, and inlet velocity is upward)
                phi_m_upwind_average = upwind_then_average(1, phi(j, i), 1, phi(j, i + 1),
                                                           0, m_list[k_next, j, i], 0,
                                                           m_list[k_next, j, i + 1], 0, u_w_list[k, j, i + 1],
                                                           direction = 'horizontal')
                A_reduced[eq_idx_water_x, uw_idx] += (phi_m_upwind_average * miu_w / (dy * dy *
                                                      phi_at_interface(phi(j, i), phi(j, i + 1))))

            # When we at the rightmost column, the velocity on the wall is zero.

            # Coefficient for u_w(j - 1, i + 1/2)^(k + 1) at second dy parenthesis - only if that interface is active.
            if j > 0 and i < N_x - 1:
                uw_prev_pos = uw_pos_map[j - 1, i]                             # Position in the active list.
                if uw_prev_pos >= 0:                                           # Check if the j - 1 interface is active.
                    uw_prev_idx = uw_start_id + uw_prev_pos                    # Position in the reduced system.
                    phi_m_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1),
                                                               phi(j, i + 1), m_list[k_next, j - 1, i],
                                                               m_list[k_next, j, i], m_list[k_next, j - 1, i + 1],
                                                               m_list[k_next, j, i + 1], u_w_list[k, j - 1, i + 1],
                                                               u_w_list[k, j, i + 1], direction = 'horizontal')
                    A_reduced[eq_idx_water_x, uw_prev_idx] -= (phi_m_upwind_average * miu_w / (dy * dy *
                                                               phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are at the bottom row, we assume there are a ghost row below that, where m = 0, phi = 1,
                # and u_w is 0. (Full of gas, and inlet velocity is upward)
                # So we don't need to add anything to the A_reduced matrix.

            # Coefficient for v_w(j - 1/2, i + 1)^k at the second dy parenthesis.
            if j > 0 and i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1),
                                                           phi(j, i + 1), m_list[k_next, j - 1, i],
                                                           m_list[k_next, j, i], m_list[k_next, j - 1, i + 1],
                                                           m_list[k_next, j, i + 1], u_w_list[k, j - 1, i + 1],
                                                           u_w_list[k, j, i + 1], direction = 'horizontal')
                b_reduced[eq_idx_water_x] -= (phi_m_upwind_average * miu_w * v_w_list[k, j, i + 1] / (dx * dy *
                                              phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == 0 and i < N_x - 1:
                # When we are at the bottom row, we assume there are a ghost row below that, where m = 0, phi = 1,
                # and u_w is 0. (Full of gas, and inlet velocity is upward)
                phi_m_upwind_average = upwind_then_average(1, phi(j, i), 1, phi(j, i + 1),
                                                           0, m_list[k_next, j, i], 0,
                                                           m_list[k_next, j, i + 1], 0, u_w_list[k, j, i + 1],
                                                           direction = 'horizontal')
                b_reduced[eq_idx_water_x] -= (phi_m_upwind_average * miu_w * v_w_list[k, j, i + 1] / (dx * dy *
                                              phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for v_w(j - 1/2, i)^k  at the second dy parenthesis.
            if j > 0 and i < N_x - 1:
                phi_m_upwind_average = upwind_then_average(phi(j - 1, i), phi(j, i), phi(j - 1, i + 1),
                                                           phi(j, i + 1), m_list[k_next, j - 1, i],
                                                           m_list[k_next, j, i], m_list[k_next, j - 1, i + 1],
                                                           m_list[k_next, j, i + 1], u_w_list[k, j - 1, i + 1],
                                                           u_w_list[k, j, i + 1], direction = 'horizontal')
                b_reduced[eq_idx_water_x] += (phi_m_upwind_average * miu_w * v_w_list[k, j, i] / (dx * dy *
                                              phi_at_interface(phi(j, i), phi(j, i + 1))))

            elif j == 0 and i < N_x - 1:
                # When we are at the bottom row, we assume there are a ghost row below that, where m = 0, phi = 1,
                # and u_w is 0. (Full of gas, and inlet velocity is upward)
                phi_m_upwind_average = upwind_then_average(1, phi(j, i), 1, phi(j, i + 1),
                                                           0, m_list[k_next, j, i], 0,
                                                           m_list[k_next, j, i + 1], 0, u_w_list[k, j, i + 1],
                                                           direction = 'horizontal')
                b_reduced[eq_idx_water_x] += (phi_m_upwind_average * miu_w * v_w_list[k, j, i] / (dx * dy *
                                              phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for u_w(j, i + 3/2)^(k + 1) at the first dx parenthesis - only if that interface is active.
            if i < N_x - 2:
                uw_next_pos = uw_pos_map[j, i + 1]                        # Position in the active list.
                if uw_next_pos >= 0:                                      # Check if the (j, i + 1) interface is active.
                    uw_next_idx = uw_start_id + uw_next_pos               # Position in the reduced system.
                    A_reduced[eq_idx_water_x, uw_next_idx] -= ((2 * miu_w + kappa_w) * phi(j, i + 1) / (dx * dx *
                                                                phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are at the i = N_x - 2 column, the velocity of the next interface is velocity on the wall,
                # which is zero. So we don't need to add anything to the A_reduced matrix.

            # Coefficient for u_w(j, i + 1/2)^(k + 1) at the first dx parenthesis.
            if i < N_x - 1:
                A_reduced[eq_idx_water_x, uw_idx] += ((2 * miu_w + kappa_w) * phi(j, i + 1) / (dx * dx *
                                                       phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for v_w(j + 1/2, i + 1)^k at the first dx parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_water_x] += (kappa_w * phi(j, i + 1) * v_w_list[k, j + 1, i + 1] / (dy * dx *
                                              phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are the rightmost column, the velocity on the wall is zero.

            # Coefficient for v_w(j - 1 /2, i + 1)^k at the first dx parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_water_x] -= (kappa_w * phi(j, i + 1) * v_w_list[k, j, i + 1] / (dy * dx *
                                              phi_at_interface(phi(j, i), phi(j, i + 1))))

                # When we are the rightmost column, the velocity on the wall is zero.

            # Coefficient for u_w(j, i + 1/2)^(k + 1) at the second dx parenthesis.
            if i < N_x - 1:
                A_reduced[eq_idx_water_x, uw_idx] += ((2 * miu_w + kappa_w) * phi(j, i) / (dx * dx *
                                                      phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for u_w(j, i - 1/2)^(k + 1) at the second dx parenthesis - only if that interface is active.
            if 0 < i < N_x - 1:
                uw_prev_pos = uw_pos_map[j, i - 1]                        # Position in the active list.
                if uw_prev_pos >= 0:                                      # Check if the (j, i - 1) interface is active.
                    uw_prev_idx = uw_start_id + uw_prev_pos               # Position in the reduced system.
                    A_reduced[eq_idx_water_x, uw_prev_idx] -= ((2 * miu_w + kappa_w) * phi(j, i) / (dx * dx *
                                                               phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for v_w(j + 1/2, i)^k at the second dx parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_water_x] -= (kappa_w * phi(j, i) * v_w_list[k, j + 1, i] / (dy * dx *
                                              phi_at_interface(phi(j, i), phi(j, i + 1))))

            # Coefficient for v_w(j - 1/2, i)^k at the second dx parenthesis.
            if i < N_x - 1:
                b_reduced[eq_idx_water_x] += (kappa_w * phi(j, i) * v_w_list[k, j, i] / (dy * dx *
                                              phi_at_interface(phi(j, i), phi(j, i + 1))))

                
        #=========================================== Solve Reduced System ==============================================
        # Numerical conditioning
        A_csr = A_reduced.tocsr()

        # Non-finite check (sparse-safe)
        if not (np.isfinite(A_csr.data).all() and np.isfinite(b_reduced).all()):
            raise np.linalg.LinAlgError("Non-finite entries in the reduced system.")

        # Two-sided equilibration (same idea as your dense version)
        eps = 1E-15

        # --- robust row/col scaling (works across SciPy versions) ---
        row_max = np.abs(A_csr).max(axis=1)
        if sp.issparse(row_max):
            row_max = row_max.toarray()
        row_norm = np.maximum(np.asarray(row_max).ravel(), eps)

        A_scaled = sp.diags(1.0 / row_norm) @ A_csr
        b_scaled = b_reduced / row_norm

        col_max = np.abs(A_scaled).max(axis=0)
        if sp.issparse(col_max):
            col_max = col_max.toarray()
        col_norm = np.maximum(np.asarray(col_max).ravel(), eps)

        A_final = A_scaled @ sp.diags(1.0 / col_norm)
        b_final = b_scaled

        try:
            x_scaled = spsolve(A_final, b_final)
            x_reduced = x_scaled / col_norm
        except Exception as e:
            raise RuntimeError(f"Sparse solve failed at k = {k}, q = {q}.") from e

        #===== Scatter Solution Back to Full Arrays =====
        # Extract pressures (first n_pressure_vars entries)
        P_w_list[k_next, :N_y - 1, :] = x_reduced[:vg_start_id].reshape(N_y - 1, N_x)
        P_w_list[k_next, -1, :] = P_top
        P_g_list[k_next, -1, :] = P_top

        # Initialize all velocities to zero (inactive interfaces will remain zero).
        v_g_list[k_next, 1:, :] = 0.0
        v_w_list[k_next, 1:, :] = 0.0
        u_g_list[k_next, :, :] = 0.0
        u_w_list[k_next, :, :] = 0.0

        # Scatter active gas vertical velocities.
        if n_active_vg > 0:
            v_g_values = x_reduced[vg_start_id: vw_start_id]
            j_idx = active_vg_indices[:, 0]
            i_idx = active_vg_indices[:, 1]
            v_g_list[k_next, j_idx + 1, i_idx] = v_g_values  # j + 1 because v_g[j + 1, i] is at interface (j + 1/2, i).

        # Scatter active water vertical velocities.
        if n_active_vw > 0:
            v_w_values = x_reduced[vw_start_id: ug_start_id]
            j_idx = active_vw_indices[:, 0]
            i_idx = active_vw_indices[:, 1]
            v_w_list[k_next, j_idx + 1, i_idx] = v_w_values  # j + 1 because v_w[j + 1, i] is at interface (j + 1/2, i).

        # Scatter active gas horizontal velocities.
        if n_active_ug > 0:
            u_g_values = x_reduced[ug_start_id: uw_start_id]
            j_idx = active_ug_indices[:, 0]
            i_idx = active_ug_indices[:, 1]
            u_g_list[k_next, j_idx, i_idx + 1] = u_g_values  # i + 1 because u_g[j, i + 1] is at interface (j, i + 1/2).

        # Scatter active water horizontal velocities.
        if n_active_uw > 0:
            u_w_values = x_reduced[uw_start_id: ]
            j_idx = active_uw_indices[:, 0]
            i_idx = active_uw_indices[:, 1]
            u_w_list[k_next, j_idx, i_idx + 1] = u_w_values  # i + 1 because u_w[j, i + 1] is at interface (j, i + 1/2).

        v_g_list[k_next, 0, :] = u_g_in                      # Gas vertical velocity at the inlet is constant.
        v_w_list[k_next, 0, :] = 0.0                         # Water vertical velocity at the inlet is constant.
        u_g_list[k_next, :, 0] = 0.0                         # Gas horizontal velocity at the left wall is zero.
        u_g_list[k_next, :, -1] = 0.0                        # Gas horizontal velocity at the right wall is zero.
        u_w_list[k_next, :, 0] = 0.0                         # Water horizontal velocity at the left wall is zero.
        u_w_list[k_next, :, -1] = 0.0                        # Water horizontal velocity at the right wall is zero.
        v_g_list[k_next, :, 0] = 0.0                         # Gas vertical velocity at the left wall is zero.
        v_g_list[k_next, :, -1] = 0.0                        # Gas vertical velocity at the right wall is zero.
        v_w_list[k_next, :, 0] = 0.0                         # Water vertical velocity at the left wall is zero.
        v_w_list[k_next, :, -1] = 0.0                        # Water vertical velocity at the right wall is zero.
        v_g_list[k_next, -1, :] = v_g_list[k_next, -2, :]    # Zero gradient for gas vertical velocity at the top row.
        v_w_list[k_next, -1, :] = v_w_list[k_next, -2, :]    # Zero gradient for water vertical velocity at the top row.

        # Update final saturations using new pressures. (Equations A17-A19)
        for j in range(N_y):
            for i in range(N_x):
                # Ensure masses are positive.
                m_list[k_next, j, i] = max(0, m_list[k_next, j, i])
                n_list[k_next, j, i] = max(0, n_list[k_next, j, i])

                rho_w_list[k_next, j, i] = P_w_list[k_next, j, i] / C_w + rho_wr
                s_w_star = m_list[k_next, j, i] / rho_w_list[k_next, j, i]
                if cement_check(y_list[j]):                                               # Only in the cemented region.
                    P_c_k_next_star = -P_star_c1 * np.log(delta_1 + (s_w_star / a_1))

                else:
                    P_c_k_next_star = 0.0

                P_g_k_next_star = P_w_list[k_next, j, i] + P_c_k_next_star
                rho_g_star = P_g_k_next_star / C_g
                s_g_star = n_list[k_next, j, i] / rho_g_star

                # Normalize final saturations. (Equation A19)
                total_sat_new = s_w_star + s_g_star
                if total_sat_new > 0:
                    S_w_list[k_next, j, i] = s_w_star / total_sat_new
                    S_g_list[k_next, j, i] = s_g_star / total_sat_new

                else:
                    print("Implicit, WARNING: Total saturation is zero at time step", k, "and cell", j)
                    break

                # Update capillary pressure for the cemented region at the timestep k + 1.
                if cement_check(y_list[j]):                                               # Only in the cemented region.
                    Pc_list[k_next, j, i] = -P_star_c1 * np.log(delta_1 + (S_w_list[k_next, j, i] / a_1))

                else:
                    Pc_list[k_next, j, i] = 0.0

                P_g_list[k_next, j, i] = P_w_list[k_next, j, i] + Pc_list[k_next, j, i]
                rho_g_list[k_next, j, i] = P_g_list[k_next, j, i] / C_g

        # Convergence test (skip after first pass)
        if it > 0:
            dPw = np.max(np.abs(P_w_list[k_next, :N_y - 1, :] - Pw_prev))
            dvg = np.max(np.abs(v_g_list[k_next, 1:, :] - vg_prev))
            dvw = np.max(np.abs(v_w_list[k_next, 1:, :] - vw_prev))
            dug = np.max(np.abs(u_g_list[k_next, :, 1: N_x] - ug_prev))
            duw = np.max(np.abs(u_w_list[k_next, :, 1: N_x] - uw_prev))

            if (dPw < tol_pw) and (dvg < tol_u) and (dvw < tol_u) and (dug < tol_u) and (duw < tol_u):
                break                                                                                       # Converged.

    # End of the Step-2 iteration loop.

    # Validate saturations at timestep k + 1 (final, after convergence).
    validate_saturations(S_w_list[k_next, :, :], S_g_list[k_next, :, :], k + q)

    # Calculate output quantities.
    vol_g = np.mean(S_g_list[k_next, :, :])
    volume_gas[k_next] = vol_g
    max_sg[k_next] = np.max(S_g_list[k_next, :, :])
    max_ug[k_next] = np.max(u_g_list[k_next, :, :])
    outlet_pressure_gas[k_next] = np.mean(P_g_list[k_next, -1, :])
    outlet_pressure_water[k_next] = np.mean(P_w_list[k_next, -1, :])
    inlet_pressure_gas[k_next] = np.mean(P_g_list[k_next, 0, :])
    inlet_pressure_water[k_next] = np.mean(P_w_list[k_next, 0, :])
    outlet_gas_velocity[k_next] = np.mean(v_g_list[k_next, -1, :])
    outlet_water_velocity[k_next] = np.mean(v_w_list[k_next, -1, :])
    inlet_gas_velocity[k_next] = np.mean(v_g_list[k_next, 0, :])
    inlet_water_velocity[k_next] = np.mean(v_w_list[k_next, 0, :])

#-----------------------------------------------------------------------------------------------------------------------
#================================================== Plot results =======================================================
#-----------------------------------------------------------------------------------------------------------------------
# Initial hydrostatic pressure distribution.
plt.figure('Initial hydrostatic pressure distribution')
plt.title('Initial hydrostatic pressure', fontweight = 'bold', fontsize = 24)
plt.xlabel('Depth (meter)')
plt.ylabel('Pressure (Pa)')
plt.xlim(0, L_k)
plt.plot(y_list[-1: 0: -1], P_g_list[0, :-1, 1], label = 'Initial hydrostatic pressure')
plt.grid(True, 'major')
plt.minorticks_on()
plt.grid(True, 'minor', axis = 'both')

# Pressure distribution over time
plt.figure('Pressure distribution over time')
plt.title('Outlet Pressure', fontsize = 24, fontweight = 'bold')
plt.xlabel('Time (second)')
plt.ylabel('Pressure (Pa)')
# Simulated gas pressure at wellhead (j = N_y - 1) for the computed time steps.
time_days = np.arange(0, M + 1, 2) * dt / 2                                                # Convert time steps to days.
plt.xlim(0, time_days[-1])
plt.plot(time_days, outlet_pressure_gas[::2], label = f'Gas, N_y = {N_y}, N_x = {N_x}, w = {w}, dt = {dt:.2e} s')
plt.plot(time_days, outlet_pressure_water[::2], label = f'Water, '
                                                        f'Iwa = {I_w_a:.2e}, Iga = {I_g_a:.2e}, Ia = {I_a:.2e}')
plt.legend()
plt.grid(True, 'major')
plt.minorticks_on()
plt.grid(True, 'minor', axis = 'both')

# Inlet Pressure
plt.figure('Inlet Pressure')
plt.title('Inlet Pressure', fontsize = 24, fontweight = 'bold')
plt.xlabel('Time (second)')
plt.ylabel('Pressure (Pa)')
plt.xlim(0, time_days[-1])
plt.plot(time_days, inlet_pressure_gas[::2], label = 'Gas')
plt.plot(time_days, inlet_pressure_water[::2], label = 'Water')
plt.legend()
plt.grid(True, 'major')
plt.minorticks_on()
plt.grid(True, 'minor', axis = 'both')

# Gas volume fraction
plt.figure('Gas volume fraction')
plt.title('Gas volume fraction', fontsize = 24, fontweight = 'bold')
plt.xlabel('Time (second)')
plt.ylabel('Gas volume fraction')
plt.plot(time_days, volume_gas[::2])
plt.grid(True, 'major')
plt.minorticks_on()
plt.grid(True, 'minor', axis = 'both')

# Velocity at the inlet
plt.figure('Inlet Velocity')
plt.title('Inlet Velocity', fontsize = 24, fontweight = 'bold')
plt.xlabel('Time (second)')
plt.ylabel('Velocity (m/s)')
plt.plot(time_days, inlet_gas_velocity[::2], label = 'Gas')
plt.plot(time_days, inlet_water_velocity[::2], label = 'Water')
plt.legend()
plt.grid(True, 'major')
plt.minorticks_on()
plt.grid(True, 'minor', axis = 'both')

# Velocity at the outlet
plt.figure('Outlet Velocity')
plt.title('Outlet Velocity', fontsize = 24, fontweight = 'bold')
plt.xlabel('Time (second)')
plt.ylabel('Velocity (m/s)')
plt.plot(time_days, outlet_gas_velocity[::2], label = 'Gas')
plt.plot(time_days, outlet_water_velocity[::2], label = 'Water')
plt.legend()
plt.grid(True, 'major')
plt.minorticks_on()
plt.grid(True, 'minor', axis = 'both')

plt.show()

# =================== 2D Visualization: GAS (BUBBLES) + GAS (COLORMAP) | sampled GIFs ===================
# --------------------------------------------------------------------------------------------------------
import os, math
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm


# -------------------- Plot scaling (IMPORTANT) --------------------
# If you use x_scale = L/W, the displayed width becomes constant (=L) and changing w does not show up.
# We instead scale x by a FIXED reference width so displayed width changes with W:
#   x_plot = x * (L / W_REF)
#   W_plot = W * (L / W_REF)  -> varies with W as desired.
W_REF_FOR_PLOT = 0.02  # meters (your "standard" width). Keep constant.


def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _rng_for(frame_id: int, j: int, i: int, seed: int = 123456789):
    # Fast deterministic RNG per (frame, cell)
    s = (seed
         + 1315423911 * int(frame_id)
         + 2654435761 * (int(j) + 1)
         + 97531 * (int(i) + 1)) & 0xFFFFFFFF
    return np.random.default_rng(s)


def _pack_bubbles_limited(x0, y0, cw, ch, gas_frac, rng,
                          max_bubbles_per_cell=2,
                          r_max_frac=0.35,
                          max_place_tries=30):
    """
    Pack up to max_bubbles_per_cell circles into a cell so total bubble area ~ gas_frac * cell_area,
    with bubble radius <= r_max_frac * min(cw, ch).
    """
    gas_frac = float(np.clip(gas_frac, 0.0, 1.0))
    if gas_frac <= 1e-12:
        return []

    cell_area = cw * ch
    target = gas_frac * cell_area
    base = min(cw, ch)

    r_max = min(0.45 * base, float(np.clip(r_max_frac, 0.01, 0.45)) * base)
    if r_max <= 1e-12:
        return []

    n_need = int(math.ceil(target / (math.pi * r_max * r_max)))
    n = int(np.clip(n_need, 1, max_bubbles_per_cell))

    bubbles = []
    remaining = target

    def overlaps(cx, cy, r):
        for (px, py, pr) in bubbles:
            if (cx - px) ** 2 + (cy - py) ** 2 < (r + pr) ** 2:
                return True
        return False

    for k in range(n):
        left = n - k
        r_goal = math.sqrt(max(remaining, 0.0) / (math.pi * left))
        r = min(r_max, r_goal * float(rng.uniform(0.85, 1.15)))

        # keep visible
        r = max(r, 0.04 * base)

        if k == n - 1:
            r = min(r_max, math.sqrt(max(remaining, 0.0) / math.pi))

        placed = False
        for _ in range(max_place_tries):
            cx = float(rng.triangular(x0 + r, x0 + 0.5 * cw, x0 + cw - r))
            cy = float(rng.triangular(y0 + r, y0 + 0.5 * ch, y0 + ch - r))
            if not overlaps(cx, cy, r):
                bubbles.append((cx, cy, r))
                remaining -= math.pi * r * r
                placed = True
                break

        if not placed:
            cx = float(rng.triangular(x0 + r, x0 + 0.5 * cw, x0 + cw - r))
            cy = float(rng.triangular(y0 + r, y0 + 0.5 * ch, y0 + ch - r))
            bubbles.append((cx, cy, r))
            remaining -= math.pi * r * r

        if remaining <= 0:
            break

    return bubbles


# --------------------------------- BUBBLES RENDER ---------------------------------
def _render_gas_grid_frame(Sg2D, *, L, W, nx, ny, k_id,
                           time_seconds=None, dt_full_seconds=None,
                           max_bubbles_per_cell=2,
                           r_max_frac=0.35,
                           out_png=None, dpi=160):
    Sg = np.asarray(Sg2D, dtype=float)
    Sg = np.clip(Sg, 0.0, 1.0)

    x_scale = L / float(W_REF_FOR_PLOT)
    W_plot = W * x_scale

    cw = (W / nx) * x_scale
    ch = (L / ny)

    fig, ax = plt.subplots(figsize=(6, 10), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.add_patch(Rectangle((0, 0), W_plot, L, facecolor='royalblue', edgecolor='none', alpha=0.25))
    ax.add_patch(Rectangle((0, 0), W_plot, L, facecolor='none', edgecolor='black', linewidth=1.0))

    for j in range(ny):
        y0 = j * ch
        for i in range(nx):
            x0 = i * cw
            s = float(Sg[j, i])
            if s <= 1e-12:
                continue
            rng = _rng_for(int(k_id), j, i)
            bubbles = _pack_bubbles_limited(
                x0, y0, cw, ch, s, rng,
                max_bubbles_per_cell=max_bubbles_per_cell,
                r_max_frac=r_max_frac
            )
            for (cx, cy, r) in bubbles:
                ax.add_patch(Circle((cx, cy), r, facecolor='red', edgecolor='none', alpha=0.90, zorder=5))

    x_edges = np.linspace(0.0, W_plot, nx + 1)
    y_edges = np.linspace(0.0, L, ny + 1)
    ax.vlines(x_edges, 0.0, L, colors='black', linewidth=0.25, alpha=0.85, zorder=10)
    ax.hlines(y_edges, 0.0, W_plot, colors='black', linewidth=0.25, alpha=0.85, zorder=10)

    ax.set_xlim(0.0, W_plot)
    ax.set_ylim(0.0, L)
    ax.set_aspect('equal')
    ax.axis('off')

    full_pos = int(k_id // 2)
    if time_seconds is not None and full_pos < len(time_seconds):
        t_sec = float(time_seconds[full_pos])
    elif dt_full_seconds is not None:
        t_sec = full_pos * float(dt_full_seconds)
    else:
        t_sec = float(full_pos)

    ax.text(
        W_plot / 2, L * 0.985,
        f"Gas Saturation (red bubbles)\n t = {t_sec:.3f} s",
        ha='center', va='top', fontsize=10, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.80, edgecolor='none', boxstyle='round'),
        zorder=20
    )

    if out_png is not None:
        fig.savefig(out_png, facecolor='white', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def export_gas_grid_gif_sampled(S_g_list, *, L, W, nx, ny,
                                out_dir="Simulation_Results_2D",
                                num_frames=100,
                                fps=10,
                                time_seconds=None,
                                dt_full_seconds=None,
                                max_bubbles_per_cell=2,
                                r_max_frac=0.35,
                                clear_existing=True):
    _ensure_dir(out_dir)
    frames_dir = os.path.join(out_dir, "gas_frames_grid")
    _ensure_dir(frames_dir)

    if clear_existing:
        for fn in os.listdir(frames_dir):
            if fn.endswith(".png"):
                try:
                    os.remove(os.path.join(frames_dir, fn))
                except Exception:
                    pass

    Sg = np.asarray(S_g_list)
    T = Sg.shape[0]

    full_ids_all = np.arange(0, T, 2, dtype=int)
    if full_ids_all.size == 0:
        raise RuntimeError("No even indices found in S_g_list.")

    num_frames = int(min(num_frames, full_ids_all.size))
    pick = np.linspace(0, full_ids_all.size - 1, num_frames).round().astype(int)
    full_ids = full_ids_all[np.unique(pick)]

    gif_path = os.path.join(out_dir, "gas_flow_grid.gif")
    with iio.get_writer(gif_path, mode="I", fps=fps) as writer:
        for n, k_id in enumerate(full_ids, start=1):
            png_path = os.path.join(frames_dir, f"gas_grid_{k_id:06d}.png")

            _render_gas_grid_frame(
                Sg[k_id],
                L=L, W=W, nx=nx, ny=ny, k_id=int(k_id),
                time_seconds=time_seconds,
                dt_full_seconds=dt_full_seconds,
                max_bubbles_per_cell=max_bubbles_per_cell,
                r_max_frac=r_max_frac,
                out_png=png_path,
                dpi=160
            )

            writer.append_data(iio.imread(png_path))

            if n % 10 == 0 or n == 1 or n == len(full_ids):
                print(f"[BUBBLES {n}/{len(full_ids)}] k_id={k_id} -> {png_path}")

    print("Saved:", gif_path)
    print(f"Exported {len(full_ids)} frames (sampled from {full_ids_all.size} full steps).")


# --------------------------------- COLORMAP RENDER ---------------------------------
# Pure blue -> red
CMAP_SG = LinearSegmentedColormap.from_list("blue_to_red", ["blue", "red"], N=256)


def _render_sg_colormap_frame(Sg2D, *, L, W, nx, ny, k_id,
                              time_seconds=None, dt_full_seconds=None,
                              cmap=CMAP_SG,
                              vmax_sg=1.0,
                              gamma=0.65,               # <1 => more detail at low Sg (keep vmax!)
                              out_png=None, dpi=160,
                              draw_grid=True,
                              show_colorbar=True):
    """
    Blue->Red colormap (no bubbles). Uses vmax_sg (NOT fixed at 1).
    gamma < 1 stretches low values for more detail.
    """
    Sg = np.asarray(Sg2D, dtype=float)
    Sg = np.clip(Sg, 0.0, None)

    vmax = float(vmax_sg)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(Sg))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1e-12

    # clip to vmax (so a single spike doesn't ruin colors inside frame)
    Sg = np.clip(Sg, 0.0, vmax)

    x_scale = L / float(W_REF_FOR_PLOT)
    W_plot = W * x_scale

    x_edges = np.linspace(0.0, W_plot, nx + 1)
    y_edges = np.linspace(0.0, L, ny + 1)

    fig, ax = plt.subplots(figsize=(6, 10), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # More detail at low Sg without changing vmax
    if gamma is None or float(gamma) == 1.0:
        norm = Normalize(vmin=0.0, vmax=vmax)
    else:
        norm = PowerNorm(gamma=float(gamma), vmin=0.0, vmax=vmax)

    m = ax.pcolormesh(
        x_edges, y_edges, Sg,
        cmap=cmap, norm=norm,
        shading="auto"
    )

    if draw_grid:
        ax.vlines(x_edges, 0.0, L, colors='black', linewidth=0.25, alpha=0.85, zorder=10)
        ax.hlines(y_edges, 0.0, W_plot, colors='black', linewidth=0.25, alpha=0.85, zorder=10)

    ax.set_xlim(0.0, W_plot)
    ax.set_ylim(0.0, L)
    ax.set_aspect('equal')
    ax.axis('off')

    full_pos = int(k_id // 2)
    if time_seconds is not None and full_pos < len(time_seconds):
        t_sec = float(time_seconds[full_pos])
    elif dt_full_seconds is not None:
        t_sec = full_pos * float(dt_full_seconds)
    else:
        t_sec = float(full_pos)

    ax.text(
        W_plot / 2, L * 0.985,
        f"Gas Saturation (colormap)\n t = {t_sec:.3f} s   (vmax={vmax:.3g}, gamma={gamma})",
        ha='center', va='top', fontsize=10, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.80, edgecolor='none', boxstyle='round'),
        zorder=20
    )

    if show_colorbar:
        cb = fig.colorbar(m, ax=ax, fraction=0.035, pad=0.01)
        cb.set_label("Sg", rotation=90)

    if out_png is not None:
        fig.savefig(out_png, facecolor='white', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def export_gas_colormap_gif_sampled(S_g_list, *, L, W, nx, ny,
                                    out_dir="Simulation_Results_2D",
                                    num_frames=100,
                                    fps=10,
                                    time_seconds=None,
                                    dt_full_seconds=None,
                                    cmap=CMAP_SG,
                                    max_sg=None,
                                    vmax_sg=None,
                                    gamma=0.65,
                                    clear_existing=True,
                                    draw_grid=True,
                                    show_colorbar=True):
    """
    COLORMAP ONLY (no bubbles).
    vmax priority:
      1) vmax_sg argument
      2) np.max(max_sg)  (your domain-max-per-k array)
      3) np.nanmax(S_g_list)
    """
    _ensure_dir(out_dir)
    frames_dir = os.path.join(out_dir, "gas_frames_colormap")
    _ensure_dir(frames_dir)

    if clear_existing:
        for fn in os.listdir(frames_dir):
            if fn.endswith(".png"):
                try:
                    os.remove(os.path.join(frames_dir, fn))
                except Exception:
                    pass

    Sg = np.asarray(S_g_list)
    T = Sg.shape[0]

    # Decide vmax ONCE (global scale across all frames)
    if vmax_sg is None:
        if max_sg is not None:
            vmax_sg = float(np.max(max_sg))
        else:
            vmax_sg = float(np.nanmax(Sg))
    if not np.isfinite(vmax_sg) or vmax_sg <= 0.0:
        vmax_sg = 1e-12

    full_ids_all = np.arange(0, T, 2, dtype=int)
    if full_ids_all.size == 0:
        raise RuntimeError("No even indices found in S_g_list.")

    num_frames = int(min(num_frames, full_ids_all.size))
    pick = np.linspace(0, full_ids_all.size - 1, num_frames).round().astype(int)
    full_ids = full_ids_all[np.unique(pick)]

    gif_path = os.path.join(out_dir, "gas_flow_colormap.gif")
    with iio.get_writer(gif_path, mode="I", fps=fps) as writer:
        for n, k_id in enumerate(full_ids, start=1):
            png_path = os.path.join(frames_dir, f"gas_cmap_{k_id:06d}.png")

            _render_sg_colormap_frame(
                Sg[k_id],
                L=L, W=W, nx=nx, ny=ny, k_id=int(k_id),
                time_seconds=time_seconds,
                dt_full_seconds=dt_full_seconds,
                cmap=cmap,
                vmax_sg=vmax_sg,          # <-- THIS is the key fix
                gamma=gamma,              # <-- more detail at low Sg
                out_png=png_path,
                dpi=160,
                draw_grid=draw_grid,
                show_colorbar=show_colorbar
            )

            writer.append_data(iio.imread(png_path))

            if n % 10 == 0 or n == 1 or n == len(full_ids):
                print(f"[COLORMAP {n}/{len(full_ids)}] k_id={k_id} -> {png_path}")

    print("Saved:", gif_path)
    print(f"Exported {len(full_ids)} frames (sampled from {full_ids_all.size} full steps).")
    print(f"Colormap scale used: vmin=0, vmax={vmax_sg:.6g}, gamma={gamma}")


# ------------------------------- Execution calls --------------------------------
# NOTE: in your file, time_days is actually in seconds (despite the name/comment).
export_gas_grid_gif_sampled(
    S_g_list,
    L=L_k, W=w, nx=N_x, ny=N_y,
    out_dir="Simulation_Results_2D",
    num_frames=100,
    fps=10,
    time_seconds=time_days,
    dt_full_seconds=None,
    max_bubbles_per_cell=2,
    r_max_frac=0.35,
    clear_existing=True
)

export_gas_colormap_gif_sampled(
    S_g_list,
    L=L_k, W=w, nx=N_x, ny=N_y,
    out_dir="Simulation_Results_2D",
    num_frames=100,
    fps=10,
    time_seconds=time_days,
    cmap=CMAP_SG,
    max_sg=max_sg,        # <-- ensures vmax = np.max(max_sg)
    vmax_sg=None,
    gamma=0.65,           # smaller => more detail at low Sg (try 0.5 if you want stronger)
    clear_existing=True,
    draw_grid=True,
    show_colorbar=True
)

# -------------------- Runtime summary (END) --------------------
RUN_END = time.perf_counter()
elapsed = RUN_END - RUN_START

h = int(elapsed // 3600)
m = int((elapsed % 3600) // 60)
s = elapsed % 60

print("\n" + "-" * 60)
print(f"Total runtime: {h:02d}:{m:02d}:{s:06.3f}  (hh:mm:ss.sss)")
print("-" * 60)
print(f'max ug = {np.max(max_ug)}')
# --------------------------------------------------------------------------------------------------------