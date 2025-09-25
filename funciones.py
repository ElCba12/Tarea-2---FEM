import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# =============================================================
# 1) Elemento de viga Euler–Bernoulli (puro flexión) 2 nodos
#    DOF por nodo: [v, theta]
# =============================================================

def beam_eb_local_stiffness(E: float, I: float, L: float) -> np.ndarray:
    if L <= 0:
        raise ValueError("L debe ser > 0")
    EI = E * I
    L2, L3 = L*L, L*L*L
    k = np.array([
        [ 12*EI/L3,   6*EI/L2,  -12*EI/L3,   6*EI/L2],
        [  6*EI/L2,   4*EI/L,    -6*EI/L2,   2*EI/L ],
        [-12*EI/L3,  -6*EI/L2,   12*EI/L3,  -6*EI/L2],
        [  6*EI/L2,   2*EI/L,    -6*EI/L2,   4*EI/L ],
    ], dtype=float)
    return k

# -------------------------------------------------------------
# Cargas equivalentes consistentes
# q: N/m POSITIVO HACIA ABAJO
# P: N POSITIVO HACIA ABAJO
# -------------------------------------------------------------

def equiv_uniform_load(q: float, L: float) -> np.ndarray:
    Fvi = -q * L/2
    Fvj = -q * L/2
    Mi  = -q * L**2 / 12
    Mj  =  q * L**2 / 12
    return np.array([Fvi, Mi, Fvj, Mj], dtype=float)


def hermite_shapes(xi: float, L: float) -> Tuple[float, float, float, float]:
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = L * (xi - 2*xi**2 + xi**3)
    N3 = 3*xi**2 - 2*xi**3
    N4 = L * (-xi**2 + xi**3)
    return N1, N2, N3, N4

# Derivadas para cinemática: v, theta=dw/dx, curvatura=d2w/dx2

def hermite_shapes_all(xi: float, L: float):
    # N
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = L * (xi - 2*xi**2 + xi**3)
    N3 = 3*xi**2 - 2*xi**3
    N4 = L * (-xi**2 + xi**3)
    # dN/dx
    dN1dx = (-6*xi + 6*xi**2) / L
    dN2dx = (1 - 4*xi + 3*xi**2)           # L cancela
    dN3dx = ( 6*xi - 6*xi**2) / L
    dN4dx = (-2*xi + 3*xi**2)              # L cancela
    # d2N/dx2
    d2N1dx2 = (-6 + 12*xi) / L**2
    d2N2dx2 = (-4 + 6*xi) / L
    d2N3dx2 = ( 6 - 12*xi) / L**2
    d2N4dx2 = (-2 + 6*xi) / L
    return (N1, N2, N3, N4, dN1dx, dN2dx, dN3dx, dN4dx, d2N1dx2, d2N2dx2, d2N3dx2, d2N4dx2)


def equiv_point_load(P: float, a: float, L: float) -> np.ndarray:
    if not (0.0 < a < L):
        raise ValueError("a debe estar en (0, L)")
    xi = a / L
    N1, N2, N3, N4 = hermite_shapes(xi, L)
    return -P * np.array([N1, N2, N3, N4], dtype=float)

# =============================================================
# 2) Malla 1D, ensamble, BC y solución
# =============================================================

def make_uniform_beam_mesh(L: float, ne: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ne < 1:
        raise ValueError("ne ≥ 1")
    x = np.linspace(0.0, L, ne+1)
    elements = np.column_stack([np.arange(0, ne), np.arange(1, ne+1)])
    Le = np.diff(x)
    return x, elements, Le


def assemble_KF_beam(E: float, I: float, x: np.ndarray, elements: np.ndarray,
                     q_uniform: Optional[float] = None,
                     pointloads: Optional[List[Tuple[float, float]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    N = x.size
    ndof = 2*N
    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros((ndof,), dtype=float)

    for e_idx, (i, j) in enumerate(elements):
        L_e = x[j] - x[i]
        ke = beam_eb_local_stiffness(E, I, L_e)
        edofs = np.array([2*i, 2*i+1, 2*j, 2*j+1], dtype=int)
        K[np.ix_(edofs, edofs)] += ke
        if q_uniform is not None and q_uniform != 0.0:
            fe = equiv_uniform_load(q_uniform, L_e)
            F[edofs] += fe

    if pointloads:
        for (xp, P) in pointloads:
            hit = np.where(np.isclose(x, xp))[0]
            if hit.size == 1:  # coincide con nodo
                n = int(hit[0])
                F[2*n] += -P
                continue
            e = np.searchsorted(x, xp) - 1
            if e < 0 or e >= elements.shape[0]:
                raise ValueError("x de punto fuera del dominio [0, L]")
            i, j = elements[e]
            L_e = x[j] - x[i]
            a = xp - x[i]
            fe = equiv_point_load(P, a, L_e)
            edofs = np.array([2*i, 2*i+1, 2*j, 2*j+1], dtype=int)
            F[edofs] += fe

    return K, F


def apply_fixed_ends_bc(K: np.ndarray, F: np.ndarray, N: int):
    ndof = 2*N
    fixed = np.array([0, 1, 2*(N-1), 2*(N-1)+1], dtype=int)
    all_idx = np.arange(ndof, dtype=int)
    free = np.setdiff1d(all_idx, fixed, assume_unique=True)
    if free.size == 0:
        U = np.zeros(ndof)
        R = -F.copy()
        return U, R, free, fixed
    Kff = K[np.ix_(free, free)]
    Ff = F[free]
    u_free = np.linalg.solve(Kff, Ff)
    U = np.zeros(ndof)
    U[free] = u_free
    R = K @ U - F
    return U, R, free, fixed

# =============================================================
# 3) Post-proceso: campos internos y gráficos
# =============================================================

def element_end_forces_local(E: float, I: float, L_e: float, u_e: np.ndarray,
                             q_uniform: float = 0.0) -> np.ndarray:
    k_loc = beam_eb_local_stiffness(E, I, L_e)
    f_loc = equiv_uniform_load(q_uniform, L_e) if q_uniform else np.zeros(4)
    return k_loc @ u_e - f_loc  # [V_i, M_i, V_j, M_j]

# Muestrea v(x), theta(x), M(x), V(x) dentro de UN elemento

def sample_element_fields(E: float, I: float, x_i: float, x_j: float,
                          u_e: np.ndarray, q_uniform: float = 0.0,
                          nper: int = 25):
    L = x_j - x_i
    xi = np.linspace(0.0, 1.0, nper)
    xs = x_i + xi * L

    # Kinemática (Hermite)
    v_list, th_list, curv_list = [], [], []
    for s in xi:
        N1,N2,N3,N4, dN1dx,dN2dx,dN3dx,dN4dx, d2N1dx2,d2N2dx2,d2N3dx2,d2N4dx2 = hermite_shapes_all(s, L)
        v  = N1*u_e[0] + N2*u_e[1] + N3*u_e[2] + N4*u_e[3]
        th = dN1dx*u_e[0] + dN2dx*u_e[1] + dN3dx*u_e[2] + dN4dx*u_e[3]
        k  = d2N1dx2*u_e[0] + d2N2dx2*u_e[1] + d2N3dx2*u_e[2] + d2N4dx2*u_e[3]
        v_list.append(v); th_list.append(th); curv_list.append(k)

    v_arr = np.array(v_list)
    th_arr = np.array(th_list)
    M_disp = E*I*np.array(curv_list)  # M por cinemática (lineal por elemento)

    # Estática a partir de fuerzas de extremo
    s_loc = element_end_forces_local(E, I, L, u_e, q_uniform=q_uniform)
    Vi, Mi, Vj, Mj = s_loc
    s = xi * L
    V_stat = Vi - q_uniform * s
    M_stat = Mi + Vi * s - 0.5 * q_uniform * s**2

    return xs, v_arr, th_arr, M_stat, V_stat

# ---------- Gráficos ----------

def plot_deflection(x: np.ndarray, elements: np.ndarray, U: np.ndarray,
                    scale: float = 1.0, nper: int = 20):
    X, Y = [], []
    for (i, j) in elements:
        xi = np.linspace(0, 1, nper)
        L_e = x[j] - x[i]
        v1, th1, v2, th2 = U[2*i:2*i+2].tolist() + U[2*j:2*j+2].tolist()
        for s in xi:
            N1, N2, N3, N4 = hermite_shapes(s, L_e)
            w = N1*v1 + N2*th1 + N3*v2 + N4*th2
            X.append(x[i] + s*L_e)
            Y.append(scale * w)
    plt.figure(figsize=(8,3))
    plt.plot(x, 0*x, lw=1)
    plt.plot(X, Y, lw=2)
    plt.xlabel('x [m]'); plt.ylabel('v [m] (esc.)')
    plt.title('Viga – desplazamiento')
    plt.grid(True, ls=':'); plt.tight_layout(); plt.show()


def plot_rotation_beam(x: np.ndarray, elements: np.ndarray, U: np.ndarray, nper: int = 40):
    X, TH = [], []
    for (i, j) in elements:
        u_e = np.array([U[2*i], U[2*i+1], U[2*j], U[2*j+1]])
        xs, _, th, _, _ = sample_element_fields(E=1.0, I=1.0, x_i=x[i], x_j=x[j], u_e=u_e, q_uniform=0.0, nper=nper)
        # Nota: rotación no depende de E,I ni q en su definición geométrica
        X.extend(xs.tolist()); TH.extend(th.tolist())
    plt.figure(figsize=(8,3))
    plt.plot(X, TH, lw=2)
    plt.xlabel('x [m]'); plt.ylabel('θ [rad]')
    plt.title('Viga – rotación')
    plt.grid(True, ls=':'); plt.tight_layout(); plt.show()


def plot_moment_beam(x: np.ndarray, elements: np.ndarray, U: np.ndarray,
                      E: float, I: float, q_uniform: float = 0.0, nper: int = 60):
    X, M = [], []
    for (i, j) in elements:
        u_e = np.array([U[2*i], U[2*i+1], U[2*j], U[2*j+1]])
        xs, _, _, M_stat, _ = sample_element_fields(E, I, x[i], x[j], u_e, q_uniform=q_uniform, nper=nper)
        X.extend(xs.tolist()); M.extend(M_stat.tolist())
    plt.figure(figsize=(8,3))
    plt.plot(X, M, lw=2)
    plt.xlabel('x [m]'); plt.ylabel('M [N·m] (signo CCW +)')
    plt.title('Viga – diagrama de momentos (estática)')
    plt.grid(True, ls=':'); plt.tight_layout(); plt.show()


def plot_shear_beam(x: np.ndarray, elements: np.ndarray, U: np.ndarray,
                     E: float, I: float, q_uniform: float = 0.0, nper: int = 60):
    X, V = [], []
    for (i, j) in elements:
        u_e = np.array([U[2*i], U[2*i+1], U[2*j], U[2*j+1]])
        xs, _, _, _, V_stat = sample_element_fields(E, I, x[i], x[j], u_e, q_uniform=q_uniform, nper=nper)
        X.extend(xs.tolist()); V.extend(V_stat.tolist())
    plt.figure(figsize=(8,3))
    plt.plot(X, V, lw=2)
    plt.xlabel('x [m]'); plt.ylabel('V [N] (positivo ↑)')
    plt.title('Viga - diagrama de corte (estática)')
    plt.grid(True, ls=':'); plt.tight_layout(); plt.show()
