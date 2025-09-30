import numpy as np
import matplotlib.pyplot as plt


# =============================
# Utilidades de geometría
# =============================

def rot_cs(x1,y1,x2,y2):
    L = ((x2-x1)**2 + (y2-y1)**2)**0.5
    c = (x2-x1)/L
    s = (y2-y1)/L

    return L, c, s

def T_beam(c,s):
    T = np.array([
        [ c, -s, 0,  0,  0, 0],
        [ s,  c, 0,  0,  0, 0],
        [ 0,  0, 1,  0,  0, 0],
        [ 0,  0, 0,  c, -s, 0],
        [ 0,  0, 0,  s,  c, 0],
        [ 0,  0, 0,  0,  0, 1],
    ], dtype=float)

    return T


# =============================
# Elemento viga Euler–Bernoulli 2D
# ==============================

def k_local_beam(E,A,I,L):
    EA_L  = E*A/L
    EI_L3 = E*I/L**3
    EI_L2 = E*I/L**2
    EI_L  = E*I/L
    k = np.array([
        [ EA_L,      0,       0, -EA_L,      0,       0],
        [    0, 12*EI_L3,  6*EI_L2,    0, -12*EI_L3,  6*EI_L2],
        [    0,  6*EI_L2,   4*EI_L,    0,  -6*EI_L2,   2*EI_L],
        [-EA_L,     0,       0,  EA_L,      0,       0],
        [    0, -12*EI_L3, -6*EI_L2,   0,  12*EI_L3, -6*EI_L2],
        [    0,  6*EI_L2,   2*EI_L,    0,  -6*EI_L2,   4*EI_L]
    ], dtype=float)

    return k

def fixed_end_loads_uniform(qx_local, qy_local, L):
    # Axial uniforme -> solo fuerzas axiales en nodos
    fx = -qx_local * L / 2.0
    f_ax = np.array([fx, 0, 0, fx, 0, 0], dtype=float)

    fy = -qy_local * L / 2.0
    mz = qy_local * L**2 / 12.0
    f_tr = np.array([0, fy,  mz, 0, fy, -mz], dtype=float)

    return f_ax + f_tr  # local


# =============================
# Armado del problema de marco
# =============================

def build_roof_frame(L, alpha_deg):
    a = np.deg2rad(alpha_deg)
    x_r = L/2.0
    y_r = L/2.0 + x_r*np.tan(a)

    nodes = np.array([
        [0.0,   0.0],        # 0
        [0.0,   L/2.0],      # 1
        [L/2.0, y_r],        # 2
        [L,     L/2.0],      # 3
        [L,     0.0],        # 4
    ], dtype=float)
    
    base_elems = [
        (0,1),  # columna izq
        (1,2),  # viga izq (esta es la cargada)
        (2,3),  # viga der
        (3,4),  # columna der
    ]

    return nodes, base_elems

def subdivide_element(nodes, elem, nsub):
    i, j = elem
    p0, p1 = nodes[i], nodes[j]
    pts = [p0 + (p1 - p0) * (k/nsub) for k in range(1, nsub)]
    # nuevos nodos al final de la lista
    start_idx = len(nodes)
    new_nodes = np.vstack([nodes] + pts) if pts else nodes.copy()
    # conectividades
    indices = [i] + list(range(start_idx, start_idx+len(pts))) + [j]
    new_elems = [(indices[k], indices[k+1]) for k in range(len(indices)-1)]

    return new_nodes, new_elems

def mesh_with_refinement_on_left_roof(L, alpha_deg, ne_left_roof=4):
    nodes, base = build_roof_frame(L, alpha_deg)
    elems = []

    # columna izq
    nodes, e_colL = subdivide_element(nodes, base[0], 1)   
    elems += e_colL
    # viga izq 
    nodes, e_roofL = subdivide_element(nodes, base[1], ne_left_roof)
    idx_left_roof_first = len(elems)
    elems += e_roofL
    # viga der
    nodes, e_roofR = subdivide_element(nodes, base[2], 1)
    elems += e_roofR
    # columna der
    nodes, e_colR = subdivide_element(nodes, base[3], 1)
    elems += e_colR

    # guarda rango de elementos que pertenecen a la viga inclinada izquierda
    left_roof_elem_ids = list(range(idx_left_roof_first, idx_left_roof_first+len(e_roofL)))

    return nodes, elems, left_roof_elem_ids


# =============================
# Ensamble global y solución
# ============================

def assemble_solve(nodes, elems, E, A, I,
                   q_global,                 
                   left_roof_elem_ids=(),
                   encastres=(0,4)):

    qx_g, qy_g = float(q_global[0]), float(q_global[1])
       
    nn = len(nodes); ndof = 3*nn
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    # para postproceso
    elem_data = []

    # Ensamble
    for e_id,(i,j) in enumerate(elems):
        x1,y1 = nodes[i]; x2,y2 = nodes[j]
        L, c, s = rot_cs(x1,y1,x2,y2)
        T = T_beam(c,s)
        kL = k_local_beam(E,A,I,L)
        kG = T.T @ kL @ T

        # DOFs globales del elemento
        gdofs = np.r_[3*i+np.arange(3), 3*j+np.arange(3)]
        # ensamblar rigidez
        K[np.ix_(gdofs,gdofs)] += kG

        # Cargas distribuidas si el elemento está en la viga cargada
        f_fixed_local = np.zeros(6)

        if e_id in left_roof_elem_ids and (abs(qx_g) > 0 or abs(qy_g) > 0):
            # ex'=(c,s), ey'=(-s,c)
            qx_loc =  qx_g*c + qy_g*s
            qy_loc = -qx_g*s + qy_g*c
            f_fixed_local = fixed_end_loads_uniform(qx_loc, qy_loc, L)
            F[np.ix_(gdofs)] += T.T @ f_fixed_local

        elem_data.append({
            'L':L,'c':c,'s':s,'T':T,'k_local':kL,
            'gdofs':gdofs,'f_fixed_local':f_fixed_local
        })

    # Condiciones de borde (encastres: u=v=θ=0)
    fixed = []
    for n in encastres:
        fixed += [3*n, 3*n+1, 3*n+2]
    fixed = np.array(sorted(set(fixed)), dtype=int)
    free  = np.array([d for d in range(ndof) if d not in fixed], dtype=int)

    # Resolver
    u = np.zeros(ndof)
    if free.size:
        Kff = K[np.ix_(free,free)]
        Ff  = F[free]
        u[free] = np.linalg.solve(Kff, Ff)

    # Reacciones
    R = K @ u - F

    return u, R, elem_data


# ================================
# Esfuerzos internos y diagramas N,V,M
# ================================

def element_end_forces_local(u, ed):
    gd = ed['gdofs']; T = ed['T']; kL = ed['k_local']; ff = ed['f_fixed_local']
    d_local = T @ u[gd]
    f_local = kL @ d_local - ff

    return f_local

def diagrams_per_element(u, elem_data, nsamples=2):
    results = []
    for ed in elem_data:
        L=ed['L']; c=ed['c']; s=ed['s']; T=ed['T']; ff=ed['f_fixed_local']
        qx = 2*ff[0]/L if L>0 else 0.0          
        qy = 2*ff[1]/L if L>0 else 0.0

        fL = element_end_forces_local(u, ed)
        N1, V1, M1, N2, V2, M2 = fL

        xs = np.linspace(0.0, L, max(2,nsamples))
        N = N1 - qx*xs
        V = V1 - qy*xs
        M = M1 + V1*xs - 0.5*qy*xs**2

        results.append({'L':L,'c':c,'s':s,'x':xs,'N':N,'V':V,'M':M,'f_end':fL})

    return results


# ==============================
# Postpro: gráfico del pórtico
# ==============================

def _beam_shape_funcs(L, x):
    if L <= 0.0:
        return np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])

    xi = x / float(L)
    # axial lineal
    N1a = 1.0 - xi
    N2a = xi
    # Hermite cúbicas (Euler-Bernoulli)
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = L*(xi - 2*xi**2 + xi**3)
    N3 = 3*xi**2 - 2*xi**3
    N4 = L*(-xi**2 + xi**3)

    return np.array([N1a, N2a]), np.array([N1, N2, N3, N4])

def _element_deformed_curve(x1, y1, x2, y2, ueg_global, T_beam_func, ns=40):
    L = np.hypot(x2 - x1, y2 - y1)
    if L == 0.0:
        return np.array([x1, x2]), np.array([y1, y2])

    c = (x2 - x1) / L
    s = (y2 - y1) / L

    # GLOBAL -> LOCAL
    T = T_beam_func(c, s)
    ueL = T_beam_func(c, s).T @ ueg_global  

    xs = np.linspace(0.0, L, ns)
    Xd, Yd = [], []

    for x in xs:
        Na, Nv = _beam_shape_funcs(L, x)
        u_ax = Na @ np.array([ueL[0], ueL[3]])                   
        v_tr = Nv @ np.array([ueL[1], ueL[2], ueL[4], ueL[5]])   

        # Volver a coordenadas globales del punto (geom. + deformaciÃ³n)
        X = x1 + c*(x + u_ax) - s*(v_tr)
        Y = y1 + s*(x + u_ax) + c*(v_tr)
        Xd.append(X); Yd.append(Y)

    return np.array(Xd), np.array(Yd)

def plot_portico(nodes, elements, U=None, scale=1.0, index_base=0,
                 dof_per_node=3, annotate=True, ax=None,
                 color_orig='0.65', color_def='C0', lw=2.0, ms=4,
                 T_beam_func=None):

    nodes = np.asarray(nodes, float)
    elements = np.asarray(elements, int)
    off = 1 if index_base == 1 else 0

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Geometría original
    for e in elements:
        i, j = int(e[0] - off), int(e[1] - off)
        x1, y1 = nodes[i]; x2, y2 = nodes[j]
        ax.plot([x1, x2], [y1, y2], color=color_orig, lw=1.25, solid_capstyle='round')
    if annotate:
        for k, (x, y) in enumerate(nodes, start=off):
            ax.plot([x], [y], 'o', color=color_orig, ms=ms)
            ax.text(x, y, f"{k}", ha='left', va='bottom', fontsize=9, color='k',
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.6))

    # Deformada
    if U is not None:
        assert T_beam_func is not None
        U = np.asarray(U, float).reshape(-1)

        for e in elements:
            i, j = int(e[0] - off), int(e[1] - off)
            x1, y1 = nodes[i]; x2, y2 = nodes[j]

            # DOFs globales de los nodos i y j
            idx_i = slice(i*dof_per_node, (i+1)*dof_per_node)
            idx_j = slice(j*dof_per_node, (j+1)*dof_per_node)

            ueg = np.r_[U[idx_i], U[idx_j]].copy()
            # escalar SOLO ux, uy (no Î¸)
            ueg[[0, 1, 3, 4]] *= float(scale)

            Xd, Yd = _element_deformed_curve(x1, y1, x2, y2, ueg, T_beam_func, ns=60)
            ax.plot(Xd, Yd, color=color_def, lw=lw)

        # nodos deformados (puntos)
        nn = nodes.shape[0]
        for i in range(nn):
            ux = U[i*dof_per_node + 0] * scale
            uy = U[i*dof_per_node + 1] * scale
            ax.plot([nodes[i,0] + ux], [nodes[i,1] + uy], 'o', color=color_def, ms=ms*0.95)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, ls='--', alpha=0.35)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Pórtico: Geometría original y deformada')

    return ax


# ==============================
# FUNCIONES PORTICO CON TENSOR
# ==============================

def k_local_tie_axial(E, A, L):

    if L <= 0.0:
        return np.zeros((6,6), dtype=float)
    k = np.zeros((6,6), dtype=float)
    k_ax = E*A/L
    k[0,0] =  k_ax;  k[0,3] = -k_ax
    k[3,0] = -k_ax;  k[3,3] =  k_ax

    return k

def add_horizontal_tie(nodes, elems, n1=1, n2=3):
    elems_new = list(elems)
    tie_elem_ids = [len(elems_new)]
    elems_new.append((n1, n2))

    return elems_new, tie_elem_ids

def assemble_solve_with_tie(nodes, elems, E, A, I,
                            q_global=None,
                            left_roof_elem_ids=(),
                            tie_elem_ids=(),
                            A_tie_factor=0.5,
                            encastres=(0,4)):
    
    qx_g, qy_g = float(q_global[0]), float(q_global[1])

    nn = len(nodes); ndof = 3*nn
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    elem_data = []
    A_tie = A * float(A_tie_factor)

    for e_id, (i, j) in enumerate(elems):
        x1, y1 = nodes[i]; x2, y2 = nodes[j]
        L, c, s = rot_cs(x1, y1, x2, y2)
        T = T_beam(c, s)

        # Selección del tipo de elemento
        if e_id in tie_elem_ids:
            # Elemento axial (tensor)
            kL = k_local_tie_axial(E, A_tie, L)
            f_fixed_local = np.zeros(6)  
        else:
            # Elemento viga (pórtico)
            kL = k_local_beam(E, A, I, L)
            if e_id in left_roof_elem_ids and (abs(qx_g) > 0 or abs(qy_g) > 0):
                qx_loc =  qx_g*c + qy_g*s
                qy_loc = -qx_g*s + qy_g*c
                f_fixed_local = fixed_end_loads_uniform(qx_loc, qy_loc, L)
            else:
                f_fixed_local = np.zeros(6)

        kG = T.T @ kL @ T
        gdofs = np.r_[3*i+np.arange(3), 3*j+np.arange(3)]

        K[np.ix_(gdofs, gdofs)] += kG
        if np.any(f_fixed_local):
            F[np.ix_(gdofs)] += T.T @ f_fixed_local

        elem_data.append({
            'L': L, 'c': c, 's': s, 'T': T, 'k_local': kL,
            'gdofs': gdofs, 'f_fixed_local': f_fixed_local,
            'is_tie': (e_id in tie_elem_ids)
        })

    # BCs
    fixed = []
    for n in encastres:
        fixed += [3*n, 3*n+1, 3*n+2]
    fixed = np.array(sorted(set(fixed)), dtype=int)
    free = np.array([d for d in range(ndof) if d not in fixed], dtype=int)

    u = np.zeros(ndof)
    if free.size:
        u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    R = K @ u - F
    return u, R, elem_data

def plot_portico_with_tie(nodes, elements, U=None, scale=1.0,
                          tie_elem_ids=(), index_base=0, dof_per_node=3,
                          annotate=True, ax=None,
                          color_orig='0.65', color_def='C0', lw=2.0, ms=4):

    nodes = np.asarray(nodes, float)
    elements = np.asarray(elements, int)
    off = 1 if index_base == 1 else 0
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # malla original
    for e_id, e in enumerate(elements):
        i, j = int(e[0] - off), int(e[1] - off)
        x1, y1 = nodes[i]; x2, y2 = nodes[j]
        ax.plot([x1, x2], [y1, y2], color=color_orig, lw=1.25)
    if annotate:
        for k, (x, y) in enumerate(nodes, start=off):
            ax.plot([x], [y], 'o', color=color_orig, ms=ms)
            ax.text(x, y, f"{k}", ha='left', va='bottom', fontsize=9,
                    color='k', bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.6))

    if U is not None:
        U = np.asarray(U, float).reshape(-1)
        for e_id, e in enumerate(elements):
            i, j = int(e[0] - off), int(e[1] - off)
            x1, y1 = nodes[i]; x2, y2 = nodes[j]

            # DOFs
            idx_i = slice(i*dof_per_node, (i+1)*dof_per_node)
            idx_j = slice(j*dof_per_node, (j+1)*dof_per_node)
            ueg = np.r_[U[idx_i], U[idx_j]].copy()
            ueg[[0, 1, 3, 4]] *= float(scale)  

            # puntos deformados de los nodos
            Xi = nodes[i,0] + ueg[0]; Yi = nodes[i,1] + ueg[1]
            Xj = nodes[j,0] + ueg[3]; Yj = nodes[j,1] + ueg[4]

            if e_id in tie_elem_ids:
                ax.plot([Xi, Xj], [Yi, Yj], color=color_def, lw=lw)
            else:
                Xd, Yd = _element_deformed_curve(x1, y1, x2, y2, ueg, T_beam, ns=60)
                ax.plot(Xd, Yd, color=color_def, lw=lw)

        # puntos deformados
        for i in range(nodes.shape[0]):
            ax.plot([nodes[i,0] + U[i*dof_per_node + 0]*scale],
                    [nodes[i,1] + U[i*dof_per_node + 1]*scale],
                    'o', color=color_def, ms=ms*0.95)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, ls='--', alpha=0.35)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Pórtico con tensor: geometría y deformada')

    return ax

# ============================
# Comparación de desplazamientos 
# ============================

def node_dofs(node_id, dof_per_node=3, index_base=0):
    i = node_id - index_base

    return i*dof_per_node, i*dof_per_node+1, i*dof_per_node+2

def get_node_disp(U, node_id, dof_per_node=3, index_base=0):
    iux, iuy, ith = node_dofs(node_id, dof_per_node, index_base)
    ux = float(U[iux]); uy = float(U[iuy]); th = float(U[ith])
    umag = (ux**2 + uy**2) ** 0.5

    return ux, uy, th, umag

def report_nodes_disp(U, nodes=(1,2,3), index_base=0, dof_per_node=3, units=('m','rad')):
    print(f"{'Nodo':>4} | {'ux ['+units[0]+']':>12} {'uy ['+units[0]+']':>12} {'|u| ['+units[0]+']':>12} {'theta ['+units[1]+']':>12}")
    print("-"*60)
    for n in nodes:
        ux, uy, th, um = get_node_disp(U, n, dof_per_node, index_base)
        print(f"{n:>4} | {ux:12.6e} {uy:12.6e} {um:12.6e} {th:12.6e}")

def barplot_nodes_disp(U_list, labels, nodes=(1,2,3), index_base=0, dof_per_node=3):
    nodes = list(nodes)
    k = len(nodes); m = len(U_list)
    UM = np.zeros((m, k))
    for i, U in enumerate(U_list):
        for j, n in enumerate(nodes):
            _, _, _, um = get_node_disp(U, n, dof_per_node, index_base)
            UM[i, j] = um

    x = np.arange(k)
    w = 0.8 / m
    fig, ax = plt.subplots(figsize=(7,4))
    for i in range(m):
        ax.bar(x + i*w - 0.4 + w/2, UM[i, :], width=w, label=labels[i])
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in nodes])
    ax.set_xlabel("Nodo"); ax.set_ylabel("|u| [m]") 
    ax.set_title("Comparación de desplazamiento nodal")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    
    return ax



