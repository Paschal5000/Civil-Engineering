import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from fpdf import FPDF

# ----------- Input Functions ----------- #
def collect_user_data():
    print("FRAME INPUT DATA COLLECTION")
    num_nodes = int(input("Enter number of nodes: "))
    nodes = {}
    for i in range(1, num_nodes + 1):
        x = float(input(f"Enter x-coordinate of node {i}: "))
        y = float(input(f"Enter y-coordinate of node {i}: "))
        nodes[i] = (x, y)

    num_elements = int(input("Enter number of elements: "))
    elements = {}
    for i in range(1, num_elements + 1):
        n1 = int(input(f"Enter start node of element {i}: "))
        n2 = int(input(f"Enter end node of element {i}: "))
        E = float(input(f"Enter Young's modulus E for element {i} (kN/m²): "))
        A = float(input(f"Enter cross-sectional area A for element {i} (m²): "))
        I = float(input(f"Enter moment of inertia I for element {i} (m⁴): "))
        elements[i] = {'nodes': (n1, n2), 'E': E, 'A': A, 'I': I}

 # Collect load data for each element (UDL, midspan load, or no load)
        print(f"\nElement {i}: 1 - UDL, 2 - Midspan Load, 0 - None")
        load_type = int(input("  Load type: "))
        if load_type == 1:
            w = float(input("  Enter UDL (kN/m): "))
            elements[i]['load'] = {'type': 'UDL', 'value': w}
        elif load_type == 2:
            P = float(input("  Enter point load (kN) at midspan: "))
            elements[i]['load'] = {'type': 'Midspan', 'value': P}
        else:
            elements[i]['load'] = {'type': 'None', 'value': 0}
    return nodes, elements

def get_constrained_dofs():
    print("\nEnter constrained DOFs (space-separated, zero-indexed):")
    print("For each node, DOFs are: u=3n, v=3n+1, theta=3n+2 (where n starts from 0)")
    return list(map(int, input("Constrained DOFs: ").strip().split()))

def get_nodal_loads(total_dof):
    print("\nEnter nodal loads (only vertical for now).")
    F = np.zeros(total_dof)
    node = int(input("Enter node number where vertical load is applied: "))
    val = float(input("Enter vertical load value in kN (downward is negative): "))
    dof_index = 3 * (node - 1) + 1  # vertical DOF
    F[dof_index] = val
    return F

# ----------- Structural Analysis ----------- #
def get_dof_map(nodes, elements):
    dof_map = {}
    for eid, element in elements.items():
        n1, n2 = element['nodes']
        dofs_n1 = [3 * (n1 - 1) + i for i in range(3)]
        dofs_n2 = [3 * (n2 - 1) + i for i in range(3)]
        dof_map[eid] = dofs_n1 + dofs_n2
    return dof_map

def calculate_k_local_global(nodes, elements):
    k_local_global = {}
    for eid, element in elements.items():
        n1, n2 = element['nodes']
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        cos = (x2 - x1) / L
        sin = (y2 - y1) / L
        E, A, I = element['E'], element['A'], element['I']

        k = np.zeros((6, 6))
        k[0, 0] = k[3, 3] = A * E / L
        k[0, 3] = k[3, 0] = -A * E / L
        k[1, 1] = k[4, 4] = 12 * E * I / L**3
        k[1, 4] = k[4, 1] = -12 * E * I / L**3
        k[1, 2] = k[2, 1] = 6 * E * I / L**2
        k[1, 5] = k[5, 1] = 6 * E * I / L**2
        k[2, 2] = k[5, 5] = 4 * E * I / L
        k[2, 5] = k[5, 2] = 2 * E * I / L
        k[2, 4] = k[4, 2] = -6 * E * I / L**2
        k[4, 5] = k[5, 4] = -6 * E * I / L**2

        T = np.array([
            [ cos,  sin, 0,    0,    0, 0],
            [-sin,  cos, 0,    0,    0, 0],
            [   0,    0, 1,    0,    0, 0],
            [   0,    0, 0,  cos,  sin, 0],
            [   0,    0, 0, -sin,  cos, 0],
            [   0,    0, 0,    0,    0, 1]
        ])
        k_global = T.T @ k @ T
        k_local_global[eid] = {'k_local': k, 'k_global': k_global, 'T': T}
    return k_local_global

def assemble_global_stiffness_matrix(nodes, elements, dof_map, k_local_global):
    total_dof = 3 * len(nodes)
    K_global = np.zeros((total_dof, total_dof))
    for eid, dofs in dof_map.items():
        k_elem = k_local_global[eid]['k_global']
        for i in range(6):
            for j in range(6):
                K_global[dofs[i], dofs[j]] += k_elem[i, j]
    return K_global

def apply_boundary_conditions(K, F, constrained_dofs):
    for dof in constrained_dofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1
        F[dof] = 0
    return K, F

def solve_member_forces(displacements, dof_map, k_local_global):
    member_forces = {}
    for eid, dofs in dof_map.items():
        d_global = displacements[dofs]
        T = k_local_global[eid]['T']
        k_local = k_local_global[eid]['k_local']
        d_local = T @ d_global
        f_local = k_local @ d_local
        member_forces[eid] = f_local
    return member_forces

# ----------- Output Display ----------- #
def display_results(displacements, member_forces):
    print("\nNode Displacements:")
    disp_table = []
    for i in range(len(displacements)//3):
        u, v, t = displacements[3*i:3*i+3]
        disp_table.append([i+1, u, v, t])
    print(tabulate(disp_table, headers=["Node", "u (m)", "v (m)", "theta (rad)"], tablefmt="grid"))

    print("\nMember Forces:")
    force_table = []
    for eid, f in member_forces.items():
        force_table.append([eid, f[0], f[1], f[2], f[3], f[4], f[5]])
    print(tabulate(force_table, headers=["Element", "F1", "F2", "M1", "F3", "F4", "M2"], tablefmt="grid"))

def generate_pdf_report(displacements, member_forces):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, "Portal Frame Structural Analysis Report", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", size=9)
    pdf.cell(200, 10, "Node Displacements", ln=True)
    headers = ["Node", "u (m)", "v (m)", "theta (rad)"]
    for h in headers:
        pdf.cell(45, 8, h, border=1)
    pdf.ln()
    for i in range(len(displacements)//3):
        row = [i+1, *displacements[3*i:3*i+3]]
        for val in row:
            pdf.cell(45, 8, f"{val:.6f}", border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.cell(200, 10, "Member Forces", ln=True)
    headers2 = ["Element", "F1", "F2", "M1", "F3", "F4", "M2"]
    for h in headers2:
        pdf.cell(27, 8, h, border=1)
    pdf.ln()
    for eid, f in member_forces.items():
        pdf.cell(27, 8, str(eid), border=1)
        for val in f:
            pdf.cell(27, 8, f"{val:.3f}", border=1)
        pdf.ln()

    pdf.output("portal_frame_report.pdf")
    print("PDF report saved as 'portal_frame_report.pdf'.")

# ----------- Plotting ----------- #
def plot_structure(nodes, elements, displacements, scale=100):
    fig, ax = plt.subplots()
    for eid, element in elements.items():
        n1, n2 = element['nodes']
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        ax.plot([x1, x2], [y1, y2], 'k--', label='Original' if eid == 1 else "")
        dx1, dy1 = displacements[3*(n1-1)], displacements[3*(n1-1)+1]
        dx2, dy2 = displacements[3*(n2-1)], displacements[3*(n2-1)+1]
        ax.plot([x1 + dx1*scale, x2 + dx2*scale], [y1 + dy1*scale, y2 + dy2*scale], 'r-', label='Deformed' if eid == 1 else "")

    ax.set_aspect('equal')
    ax.set_title("Portal Frame: Original vs Deformed")
    ax.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

# ----------- Main Driver ----------- #
def run_frame_analysis():
    nodes, elements = collect_user_data()
    dof_map = get_dof_map(nodes, elements)
    k_local_global = calculate_k_local_global(nodes, elements)
    K_global = assemble_global_stiffness_matrix(nodes, elements, dof_map, k_local_global)
    total_dof = 3 * len(nodes)
    F_global = get_nodal_loads(total_dof)
    constrained_dofs = get_constrained_dofs()
    K_mod, F_mod = apply_boundary_conditions(K_global.copy(), F_global.copy(), constrained_dofs)
    displacements = np.linalg.solve(K_mod, F_mod)
    member_forces = solve_member_forces(displacements, dof_map, k_local_global)
    display_results(displacements, member_forces)
    generate_pdf_report(displacements, member_forces)
    plot_structure(nodes, elements, displacements)

if __name__ == "__main__":
    run_frame_analysis()
