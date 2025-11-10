"""
Quad4 FEM (2D) — minimal, readable, single-file implementation
- 4-node bilinear quadrilateral, plane stress/strain
- Inputs: coordinates, connectivity, material, thickness, BCs, nodal loads
- Outputs: nodal displacements, element stresses; quick plot of undeformed/deformed mesh

Usage: run the file directly to execute the demo at bottom, or import and call run_fem(...)
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --------------------------- Core math helpers ---------------------------

def kron2I(n: int) -> np.ndarray:
    """Kronecker product helper: identity(n) kron identity(2) to map node->(ux,uy)."""
    return np.kron(np.eye(n), np.eye(2))

# --------------------------- Element formulation ---------------------------

@dataclass
class Material:
    E: float          # Young's modulus
    nu: float         # Poisson's ratio
    plane_stress: bool = True

    def D(self) -> np.ndarray:
        E, nu = self.E, self.nu
        if self.plane_stress:
            c = E / (1 - nu**2)
            return c * np.array([[1,  nu, 0],
                                  [nu, 1,  0],
                                  [0,  0, (1-nu)/2]])
        else:  # plane strain
            c = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
            a = (1 - nu)
            b = nu
            g = (1 - 2*nu)/2
            return c * np.array([[1,   b/a, 0],
                                  [b/a, 1,  0],
                                  [0,   0,  g/a]])

@dataclass
class Quad4:
    coords: np.ndarray       # (nnode, 2)
    conn:   np.ndarray       # (nelem, 4) node indices (0-based)
    material: Material
    t: float = 1.0           # thickness

    def _shape(self, xi: float, eta: float):
        """Shape functions N (4,) and derivatives dN/dxi, dN/deta."""
        N = np.array([
            0.25*(1-xi)*(1-eta),
            0.25*(1+xi)*(1-eta),
            0.25*(1+xi)*(1+eta),
            0.25*(1-xi)*(1+eta)
        ])
        dN_dxi = np.array([
            -0.25*(1-eta),
             0.25*(1-eta),
             0.25*(1+eta),
            -0.25*(1+eta)
        ])
        dN_deta = np.array([
            -0.25*(1-xi),
            -0.25*(1+xi),
             0.25*(1+xi),
             0.25*(1-xi)
        ])
        return N, dN_dxi, dN_deta

    def _Bmatrix(self, xe: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float, np.ndarray]:
        """Return B (3x8), detJ, and N (for later post-proc)."""
        N, dN_dxi, dN_deta = self._shape(xi, eta)
        J = np.zeros((2,2))
        J[0,0] = np.dot(dN_dxi,  xe[:,0]); J[0,1] = np.dot(dN_dxi,  xe[:,1])
        J[1,0] = np.dot(dN_deta, xe[:,0]); J[1,1] = np.dot(dN_deta, xe[:,1])
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Element with non-positive Jacobian. Check node ordering (counter-clockwise) and mesh quality.")
        invJ = np.linalg.inv(J)
        # dN/dx, dN/dy
        grad = np.vstack((dN_dxi, dN_deta)).T @ invJ.T  # (4,2)
        B = np.zeros((3, 8))
        for a in range(4):
            i = 2*a
            B[0, i  ] = grad[a,0]        # dNa/dx
            B[1, i+1] = grad[a,1]        # dNa/dy
            B[2, i  ] = grad[a,1]        # dNa/dy
            B[2, i+1] = grad[a,0]        # dNa/dx
        return B, detJ, N

    def element_stiffness(self, e: int) -> np.ndarray:
        D = self.material.D()
        t = self.t
        nodes = self.conn[e]
        xe = self.coords[nodes, :]  # (4,2)
        Ke = np.zeros((8,8))
        # 2x2 Gauss
        pts = [-np.sqrt(1/3), np.sqrt(1/3)]
        wts = [1.0, 1.0]
        for i, xi in enumerate(pts):
            for j, eta in enumerate(pts):
                B, detJ, _ = self._Bmatrix(xe, xi, eta)
                Ke += B.T @ D @ B * detJ * wts[i] * wts[j] * t
        return Ke

    def assemble(self) -> np.ndarray:
        nnode = self.coords.shape[0]
        ndof = 2*nnode
        K = np.zeros((ndof, ndof))
        for e in range(self.conn.shape[0]):
            Ke = self.element_stiffness(e)
            # DOF map for element: [u1,v1,u2,v2,u3,v3,u4,v4]
            edofs = np.ravel(np.column_stack((2*self.conn[e], 2*self.conn[e]+1)))
            for a in range(8):
                A = edofs[a]
                for b in range(8):
                    B = edofs[b]
                    K[A,B] += Ke[a,b]
        return K

    def element_strain_stress(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute strain, stress (Voigt) at 2x2 Gauss points for each element.
        Returns: (nelem,4,3) strain, (nelem,4,3) stress, (nelem,4,2) gauss-point (xi,eta)
        """
        D = self.material.D()
        ne = self.conn.shape[0]
        strain = np.zeros((ne,4,3))
        stress = np.zeros((ne,4,3))
        gp     = np.zeros((ne,4,2))
        pts = [-np.sqrt(1/3), np.sqrt(1/3)]
        idx = 0
        for e in range(ne):
            nodes = self.conn[e]
            xe = self.coords[nodes, :]
            ue = U[np.ravel(np.column_stack((2*nodes, 2*nodes+1)))]
            ue = ue.reshape(8)
            k = 0
            for xi in pts:
                for eta in pts:
                    B, detJ, N = self._Bmatrix(xe, xi, eta)
                    eps = B @ ue
                    sig = D @ eps
                    strain[e,k,:] = eps
                    stress[e,k,:] = sig
                    gp[e,k,:] = [xi, eta]
                    k += 1
            idx += 1
        return strain, stress, gp

# --------------------------- Boundary conditions & solver ---------------------------

def apply_dirichlet(K: np.ndarray, F: np.ndarray, bcs: list[tuple[int,int,float]]) -> tuple[np.ndarray,np.ndarray,list[int]]:
    """Apply u_d prescribed at (node i, dof j: 0->ux,1->uy). Returns reduced system and free dof list."""
    n = K.shape[0]
    fixed = []
    ubar = np.zeros(n)
    for (node, comp, val) in bcs:
        dof = 2*node + comp
        fixed.append(dof)
        ubar[dof] = val
    fixed = np.array(sorted(set(fixed)))
    free  = np.array([i for i in range(n) if i not in fixed])

    # Modify RHS: K_ff * Uf = Ff - K_fc * Uc
    Kff = K[np.ix_(free, free)]
    Kfc = K[np.ix_(free, fixed)]
    Ff  = F[free] - Kfc @ ubar[fixed]
    return Kff, Ff, free.tolist()

# --------------------------- Plotting ---------------------------

def plot_mesh(coords: np.ndarray, conn: np.ndarray, U: np.ndarray | None = None, scale: float | None = None,
              title: str = "Mesh", show_nodes: bool = True):
    fig, ax = plt.subplots()
    # Undeformed
    for e in range(conn.shape[0]):
        nodes = conn[e]
        cyc = np.r_[nodes, nodes[0]]
        ax.plot(coords[cyc,0], coords[cyc,1], '-o', lw=1.0, ms=3)
    if U is not None:
        n = coords.shape[0]
        u = U[0::2]; v = U[1::2]
        if scale is None:
            # heuristic scale
            L = np.max(coords, axis=0) - np.min(coords, axis=0)
            mag = np.sqrt(u**2+v**2).max() if np.any(U) else 1.0
            scale = 0.1 * np.linalg.norm(L) / (mag + 1e-12)
        defC = coords + scale*np.c_[u, v]
        for e in range(conn.shape[0]):
            nodes = conn[e]
            cyc = np.r_[nodes, nodes[0]]
            ax.plot(defC[cyc,0], defC[cyc,1], '--', lw=2.0)
    if show_nodes:
        for i,(x,y) in enumerate(coords):
            ax.text(x, y, f" {i}", fontsize=8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.show()

# --------------------------- Driver ---------------------------

def run_fem(coords: np.ndarray,
            conn: np.ndarray,
            material: Material,
            thickness: float,
            bcs: list[tuple[int,int,float]],
            loads: np.ndarray) -> dict:
    """High-level solver.
    Parameters
    ----------
    coords : (nnode,2)
    conn   : (nelem,4)  (0-based node ids)
    material : Material
    thickness : float
    bcs : list of (node, comp, value) with comp=0->ux, 1->uy
    loads : (2*nnode,) nodal force vector [Fx1, Fy1, Fx2, Fy2, ...]
    """
    quad = Quad4(coords=coords, conn=conn, material=material, t=thickness)
    K = quad.assemble()
    F = loads.copy()

    Kff, Ff, free = apply_dirichlet(K, F, bcs)
    Uf = np.linalg.solve(Kff, Ff) if Ff.size else np.array([])

    # Assemble full U
    U = np.zeros_like(F)
    U[free] = Uf

    strain, stress, gp = quad.element_strain_stress(U)

    return {
        'U': U,
        'K': K,
        'strain': strain,
        'stress': stress,
        'gauss_points': gp,
    }

# --------------------------- Demo ---------------------------
if __name__ == "__main__":
    # Simple 2x1 mesh of 4-node quads on a 2x1 rectangle
    # 3 x 2 grid nodes => 6 nodes (0..5)
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0])
    X, Y = np.meshgrid(xs, ys)
    coords = np.c_[X.ravel(order='C'), Y.ravel(order='C')]  # (6,2)

    # Elements (counter-clockwise):
    # lower-left: nodes [0,1,4,3], lower-right: [1,2,5,4]
    conn = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, 4]
    ], dtype=int)

    # Material & thickness
    mat = Material(E=210e9, nu=0.3, plane_stress=True)
    t = 0.01

    # Boundary conditions: fix left edge (nodes 0 and 3), ux=uy=0
    bcs = [(0,0,0.0), (0,1,0.0), (3,0,0.0), (3,1,0.0)]

    # Loads: apply downward Fy = -1000 N at right-edge nodes (2 and 5)
    nnode = coords.shape[0]
    F = np.zeros(2*nnode)
    F[2*2+1] = -1e3
    F[2*5+1] = -1e3

    res = run_fem(coords, conn, mat, t, bcs, F)
    U = res['U']
    print("Max |U| =", np.linalg.norm(U.reshape(-1,2), axis=1).max())

    # Plot
    plot_mesh(coords, conn, U, title="Quad4: undeformed (solid) & deformed (dashed)")
    print("Max |U| =", np.linalg.norm(U.reshape(-1,2), axis=1).max())
