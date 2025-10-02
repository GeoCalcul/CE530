import numpy as np

# Input data
coordinates = np.array([
	[0, 0],
	[4, 0],
	[8, 0],
	[4, -6]
])
connectivity = np.array([
	[0, 3],
	[1, 3],
	[2, 3]
])
E = 1  # Young's Modulus (Pa)
A = 1  # Cross-sectional area (m^2)

# Define supports (0 for not supported, 1 for supported)
supports = np.array([
	[1, 1],
	[1, 1],
	[1, 1],
	[0, 0]
])

# Define applied loads (0 for no load, specify direction and value)
applied_loads = np.array([
	[0, 0],
	[0, 0],  # Node 1, applied horizontal load
	[0, 0],  # Node 2, applied vertical load
	[100, -100]
])

# Create freedom matrix
num_nodes = len(coordinates)
num_dofs = 2 * num_nodes
num_elements = len(connectivity)
dofs = np.zeros((num_elements, 4), dtype=int)
NodeDof = np.zeros((num_nodes, 2), dtype=int)

# Initialize global stiffness matrix and force vector
KG = np.zeros((num_dofs, num_dofs))
kt_global = np.zeros((num_dofs, num_dofs))
F_global = np.zeros(num_dofs)

# Calculate Dof Matrix
for i, (node1, node2) in enumerate(connectivity):
	dofs[i, :] = np.array([2 * node1, 2 * node1 + 1, 2 * node2, 2 * node2 + 1])
	NodeDof[node1, 0] = 2 * node1
	NodeDof[node1, 1] = 2 * node1 + 1
	NodeDof[node2, 0] = 2 * node2
	NodeDof[node2, 1] = 2 * node2 + 1

print(NodeDof)
# Calculate element lengths and stiffness matrices
for i, (node1, node2) in enumerate(connectivity):
	x1, y1 = coordinates[node1]
	x2, y2 = coordinates[node2]

	L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
	c = (x2 - x1) / L
	s = (y2 - y1) / L

	k_local = (E * A / L) * np.array([
		[c ** 2, c * s, -c ** 2, -c * s],
		[c * s, s ** 2, -c * s, -s ** 2],
		[-c ** 2, -c * s, c ** 2, c * s],
		[-c * s, -s ** 2, c * s, s ** 2]
	])

	# Assemble local stiffness matrix into global stiffness matrix
	KG[np.ix_(dofs[i, :], dofs[i, :])] += k_local
	# print(k_local)
	# print(np.ix_(dofs[i,:], dofs[i,:]) )
	# print(K_global)

print(KG)
kt_global = KG * 1
# Apply boundary conditions and applied loads
for node in range(num_nodes):
	for i in range(2):
		if supports[node, i] == 1:
			fixed_dof = NodeDof[node, i]
			kt_global[fixed_dof, :] = 0
			kt_global[:, fixed_dof] = 0
			kt_global[fixed_dof, fixed_dof] = 1
			F_global[fixed_dof] = 0
		else:
			applied_force = applied_loads[node, i]
			F_global[2 * node + i] = applied_force

print(KG)
# Solve for displacements
displacement = np.linalg.solve(kt_global, F_global)
print(KG)
# Calculate reaction forces
reaction_forces = np.dot(KG, displacement)

print(KG)
print("Displacements (mm):")
print(displacement * 1000)
print("Reaction Forces (N):")
print(reaction_forces)

