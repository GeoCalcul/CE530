import numpy as np
!pip install -Uqq ipdb
import ipdb
# Define the connectivity information (which nodes are connected by each element)
# For example, element 1 connects nodes 1 and 2, and element 2 connects nodes 2 and 3.
connectivity = [(0,1), (1, 2), (2, 3), (3, 4)]
# Define which nodes have supports (here, we assume nodes 1 and 4 have supports).
nodes_with_supports = [0]
# Define material properties, element lengths, etc.
E = np.array([10000, 10000, 10000, 10000, 10000])
A = np.array([1, 1, 1, 1, 1])
L = np.array([1, 1, 1, 1, 1])

# Define the degree of freedom (DOF) matrix for each node.
# In this example, we assume that each node has 2 DOFs (displacement in x and y).
# You can adjust this based on your problem.
num_nodes = max(max(connectivity))+1
num_dofs_per_node = 1
DOF_matrix = np.zeros((num_nodes, num_dofs_per_node))
 
# Set DOFs to -1 for nodes with supports
for node in nodes_with_supports:
    DOF_matrix[node]=-1

i=0
for node in range(num_nodes*num_dofs_per_node) :
    if     DOF_matrix[node] !=-1:
           DOF_matrix[node]=i
           i+=1


ipdb.set_trace()
# Initialize the global stiffness matrix Kglobal as a zeros matrix
dofs =  max(max(DOF_matrix)) +1
Kuu = np.zeros((int(dofs), int(dofs)))

# Assemble the global stiffness matrix by adding the element stiffness matrices
for j, (node1, node2) in enumerate(connectivity):

    # Define the individual element stiffness matrices (e.g., for two elements)
    k = (E[j] * A[j] / L[j]) * np.array([[1, -1], [-1, 1]])
    Kuu[(int(DOF_matrix[(node1)]), int(DOF_matrix[(node1) ]))]+= k[0, 0]
    Kuu[(int(DOF_matrix[(node1)]), int(DOF_matrix[(node2) ]))]+= k[0, 1]
    Kuu[(int(DOF_matrix[(node2)]), int(DOF_matrix[(node1) ]))]+= k[1, 0]
    Kuu[(int(DOF_matrix[(node2)]), int(DOF_matrix[(node2) ]))]+= k[1, 1]



# Print the global stiffness matrix
print("Global Stiffness Matrix (Kuu):\n", Kuu)
