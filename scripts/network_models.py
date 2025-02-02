### Script intended to refine reconstruction pipeline

import numpy asnp
import networkx as nx

def hyperbolic_dist(x1,x2,xi=1):
    r1, theta1 = x1[0], x1[1]
    r2, theta2 = x2[0], x2[1]
    delta = np.pi - np.abs(np.pi - np.abs(theta1-theta2))
#     right_side = np.cosh(xi*r1)*np.cosh(xi*r2)-np.sinh(xi*r1)*np.sinh(xi*r2)*np.cos(delta)
#     return np.arccosh(right_side)/xi
    return r1+r2+np.log(delta/2)
#     r1, theta1 = x1[0], x1[1]
#     r2, theta2 = x2[0], x2[1]
#     delta = np.pi - np.abs(np.pi - np.abs(theta1-theta2))
#     right_side = np.cosh(xi*r1)*np.cosh(xi*r2)-np.sinh(xi*r1)*np.sinh(xi*r2)*np.cos(delta)
#     return np.arccosh(right_side)/xi

def temp_dist(x1,x2,T,R):
    return 1/(1+np.exp((hyperbolic_dist(x1,x2)-R)/(2*T)))

def R(x,T,beta,j,m):
    return x[0] - np.log((2*T*(1-np.exp(-(1-beta)*np.log(j+1))))/(np.sin(T*np.pi)*m*(1-beta)))

# def pso(N,m,T,beta,xi=1):
#     """
#     PSO algorithm for scale-free clustered networks.
    
#     Parameters:
#         N (int) - number of nodes
#         m (float) - half of the average node degree
#         T (float) - temperature
#         beta (float) popularity
#         xi (float) - curvature
#     """
#     # Initialize network
#     g = nx.Graph()

#     x = np.log(1+np.arange(N)).reshape(N,1)
#     y = 2*np.pi*np.random.random(size=N).reshape(N,1)

#     coor = np.hstack((x,y))

#     # Add nodes
#     for i in range(N):
#         # Add node
#         g.add_node(i)
#         if i == 0:
#             continue
#         if i == 1:
#             g.add_edge(0,1)
#             continue
#         # Increase coordinates of other nodes
#         coor[:i,0] = beta*coor[:i,0] + (1-beta) * coor[i,0]
#         # Connect to closest nodes
#         if T == 0:
#             # Get closest nodes
#             dist = np.apply_along_axis(hyperbolic_dist,1,coor[:i],x2=coor[i])
#             nodes = np.argsort(dist)[:m]
#             for x in nodes:
#                 g.add_edge(i,x)
#         else:
#             # Get R
#             cur_R = R(coor[i],T,beta,i,m)
#             # Choose random nodes
#             count = 0
#             nodes = np.arange(i)
#             val = np.min([i-1,m])
#             while count != val:
#                 # Choose a node
#                 cur_x = np.random.choice(nodes)
#                 # Get probability
#                 p = temp_dist(coor[cur_x],coor[i],T,cur_R)
#                 if np.random.random() < p:
#                     g.add_edge(cur_x,i)
#                     nodes = nodes[nodes!=cur_x]
#                     count += 1
#     return g
def pso(N, m, T, beta, xi=1):
    """
    PSO algorithm for scale-free clustered networks using graph-tool.

    Parameters:
        N (int) - number of nodes
        m (int) - number of edges per new node
        T (float) - temperature
        beta (float) - popularity
        xi (float) - curvature
    """
    g = gt.Graph(directed=False)

    # Add node property for coordinates
    x_prop = g.new_vertex_property("double")  # Radial coordinate
    y_prop = g.new_vertex_property("double")  # Angular coordinate

    # Initialize node positions
    x = np.log(1 + np.arange(N)).reshape(N, 1)
    y = 2 * np.pi * np.random.random(size=N).reshape(N, 1)
    coor = np.hstack((x, y))

    # Add nodes to graph-tool
    vertices = [g.add_vertex() for _ in range(N)]

    for i in range(N):
        x_prop[vertices[i]] = coor[i, 0]
        y_prop[vertices[i]] = coor[i, 1]

        if i == 0:
            continue
        if i == 1:
            g.add_edge(vertices[0], vertices[1])
            continue

        # Update other nodes' coordinates
        coor[:i, 0] = beta * coor[:i, 0] + (1 - beta) * coor[i, 0]

        if T == 0:
            # Connect to m closest nodes
            dist = np.array([hyperbolic_dist(coor[j], coor[i]) for j in range(i)])
            closest_nodes = np.argsort(dist)[:m]
            for j in closest_nodes:
                g.add_edge(vertices[i], vertices[j])
        else:
            # Get radius R
            cur_R = R(coor[i], T, beta, i, m)
            available_nodes = list(range(i))
            count = 0

            while count < min(i, m):
                chosen = np.random.choice(available_nodes)
                if np.random.random() < temp_dist(coor[chosen], coor[i], T, cur_R):
                    g.add_edge(vertices[i], vertices[chosen])
                    available_nodes.remove(chosen)
                    count += 1

    # Attach node properties to graph
    g.vertex_properties["x"] = x_prop
    g.vertex_properties["y"] = y_prop

    return g

# Example usage
g = pso(N=100, m=3, T=0.5, beta=0.8)

# Draw the graph
graph_draw(g, output_size=(600, 600), output="pso_graph.png")
