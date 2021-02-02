import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# initialize
erdos_renyi_graphs = None
air_traffic_graph = None
acad_collab_graph = None

num_nodes = 1226
edge_probs = [0.001, 0.003, 0.01, 0.05, 0.1]


def generate_erdos_renyi_graph(n=1226, p=0.003):
    """
    Return an instance of Erdos-Renyi graph of size ``n`` and edge probability ``p``.

    INPUT:
    - ``n`` -- number of nodes
    - ``p`` -- probability of an edge between every pair of nodes

    OUTPUT:
    - NetworkX graph object

    """

    graph = nx.erdos_renyi_graph(n=n,p=p,seed=123)
    return graph


def load_real_world_network(name="air_traffic"):
    """
    Read from file and return ``name`` graph.

    INPUT:
    - ``name`` -- name of real-world network

    OUTPUT:
    - NetworkX graph object

    """

    graph = nx.read_edgelist(path=name+'.edgelist')
    return graph


def compute_clustering_coefficient(graph=air_traffic_graph):
    """
    Compute average clustering coefficient of ``graph``.

    INPUT:
    - ``graph`` -- NetworkX graph object

    OUTPUT:
    - average clustering coefficient of ``graph`` (type: float)

    """

    # Initial C
    C=0
    
    # Remove all the self loops, which will disturb our calculation
    graph.remove_edges_from(nx.selfloop_edges(graph))
    
    # Start to calculate the clustering coefficient one by one
    for node in nx.nodes(graph):
        # Get the degree and neighbors list of node in graph
        degree = nx.degree(graph,node)
        neighbors = set(nx.neighbors(graph,node))
        
        exis_edges = 0 # (2X) the number of edges of between the neighbours of node
        poss_edges = degree*(degree-1)
        
        # For node whose degree is less than 2
        if degree < 2:
            clus_coeff = 0
        
        # For node whose degree is more than 1
        else:
            for i in neighbors:
                for k in set(nx.neighbors(graph,i)):
                    if k in neighbors:
                        exis_edges += 1
            # Get the clustering coefficient of this node
            clus_coeff = exis_edges/poss_edges
        
        # Get the SUM of clustering coefficient
        C += clus_coeff
    
    # Get the average clustering coefficient
    C = C/nx.number_of_nodes(graph)
    
    # Check the answer with NetworkX function
    if (C != nx.average_clustering(graph)):
        print ("The average clustering coefficient is wrong")
    
    return C


def get_data_to_plot(graph=air_traffic_graph):
    """
    Return degree values and cumulative count of number of nodes for each degree value.

    INPUT:
    - ``graph`` -- NetworkX graph object

    OUTPUT:
    - ``x_graph`` -- degree values of nodes in ``graph`` in sorted order (type: list)
    - ``y_graph`` -- number of nodes of degree `d` for all degree values `d` (type: list)
    """

    degree_values = set(i[1] for i in nx.degree(graph))
    
    x_graph = sorted(list(degree_values))

    y_graph = [0]*len(degree_values)

    for k,v in nx.degree(graph):
        index = x_graph.index(v)
        y_graph[index]+=1

    return x_graph, y_graph


def degree_distribution_plot():
    """
    Draw degree distribution plot.
    """

    x_ER, y_ER = get_data_to_plot(erdos_renyi_graphs[0.01])
    x_air, y_air = get_data_to_plot(air_traffic_graph)
    x_aca, y_aca = get_data_to_plot(acad_collab_graph)
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
    ax1.loglog(x_ER, y_ER, color='r', label='ER G(1226, 0.01)')
    ax2.loglog(x_air, y_air, color = 'g', label = 'Air Traffic')
    ax3.loglog(x_aca, y_aca, color = 'b', label = 'Academic Collaboration')

    for ax in (ax1, ax2, ax3):
        ax.tick_params(direction='in', length=5, width=1)
        ax.set_xlabel('Degree_log')
        ax.set_ylabel('Frequency_log')
        ax.legend()

    plt.suptitle('Degree Distribution on a log-log scale')
    plt.savefig("Degree_Distribution_loglog.png", dpi=250)
    plt.show()
    return 0



def print_num_nodes_edges():
    """
    Print number of nodes and edges of all graphs.
    """

    for prob in erdos_renyi_graphs:
        print("Erdos Renyi Graph G({}, {}): Numbers of nodes and edges = {}, {}".format(num_nodes, prob, erdos_renyi_graphs[prob].number_of_nodes(), erdos_renyi_graphs[prob].number_of_edges()))
    
    print("Air-Traffic Graph: Numbers of nodes and edges = {}, {}".format(air_traffic_graph.number_of_nodes(), air_traffic_graph.number_of_edges()))

    print("Academic Collaboration Graph: Numbers of nodes and edges = {}, {}".format(acad_collab_graph.number_of_nodes(), acad_collab_graph.number_of_edges()))

    return 0


def problem_1():
    """
    Code for Problem 1 deliverables.
    """
    global num_nodes, edge_probs
    global erdos_renyi_graphs, air_traffic_graph, acad_collab_graph

    # generate Erdos-Renyi random graphs
    erdos_renyi_graphs = {}
    for p in edge_probs:
        erdos_renyi_graphs[p] = generate_erdos_renyi_graph(n=num_nodes, p=p)

    # load real-world networks
    air_traffic_graph = load_real_world_network(name="air_traffic")
    acad_collab_graph = load_real_world_network(name="academic_collaboration")

    print_num_nodes_edges()

    # compute average clustering coefficient
    C_erdos_renyi = compute_clustering_coefficient(erdos_renyi_graphs[0.1])
    C_air_traffic = compute_clustering_coefficient(air_traffic_graph)
    C_acad_collab = compute_clustering_coefficient(acad_collab_graph)

    print("Clustering coefficient of Erdos-Renyi graph G(1226, 0.1) = {:.4f}".format(C_erdos_renyi))
    print("Clustering coefficient of Air-Traffic graph = {:.4f}".format(C_air_traffic))
    print("Clustering coefficient of Academic Collaboration graph = {:.4f}".format(C_acad_collab))

    # plot degree distribution
    degree_distribution_plot()

if __name__ == '__main__':
    problem_1()

