import networkx as nx
import matplotlib.pyplot as plt

def create_causal_graph(graph_edges, columns, output_file='causal_graph.png'):
    # Create a mapping from node names to column names
    node_mapping = {f"X{i+1}": columns[i] for i in range(len(columns))}

    # Create a directed graph
    G = nx.DiGraph()

    # Parse edges from the input format and add them to the graph
    for edge in graph_edges:
        # Extract source, target, and edge type from the input string
        parts = edge.split(' ')
        source = parts[0]
        target = parts[-1]
        edge_type = parts[1]

        # Map node names to column names
        source_col = node_mapping.get(source, source)
        target_col = node_mapping.get(target, target)

        # Add edges to the graph based on edge type
        if edge_type == 'o->':
            G.add_edge(source_col, target_col)
        elif edge_type == 'o-o':
            G.add_edge(source_col, target_col)
            G.add_edge(target_col, source_col)

    # Draw the graph
    pos = nx.spring_layout(G)  # Position the nodes
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=10, font_weight="bold")
    plt.title("Causal Graph with Column Names")
    
    # Save and display the graph
    plt.savefig(output_file)
    print(f"Causal graph saved as '{output_file}'.")
    plt.show()

# Example usage
graph_edges = [
    "X2 o-> X1",
    "X3 o-> X1",
    "X3 o-o X5",
    "X4 o-o X5"
]

columns = [
    "Action Units", "Estimated Emotion (7 class)", "glasses", "fringe", "gender"
]

create_causal_graph(graph_edges, columns)
