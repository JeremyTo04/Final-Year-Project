import pandas as pd
from pgmpy.estimators import PC
import matplotlib.pyplot as plt
import networkx as nx

# Load your dataset
data = pd.read_csv('encoded_microexpression_data.csv')

# Ensure all columns are categorical
for column in data.columns:
    data[column] = data[column].astype('category')

# Drop columns with all NaN values or constant values
data = data.dropna(axis=1, how='all')
data = data.loc[:, (data != data.iloc[0]).any()]

# Initialize the PC algorithm
estimator = PC(data)

try:
    # Estimate the structure using PC (not FCI)
    pc_model = estimator.estimate()

    # Convert the DAG to a networkx graph
    nx_graph = nx.DiGraph()
    nx_graph.add_edges_from(pc_model.edges())

    # Debugging: Print out the edges
    print("Edges in the estimated DAG:", pc_model.edges())

    # Set up the plot
    plt.figure(figsize=(12, 8))  # Adjust the size as needed

    # Compute layout
    pos = nx.spring_layout(nx_graph)

    # Draw the graph with smaller nodes
    nx.draw(nx_graph, pos, with_labels=True, node_color='skyblue', 
            node_size=800,  # Smaller node size
            edge_color='black', font_size=8, font_weight='bold')

    plt.title("Causal Graph using PC Algorithm")

    # Save the plot
    plt.savefig('pc_causal_graph.png', format='png', bbox_inches='tight')

    # Show the plot
    plt.show()

except ValueError as e:
    print(f"Error during estimation: {e}")
