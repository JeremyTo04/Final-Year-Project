import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils

# Load your dataset
data = pd.read_csv('encoded_microexpression_data.csv')

# Ensure all columns are categorical
for column in data.columns:
    data[column] = data[column].astype('category')

# Drop columns with all NaN values or constant values
data = data.dropna(axis=1, how='all')
data = data.loc[:, (data != data.iloc[0]).any()]

# Convert DataFrame to numpy array
data_np = data.to_numpy()

# Run FCI algorithm with default parameters
g, edges = fci(data_np)

# Convert the graph to pydot format
pdy = GraphUtils.to_pydot(g)

# Rename the nodes in the graph to match the original column names
node_names = data.columns
for i, node in enumerate(pdy.get_nodes()):
    node_name = node.get_name().strip('"')  # Remove quotes
    if i < len(node_names):  # Ensure there are enough column names for the nodes
        node.set_name(f'"{node_names[i]}"')

# Save the graph as PNG
pdy.write_png('fci_causal_graph.png')

print("Causal graph saved as 'fci_causal_graph.png'.")
