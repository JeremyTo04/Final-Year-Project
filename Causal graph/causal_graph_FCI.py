import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils

# Load your dataset
data = pd.read_csv("encoded_microexpression_data_4dme.csv")

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

# print(g)

for edge in edges:
    print(edge.properties)

# print(type(g))

# # Convert the graph to pydot format and save as PNG
pdy = GraphUtils.to_pydot(g, labels=["Action Units", "Ethnicity", "glasses", "fringe", "gender", "emotion"])
pdy.write_png('fci_causal_graph_4dme.png')

print("Causal graph saved as 'fci_causal_graph.png'.")