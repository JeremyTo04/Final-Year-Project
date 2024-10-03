import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils

# Load your dataset
data = pd.read_csv(r"C:\Users\jeret\OneDrive\Documents\GitHub\Final-Year-Project\final_microexpression_data_CASME2.csv")

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

print("Dataset columns:", data.columns)
print("Graph nodes:", g.nodes)

# Check labels
labels = ["Action Units", "Emotion", "age", "gender", "ethnicity", "fringe", "glasses", "Predicted Emotion"]
print("Number of labels:", len(labels))

# # Convert the graph to pydot format and save as PNG
pdy = GraphUtils.to_pydot(g, labels=["Action Units", "Emotion", "age", "gender", "fringe", "glasses", "Predicted Emotion"])
pdy.write_png('final_causal_graph_CASME2.png')

print("Causal graph saved as 'final_causal_graph_CASME2.png'.")