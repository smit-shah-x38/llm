# Import libraries
import spacy
import textacy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Define text
text = "Lionel Messi is an Argentine professional footballer who plays as a forward and captains both Spanish club Barcelona and the Argentina national team."

# Create spacy doc object
doc = nlp(text)

# Extract entities
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(entities)
# [('Lionel Messi', 'PERSON'), ('Argentine', 'NORP'), ('Spanish', 'NORP'), ('Barcelona', 'ORG'), ('Argentina', 'GPE')]

# Extract relations
relations = textacy.extract.subject_verb_object_triples(doc)
print(list(relations))
# [(Lionel Messi, plays, forward), (Lionel Messi, captains, club), (Lionel Messi, captains, team)]

# Create a dataframe of relations
kg_df = pd.DataFrame(list(relations), columns=["source", "relation", "target"])
print(kg_df)
#         source relation   target
# 0  Lionel Messi    plays  forward
# 1  Lionel Messi  captains     club
# 2  Lionel Messi  captains     team

# Create a networkx graph from the dataframe
G = nx.from_pandas_edgelist(
    kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph()
)

# Plot the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color="skyblue", edge_cmap=plt.cm.Blues, pos=pos)
plt.show()
