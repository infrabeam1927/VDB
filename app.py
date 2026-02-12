import json
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import chromadb
from pprint import pprint

# load the sample data
with open('posts.json', 'r') as f:
    posts = json.load(f)
    
# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# get embeddings per topic
j_embeddings = {}
for topic, v in posts.items():
    j_embeddings[topic] = model.encode(posts[topic])

# combine embeddings in single array
topics = list(posts.keys())  # Use actual topics from the JSON
embeddings_list = [j_embeddings[topic] for topic in topics]
embeddings = np.vstack(embeddings_list)

# Perform TSNE to reduce to 2 components
tsne_model = TSNE(n_components=2, random_state=42)
tsne_embeddings_values = tsne_model.fit_transform(embeddings)

# Build color labels based on actual topic sizes
col_topics = []
for topic in topics:
    col_topics.extend([topic] * len(posts[topic]))

fig = px.scatter(
    x = tsne_embeddings_values[:,0], 
    y = tsne_embeddings_values[:,1],
    color = col_topics,
)

fig.update_traces(marker=dict(size=13))  # Increase the marker size uniformly


fig.update_layout(
    xaxis=dict(showticklabels=False, title=''),
    yaxis=dict(showticklabels=False, title=''),
    #showlegend=False,
    autosize=False,
    #width=600,  # Width of the plot
    #height=600,  # Height of the plot
    margin=dict(l=50, r=50, b=50, t=50, pad=4)  # Margins
)
fig.show()
