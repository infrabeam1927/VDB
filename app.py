import json
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import chromadb
from pprint import pprint
from datetime import datetime

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
# Set perplexity to be less than n_samples
n_samples = embeddings.shape[0]
perplexity = min(30, max(5, (n_samples - 1) // 3))
tsne_model = TSNE(n_components=2, random_state=42, perplexity=perplexity)
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


client = chromadb.Client()

collection_name = "Random-dump"

try:
    client.delete_collection(name=collection_name)
    print(f"COLLECTION {collection_name} DELETED")
except:
    print(f"COLLECTION {collection_name} DIDNT EXIST YET")

collection = client.create_collection(
      name=collection_name,
      metadata={"hnsw:space": "cosine"}
  )

# fill vector database
for k in j_embeddings.keys():
    print(f"Add stuff for topic {k}")
    num_elements = len(posts[k])
    collection.add(
        embeddings = j_embeddings[k],
        documents=posts[k],
        metadatas=[{"topic": k}]*num_elements,
        ids=[f"{i:02}__{k}" for i in range(num_elements)],
    )

results = collection.query(
    query_texts=["Buy-In Annuity"],
    n_results=3,
)

results2 = collection.query(
    query_texts=["Are there any posts about Premium Adjustments?"],
    n_results=2,
)

# Save results with timestamped filename
import os

# Create timestamped output filename
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"outputs_{timestamp_str}.json"

# Create entries with timestamps and distance analysis
def extract_distance_analysis(results):
    """Extract and analyze the distances from query results"""
    if 'distances' in results:
        distances = results['distances'][0] if results['distances'] else []
        avg_distance = sum(distances) / len(distances) if distances else 0
        analysis = f"Retrieved {len(distances)} results with cosine distances: {[f'{d:.4f}' for d in distances]}. Average semantic distance: {avg_distance:.4f} (lower = more similar)"
        return analysis
    return "Distance information not available"

queries = [
    {
        
        "query": "Buy-In Annuity",
        "distance_analysis": extract_distance_analysis(results),
        "results": results
    },
    {
        
        "query": "Are there any posts about Premium Adjustments?",
        "distance_analysis": extract_distance_analysis(results2),
        "results": results2
    }
]

# Save to timestamped file
with open(output_filename, 'w') as f:
    json.dump(queries, f, indent=2)

print(f"Results saved to {output_filename}")
pprint(results)
pprint(results2)