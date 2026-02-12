# VDB - Vector Database Semantic Search

A Python-based semantic search system that uses sentence embeddings and vector databases to query and analyze pension-related documents. This project combines embeddings visualization with ChromaDB for efficient semantic similarity searches.

## Overview

VDB processes a collection of documents from `posts.json`, generates semantic embeddings, creates a visual 2D representation using t-SNE, and stores everything in a vector database for intelligent querying.

### Key Features

- **Semantic Embeddings**: Uses `SentenceTransformer` (all-MiniLM-L6-v2) to convert text into dense vector representations
- **Visualization**: t-SNE dimensionality reduction to plot documents in 2D space by semantic similarity
- **Vector Database**: ChromaDB with cosine similarity for fast semantic search
- **Query Analysis**: Semantic distance measurements showing how similar results are to queries
- **Timestamped Outputs**: Each query result is saved with timestamp and semantic distance analysis

## Data Structure

### posts.json

Contains 4 categories of pension and regulatory documents:

1. **Canadian Regulatory Framework & Compliance** (6 posts)
   - OSFI regulations, CAPSA guidelines, fiduciary duties, CRA requirements, ESG factors, MJPP

2. **Premium Adjustments (PAPT) & Data True-Ups** (6 posts)
   - Premium adjustment accounting, data verification, death-before-closing events, yield-to-maturity, salary-linked benefits

3. **Buy-In vs. Buy-Out Comparisons** (6 posts)
   - Insurance policies, legal responsibility, member communication, asset management, accounting treatment, Assuris Protection

4. **PRT Policy & Operational Tables** (6 posts)
   - Policy comparison, annuity factors, mortality tables, SLA definitions, jurisdictional summaries, true-up reconciliation

## Project Workflow

```
posts.json 
    ↓
Generate Embeddings (SentenceTransformer)
    ↓
    ├→ Visualize (t-SNE + Plotly scatter plot)
    └→ Store in ChromaDB (vector database)
         ↓
    Execute Semantic Queries
         ↓
    Calculate Semantic Distances
         ↓
    Save to outputs_TIMESTAMP.json
```

## How It Works

### 1. **Embedding Generation**
- Each post is converted to a 384-dimensional vector using `SentenceTransformer`
- All embeddings are combined into a single matrix

### 2. **Visualization**
- t-SNE reduces embeddings from 384D to 2D
- Perplexity is automatically adjusted based on sample count
- Results plotted with Plotly, colored by topic

### 3. **Vector Storage**
- ChromaDB creates an in-memory collection with cosine metric
- Documents, embeddings, and metadata stored for retrieval
- Cleaned and recreated on each run

### 4. **Semantic Search**
- Queries are embedded using the same model
- Vector DB returns top N results sorted by cosine distance
- Distance closer to 0 = more semantically similar

### 5. **Output Generation**
- Results saved to timestamped JSON file: `outputs_YYYY-MM-DD_HH-MM-SS.json`
- Each result includes:
  - Query text
  - Semantic distances for each retrieved document
  - Average distance (lower = better match)
  - Full search results from ChromaDB

## Installation

### Requirements
```
numpy
plotly
sentence-transformers
scikit-learn
chromadb
```

### Setup
```bash
pip install -r requirements.txt
```

## Usage

### Run the Application
```bash
python app.py
```

This will:
1. Load posts.json
2. Generate and visualize embeddings (opens Plotly figure)
3. Create ChromaDB collection
4. Execute predefined queries
5. Save results to a timestamped output file
6. Print results to console

### Example Output File

`outputs_2026-02-12_05-51-46.json`:
```json
[
  {
    "query": "Buy-In Annuity",
    "distance_analysis": "Retrieved 3 results with cosine distances: ['0.1234', '0.3456', '0.5678']. Average semantic distance: 0.3456 (lower = more similar)",
    "results": {
      "documents": [...],
      "distances": [[0.1234, 0.3456, 0.5678]],
      "metadatas": [...]
    }
  },
  ...
]
```

## Customizing Queries

Edit the query configuration in `app.py` (lines with `results = collection.query(...)`):

```python
# Current queries:
results = collection.query(
    query_texts=["Buy-In Annuity"],
    n_results=3,
)

results2 = collection.query(
    query_texts=["Are there any posts about Premium Adjustments?"],
    n_results=2,
)
```

### Tips for Good Query Results

- **Exact match → Distance ~0**: `"Premium Adjustment for Pension Risk Transfer (PAPT)"`
- **Close semantic meaning → Distance 0.1-0.3**: `"annuity purchase"`
- **Related but broader → Distance 0.3-0.6**: `"pension regulations"`
- **Unrelated → Distance 0.6+**: `"apple computer"`

## Project Structure

```
/workspaces/VDB/
├── app.py                    # Main application
├── posts.json               # Document collection
├── README.md                # This file
└── outputs_*.json           # Query results (generated on each run)
```

## Key Technologies

- **SentenceTransformers**: Fast, efficient sentence embeddings
- **ChromaDB**: Lightweight vector database with cosine similarity
- **scikit-learn**: t-SNE for dimensionality reduction
- **Plotly**: Interactive 2D visualization
- **NumPy**: Numerical operations on embeddings

## Output Interpretation

### Semantic Distance Metrics

- **0.0 - 0.15**: Highly similar, likely exact matches or very close semantic meaning
- **0.15 - 0.40**: Related content, moderately similar
- **0.40 - 0.70**: Loosely related, some topical overlap
- **0.70 - 1.0**: Unrelated or orthogonal concepts

Lower distances indicate better semantic matches for your query.

## Notes

- ChromaDB collection is deleted and recreated on each run
- Output files are persistent and accumulate over time (organized by timestamp)
- t-SNE is non-deterministic unless using a fixed random seed (seed=42 used)
- Visualizations display automatically; close browser/Plotly to continue execution