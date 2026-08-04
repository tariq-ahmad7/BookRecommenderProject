# Book Recommendation System

## Overview 

I built a project that uses semantic search to recommend books based on natural-language descriptions. Instead of searching by title, author, or genre, a user describes the kind of book they're in the mood for (e.g. "a story about forgiveness"), and the model retrieves books whose descriptions are the closest semantic match. Results can also be filtered by category and emotional tone.

## Technologies Used 
- Python
- PyTorch
- pandas
- NumPy
- LangChain
- OpenAI Embeddings
- Chroma
- Gradio
- Jupyter Notebook
- Render

## Features: 

- Natural-language book search ("a story about forgiveness")
- Category filtering (e.g. Fiction, Nonfiction)
- Emotional tone filtering (Happy, Surprising, Angry, Suspenseful, Sad)
- Results with book cover, title, author(s), and a brief description
  
## Approach

I built a semantic recommendation pipeline rather than a traditional keyword search. To get from raw book metadata to a working app, the model uses:

- Data cleaning to organize raw book metadata (titles, authors, descriptions, thumbnails) into a usable dataset
- Zero-shot classification to assign each book a simplified category
- Sentiment analysis to score each book's description across five emotional tones (joy, sadness, fear, anger, surprise)
- OpenAI embeddings to convert each book description into a vector representation
- A Chroma vector store to index those embeddings for fast similarity search
- A Gradio interface to take a user's query, retrieve the closest-matching books, apply category/tone filters, and display results

Important Note

Because recommendations are driven by semantic similarity rather than exact keyword matching, results are ranked by how closely a book's description matches the meaning of the query, not by how many words overlap. A query like "a story about forgiveness" can return books that never use the word "forgiveness" at all, as long as the underlying theme is a close semantic match.

Results

The pipeline processes book metadata through category classification and sentiment scoring, then indexes the resulting descriptions as embeddings in a Chroma vector store. The deployed app returns the top matching books for a given query, filtered by the user's selected category and tone.
