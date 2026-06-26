import chromadb
import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Step 1: Read the document
with open("knowledge.txt", "r") as f:
    text = f.read()

# Step 2: Chunk it — splitting by line is simple and works for our small test file
chunks = [line.strip() for line in text.split("\n") if line.strip()]

print(f"Loaded {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"  [{i}] {chunk}")

# Step 3: Set up Chroma and store chunks
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="knowledge_base")

# Add each chunk with a unique ID
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"\nStored {len(chunks)} chunks in Chroma.")

# Peek at the raw stored data, including embeddings
results = collection.get(
    ids=["chunk_0"],
    include=["embeddings", "documents"]
)

print("\nID:", results["ids"][0])
print("Document text:", results["documents"][0])
print("Embedding (first 10 numbers):", results["embeddings"][0][:10])
print("Embedding length:", len(results["embeddings"][0]))

# If you want to see ALL chunks at once, not just one:
# all_data = collection.get(include=["embeddings", "documents"])
# for i, doc in enumerate(all_data["documents"]):
#     print(f"\n{all_data['ids'][i]}: {doc}")
#     print(f"  Vector (first 5): {all_data['embeddings'][i][:5]}")

# Step 4: Ask a question and retrieve the most relevant chunks
#What's happening: Chroma takes your question, converts it into a vector the same way it converted your chunks, 
# then mathematically compares that vector against all 5 stored chunks to find which ones are closest in meaning. 
# You should see it correctly pull back the line about the 45-day return policy as the top match — even though your question used completely different words ("how long do I have" vs "45 days of purchase"). 
# That's the part that's actually doing real work here — semantic matching, not keyword search.

# question = "How long do I have to return something?"
# question = "Can I return something I bought on sale?"

"""
RAG = Retrieval-Augmented Generation. It's not a tool or library — it's a pattern, made of three steps:

Retrieval — find relevant information from somewhere (a document, database, etc.)
Augmentation — insert that retrieved information into the prompt you send to the LLM
Generation — the LLM writes an answer, using that inserted context

Where Chroma fits: Chroma is just a tool that helps with step 1 (Retrieval). It stores your chunks as vectors and finds the closest matches to a question. That's it — Chroma has never talked to Claude, has no idea what an LLM is, and doesn't generate any text. It's purely a search mechanism.
What you've built so far: only Retrieval. You asked Chroma a question, it handed back matching text chunks — and that's where your script currently stops. You're seeing the right sentences get pulled out, but no one has shown those sentences to Claude yet, and Claude hasn't written an answer using them.
What's still missing — Augmentation + Generation:
"""

# Augmentation: stuff the retrieved chunks into a prompt

while True:
    question = input("\nAsk about the document (or 'quit'): ")
    if question.lower() == "quit":
        break

    results = collection.query(query_texts=[question], n_results=2)
    context = "\n".join(results["documents"][0])

    prompt = f"""Answer the question using only the information below.

Context:
{context}

Question: {question}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nClaude:", response.content[0].text)