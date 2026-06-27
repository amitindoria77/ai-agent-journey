import chromadb
import anthropic
import os
from eval_cases import test_cases

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Reconnect to the same Chroma collection you already built
#chroma_client = chromadb.Client()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

def ask_question(question):
    results = collection.query(query_texts=[question], n_results=4)
    context = "\n".join(results["documents"][0])

    print(f"\n   [DEBUG] Retrieved chunks for '{question}':")
    for doc in results["documents"][0]:
        print(f"     - {doc[:100]}...")

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
    return response.content[0].text

# Run every test case and score it
passed = 0
failed = 0

for i, case in enumerate(test_cases):
    question = case["question"]
    expected_keywords = case["expected_keywords"]

    answer = ask_question(question)

    # Check how many expected keywords actually appear in the answer
    found = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer.lower()]

    if len(missing) == 0:
        status = "PASS"
        passed += 1
    else:
        status = "FAIL"
        failed += 1

    print(f"\n[{i}] {status} — {question}")
    print(f"   Answer: {answer[:150]}...")
    if missing:
        print(f"   Missing keywords: {missing}")

print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)}")


import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
print(client.list_collections())
