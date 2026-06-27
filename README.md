# AI Agent Journey: From Test Automation Engineer to AI Agent Builder

## What this project is

This repo documents a hands-on learning path from **test automation engineering** into **AI agent engineering**, built entirely with free tools (GitHub Codespaces, Claude API, ChromaDB). It's not a finished product — it's a structured set of small projects, each one teaching a core building block of how modern AI agents work, plus a real debugging log of every bug hit along the way.

If you're coming from a QA/testing background, this is meant to show that your instincts (break things, verify outputs, write test cases) map directly onto AI engineering — they just point at a different kind of system.

---

## The four things we built, in order

### 1. A basic chatbot (`chat.py`)
The simplest possible thing: take user input, send it to Claude, print the answer, loop until the user types `quit`.

###python

import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": user_input}]
    )
    print("Claude:", response.content[0].text)
```

**Why each line matters:**
- `anthropic.Anthropic(api_key=...)` — creates an authenticated connection object. No network call happens yet; this just sets up *how* future requests will be signed.
- `os.environ["ANTHROPIC_API_KEY"]` — reads the API key from an environment variable instead of hardcoding it in the file. This keeps secrets out of your code (and out of GitHub, where anyone could see them).
- `while True:` — Python scripts normally run once top-to-bottom and exit. Wrapping the conversation in `while True` keeps it alive so you can ask multiple questions in one session.
- `client.messages.create(...)` — the actual network call. Everything (`model`, `max_tokens`, `messages`) gets sent as JSON to Anthropic's API.
- `response.content[0].text` — Claude's reply comes back as a list of "content blocks" (it can mix text, tool requests, etc.). `[0].text` grabs the text from the first block.

**Goal of this step:** prove you can talk to an LLM from your own code, not just through a chat website.

---

### 2. A multi-tool agent (`agent.py`)
This is the actual core skill of "AI agent engineering": teaching Claude to **delegate** tasks to real functions (a calculator, a weather lookup) instead of guessing answers itself.

```python
import anthropic
import os
import requests

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# The "menu" of tools Claude is allowed to ask for.
# This is a description sent TO Claude — it is not code Claude runs.
tools = [
    {
        "name": "calculator",
        "description": "Perform a basic math calculation. Use this whenever the user asks a math question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression to evaluate, e.g. '47 * 12'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a city. Use this whenever the user asks about weather.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"}
            },
            "required": ["city"]
        }
    }
]

# These are the REAL functions. Claude never sees or runs this code —
# it only sees the descriptions above.
def calculator(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def get_weather(city):
    geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}").json()
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    weather = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    ).json()
    temp = weather["current_weather"]["temperature"]
    return f"{temp}°C in {city}"

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break

    messages = [{"role": "user", "content": user_input}]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        tools=tools,
        messages=messages
    )

    if response.stop_reason == "tool_use":
        # Claude can ask for MULTIPLE tools in one turn — collect all of them
        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
        tool_results = []

        for tool_use_block in tool_use_blocks:
            tool_name = tool_use_block.name
            tool_input = tool_use_block.input

            print(f"Claude wants to call: {tool_name} with {tool_input}")

            # THIS is where local code actually executes — Claude has no
            # involvement in this line at all.
            if tool_name == "calculator":
                result = calculator(tool_input["expression"])
            elif tool_name == "get_weather":
                result = get_weather(tool_input["city"])

            print(f"Local result: {result}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": result
            })

        # Send everything back: Claude's own request + our results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        final_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            tools=tools,
            messages=messages
        )

        print("Claude:", final_response.content[0].text)
    else:
        print("Claude:", response.content[0].text)
```

**The mental model that matters most here:**
"Tool" is just the generic name for "a function Claude can ask someone to run." Claude never executes anything. The full round trip is:

```
You ask → Claude decides it needs a tool, and asks for it (1st API call)
       → your code runs the real function locally
       → your code sends the result back (2nd API call)
       → Claude writes the final answer using that result
```

Two separate network calls per tool-using question, with your own code doing all the real computation in between.

---

### 3. A RAG pipeline on a real document (`rag.py`)
RAG = **Retrieval-Augmented Generation**. It's not a library, it's a pattern with three steps:

1. **Retrieval** — find the relevant pieces of a document for a given question
2. **Augmentation** — paste those pieces into the prompt sent to Claude
3. **Generation** — Claude writes an answer grounded in that retrieved text, instead of guessing from training data

We used **ChromaDB** as the retrieval engine: it converts text into embeddings (lists of numbers that capture meaning) and finds the closest match to a question by comparing those numbers.

```python
import chromadb
import anthropic
import os
import re

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

with open("CV_Automation_Engineer.txt", "r") as f:
    text = f.read()

# --- Chunking strategy (see "Bugs we hit" below for why this evolved) ---
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

# Job entries look like "Company (Location) – Title (Month YYYY – Month YYYY)"
job_header_pattern = re.compile(r'\([A-Za-z]+\.?\s*\d{4}\s*[–-]\s*[A-Za-z]*\.?\s*\d{4}\)')

chunks = []
current_job_chunk = None

for para in paragraphs:
    # Education gets a special label so it can be found by keyword OR meaning
    if "Master" in para and "Bachelor" in para:
        if current_job_chunk is not None:
            chunks.append(current_job_chunk.strip())
            current_job_chunk = None
        chunks.append("Education: " + para)
        continue

    if job_header_pattern.search(para):
        # A new job starts — flush whatever job we were building
        if current_job_chunk is not None:
            chunks.append(current_job_chunk.strip())
        current_job_chunk = para
    elif current_job_chunk is not None:
        # Still inside the same job — keep merging its paragraphs together
        current_job_chunk += "\n\n" + para
    else:
        chunks.append(para)

if current_job_chunk is not None:
    chunks.append(current_job_chunk.strip())

# --- Store in a PERSISTENT Chroma database (saved to disk, not memory) ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"Stored {len(chunks)} chunks in Chroma.")

while True:
    question = input("\nAsk about the document (or 'quit'): ")
    if question.lower() == "quit":
        break

    # Retrieval: find the 4 most relevant chunks for this question
    results = collection.query(query_texts=[question], n_results=4)
    context = "\n".join(results["documents"][0])

    # Augmentation: stuff those chunks into the prompt
    prompt = f"""Answer the question using only the information below.

Context:
{context}

Question: {question}
"""

    # Generation: Claude answers using only what was retrieved
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nClaude:", response.content[0].text)
```

**Why job-aware chunking instead of "split on every blank line":**
A resume has internal paragraph breaks within a single job (header → intro → bullet list). Splitting on *every* blank line shreds one job into 2-3 disconnected chunks, so a question like "what did they do at Atos UK" could retrieve the job header with no bullet points. The regex detects job headers by their date pattern and merges every paragraph in between into one chunk, until the next job header appears.

---

### 4. An automated eval suite (`eval_cases.py` + `run_evals.py`)
This is where a QA background becomes a real advantage. Instead of manually asking one question at a time and eyeballing the answer, we wrote a fixed list of test cases and a script that scores every answer automatically.

`eval_cases.py`:
```python
test_cases = [
    {
        "question": "What testing tools does this person have experience with?",
        "expected_keywords": ["Selenium", "QTP", "TestNG", "Jenkins"]
    },
    {
        "question": "What did they do at Atos UK?",
        "expected_keywords": ["UK Post Office", "Page object model", "Backoffice"]
    },
    {
        "question": "What did they study at university?",
        "expected_keywords": ["Data Analytics", "National College of Ireland"]
    },
    {
        "question": "What programming languages do they know?",
        "expected_keywords": ["Python", "Java", "SQL"]
    },
]
```

`run_evals.py` (the runner):
```python
import chromadb
import anthropic
import os
from eval_cases import test_cases

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

def ask_question(question):
    results = collection.query(query_texts=[question], n_results=4)
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
    return response.content[0].text

passed = 0
failed = 0

for i, case in enumerate(test_cases):
    answer = ask_question(case["question"])
    missing = [kw for kw in case["expected_keywords"] if kw.lower() not in answer.lower()]

    status = "PASS" if not missing else "FAIL"
    passed += (status == "PASS")
    failed += (status == "FAIL")

    print(f"\n[{i}] {status} — {case['question']}")
    print(f"   Answer: {answer[:150]}...")
    if missing:
        print(f"   Missing keywords: {missing}")

print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)}")
```

**Why "keyword match" instead of "exact answer match":** Claude phrases things differently every time, so checking for an exact string match is too brittle. Checking whether expected *keywords* appear anywhere in the answer is a simple, practical proxy for "is this answer actually grounded in the right information" — close to how a smoke test checks for key indicators rather than a byte-for-byte match.

---

## Real bugs we hit, and how we diagnosed + fixed each one

This is the part most tutorials skip — but it's the part that's actually closest to real engineering work.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `KeyError: 'ANTHROPIC_API_KEY'` | Codespaces secret wasn't authorized for this specific repo, or was added after the Codespace was created | Authorize the secret for the repo in GitHub settings, then restart/recreate the Codespace (secrets only load on creation) |
| 2 | `ModuleNotFoundError: No module named 'anthropic'` after install | Codespace was using a *different* Python interpreter path than the one `pip install` ran against | Install and run with the exact same interpreter path, or just use `python3` consistently everywhere instead of mixing full paths |
| 3 | Script only printed Claude's tool *request*, never a final answer | Old code stopped after the first API call — it never executed the local function or sent a follow-up request | Added the second half of the loop: run the local function, package the result as a `tool_result`, send a second API call |
| 4 | `NameError: name 'result' is not defined` | Dispatch logic (`if tool_name == "calculator"`) only had a branch for one tool; the other tool (`get_weather`) matched nothing | Added the missing `elif` branch for the second tool |
| 5 | Asking a math-only question silently did nothing (no error, no output) | An `if` block meant to be *inside* the outer "tool_use" check was accidentally at zero indentation, creating a second, disconnected `if/elif/else`. The calculator branch ended up containing only one line with nothing after it | Fixed indentation so the dispatch logic and everything after it (print, append, second API call) all live inside the same outer block |
| 6 | `anthropic.BadRequestError: tool_use ids were found without tool_result blocks` | When asking a question that needed *two* tools at once, the code only grabbed and answered the *first* `tool_use` block (`next(...)`), leaving the second one unanswered | Rewrote to loop over **all** `tool_use` blocks, collect a `tool_result` for each, and send them all back together in one message |
| 7 | Combined weather+math question only ever returned the weather answer | A `for` loop's body wasn't fully indented — only two lines (`tool_name =`, `tool_input =`) were actually inside the loop; everything else ran once, after the loop, using only the *last* tool's values | Indented the entire per-tool block (print, dispatch, append) so it runs once *per* tool call, not once total |
| 8 | `FileNotFoundError: knowledge.txt` | A heredoc command meant to create the file was run inside the Python REPL by accident, so it never actually created the file | Confirmed with `ls`, then re-ran the file-creation command directly in bash |
| 9 | `"file.txt" may be a binary file` when viewing with `cat`/`less` | A `.doc` file had been renamed to `.txt` without actually being re-exported as plain text, so it still contained binary formatting | Used the application's real **Export** function (not "Save As") to produce genuine plain UTF-8 text |
| 10 | Eval script said "the context provided appears to be empty" for every question | `chromadb.Client()` creates an **in-memory** database — every time a *new* Python process (`rag.py` vs `run_evals.py`) ran, it started with a completely empty collection | Switched to `chromadb.PersistentClient(path="./chroma_db")` in both scripts, so data is saved to disk and shared across runs |
| 11 | `chromadb.errors.InternalError: Collection already exists` | `create_collection(...)` fails if a collection with that name already exists on disk from a previous run | Switched to `get_or_create_collection(...)`, which is safe to call repeatedly |
| 12 | Eval question "what did they study at university" failed, even though retrieval looked "close" | The actual education chunk never contains the word "university" — but a *different* chunk (online certifications) literally contains the phrase "University of Michigan." The embedding model partially favored that literal word match over the semantically correct chunk | Prefixed the education chunk with an explicit label (`"Education: "`) so it carries an unambiguous keyword signal, not just a meaning signal |
| 13 | Eval question "what did they do at Atos UK" kept failing even after other fixes | Paragraph-based chunking (split on every blank line) split one job into 2–3 separate chunks (header / intro / bullets) because the original resume has internal blank lines within a single job entry | Replaced naive paragraph splitting with **job-aware chunking**: detect job headers by their date-range pattern and merge every paragraph until the next header into a single chunk |

**The two genuinely "AI-specific" bugs (12 and 13) are worth remembering as named failure categories**, because they show up in every real RAG system, not just this resume:
- **Lexical vs. semantic mismatch** — literal word overlap can outrank true meaning-based relevance
- **Chunk fragmentation** — splitting documents at the wrong boundaries scatters related information across multiple chunks, so none of them retrieve as a complete answer

---

## Final result

```
[0] PASS — What testing tools does this person have experience with?
[1] PASS — What did they do at Atos UK?
[2] PASS — What did they study at university?
[3] PASS — What programming languages do they know?

Results: 4 passed, 4 failed out of 4 → 4 passed, 0 failed out of 4
```

A working RAG agent over a real document, with an automated eval suite proving the retrieval and answers are accurate — not just "it looked right when I tried it once."

---

## What doesn't scale here (on purpose) — and what would replace it in production

This project intentionally takes shortcuts that wouldn't survive contact with thousands of real documents. Worth being honest about that:

- **Manual regex for job headers** → in production, you'd use overlapping sliding-window chunks (e.g. 500 characters with 100 characters of overlap) so boundary information is rarely fully lost, regardless of document structure.
- **Single embedding-based retrieval** → in production, you'd combine vector search with traditional keyword search (hybrid search) so literal and semantic matches are both covered automatically, for any document.
- **Manually labeling one chunk as "Education:"** → in production, structure-aware ingestion (parsing real headers/sections at extraction time) replaces one-off hardcoded fixes.
- **Editing one document by hand** → in production, you fix the *pipeline* once (better chunking, hybrid search, reranking) and validate the fix against a fixed eval set, so the improvement applies to every document automatically.

The eval suite is the part that actually scales: once built, it's reused every time the pipeline changes, to prove improvements generalize — not just that one test passed once.

---

## How to run this yourself

1. Clone this repo into a GitHub Codespace
2. Add your Anthropic API key as a Codespaces secret named `ANTHROPIC_API_KEY`, authorized for this repo
3. Install dependencies:
   ```bash
   python3 -m pip install anthropic requests chromadb
   ```
4. Run each script in order:
   ```bash
   python3 chat.py        # Week 1: basic chatbot
   python3 agent.py       # Week 2: multi-tool agent
   python3 rag.py         # Week 3: RAG over a document (type 'quit' to exit the prompt loop)
   python3 run_evals.py   # Week 4: automated eval suite
   ```

## What's next

- Add 5-10 more eval cases covering trickier edge cases
- Try a hybrid search approach (keyword + semantic) instead of the manual "Education:" label hack
- Rebuild the same pipeline using LangChain or LangGraph to compare how much boilerplate a framework removes
- Deploy as a live demo on Hugging Face Spaces