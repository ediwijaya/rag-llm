# RAG with Weaviate — Learning Tasks

6 progressive tasks to prove end-to-end RAG capability. All run locally with Docker (Weaviate + Ollama).

```
[1] ──▶ [2] ──▶ [3] ──▶ [4] ──▶ [5] ──▶ [6]
 Data    Search   Control  Generate  Hybrid  Scale
```

---

## Task 1: Build Your Own Dataset

**Core Skill:** Data modeling for vector search — understanding what makes good vectorizable content.

### Concept

Vector databases add a critical question beyond traditional schema design: **"What text will be turned into a vector?"**

- Vectorize **descriptive, natural-language fields** (descriptions, summaries, reviews)
- Store but don't vectorize **structured fields** (IDs, prices, dates, categories) — use these as filters instead
- More text ≠ better vectors. A focused 2-sentence description often outperforms a 5-page document because the meaning is diluted

```
GOOD for vectorization:
  "A coming-of-age drama about a teenager navigating identity and family expectations in 1990s rural Japan"

BAD for vectorization:
  "Movie ID: 4821 | Released: 2003-04-15 | Runtime: 118 | Budget: 12000000"
```

### Task

Build a **recipe collection** with at least **10 recipes** from different cuisines. Each recipe object must have:
- `title` (string)
- `description` (string — 2-3 sentences describing the dish, its flavor, and origin)
- `cuisine` (string — e.g., "Italian", "Thai")
- `difficulty` (string — "easy", "medium", "hard")
- `ingredients` (string — comma-separated list of main ingredients)

Create `playground/weaviate/task1_dataset.ipynb` that:
1. Connects to local Weaviate
2. Deletes the collection if it already exists (for re-runnability)
3. Creates a `Recipe` collection with `text2vec_ollama`
4. Batch imports all 10+ recipes
5. Prints the count of imported objects to verify

### Rubric

| Criteria | Pass | Fail |
|---|---|---|
| **10+ diverse recipes** | At least 10 recipes across 3+ cuisines | Fewer than 10, or all same cuisine |
| **Rich descriptions** | Each description is 2-3 sentences with flavor/origin/context | One-word or purely factual descriptions ("chicken dish") |
| **Proper schema** | All 5 properties present on every object | Missing properties or inconsistent shape |
| **Idempotent script** | Deletes existing collection before creating — can run twice without error | Crashes on second run with "collection already exists" |
| **Batch import** | Uses `batch.fixed_size()` or `batch.dynamic()` | Inserts one-by-one without batching |
| **Verification** | Prints object count after import | No verification that import succeeded |

---

## Task 2: Semantic Search Precision

**Core Skill:** Tuning retrieval — distance thresholds, limits, filters, and understanding when semantic search fails.

### Concept

When you call `near_text("spicy Asian food")`, Weaviate:
1. Embeds your query into a vector
2. Computes **cosine distance** between your query vector and every stored vector
3. Returns the closest ones

```
Distance scale (cosine):
  0.0  ── identical meaning
  0.5  ── loosely related
  1.0  ── completely unrelated
  2.0  ── opposite meaning
```

**`limit`** — always returns N results, even if some are irrelevant.
**`auto_limit`** — uses clustering to return only the "natural" group of nearest results.
**Filters** — constrain results *before* vector ranking (e.g., only "Thai" cuisine, then rank by similarity).

### Task

Using your Task 1 recipe collection, create `playground/weaviate/task2_search.ipynb` with **5 queries**, each in its own cell:

1. **Basic near_text** — Search for a concept (e.g., "comfort food for winter") with `limit=3`. Print each result's title and its **distance** (`obj.metadata.distance`)
2. **Surprising match** — Find a query where the top result is semantically correct but wouldn't match with keyword search (e.g., "something to eat after gym" returning a high-protein recipe even though "gym" appears nowhere in the data)
3. **Filter + search** — Combine `near_text` with a `Filter.by_property("difficulty").equal("easy")` to find easy recipes matching a concept
4. **near_object** — Take the UUID of one result and use `near_object(obj_uuid)` to find "more like this"
5. **Failure case** — Show a query where semantic search gives a **wrong or irrelevant** result. Add a markdown cell explaining *why* it failed

### Rubric

| Criteria | Pass | Fail |
|---|---|---|
| **Distance printed** | Every query prints distance alongside results | Results shown without distance |
| **Surprising match works** | Demonstrates semantic understanding beyond keywords | Query and data share obvious keywords |
| **Filter applied correctly** | Uses `weaviate.classes.query.Filter` properly, results respect the filter | Filter ignored or hardcoded wrong |
| **near_object works** | Uses a real UUID from a previous query result | Hardcodes a fake UUID or skips |
| **Failure case analyzed** | Shows a bad result AND explains why vectors failed here | No failure shown, or no explanation |

---

## Task 3: Multi-Property Vectorization

**Core Skill:** Controlling *what* gets embedded — not all fields should become vectors.

### Concept

By default, Weaviate concatenates **all text properties** and vectorizes the combined string. This means structured fields like `cuisine: "Thai"` and `difficulty: "easy"` pollute the semantic vector.

Control this with **`source_properties`**:

```python
Configure.Vectors.text2vec_ollama(
    model="nomic-embed-text",
    source_properties=["description"],  # only embed this field
)
```

**When to include multiple properties:**
- Include `title` + `description` when titles carry semantic meaning
- Exclude `title` when titles are codes or names
- Never include fields with controlled vocabulary (difficulty, category) — use filters for those

### Task

Create `playground/weaviate/task3_vectorization.ipynb` that:

1. Creates **two collections** from the same recipe data:
   - `RecipeFullVec` — default vectorization (all text properties)
   - `RecipeDescOnly` — `source_properties=["description"]` only
2. Imports identical data into both
3. Runs the **same 3 queries** against both collections side-by-side
4. For each query, prints results from both collections with distances
5. Adds a **markdown cell per query** comparing results

**Suggested queries:** `"easy Thai"`, `"rich creamy sauce"`, `"Italian"`

### Rubric

| Criteria | Pass | Fail |
|---|---|---|
| **Two collections created** | Both exist with different `source_properties` config | Only one, or both configured identically |
| **Same data in both** | Identical recipe objects in both collections | Different data makes comparison invalid |
| **Side-by-side comparison** | Each query shows results from both collections | Results shown separately with no comparison |
| **Analysis per query** | Markdown cell explains which collection won and why | No analysis, or generic "both are similar" |
| **At least one clear difference** | Demonstrates at least one query where results meaningfully differ | All queries return identical rankings |

---

## Task 4: RAG Pipeline End-to-End

**Core Skill:** Combining retrieval + generation — prompt engineering for grounded answers.

### Concept

RAG is two phases: **retrieve** relevant context, then **generate** an answer from it.

**Two prompt modes in Weaviate:**

- **`grouped_task`** — All retrieved objects sent as one batch to LLM. Use for: summaries, comparisons, "pick the best one"
- **`single_prompt`** — LLM called **once per object** with a `{property}` template. Use for: transforming each result individually

```python
# grouped_task: one LLM call, all results as context
response = collection.generate.near_text(
    query="healthy dinner", limit=3,
    grouped_task="Compare these recipes and recommend the healthiest.",
)

# single_prompt: one LLM call PER result
response = collection.generate.near_text(
    query="healthy dinner", limit=3,
    single_prompt="Rewrite this recipe for nut allergies: {description}",
)
```

**Hallucination risk:** The LLM might invent facts not in context. Good RAG prompts say *"Only use the provided information"* or *"If the answer is not in the context, say so."*

### Task

Create `playground/weaviate/task4_rag.ipynb` that demonstrates:

1. **grouped_task** — Retrieve 3 recipes and ask LLM to *"Create a 3-course dinner menu, explaining why they pair well"*
2. **single_prompt** — Retrieve 3 recipes, template: *"Write a restaurant menu description for: {title}. Base it on: {description}"*
3. **Grounded Q&A** — Ask a factual question about your data. Verify the LLM's answer is correct
4. **Hallucination test** — Ask a question the context **cannot answer** (e.g., calorie count when not stored). Document whether the LLM hallucinated
5. **Collection-level generative config** — Bind llama3.2 to the collection at creation time via `generative_config=Configure.Generative.ollama(...)`. Show `.generate.near_text()` works without `generative_provider`

### Rubric

| Criteria | Pass | Fail |
|---|---|---|
| **grouped_task works** | Coherent multi-recipe response | Error, or response ignores context |
| **single_prompt works** | Each object has its own generated text using `{property}` templates | Same generic response for all |
| **Grounded Q&A verified** | LLM's answer verified against actual data | Answer taken at face value |
| **Hallucination documented** | Shows what LLM said when context insufficient, with analysis | Skipped or not analyzed |
| **Collection-level generative** | Collection created with `generative_config`, query works without provider | Still passing provider at query time |

---

## Task 5: Hybrid Search

**Core Skill:** Combining keyword (BM25) + vector search — knowing when each is better.

### Concept

Vector search finds **meaning**. Keyword search finds **exact words**. Hybrid combines both.

```python
response = collection.query.hybrid(
    query="Tom Yum",
    alpha=0.5,  # 0.0 = pure keyword, 1.0 = pure vector
    limit=3,
)
```

```
alpha = 0.0  ──── pure BM25 (keyword)
alpha = 0.5  ──── balanced blend
alpha = 1.0  ──── pure vector (same as near_text)
```

| Query Type | Best Method |
|---|---|
| Concept/vibe ("comfort food") | `near_text` (vector) |
| Exact name ("Pad Thai") | `bm25` (keyword) |
| Mix ("easy Pad Thai alternatives") | `hybrid` |

### Task

Create `playground/weaviate/task5_hybrid.ipynb` that:

1. **BM25 search** — `collection.query.bm25(query="...", limit=3)` for a query containing an exact recipe name. Print results with **score** (`obj.metadata.score`)
2. **Same query with near_text** — Compare which found the target recipe and at what distance
3. **Alpha sweep** — Same query with `alpha` values `[0.0, 0.25, 0.5, 0.75, 1.0]`. Show how results shift
4. **Find the sweet spot** — A query where hybrid outperforms both pure methods. Explain why
5. **RAG with hybrid** — Use `.generate.hybrid(...)` for generation with hybrid-retrieved context

### Rubric

| Criteria | Pass | Fail |
|---|---|---|
| **BM25 works** | Results ranked by keyword relevance with scores | Error or no scores |
| **Same query, both methods** | Explicit BM25 vs near_text comparison for identical query | Different queries used |
| **Alpha sweep** | All 5 alpha values tested, results shown | Fewer than 3 values |
| **Sweet spot identified** | One query where hybrid beats both, with explanation | No such query, or no explanation |
| **Hybrid RAG** | `.generate.hybrid()` used successfully | Only `.generate.near_text()` |

---

## Task 6: Multi-Collection RAG

**Core Skill:** Cross-collection retrieval — real-world RAG where data lives in multiple places.

### Concept

In production, data is rarely in one table. RAG gets powerful when you retrieve from **multiple sources** and combine context.

```
User: "What's a good Italian recipe that people loved?"

    ┌──────────┐         ┌──────────┐
    │ Recipes  │         │ Reviews  │
    └────┬─────┘         └────┬─────┘
         │ near_text          │ near_text
         ▼                    ▼
   [Carbonara]          ["Amazing! Best
    [Tiramisu]           pasta ever"]
         │                    │
         └────────┬───────────┘
                  ▼
           LLM receives BOTH
                  │
                  ▼
         "I recommend Carbonara —
          reviewers called it 'amazing'"
```

### Task

Create `playground/weaviate/task6_multi.ipynb` that:

1. **Create `Review` collection** with: `recipe_name`, `reviewer`, `rating` (int), `comment` (string). Import **15+ reviews** across your recipes
2. **Single-collection baseline** — Ask a question needing review data using only Recipe collection. Show the LLM **cannot answer properly**
3. **Multi-collection retrieval** — For the same question: retrieve from both collections, combine into one prompt, generate
4. **Compare responses** — Side-by-side: single-collection vs multi-collection answer with analysis
5. **Helper function** — Write `multi_rag(query, recipe_limit, review_limit)` encapsulating the pipeline

### Rubric

| Criteria | Pass | Fail |
|---|---|---|
| **Review collection created** | 15+ reviews across multiple recipes | Fewer than 15, or all same recipe |
| **Baseline shown** | LLM fails/hallucinates without review context | Skipped baseline |
| **Multi-collection retrieval** | Retrieves from both collections, combines into one prompt | Only one collection queried |
| **Side-by-side comparison** | Both responses shown with analysis | Only multi-collection shown |
| **Helper function** | Reusable `multi_rag()` with parameters | Inline code without encapsulation |

---

## Completion Checklist

```
[ ] Task 1 — playground/weaviate/task1_dataset.ipynb
[ ] Task 2 — playground/weaviate/task2_search.ipynb
[ ] Task 3 — playground/weaviate/task3_vectorization.ipynb
[ ] Task 4 — playground/weaviate/task4_rag.ipynb
[ ] Task 5 — playground/weaviate/task5_hybrid.ipynb
[ ] Task 6 — playground/weaviate/task6_multi.ipynb
```
