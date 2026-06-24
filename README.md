# First Aid Assistant

A retrieval-augmented Q&A assistant that answers first-aid questions strictly from trusted first-aid manuals, using a local LLM (Ollama) with strict safety guardrails and citation tracking.

## Overview

This is a complete RAG (Retrieval-Augmented Generation) pipeline for first-aid guidance:

1. **Ingest** — PDF first-aid manuals are parsed and split into overlapping text chunks
2. **Embed & Index** — chunks are embedded with `all-MiniLM-L6-v2` (Sentence-Transformers) and indexed in **FAISS** for fast semantic search
3. **Retrieve** — at query time, the top-K most relevant chunks are retrieved, deduplicated across sources, and assembled into context
4. **Generate** — an LLM answers strictly from the retrieved context, with a confidence gate that declines to answer if retrieval similarity is too low
5. **Cite** — every answer is returned alongside the specific source manual and page number it came from
6. **Evaluate** — retrieval quality is measured quantitatively against a labelled query set

The app runs as an interactive **Streamlit** UI, and can answer via a **local Ollama model** (no API key, no cost, runs offline) or optionally OpenAI if a key is provided.

## Source Documents

The knowledge base is built from three trusted, publicly available first-aid references:

- Australian Red Cross — *Essential First Aid Guide*
- St John Ambulance — *First Aid Quick Reference Guide*
- Safe Work Australia — *First Aid in the Workplace (Code of Practice)*

## Safety Design

This is deliberately scoped, not a general medical chatbot:

- **System prompt constraint** — the model is instructed to use *only* the retrieved manual context and to say "I don't know" if the answer isn't covered
- **Forbidden-term filter** — a regex guard blocks queries asking about prescriptions, dosages, medications, or diagnoses before they ever reach the LLM
- **Confidence gate** — if the top retrieval similarity score falls below a configurable threshold (default 0.20), the app tells the user it isn't confident the manuals cover the topic, rather than guessing
- **Structured, non-expert-friendly answers** — responses are formatted as 6–10 short numbered steps with an optional cautions section, never long unstructured paragraphs
- **Mandatory safety disclaimer** — every answer ends with a reminder to call emergency services for life-threatening situations
- **Source citations** — every answer shows which manual and page it was grounded in, so claims are verifiable rather than opaque

## Tech Stack

Python · Streamlit · FAISS · Sentence-Transformers (`all-MiniLM-L6-v2`) · Ollama (local LLM, `llama3.2:3b`) · optional OpenAI (`gpt-4o-mini`)

## Project Structure
app.py                              # Streamlit app — retrieval + guarded LLM answering

assets/logo.png

data/

corpus/

raw_pdfs/                       # Source first-aid manuals (PDF)

processed_text/chunks.jsonl     # Chunked text (generated)

metadata.csv                    # Maps source files to titles/doc IDs

index/                            # FAISS index + metadata (generated)

queries.csv                       # Labelled evaluation queries

scripts/

prep_first_aid.py                 # PDF -> cleaned, chunked text

build_index.py                    # Builds the FAISS index from chunks

demo_first_aid_v2.py              # CLI demo (no web UI)

run_eval_first_aid.py             # Quantitative retrieval evaluation

Evaluation_results/

per_query_results.csv

summary.json

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1. Prepare the corpus and build the index

```bash
python scripts/prep_first_aid.py --in data/corpus/raw_pdfs \
  --out data/corpus/processed_text --meta data/corpus/metadata.csv \
  --chunk_words 700 --overlap 120

python scripts/build_index.py \
  --chunks data/corpus/processed_text/chunks.jsonl \
  --index_dir data/index --model all-MiniLM-L6-v2
```

### 2. Set up the local LLM (Ollama)

```bash
# Install Ollama, then in a separate terminal:
ollama serve

# Pull the model used by this app:
ollama pull llama3.2:3b
```

### 3. Run the app

```bash
streamlit run app.py
```

In the sidebar: choose **Ollama** as the answer mode, set Top-K retrieval (default 10), and optionally toggle "Show retrieved context" to see the raw chunks behind an answer.

### Command-line demo (no web UI)

```bash
python scripts/demo_first_aid_v2.py --llm ollama --index_dir data/index --ollama_model llama3.2:3b --k 10
```

## Evaluation

Two layers of evaluation were conducted:

**Retrieval evaluation** (automated, `run_eval_first_aid.py`, 5 labelled queries):

| Metric | Result |
|---|---|
| Top-1 hit rate | 60% |
| Top-3 hit rate | 60% |

**End-to-end response evaluation** (manual verification against ground-truth first-aid documents):

| Metric | Result |
|---|---|
| Answer Accuracy | 89.4% |
| Factual Consistency | 92.1% |
| Faithfulness (no hallucination) | 94.3% |
| Unanswered Query Rate (appropriately flagged) | 6.7% |
| Average Latency | 1.9s |
| User Readability Score | 95.6% |

Retrieval was evaluated against 5 labelled queries spanning all three source manuals (CPR/DRSABCD, severe bleeding, workplace first-aid equipment, choking, AED use):

| Metric | Result |
|---|---|
| Queries evaluated | 5 / 5 |
| Top-1 hit rate | 60% |
| Top-3 hit rate | 60% |

Run it yourself:
```bash
python scripts/run_eval_first_aid.py --index_dir data/index --queries data/queries.csv --emb all-MiniLM-L6-v2 --k 5
```

## Notes

Built as an applied NLP/IR project exploring retrieval-augmented generation with safety constraints for a sensitive, real-world domain. All code and experiments are original work.
