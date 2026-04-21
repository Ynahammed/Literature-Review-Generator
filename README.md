# Literature Review Generator

Generate a structured literature review from research paper abstracts using:
- **Topic Modeling** (Latent Dirichlet Allocation via scikit-learn)
- **Semantic Similarity** (Sentence-BERT / TF-IDF fallback)
- **Summarization** (DistilBART via Transformers / extractive fallback)
- **Text Preprocessing** (spaCy lemmatization + stopword removal)

---

## Project Structure

```
literature_review_generator/
├── .vscode/
│   ├── launch.json        ← VS Code run/debug configs
│   └── settings.json      ← Python interpreter & formatter
├── data/
│   └── abstracts/
│       ├── papers.json                        ← JSON input (optional)
│       └── TextCNN_Sentence_Classification.txt← Example .txt abstract
├── outputs/
│   ├── literature_review.txt                  ← Generated review (auto-created)
│   └── papers_metadata.json                   ← Processed metadata
├── src/
│   └── literature_review.py                   ← Main pipeline
├── tests/
│   └── test_pipeline.py                       ← pytest unit tests
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Create & activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the generator
```bash
python src/literature_review.py
```

The review is printed to the console **and** saved to `outputs/literature_review.txt`.

---

## Input Modes

Edit the `load_abstracts(source=...)` call in `src/literature_review.py`:

| `source`   | Description |
|------------|-------------|
| `"sample"` | Built-in 6-paper NLP demo (default) |
| `"folder"` | One `.txt` file per abstract in `data/abstracts/` |
| `"json"`   | `data/abstracts/papers.json` — list of `{title, abstract}` objects |

### JSON format
```json
[
  { "title": "My Paper", "abstract": "Abstract text here…" }
]
```

### Folder format
Create one `.txt` file per paper. The filename becomes the paper title.

---

## Running Tests
```bash
pytest tests/ -v
```

---

## VS Code Integration

- Open the folder: `File → Open Folder → literature_review_generator/`
- Select the `.venv` interpreter (Ctrl+Shift+P → *Python: Select Interpreter*)
- Run via the **Run & Debug** panel → *Run Literature Review Generator*

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/literature_review.txt` | Full structured review (5 sections) |
| `outputs/papers_metadata.json`  | Papers list used in this run |

---

## Review Sections

1. **Thematic Overview** — LDA topics with keywords and mapped papers
2. **Individual Paper Summaries** — Per-paper theme, keywords, and abstract summary
3. **Semantic Relationships** — Similar paper pairs ranked by cosine similarity
4. **Synthesized Narrative** — Unified prose connecting all papers
5. **Research Gaps & Future Directions** — Automatically inferred open problems

---

## Dependencies

| Package | Purpose |
|---------|---------|
| spaCy | Tokenization, lemmatization, stopword removal |
| scikit-learn | LDA topic modeling, TF-IDF |
| sentence-transformers | Semantic embeddings (SBERT) |
| transformers | Abstractive summarization (DistilBART) |
| torch | Backend for transformer models |
| numpy | Matrix operations |
