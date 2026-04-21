"""
Flask Web Server for Literature Review Generator
Run: python app.py
Then open: http://127.0.0.1:5000
"""

import os
import sys
import json
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

# Make sure src/ is importable
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from literature_review import (
    LiteratureReviewGenerator,
    load_abstracts,
    TextPreprocessor,
    TopicModeler,
    SemanticSimilarity,
)

app = Flask(__name__, static_folder="web/static", template_folder="web")


# ── Serve frontend ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


# ── API: Generate Review ───────────────────────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        papers = data.get("papers", [])

        if not papers:
            papers = load_abstracts("sample")

        generator = LiteratureReviewGenerator(n_topics=min(3, len(papers)))
        abstracts  = [p["abstract"] for p in papers]
        titles     = [p["title"]    for p in papers]
        proc       = TextPreprocessor()
        processed  = [proc.preprocess(a) for a in abstracts]

        # Topics
        modeler     = TopicModeler(n_topics=min(3, len(papers)))
        topic_words = modeler.fit(processed)
        doc_topics  = modeler.get_doc_topic_distribution(processed)
        topic_labels = [modeler.label_topic(tw) for tw in topic_words]

        # Similarity
        sim = SemanticSimilarity()
        pairs = sim.find_similar_pairs(abstracts, titles, threshold=0.2)

        # Summaries & keywords
        from literature_review import Summarizer
        summarizer = Summarizer()
        summaries  = [summarizer.summarize(a) for a in abstracts]
        keywords   = [proc.extract_keywords(a, top_n=6) for a in abstracts]

        import numpy as np
        papers_out = []
        for i, p in enumerate(papers):
            papers_out.append({
                "title":    p["title"],
                "abstract": p["abstract"],
                "summary":  summaries[i],
                "keywords": keywords[i],
                "topic":    topic_labels[int(np.argmax(doc_topics[i]))],
                "topic_idx": int(np.argmax(doc_topics[i])),
            })

        # Full text review
        full_review = generator.generate(papers)

        return jsonify({
            "success":      True,
            "papers":       papers_out,
            "topics":       [{"label": l, "words": w} for l, w in zip(topic_labels, topic_words)],
            "similar_pairs": [{"paper1": p1, "paper2": p2, "score": round(s, 3)} for p1, p2, s in pairs[:6]],
            "full_review":  full_review,
            "count":        len(papers),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── API: Load sample abstracts ─────────────────────────────────────────────────

@app.route("/api/samples", methods=["GET"])
def samples():
    papers = load_abstracts("sample")
    return jsonify(papers)


if __name__ == "__main__":
    print("\n  Literature Review Generator — Web Server")
    print("  Open: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
