"""
Unit tests for the Literature Review Generator pipeline.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from literature_review import (
    TextPreprocessor,
    TopicModeler,
    SemanticSimilarity,
    Summarizer,
    LiteratureReviewGenerator,
    load_abstracts,
)


SAMPLE_ABSTRACTS = [
    "Neural networks use backpropagation to learn weights through gradient descent.",
    "Transformer models rely on self-attention mechanisms for sequence modeling.",
    "BERT pre-trains deep bidirectional representations on large text corpora.",
]
SAMPLE_TITLES = ["Paper A", "Paper B", "Paper C"]


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestTextPreprocessor:
    def setup_method(self):
        self.proc = TextPreprocessor()

    def test_preprocess_returns_string(self):
        result = self.proc.preprocess(SAMPLE_ABSTRACTS[0])
        assert isinstance(result, str)

    def test_preprocess_lowercased(self):
        result = self.proc.preprocess("Hello WORLD")
        assert result == result.lower()

    def test_extract_keywords_returns_list(self):
        kws = self.proc.extract_keywords(SAMPLE_ABSTRACTS[0], top_n=5)
        assert isinstance(kws, list)
        assert len(kws) <= 5


# ── Topic Modeling ────────────────────────────────────────────────────────────

class TestTopicModeler:
    def test_fit_returns_topic_words(self):
        modeler = TopicModeler(n_topics=2)
        processed = [" ".join(a.lower().split()) for a in SAMPLE_ABSTRACTS]
        topics = modeler.fit(processed)
        assert isinstance(topics, list)
        assert len(topics) == 2
        for topic in topics:
            assert isinstance(topic, list)

    def test_doc_topic_distribution_shape(self):
        import numpy as np
        modeler = TopicModeler(n_topics=2)
        processed = [" ".join(a.lower().split()) for a in SAMPLE_ABSTRACTS]
        modeler.fit(processed)
        dist = modeler.get_doc_topic_distribution(processed)
        assert dist.shape == (len(SAMPLE_ABSTRACTS), 2)
        # Each row should sum to ~1
        assert all(abs(row.sum() - 1.0) < 1e-5 for row in dist)


# ── Similarity ────────────────────────────────────────────────────────────────

class TestSemanticSimilarity:
    def test_similarity_matrix_shape(self):
        sim = SemanticSimilarity()
        matrix = sim.compute_similarity_matrix(SAMPLE_ABSTRACTS)
        assert matrix.shape == (3, 3)

    def test_self_similarity_is_one(self):
        sim = SemanticSimilarity()
        matrix = sim.compute_similarity_matrix(SAMPLE_ABSTRACTS)
        for i in range(3):
            assert abs(matrix[i, i] - 1.0) < 1e-4

    def test_find_similar_pairs_returns_list(self):
        sim = SemanticSimilarity()
        pairs = sim.find_similar_pairs(SAMPLE_ABSTRACTS, SAMPLE_TITLES, threshold=0.0)
        assert isinstance(pairs, list)
        assert all(len(p) == 3 for p in pairs)


# ── Summarizer ────────────────────────────────────────────────────────────────

class TestSummarizer:
    def test_extractive_fallback(self):
        s = Summarizer()
        # Force extractive
        result = s._extractive_summarize(SAMPLE_ABSTRACTS[0], n_sentences=1)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_output_is_string(self):
        s = Summarizer()
        result = s.summarize(SAMPLE_ABSTRACTS[1])
        assert isinstance(result, str)
        assert len(result) > 0


# ── Load abstracts ────────────────────────────────────────────────────────────

class TestLoadAbstracts:
    def test_sample_returns_list_of_dicts(self):
        papers = load_abstracts("sample")
        assert isinstance(papers, list)
        assert len(papers) > 0
        for p in papers:
            assert "title" in p and "abstract" in p


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_generate_returns_nonempty_string(self):
        papers = [
            {"title": t, "abstract": a}
            for t, a in zip(SAMPLE_TITLES, SAMPLE_ABSTRACTS)
        ]
        gen = LiteratureReviewGenerator(n_topics=2)
        review = gen.generate(papers)
        assert isinstance(review, str)
        assert "LITERATURE REVIEW" in review
        assert len(review) > 200
