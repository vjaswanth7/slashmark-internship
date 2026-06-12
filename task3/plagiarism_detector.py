import os
import re
import string
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime

import joblib
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except Exception:
    stopwords = None
    PorterStemmer = None


@dataclass
class Document:
    file_name: str
    raw_text: str
    clean_text: str


class PlagiarismDetector:
    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=8000)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8000)
        self.source_docs: List[Document] = []
        self.word_matrix = None
        self.char_matrix = None
        self.stemmer = PorterStemmer() if PorterStemmer else None
        self.stop_words = self._load_stopwords()

    def _load_stopwords(self):
        fallback = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "and", "or", "but", "if", "in", "on", "at", "to", "for", "from", "of",
            "by", "with", "as", "this", "that", "these", "those", "it", "its", "into"
        }
        try:
            return set(stopwords.words("english")) if stopwords else fallback
        except Exception:
            return fallback

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", " ", text)
        tokens = re.findall(r"\b[a-z]{2,}\b", text)
        tokens = [t for t in tokens if t not in self.stop_words]
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(tokens)

    def load_documents(self, folder_path: str) -> List[Document]:
        docs = []
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.lower().endswith(".txt"):
                path = os.path.join(folder_path, file_name)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                docs.append(Document(file_name, raw, self.normalize_text(raw)))

        if not docs:
            raise ValueError(f"No .txt files found in {folder_path}")

        return docs

    def fit(self, source_folder: str):
        self.source_docs = self.load_documents(source_folder)
        clean_texts = [doc.clean_text for doc in self.source_docs]
        self.word_matrix = self.word_vectorizer.fit_transform(clean_texts)
        self.char_matrix = self.char_vectorizer.fit_transform(clean_texts)
        return self

    def save(self, model_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self, model_path)

    @staticmethod
    def load(model_path: str):
        return joblib.load(model_path)

    def check_text(self, text: str) -> pd.DataFrame:
        if self.word_matrix is None or self.char_matrix is None:
            raise ValueError("Model is not trained. Run train_model.py first.")

        clean = self.normalize_text(text)
        word_vec = self.word_vectorizer.transform([clean])
        char_vec = self.char_vectorizer.transform([clean])

        word_scores = cosine_similarity(word_vec, self.word_matrix)[0]
        char_scores = cosine_similarity(char_vec, self.char_matrix)[0]

        rows = []
        for i, doc in enumerate(self.source_docs):
            fuzzy_score = fuzz.token_set_ratio(clean, doc.clean_text) / 100
            final_score = (
                0.50 * word_scores[i]
                + 0.30 * char_scores[i]
                + 0.20 * fuzzy_score
            )

            rows.append({
                "source_file": doc.file_name,
                "word_ngram_cosine": round(float(word_scores[i]), 4),
                "char_ngram_cosine": round(float(char_scores[i]), 4),
                "fuzzy_match": round(float(fuzzy_score), 4),
                "final_similarity": round(float(final_score), 4),
                "similarity_percent": round(float(final_score) * 100, 2),
                "status": "FLAGGED - Possible Plagiarism"
                if final_score >= self.threshold else "OK"
            })

        return pd.DataFrame(rows).sort_values("final_similarity", ascending=False)

    def check_file(self, submission_path: str) -> pd.DataFrame:
        with open(submission_path, "r", encoding="utf-8", errors="ignore") as f:
            return self.check_text(f.read())

    def generate_report(
        self,
        results: pd.DataFrame,
        output_dir: str = "reports"
    ) -> Tuple[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = os.path.join(output_dir, f"plagiarism_report_{timestamp}.csv")
        txt_path = os.path.join(output_dir, f"plagiarism_report_{timestamp}.txt")

        results.to_csv(csv_path, index=False)

        top = results.iloc[0]
        flagged = results[results["status"].str.contains("FLAGGED")]

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("AI Plagiarism Detection Report\n")
            f.write("=" * 35 + "\n\n")
            f.write(f"Threshold: {self.threshold * 100:.0f}%\n")
            f.write(f"Highest matching source: {top['source_file']}\n")
            f.write(f"Highest similarity: {top['similarity_percent']}%\n")
            f.write(
                f"Final decision: "
                f"{'Potential plagiarism found' if len(flagged) else 'No plagiarism above threshold'}\n\n"
            )
            f.write("Detailed Results:\n")
            f.write(results.to_string(index=False))

        return csv_path, txt_path
