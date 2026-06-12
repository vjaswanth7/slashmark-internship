from plagiarism_detector import PlagiarismDetector

SOURCE_FOLDER = "data/source_corpus"
MODEL_PATH = "models/tfidf_plagiarism_model.joblib"

if __name__ == "__main__":
    detector = PlagiarismDetector(threshold=0.60)
    detector.fit(SOURCE_FOLDER)
    detector.save(MODEL_PATH)
    print(f"Model trained successfully on {len(detector.source_docs)} source documents.")
    print(f"Saved model to: {MODEL_PATH}")
