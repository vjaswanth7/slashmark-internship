import argparse
import os

from plagiarism_detector import PlagiarismDetector

MODEL_PATH = "models/tfidf_plagiarism_model.joblib"
SOURCE_FOLDER = "data/source_corpus"


def main():
    parser = argparse.ArgumentParser(description="AI Plagiarism Detector using TF-IDF, cosine similarity, and fuzzy matching")
    parser.add_argument("--submission", required=True, help="Path to submission .txt file")
    parser.add_argument("--threshold", type=float, default=0.60, help="Flagging threshold, example: 0.60")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training a new model...")
        detector = PlagiarismDetector(threshold=args.threshold)
        detector.fit(SOURCE_FOLDER)
        detector.save(MODEL_PATH)
    else:
        detector = PlagiarismDetector.load(MODEL_PATH)
        detector.threshold = args.threshold

    results = detector.check_file(args.submission)
    csv_path, txt_path = detector.generate_report(results)

    print("\nTop matches:")
    print(results.head(10).to_string(index=False))
    print(f"\nCSV report saved: {csv_path}")
    print(f"Text report saved: {txt_path}")


if __name__ == "__main__":
    main()
