"""
FAQ Chatbot
===========

A command-line chatbot that answers user questions by matching them against
a collection of FAQs for a product ("Nimbus", a fictional cloud storage app).

Pipeline (matches the 4 required steps):
  1. Collect FAQs           -> loaded from faqs.json (question/answer pairs)
  2. Preprocess text        -> NLTK: lowercase, tokenize, remove stopwords/punct, lemmatize
  3. Match user questions   -> TF-IDF vectors + cosine similarity (scikit-learn)
  4. Display best answer    -> highest-scoring FAQ's answer, with a confidence score

Usage:
    python faq_chatbot.py
"""

import json
import string
import sys
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

FAQ_PATH = Path(__file__).parent / "faqs.json"
CONFIDENCE_THRESHOLD = 0.20  # below this, we admit we don't have a good answer


def ensure_nltk_data() -> None:
    """Download required NLTK resources the first time the script runs."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


# ---------------------------------------------------------------------------
# Step 1: Collect FAQs
# ---------------------------------------------------------------------------

def load_faqs(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 2: Preprocess the text
# ---------------------------------------------------------------------------

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.punct_table = str.maketrans("", "", string.punctuation)

    def clean(self, text: str) -> str:
        """Lowercase, tokenize, strip punctuation/stopwords, lemmatize."""
        text = text.translate(self.punct_table)
        tokens = word_tokenize(text.lower())
        tokens = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok.isalpha() and tok not in self.stop_words
        ]
        return " ".join(tokens)


# ---------------------------------------------------------------------------
# Step 3: Match user questions with the most similar FAQ (cosine similarity)
# ---------------------------------------------------------------------------

class FaqMatcher:
    def __init__(self, faqs: list[dict], preprocessor: Preprocessor):
        self.faqs = faqs
        self.preprocessor = preprocessor
        # Combine the question with its keyword tags so synonyms (e.g. "money
        # back" for "refund") still match well even though TF-IDF is a
        # bag-of-words technique with no built-in sense of meaning.
        corpus = [
            preprocessor.clean(f"{faq['question']} {faq.get('keywords', '')}")
            for faq in faqs
        ]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.faq_matrix = self.vectorizer.fit_transform(corpus)

    def best_match(self, user_query: str) -> tuple[dict | None, float]:
        cleaned = self.preprocessor.clean(user_query)
        if not cleaned.strip():
            return None, 0.0

        query_vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(query_vec, self.faq_matrix)[0]

        best_idx = scores.argmax()
        best_score = scores[best_idx]
        return self.faqs[best_idx], float(best_score)

    def top_matches(self, user_query: str, k: int = 3) -> list[tuple[dict, float]]:
        cleaned = self.preprocessor.clean(user_query)
        query_vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(query_vec, self.faq_matrix)[0]
        ranked = sorted(zip(self.faqs, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:k]


# ---------------------------------------------------------------------------
# Step 4: Display the best matching answer
# ---------------------------------------------------------------------------

def run_chat(matcher: FaqMatcher) -> None:
    print("=" * 60)
    print("  Nimbus FAQ Bot  (type 'quit' or 'exit' to leave, 'help' for topics)")
    print("=" * 60)

    categories = sorted({faq["category"] for faq in matcher.faqs})
    print("Topics I can help with:", ", ".join(categories))
    print()

    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Bye!")
            break

        if not user_query:
            continue
        if user_query.lower() in {"quit", "exit"}:
            print("Bot: Bye!")
            break
        if user_query.lower() == "help":
            print("Bot: Try asking things like 'How do I reset my password?' "
                  "or 'What plans do you offer?'")
            continue

        best_faq, score = matcher.best_match(user_query)

        if best_faq is None or score < CONFIDENCE_THRESHOLD:
            print(f"Bot: I'm not confident I have a good answer for that (score={score:.2f}).")
            print("     Here are a few FAQs that might be related:")
            for faq, s in matcher.top_matches(user_query, k=3):
                print(f"       - {faq['question']}  (score={s:.2f})")
        else:
            print(f"Bot: {best_faq['answer']}")
            print(f"     [matched: \"{best_faq['question']}\"  |  cosine similarity={score:.2f}]")
        print()


def main():
    ensure_nltk_data()
    faqs = load_faqs(FAQ_PATH)
    preprocessor = Preprocessor()
    matcher = FaqMatcher(faqs, preprocessor)
    run_chat(matcher)


if __name__ == "__main__":
    sys.exit(main())
