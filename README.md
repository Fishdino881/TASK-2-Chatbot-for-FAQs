Task 2: Chatbot for FAQs

**Files:** `faqs.json`, `faq_chatbot.py`, `chatbot_ui.html`

An FAQ chatbot for a fictional cloud storage product ("Nimbus"), built two ways: a Python/NLTK command-line version and a standalone browser chat UI. Both use the same technique — **TF-IDF vectors + cosine similarity** — to match a user's question to the closest FAQ.

### Files

| File | Purpose |
|---|---|
| `faqs.json` | The FAQ knowledge base — 18 Q&A pairs across 5 categories (Getting Started, Billing & Plans, Storage & Sync, Security & Privacy, Troubleshooting), each with extra `keywords` to catch synonyms. |
| `faq_chatbot.py` | Command-line chatbot. Uses **NLTK** to preprocess text (tokenize, remove stopwords/punctuation, lemmatize) and **scikit-learn** (`TfidfVectorizer` + `cosine_similarity`) to find the best-matching FAQ. |
| `chatbot_ui.html` | A standalone browser chat interface. Re-implements the same TF-IDF + cosine similarity pipeline in plain JavaScript, so it runs entirely client-side with no server or install required. |

### Running the Python version

```bash
pip install nltk scikit-learn
python faq_chatbot.py
```

The first run automatically downloads the small NLTK datasets it needs (`punkt`, `stopwords`, `wordnet`). Then just type questions at the `You:` prompt — type `help` for topic hints, or `quit` / `exit` to leave.

Example:

```
You: how do I get my money back
Bot: We offer a full refund within 14 days of your first payment on a new
     subscription. Contact support@nimbus.io with your account email to
     request one.
     [matched: "Do you offer refunds?"  |  cosine similarity=0.35]
```

If the best match's similarity score is below a confidence threshold (0.20), the bot admits it isn't sure and lists a few related FAQs instead of guessing.

### Running the web UI

Just open `chatbot_ui.html` in a browser. It has:
- A sidebar of topic categories you can click to auto-fill a sample question
- A chat window with a typing indicator
- A **match confidence meter** under each bot answer, showing the cosine similarity score — useful for seeing how the matching technique performs
- A fallback with suggested related questions when confidence is low

### Customizing the FAQs
Edit `faqs.json` (and, if you want the web UI to match, copy the same array into the `FAQS` constant near the top of `chatbot_ui.html`'s `<script>` block). Each entry looks like:

```json
{
  "id": 1,
  "category": "Getting Started",
  "question": "How do I create a Nimbus account?",
  "answer": "Go to nimbus.io/signup, enter your email and a password...",
  "keywords": "sign up register new account create"
}
```

The `keywords` field isn't shown to the user — it just gives the matcher extra vocabulary so questions phrased differently from the FAQ (e.g. "money back" instead of "refund") still match correctly.

---

## Quick start summary

| Want to... | Open / run |
|---|---|
| Translate text in the browser | `translator.html` |
| Chat with the FAQ bot in the browser | `chatbot_ui.html` |
| Run the FAQ bot from the command line (see the NLP pipeline in Python) | `python faq_chatbot.py` |
