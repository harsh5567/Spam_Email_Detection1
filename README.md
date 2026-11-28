# Spam / Not Spam Email Classifier

Email spam detection using TF-IDF and Logistic Regression (scikit-learn).

## Experiments and results

| # | Training data | Test data | Accuracy |
|---|---|---|---|
| 1 | `spam.csv` (synthetic-looking) | 20% split of `spam.csv` | 100% |
| 1b | `spam.csv` | `hard_test_emails.csv` (subject + text only) | 87.5% |
| 2 | Enron-Spam | 20% split of Enron-Spam | 98.69% |
| 3 | Enron-Spam | `hard_test_emails.csv` | 67.5% |

Experiment 2, spam class: precision 97.65%, recall 99.66%, F1 98.64%.

## Why 100% is misleading
In `spam.csv`, no non-empty sender or subject appears in both classes, and email texts repeat often.
The model can separate the classes very easily.

## Limitations
- `hard_test_emails.csv` has only 40 hand-written synthetic emails (written with AI help after inspecting `spam.csv`).
- The Enron model marked 11 of 20 normal emails in that set as spam.
- Enron emails are old and may not reflect current spam.

## Data
- Enron-Spam: https://huggingface.co/datasets/SetFit/enron_spam (loaded by URL, not stored in this repo)

## How to run
1. `pip install -r requirements.txt` (or open the notebook in Google Colab).
2. Place `spam.csv` and `hard_test_emails.csv` in `/content/` (Colab), or change the paths.
3. Run the notebook cells from top to bottom.
