# SkillStride AI — Career Intelligence Dashboard

A Streamlit web application that maps your skills to real job market data, predicts your pay grade and industry fit, and identifies exactly which skills you need to land your target role — powered by **Sentence Transformers**, **Logistic Regression**, and **cosine similarity**.

---

## Features

- **Job Matching** — Top 3 job roles ranked by semantic similarity to your skills
- **Skill Gap Analysis** — See exactly which skills you have vs. what your target role requires
- **Readiness Score** — Percentage readiness for any target career role
- **Pay Grade Prediction** — ML model predicts your likely pay band from your skill set
- **Industry Prediction** — ML model predicts your best-fit industry
- **Dark Glassmorphism UI** — Professional dark-themed interface with animated cards

---

## How It Works

### Training Pipeline (Google Colab Notebook)

Dataset: [Jobs and Skills Mapping for Career Analysis](https://www.kaggle.com/datasets/emaadakhter/jobs-and-skills-mapping-for-career-analysis) via KaggleHub.

1. **Data Loading** — Reads job titles, required skills, pay grades, and industries from CSV
2. **Skill Parsing** — Splits concatenated title-case skill phrases using regex (e.g. `"ProblemSolvingLogicalReasoning"` → `["Problem Solving", "Logical Reasoning"]`)
3. **Job Aggregation** — Groups by job title, keeps top 10 most frequent skills per role
4. **Feature Engineering** — `MultiLabelBinarizer` converts skill lists to binary vectors
5. **Model A — Pay Grade** — Trains Logistic Regression, Random Forest, and LinearSVC; picks best by accuracy
6. **Model B — Industry** — Same pipeline; filters out industries with fewer than 20 samples
7. **Semantic Embeddings** — `SentenceTransformer('all-MiniLM-L6-v2')` encodes each job's skill set
8. **Export** — All models, encoders, embeddings, and metadata saved to `skill_gap_artifacts.pkl`

### Inference Pipeline (Streamlit App)

1. User selects skills and a target role from the sidebar
2. Skills are encoded via `SentenceTransformer` into an embedding vector
3. **Cosine similarity** ranks all job embeddings → top 3 displayed as match cards
4. Exact skill set of the target role is compared to the user's skills → matched / missing displayed
5. Pay grade and industry predicted from the binary skill vector using trained ML models

---

## Project Structure

```
skillstride/
├── app.py                    # Streamlit web application
├── skill_gap_artifacts.pkl   # Pretrained models, encoders, embeddings & job data
└── notebook.ipynb            # Training pipeline (Google Colab)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- `skill_gap_artifacts.pkl` (pre-trained, included in the project)

### Installation

```bash
git clone <your-repo-url>
cd skillstride
pip install streamlit pandas numpy scikit-learn sentence-transformers
```

### Running the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

> **Important:** `skill_gap_artifacts.pkl` must be in the same directory as `app.py`.

---

## Artifacts Saved in `.pkl`

| Key | Description |
|---|---|
| `best_clf_pg` | Best pay grade classifier |
| `best_clf_in` | Best industry classifier |
| `mlb` | MultiLabelBinarizer for pay grade model |
| `mlb_ind` | MultiLabelBinarizer for industry model |
| `le` | LabelEncoder for pay grade labels |
| `le_ind` | LabelEncoder for industry labels |
| `df_jobs` | Job title → required skills mapping |
| `job_embeddings` | Pre-computed sentence embeddings per job |
| `all_skills` | All unique skills available for selection |
| `all_job_titles` | All job titles available as targets |

---

## Retraining the Model

1. Open the notebook in Google Colab
2. Run all cells — dataset downloads automatically via KaggleHub
3. Download the generated `skill_gap_artifacts.pkl`
4. Place it next to `app.py` and restart the Streamlit app

---

## Dependencies

| Package | Purpose |
|---|---|
| streamlit | Web application framework |
| sentence-transformers | Semantic skill embeddings (`all-MiniLM-L6-v2`) |
| scikit-learn | ML models, encoders, cosine similarity |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib | Training visualizations (notebook only) |
| kagglehub | Dataset download (notebook only) |
| pickle | Artifact serialization |

---

## Notes

- Skills must be selected from the dropdown (sourced directly from the dataset) for accurate matching.
- The embedding model (`all-MiniLM-L6-v2`) is downloaded automatically on first run (~80 MB).
- All ML inference runs on CPU — no GPU required.
