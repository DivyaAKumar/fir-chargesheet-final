# FIR Chargesheet Generator

A full-stack AI-powered web application that automates the generation of legal chargesheets from First Information Reports (FIRs) for law enforcement use.


---

## Overview

Officers enter a crime description and the system automatically predicts applicable Indian Penal Code (IPC) sections using a hybrid AI approach — combining BERT semantic embeddings with a Reinforcement Learning agent and a rule-based post-filter. All FIR events are cryptographically recorded on a local SHA-256 blockchain, ensuring tamper-proof audit trails.

---

## Tech Stack

- **Backend:** Python, Flask, SQLite
- **AI / ML:** PyTorch, HuggingFace Transformers (BERT), Reinforcement Learning (REINFORCE)
- **Blockchain:** SHA-256, Proof-of-Work
- **Frontend:** HTML, CSS, Bootstrap, Jinja2
- **DevOps:** Docker, Docker Compose

---

## Key Features

- Secure login and role-based access for police officers
- FIR creation, management, and status tracking
- Evidence upload and management (images and text)
- Automated IPC section prediction from unstructured crime descriptions
- Rule-based post-filter with 200+ domain-specific rules for legal accuracy
- Blockchain audit trail — every FIR event recorded as a tamper-proof block
- Visual blockchain explorer with real-time tamper detection
- Print-ready chargesheet generation

---

## Model Performance

Evaluated across 15 crime categories including domestic violence, murder, cyber crime, kidnapping, dacoity, rape, stalking, forgery, and bribery.

| Metric | Value |
|--------|-------|
| Precision | 0.85 |
| Recall | 0.76 |
| F1-Score | 0.79 |

Perfect F1 score of 1.00 achieved on domestic violence and stalking categories.

---

## Project Structure

```
.
├── app.py                   # Main Flask application
├── blockchain.py            # SHA-256 blockchain implementation
├── chargesheet.py           # Chargesheet generation logic
├── chargesheet_rl.py        # RL model and rule-based filter
├── requirements.txt         # Python dependencies
├── Dockerfile
├── docker-compose.yml
├── static/
│   └── evidence/            # Evidence uploads
├── models/
│   ├── chargesheet_model.pth
│   ├── chargesheet_env.pth
│   └── fir_blockchain.json  # Blockchain storage
└── templates/               # HTML templates
```

---

## Prerequisites

**Option A — Docker (Recommended)**
- Docker
- Docker Compose

**Option B — Local**
- Python 3.12+
- pip

---

## Installation and Running

### Using Docker

```bash
git clone <repository-url>
cd fir-chargesheet-generator
docker-compose up --build
```

Open `http://localhost:5000` in your browser.

### Local Development

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Train the model (first time only — takes 10-20 minutes)
python chargesheet_rl.py

# Run the application
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

> **Note:** Run `python chargesheet_rl.py` before starting the app for the first time. This trains and saves the model to the `models/` folder.

---

## Default Login

| Field | Value |
|-------|-------|
| Username | police |
| Password | police123 |
| Police Station | Central Police Station |
| Designation | Inspector |

---

## Notes

- Database: SQLite stored in `instance/fir_system.db`
- Evidence files stored in `static/evidence/`
- Blockchain stored in `models/fir_blockchain.json` — auto-created on first run
- Application runs on port 5000 by default

---

## Author

Divya A Kumar
