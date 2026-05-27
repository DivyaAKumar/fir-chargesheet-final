# FIR Chargesheet Generator

A Flask-based web application for managing FIRs (First Information Reports) and generating chargesheets using a reinforcement learning model.

## Prerequisites

- Docker
- Docker Compose

## Project Structure

```
.
├── app.py                 # Main Flask application
├── chargesheet.py         # Chargesheet generation logic
├── chargesheet_rl.py      # Reinforcement learning model
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── static/               # Static files
│   └── evidence/        # Evidence uploads
├── models/              # Trained model files
└── templates/           # HTML templates
```

## Building and Running

1. Clone the repository:
```bash
git clone <repository-url>
cd fir-chargesheet-generator
```

2. Build and start the containers:
```bash
docker-compose up --build
```

3. Access the application:
- Open your browser and navigate to `http://localhost:5000`
- Login with default credentials:
  - Username: police
  - Password: police123

## Development

To run the application in development mode:

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## Features

- User authentication for police officers
- FIR creation and management
- Evidence collection and management
- Automatic chargesheet generation using RL model
- Print functionality for chargesheets

## Default Login

- Username: police
- Password: police123
- Police Station: Central Police Station
- Designation: Inspector

## Notes

- The application uses SQLite as the database
- Evidence files are stored in the `static/evidence` directory
- Trained model files should be placed in the `models` directory
- The application runs on port 5000 by default 