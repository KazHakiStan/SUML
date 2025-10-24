# SUML Projects

---

## Getting Started

Follow these steps to set up and run project locally

### 1. Clone the repository

```bash
git clone https://github.com/KazHakiStan/SUML.git
cd SUML
```

### 2. Create and Activate a Virtual Environment (recommended)

Windows (PowerShell)
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the App

Once dependencies are installed, start the Streamlit app:
```bash
streamlit run .\app\app.py
```
or on Linux/macOS:
```bash
streamlit run ./app/app.py
```

This will launch the app in your default browser, typically at:
http://localhost:8501
