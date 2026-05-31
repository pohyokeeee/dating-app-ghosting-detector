# Ghosted Predictor UI

## Requirements

* Python 3.10 or above
* Git
* VS Code (recommended)

## Project Setup

### 1. Clone the repository

```bash
git clone <repository-link>
cd Ghosted-Predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

#### Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)

```cmd
venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Make sure the model file (`flaml_best_model.pkl`) is located in the project directory.

Run:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

## Project Structure

```text
Ghosted-Predictor/
│
├── app.py
├── flaml_best_model.pkl
├── requirements.txt
└── README.md
```

## Notes

* Do not rename `flaml_best_model.pkl`.
* Ensure all required packages are installed before running the application.
* If you modify the UI, commit your changes to your own branch before creating a Pull Request.
