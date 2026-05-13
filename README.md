# Coffee Disease Intelligence

## Run on Windows

Install Python 3.12 from [python.org](https://www.python.org/downloads/windows/). During installation, tick **Add python.exe to PATH**.

## PowerShell

Open PowerShell in the project folder, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m flask --app wsgi run --debug
```

Open:

```text
http://127.0.0.1:5000
```

If PowerShell blocks the virtual environment activation, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m flask --app wsgi run --debug
```

## Command Prompt

Open Command Prompt in the project folder, then run:

```cmd
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m flask --app wsgi run --debug
```

Open:

```text
http://127.0.0.1:5000
```

## Optional Model Settings

To use a custom model path in PowerShell:

```powershell
$env:COFFEE_MODEL_PATH="C:\full\path\to\coffee_disease_efficientnetb0.keras"
$env:COFFEE_MODEL_INPUT_SIZE="224"
python -m flask --app wsgi run --debug
```

To use a custom model path in Command Prompt:

```cmd
set COFFEE_MODEL_PATH=C:\full\path\to\coffee_disease_efficientnetb0.keras
set COFFEE_MODEL_INPUT_SIZE=224
python -m flask --app wsgi run --debug
```
