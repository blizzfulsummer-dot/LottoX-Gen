# LottoX-Gen 🎱

**LottoX-Gen** is an AI-powered lottery number suggestion application built with Python and FastAPI. It provides a web-based interface for generating lottery number suggestions using the AI model running in the backend.

## ✨ Features

- 🎯 Generate lottery number suggestions.
- 🤖 Run the AI model through a Python backend.
- 🌐 Access the application through a web browser.
- 📱 Designed for a lottery suggestion experience that can be extended to mobile platforms.

## 🛠️ Requirements

Before running the application, make sure you have:

- Python 3.10 or a compatible version for your project's dependencies.
- `pip` or another Python package manager.
- The project dependencies installed.
- A modern web browser.

## 🚀 Getting Started

### 1. Clone or Download the Repository

Download the project or clone the repository, then navigate to the project directory.

```bash
cd LottoX-Gen
```

### 2. Install Dependencies

If the project includes a `requirements.txt` file, install the required packages:

```bash
pip install -r requirements.txt
```

If FastAPI and Uvicorn are not included in the project's dependencies, install them with:

```bash
pip install fastapi uvicorn
```

### 3. Start the Backend

Navigate to the backend directory containing `main.py`, if applicable, and start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

**What these options mean:**

- `main:app` — Loads the `app` FastAPI instance from `main.py`.
- `--host 0.0.0.0` — Makes the server listen on all network interfaces.
- `--port 8080` — Runs the application on port `8080`.
- `--reload` — Automatically restarts the server when Python code changes. This is intended for development.

### 4. Open the Application

Once the server has started, open your browser and visit:

- **Local access:** http://localhost:8080
- **Alternative local address:** http://127.0.0.1:8080
- **API documentation:** http://localhost:8080/docs

> **Note:** `0.0.0.0` is the address used to bind the server, not normally the address you enter in your browser. Use `localhost` or `127.0.0.1` for access from the same computer.

## 📂 Project Structure

A typical project structure may look like this:

```text
LottoX-Gen/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── ...                 # AI model and backend modules
├── index.html              # Web interface, if located here
├── script.js               # Frontend logic, if located here
└── README.md
```

*Adjust this structure to match the actual files and folders in your repository.*

## 🔧 Troubleshooting

**The `uvicorn` command is not recognized**

Install Uvicorn:

```bash
pip install uvicorn
```

Alternatively, start it through Python:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

**The server cannot find `main:app`**

Make sure you are running the command from the directory containing `main.py` and that the FastAPI application instance is named `app`.

**Port 8080 is already in use**

Run the server on another port:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8081 --reload
```

Then visit http://localhost:8081.

## ⚠️ Disclaimer

LottoX-Gen generates lottery number suggestions for informational and entertainment purposes only. Lottery outcomes are random, and AI-generated suggestions do not guarantee winning numbers or improve the odds of winning. Please play responsibly.

## 📄 License

Add your project's license here if you intend to distribute or publish the application.
