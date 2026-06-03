## First time running the project?

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

#### Linux/macOS

```bash
source ./venv/bin/activate
```

#### Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the project

### Normal run

```bash
python ./main.py
```

### Benchmark run (test multiple seeds)

Use `--num-seeds` to test seeds from `1` to `N` and keep the best one.

```bash
.
```

The best benchmark result is saved to:

```text
melhor_seed.txt
```