python -m ruff check src tests app.py 2>$null; python -m black --check src tests app.py 2>$null; Write-Host "lint done (instala ruff/black para chequeo real)"
