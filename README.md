# Image Batch Studio

A local Streamlit app for generating images with Google Gemini (Nano Banana),
Black Forest Labs FLUX.2 and OpenAI GPT Image. Prepare prompts, generate batches,
and reopen saved runs to recover unfinished work. The interface is currently in Italian.

## Features

- Manual prompts with variants, or batches imported from DOCX, XLSX/XLSM and CSV.
- Editable column mapping, prompt preview and row validation.
- Reference images and a shared style prefix.
- Model-specific controls for dimensions, resolution and output options.
- Concurrent generation with independent retries for generation, polling, download and saving.
- Collision-safe filenames, a result gallery and ZIP downloads.
- Persistent history with original settings, prompts, references, errors and attempt counts.
- Resume pending work, retry selected failures and recover existing BFL requests.

## Requirements

Python **3.13** is the verified runtime. Windows is tested locally; GitHub Actions
is configured to run the test suite on Windows and Linux. macOS installation
commands are provided, but macOS has not been verified.

You need an API key for at least one provider you intend to use. Each user supplies
their own keys. API access, model availability, quotas and charges depend on the
provider and account. The app does not include model weights or API credits.

## Install and run

Clone this repository, open its directory in a terminal, then:

### Windows (PowerShell)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints.txt
Copy-Item .env.example .env
# Edit .env and add your API keys.
.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS / Linux

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -c constraints.txt
cp .env.example .env
# Edit .env and add your API keys.
.venv/bin/python -m streamlit run app.py
```

Run these commands from the repository root. Open `http://localhost:8501`.
`constraints.txt` pins the dependency closure of the verified environment rather
than exporting unrelated packages from the developer's environment.

## Configure providers

Edit `.env`; configure only the services you use:

```dotenv
GOOGLE_KEY=
BFL_API_KEY=
OPENAI_API_KEY=
```

Keys are read locally and are not stored in run manifests. Prompts and reference
images are sent to the selected provider when you generate images. Keep `.env`
private. This app is intended for local use; it has no user authentication or
per-user credential isolation for a publicly hosted deployment.

The model registry is in `config.py`. Provider APIs and preview model IDs may
change; the local tests simulate API responses and do not certify live model
availability. OpenAI uses its image edit endpoint when references are provided;
there is no mask editor or interactive editing canvas.

## Try a batch

1. Choose a model whose API key you configured.
2. Open the batch tab and upload [`examples/prompts.csv`](examples/prompts.csv).
3. Review the column mapping and extracted prompts.
4. Optionally add references and a style prefix.
5. Limit the batch to one prompt to check the result, then generate the full batch.

A small sample still makes real, potentially billable API calls.

## History and recovery

Each run stores `output/<run>/run.json`, generated images and a `references/`
subdirectory. The manifest preserves the original model, settings, complete prompts,
status, timestamps, metadata, errors and attempt counts. Keep the whole run directory
when moving or backing up your history. `output/` is excluded from Git.

In **Storico delle generazioni**, select a run:

- **Riprendi batch** starts pending jobs and recovers recorded BFL requests.
- **Riprova i falliti** retries selected failures and shows the number of new
  generations expected. Existing BFL requests are reused when available.
- **Rigenera con nuove richieste** explicitly submits new requests for uncertain
  outcomes or unrecoverable BFL jobs, potentially incurring additional charges.
- **Chiudi visualizzazione** closes the view without deleting history.

Completed, intact images are excluded from recovery. Checksums reconcile files
saved immediately before an interruption. Missing or modified results are reported.
Manifests are replaced atomically; an operating-system lock prevents simultaneous
execution of the same run by two sessions.

A timeout or interruption during synchronous generation, or before a BFL submission
is recorded, can leave the outcome unknown. These jobs are not automatically
resubmitted. BFL result URLs may expire; recovery is not guaranteed indefinitely.
Legacy output folders without a manifest can be browsed but cannot be resumed.

The configured attempt limit includes the first attempt. Generation is retried only
on HTTP 429; ambiguous timeouts and 5xx responses are not automatically resubmitted.
BFL reads retry transient failures on the same resource. Saving failures retry the
same bytes without generating another image. Existing files are never overwritten.

## Development and tests

```bash
python -m unittest discover -s tests -v
```

Use the Python executable inside your virtual environment. Tests use temporary
folders and simulated providers/HTTP responses: no API keys or billable calls.

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `config.py` | Model registry and capabilities |
| `prompt_files.py` | File parsing, validation and filename allocation |
| `history.py` | Run manifests and persistent recovery |
| `runner.py` | Batch execution and exclusive image saving |
| `providers/` | Provider implementations and retry policy |
| `tests/` | Regression and UI tests |
| `examples/` | Generic sample prompts |

## License

[MIT](LICENSE), copyright 2026 Pietro Segreto. This license covers the project code;
third-party services, dependencies and third-party assets retain their own terms.
See [CHANGELOG.md](CHANGELOG.md) for the initial release contents.
