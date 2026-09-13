"""Parsing flessibile di file di prompt (.docx, .xlsx, .csv).

Il caso ideale è una tabella a due colonne (nome immagine | prompt), ma il
parser non lo dà per scontato:

- estrae tutte le tabelle del documento (o i fogli dell'xlsx) e lascia
  scegliere quale usare;
- prova a riconoscere automaticamente la riga di intestazione e quali colonne
  contengono nome e prompt (per keyword nell'header o, in mancanza, per
  lunghezza media del testo);
- la mappatura resta comunque modificabile dalla UI, incluse più colonne
  prompt nella stessa tabella;
- in fase di costruzione dei job valida riga per riga: prompt vuoti scartati,
  nomi mancanti generati, duplicati rinominati, caratteri non validi per i
  nomi file rimossi. Ogni anomalia è riportata con il numero di riga.

Se una cella prompt inizia con un identificativo tipo "1.2 " o "3)" e non è
stata indicata una colonna nome, l'identificativo viene usato come nome file
(compatibile con il formato del vecchio lista_prompt.docx).
"""
import csv
import io
import re
from dataclasses import dataclass, field

import openpyxl
from docx import Document

NAME_KEYWORDS = ("nome", "name", "id", "file", "titolo", "title", "codice", "img", "immagine", "image")
PROMPT_KEYWORDS = ("prompt", "descrizione", "description", "testo", "text", "scena", "scene")

# Identificativo inline a inizio cella: "1.2 testo", "3 - testo", "4) testo"
_INLINE_ID_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[\s\)\.\-:]+(\S.*)$", re.DOTALL)
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class ParsedTable:
    label: str
    rows: list[list[str]]


@dataclass
class PromptJob:
    name: str
    prompt: str
    source_row: int  # riga 1-based nella tabella originale


@dataclass
class RowIssue:
    row: int
    message: str
    skipped: bool = False


@dataclass
class ColumnGuess:
    has_header: bool
    name_col: int | None
    prompt_cols: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Estrazione tabelle
# ---------------------------------------------------------------------------

def parse_file(filename: str, data: bytes) -> list[ParsedTable]:
    """Restituisce tutte le tabelle trovate nel file."""
    lower = filename.lower()
    if lower.endswith(".docx"):
        return _parse_docx(data)
    if lower.endswith((".xlsx", ".xlsm")):
        return _parse_xlsx(data)
    if lower.endswith(".csv"):
        return _parse_csv(data)
    raise ValueError(f"Formato non supportato: {filename} (attesi .docx, .xlsx, .csv)")


def _parse_docx(data: bytes) -> list[ParsedTable]:
    doc = Document(io.BytesIO(data))
    if not doc.tables:
        raise ValueError("Il documento .docx non contiene tabelle.")
    tables = []
    for i, table in enumerate(doc.tables):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        rows = [r for r in rows if any(c for c in r)]
        if rows:
            tables.append(ParsedTable(label=f"Tabella {i + 1} ({len(rows)} righe)", rows=rows))
    if not tables:
        raise ValueError("Le tabelle del documento sono vuote.")
    return tables


def _parse_xlsx(data: bytes) -> list[ParsedTable]:
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    tables = []
    for sheet in wb.worksheets:
        rows = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if v is None else str(v).strip() for v in row]
            if any(cells):
                rows.append(cells)
        if rows:
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            tables.append(ParsedTable(label=f"Foglio '{sheet.title}' ({len(rows)} righe)", rows=rows))
    wb.close()
    if not tables:
        raise ValueError("Il file Excel non contiene righe con dati.")
    return tables


def _parse_csv(data: bytes) -> list[ParsedTable]:
    text = data.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [
        [c.strip() for c in row]
        for row in csv.reader(io.StringIO(text), dialect)
        if any(c.strip() for c in row)
    ]
    if not rows:
        raise ValueError("Il file CSV è vuoto.")
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return [ParsedTable(label=f"CSV ({len(rows)} righe)", rows=rows)]


# ---------------------------------------------------------------------------
# Euristiche su header e colonne
# ---------------------------------------------------------------------------

def guess_columns(rows: list[list[str]]) -> ColumnGuess:
    """Indovina header e ruolo delle colonne; la UI permette di correggere."""
    n_cols = max(len(r) for r in rows)
    header = [rows[0][c].lower().strip() if c < len(rows[0]) else "" for c in range(n_cols)]

    # Le keyword valgono solo su celle corte: una cella di intestazione non è
    # un paragrafo (evita falsi positivi tipo "quotidiano" che contiene "id").
    def keyword_col(keywords: tuple[str, ...], exclude: set[int]) -> int | None:
        for i, cell in enumerate(header):
            if i in exclude or not cell or len(cell) > 40:
                continue
            if any(k in cell for k in keywords):
                return i
        return None

    prompt_col = keyword_col(PROMPT_KEYWORDS, set())
    name_col = keyword_col(NAME_KEYWORDS, {prompt_col} if prompt_col is not None else set())
    has_header = prompt_col is not None or name_col is not None

    body = rows[1:] if has_header and len(rows) > 1 else rows

    def avg_len(col: int) -> float:
        return sum(len(r[col]) if col < len(r) else 0 for r in body) / max(1, len(body))

    if prompt_col is None:
        # Senza header riconoscibile: la colonna prompt è quella con il testo
        # mediamente più lungo nelle righe dati.
        prompt_col = max(range(n_cols), key=avg_len)

    if name_col is None:
        # Prima colonna non-prompt con contenuto corto e non vuoto.
        for c in range(n_cols):
            if c != prompt_col and 0 < avg_len(c) <= 40:
                name_col = c
                break

    if name_col == prompt_col:
        name_col = None

    return ColumnGuess(has_header=has_header, name_col=name_col, prompt_cols=[prompt_col])


# ---------------------------------------------------------------------------
# Costruzione e validazione dei job
# ---------------------------------------------------------------------------

def build_jobs(
    rows: list[list[str]],
    *,
    has_header: bool,
    name_col: int | None,
    prompt_cols: list[int],
    min_prompt_length: int = 3,
) -> tuple[list[PromptJob], list[RowIssue]]:
    """Trasforma la tabella in job validati (nome file, prompt)."""
    if not prompt_cols:
        raise ValueError("Seleziona almeno una colonna prompt.")

    jobs: list[PromptJob] = []
    issues: list[RowIssue] = []
    used_names: set[str] = set()
    data_rows = rows[1:] if has_header else rows

    for offset, row in enumerate(data_rows):
        row_number = offset + (2 if has_header else 1)
        for col in prompt_cols:
            cell = row[col].strip() if col < len(row) else ""
            name = row[name_col].strip() if name_col is not None and name_col < len(row) else ""

            if not cell:
                if name or any(c.strip() for c in row):
                    issues.append(RowIssue(row_number, f"colonna {col + 1}: prompt vuoto, riga scartata.", skipped=True))
                continue

            prompt = cell
            if not name:
                m = _INLINE_ID_RE.match(cell)
                if m and len(m.group(1)) <= 12:
                    name, prompt = m.group(1), m.group(2).strip()

            if len(prompt) < min_prompt_length:
                issues.append(RowIssue(row_number, f"colonna {col + 1}: prompt troppo corto ({prompt!r}), riga scartata.", skipped=True))
                continue

            if not name:
                name = f"img_{len(jobs) + 1:03d}"
                issues.append(RowIssue(row_number, f"nome mancante, assegnato '{name}'."))

            clean = sanitize_filename(name)
            if clean != name:
                issues.append(RowIssue(row_number, f"nome '{name}' normalizzato in '{clean}'."))
            name = clean

            renamed = unique_filename(name, used_names)
            if renamed != name:
                issues.append(RowIssue(row_number, f"nome duplicato '{name}', rinominato in '{renamed}'."))
                name = renamed

            jobs.append(PromptJob(name=name, prompt=prompt, source_row=row_number))

    return jobs, issues


def sanitize_filename(name: str) -> str:
    clean = _INVALID_FILENAME_CHARS.sub("_", name).strip(" .")
    clean = re.sub(r"\s+", " ", clean)
    clean = clean[:100].rstrip(" .") or "img"
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³]|CONIN\$|CONOUT\$", clean.split(".")[0], re.IGNORECASE):
        clean = "_" + clean
    return clean


def unique_filename(name: str, used_names: set[str]) -> str:
    """Riserva un nome, confrontando senza distinzione maiuscole/minuscole."""
    base = sanitize_filename(name)
    candidate = base
    suffix = 2
    while candidate.casefold() in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate.casefold())
    return candidate
