"""UI Streamlit per la generazione di immagini con Nano Banana (Gemini) e FLUX.2 (BFL).

Avvio:  streamlit run app.py
"""
import io
import os
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from image_batch_studio.config import BFL, GEMINI, OPENAI, MODELS
from image_batch_studio.prompt_files import PromptJob, build_jobs, guess_columns, parse_file
from image_batch_studio.providers import GenerationParams, create_provider
from image_batch_studio.providers.bfl_provider import dims_for_aspect
from image_batch_studio.providers.openai_provider import dims_for_aspect as openai_dims_for_aspect
from image_batch_studio.providers.openai_costs import estimate_output_cost
from image_batch_studio.history import create_run, load_run, run_saved, effective_status, local_path, is_running

OUTPUT_DIR = Path("output")

st.set_page_config(page_title="Image Generator — Nano Banana & FLUX.2", page_icon="🎨", layout="wide")

load_dotenv()
API_KEYS = {
    GEMINI: os.getenv("GOOGLE_KEY"),
    BFL: os.getenv("BFL_API_KEY"),
    OPENAI: os.getenv("OPENAI_API_KEY"),
}
ENV_VAR_BY_PROVIDER = {GEMINI: "GOOGLE_KEY", BFL: "BFL_API_KEY", OPENAI: "OPENAI_API_KEY"}
PROVIDER_NAMES = {GEMINI: "Google", BFL: "Black Forest Labs", OPENAI: "OpenAI"}

# ---------------------------------------------------------------------------
# Sidebar: modello e parametri
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Modello e parametri")

    spec = MODELS[st.selectbox(
        "Modello",
        options=list(MODELS),
        format_func=lambda k: MODELS[k].label,
    )]
    if spec.notes:
        st.caption(spec.notes)

    key = API_KEYS.get(spec.provider)
    if not key or key.lower().startswith("dummy"):
        st.error(
            f"Chiave API {PROVIDER_NAMES[spec.provider]} mancante o segnaposto: imposta "
            f"{ENV_VAR_BY_PROVIDER[spec.provider]} in .env. "
            "Le generazioni con questo modello falliranno."
        )

    aspect_ratio = st.selectbox("Aspect ratio", spec.aspect_ratios, index=spec.aspect_ratios.index("16:9") if "16:9" in spec.aspect_ratios else 0)
    resolution = st.selectbox(
        "Risoluzione" if spec.provider == GEMINI else "Risoluzione (megapixel totali)",
        spec.resolutions,
        index=spec.resolutions.index(spec.default_resolution),
    )
    if spec.provider == BFL:
        w, h = dims_for_aspect(aspect_ratio, resolution)
        st.caption(f"Dimensioni effettive inviate all'API: **{w} × {h} px**")
    elif spec.provider == OPENAI:
        w, h = openai_dims_for_aspect(aspect_ratio, resolution)
        st.caption(f"`size` inviata all'API: **{w} × {h} px**")

    params_kwargs = {"aspect_ratio": aspect_ratio, "resolution": resolution}

    if spec.thinking_levels:
        thinking = st.selectbox(
            "Livello di ragionamento", ["predefinito", *spec.thinking_levels],
            help="Più ragionamento migliora composizioni complesse ma aumenta tempi e token fatturati. "
                 "`predefinito` lascia decidere al modello.",
        )
        if thinking != "predefinito":
            params_kwargs["thinking_level"] = thinking

    if spec.provider == BFL:
        st.subheader("Parametri FLUX.2")
        seed_text = st.text_input("Seed (vuoto = casuale)", value="", help="Intero per generazioni riproducibili.")
        if seed_text.strip():
            try:
                params_kwargs["seed"] = int(seed_text)
            except ValueError:
                st.warning("Seed non valido, verrà ignorato.")
        params_kwargs["prompt_upsampling"] = st.toggle(
            "Prompt upsampling", value=False,
            help="L'API arricchisce automaticamente il prompt prima della generazione.",
        )
        params_kwargs["safety_tolerance"] = st.slider(
            "Safety tolerance", 0, 5, 2,
            help="0 = moderazione più severa, 5 = più permissiva.",
        )
        params_kwargs["output_format"] = st.selectbox("Formato output", ["png", "jpeg", "webp"])
        if spec.supports_guidance_steps:
            params_kwargs["guidance"] = st.slider("Guidance", 1.5, 10.0, 5.0, 0.5)
            params_kwargs["steps"] = st.slider("Steps", 1, 50, 50)

    if spec.provider == OPENAI:
        st.subheader(f"Parametri {spec.label}")
        params_kwargs["quality"] = st.selectbox(
            "Qualità", list(spec.qualities),
            help="`auto` lascia scegliere al modello; livelli superiori richiedono più tempo e token.",
        )
        params_kwargs["output_format"] = st.selectbox("Formato output", ["png", "jpeg", "webp"])
        if params_kwargs["output_format"] in ("jpeg", "webp"):
            params_kwargs["output_compression"] = st.slider("Compressione", 0, 100, 100)
        params_kwargs["background"] = st.selectbox(
            "Sfondo", ["auto", "transparent", "opaque"],
            help="`transparent` richiede formato png o webp.",
        )
        if params_kwargs["background"] == "transparent" and params_kwargs["output_format"] == "jpeg":
            st.warning("Lo sfondo trasparente non è supportato con jpeg: usa png o webp.")
        params_kwargs["moderation"] = st.selectbox(
            "Moderazione", ["auto", "low"],
            help="`low` applica filtri meno restrittivi.",
        )

    st.subheader("Esecuzione")
    concurrency = st.slider("Generazioni in parallelo", 1, 16, 4)
    max_retries = st.slider("Tentativi per fase", 1, 5, 3,
                            help="Retry separati per generazione, polling, download e salvataggio. Le generazioni con esito incerto non vengono reinviate automaticamente.")

params = GenerationParams(**params_kwargs)


# ---------------------------------------------------------------------------
# Area principale
# ---------------------------------------------------------------------------

st.title("🎨 Image Generator")

with st.expander(f"💡 Consigli di prompting", expanded=False):
    st.markdown(spec.prompting_hint)

col_ref, col_prefix = st.columns([1, 1])

with col_ref:
    st.subheader("🖼️ Immagini di riferimento")
    uploaded_refs = st.file_uploader(
        f"Carica fino a {spec.max_reference_images} immagini (stile/soggetto)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="ref_uploader",
    )
    if spec.provider == OPENAI:
        st.caption(
            f"Endpoint che verrà usato: **`/v1/images/"
            f"{'edits' if uploaded_refs else 'generations'}`** "
            "(passa a `edits` se carichi almeno un'immagine)."
        )
    reference_images: list[bytes] = []
    if uploaded_refs:
        if len(uploaded_refs) > spec.max_reference_images:
            st.warning(
                f"{spec.label} accetta al massimo {spec.max_reference_images} immagini di "
                f"riferimento: verranno usate solo le prime {spec.max_reference_images}."
            )
            uploaded_refs = uploaded_refs[: spec.max_reference_images]
        reference_images = [f.getvalue() for f in uploaded_refs]
        thumbs = st.columns(min(len(uploaded_refs), 6))
        for i, f in enumerate(uploaded_refs):
            thumbs[i % len(thumbs)].image(f, caption=f.name, width=110)

with col_prefix:
    st.subheader("📝 Prefisso di stile (opzionale)")
    style_prefix = st.text_area(
        "Testo anteposto a ogni prompt",
        value="",
        height=160,
        placeholder=(
            "Es.: Genera un'immagine in stile vettoriale. L'immagine fornita è un "
            "riferimento esclusivamente stilistico: replica palette, illuminazione e "
            "tratto, ma non copiare soggetto o composizione."
        ),
        help="Utile per applicare lo stesso stile a tutti i prompt di un batch.",
    )


def compose_prompt(body: str) -> str:
    return f"{style_prefix.strip()} {body.strip()}".strip() if style_prefix.strip() else body.strip()


def show_cost_estimate(count: int):
    if spec.provider != OPENAI:
        return
    width, height = openai_dims_for_aspect(params.aspect_ratio, params.resolution)
    low, high = estimate_output_cost(spec.model_id, params.quality, width, height)

    def usd(a, b):
        return f"US$ {a:.5f}" if a == b else f"US$ {a:.5f} – {b:.5f}"

    st.info(
        f"**Stima costo output immagini:** {usd(low, high)} per immagine · "
        f"**{usd(low * count, high * count)} per {count} immagini**."
    )
    st.caption(
        "Stima dal calcolatore OpenAI (15/09/2026), non un preventivo totale: "
        "esclude token del prompt e delle immagini di riferimento, imposte e ulteriori tentativi. "
        "Gli input vengono inviati per ogni variante. "
        + ("Con qualità auto l'intervallo mostra gli scenari low–massima qualità; "
           "non è un limite di spesa garantito. " if params.quality == "auto" else "")
        + "[Formula e tariffe](https://developers.openai.com/api/docs/guides/image-generation#cost-and-latency)"
    )


def execute(jobs: list[PromptJob], run_kind: str):
    """Lancia i job e registra i risultati in session_state."""
    try:
        provider = create_provider(spec, API_KEYS)
    except ValueError as exc:
        st.error(str(exc))
        return

    label = datetime.now().strftime(f"{run_kind}_{spec.key}_%Y%m%d_%H%M%S") + f"_{uuid4().hex[:12]}"
    run_dir = OUTPUT_DIR / label

    progress = st.progress(0.0, text=f"0/{len(jobs)} completate…")
    log = st.empty()

    def on_progress(done, total, result):
        progress.progress(done / total, text=f"{done}/{total} completate…")
        if result.ok:
            log.info(f"✅ {result.job.name} ({result.elapsed_s:.1f}s)")
        else:
            log.warning(f"❌ {result.job.name}: {result.error}")

    try:
        create_run(run_dir, spec, params, jobs, reference_images, concurrency, max_retries)
        data = run_saved(run_dir, API_KEYS, on_progress=on_progress, provider=provider)
        st.session_state.history_choice = run_dir.name
        ok = sum(j['status'] == 'completed' for j in data['jobs'])
        st.info(f"Run salvata: {ok}/{len(jobs)} immagini completate. Dettagli nello storico.")
    except (OSError, ValueError) as exc:
        st.error(f"Impossibile completare la run: {exc}")
    finally:
        progress.empty()
        log.empty()



tab_manual, tab_file = st.tabs(["✍️ Prompt manuale", "📄 Batch da file (.docx / .xlsx / .csv)"])

# --- Tab 1: prompt manuale -------------------------------------------------
with tab_manual:
    manual_name = st.text_input("Nome immagine", value="immagine", help="Nome del file di output (senza estensione).")
    manual_prompt = st.text_area("Prompt", height=220, placeholder="Descrivi l'immagine da generare…")
    n_variants = st.number_input("Numero di varianti", 1, 10, 1, help="Genera N immagini con lo stesso prompt.")
    show_cost_estimate(int(n_variants))
    if st.button("🚀 Genera", type="primary", disabled=not manual_prompt.strip()):
        from image_batch_studio.prompt_files import sanitize_filename
        base = sanitize_filename(manual_name or "immagine")
        jobs = [
            PromptJob(
                name=base if n_variants == 1 else f"{base}_{i + 1}",
                prompt=compose_prompt(manual_prompt),
                source_row=1,
            )
            for i in range(int(n_variants))
        ]
        execute(jobs, "single")

# --- Tab 2: batch da file ---------------------------------------------------
with tab_file:
    st.markdown(
        "Carica un file con una tabella di prompt. Caso ideale: **due colonne**, "
        "la prima con il nome dell'immagine e la seconda con il prompt. Tabelle "
        "diverse funzionano comunque: sotto puoi correggere quali colonne usare."
    )
    uploaded_file = st.file_uploader("File prompt", type=["docx", "xlsx", "xlsm", "csv"], key="prompt_file")

    if uploaded_file is not None:
        try:
            tables = parse_file(uploaded_file.name, uploaded_file.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            tables = []

        if tables:
            table = tables[0]
            if len(tables) > 1:
                table = tables[st.selectbox(
                    "Tabella / foglio da usare",
                    range(len(tables)),
                    format_func=lambda i: tables[i].label,
                )]

            guess = guess_columns(table.rows)
            n_cols = max(len(r) for r in table.rows)
            col_labels = {}
            for c in range(n_cols):
                sample = next((r[c] for r in table.rows if c < len(r) and r[c]), "")
                preview = (sample[:30] + "…") if len(sample) > 30 else sample
                col_labels[c] = f"Colonna {c + 1}" + (f' — "{preview}"' if preview else "")

            st.markdown("**Mappatura colonne** (precompilata automaticamente, correggi se serve):")
            c1, c2, c3 = st.columns([1, 1, 2])
            has_header = c1.checkbox("La prima riga è un'intestazione", value=guess.has_header)
            name_options = [None] + list(range(n_cols))
            name_col = c2.selectbox(
                "Colonna nome immagine",
                name_options,
                index=name_options.index(guess.name_col) if guess.name_col in name_options else 0,
                format_func=lambda c: "— nessuna (nomi automatici) —" if c is None else col_labels[c],
            )
            prompt_cols = c3.multiselect(
                "Colonne prompt (anche più di una)",
                options=list(range(n_cols)),
                default=[c for c in guess.prompt_cols if c is not None],
                format_func=lambda c: col_labels[c],
            )

            with st.expander("Anteprima tabella", expanded=False):
                st.dataframe(table.rows, width="stretch")

            if prompt_cols:
                jobs, issues = build_jobs(
                    table.rows,
                    has_header=has_header,
                    name_col=name_col,
                    prompt_cols=prompt_cols,
                )
                skipped = [i for i in issues if i.skipped]
                st.markdown(
                    f"**{len(jobs)} prompt validi** · {len(skipped)} righe scartate · "
                    f"{len(issues) - len(skipped)} avvisi"
                )
                if issues:
                    with st.expander(f"⚠️ Report validazione ({len(issues)} segnalazioni)"):
                        for issue in issues:
                            icon = "🚫" if issue.skipped else "ℹ️"
                            st.markdown(f"{icon} Riga {issue.row}: {issue.message}")

                if jobs:
                    with st.expander("Anteprima prompt estratti", expanded=True):
                        st.dataframe(
                            [{"nome file": j.name, "prompt": j.prompt[:160] + ("…" if len(j.prompt) > 160 else "")} for j in jobs],
                            width="stretch",
                        )

                    c_limit, c_var = st.columns(2)
                    limit = c_limit.number_input(
                        "Limita ai primi N prompt (0 = tutti)", 0, len(jobs), 0,
                        help="Utile per un dry-run prima del batch completo.",
                    )
                    variants = c_var.number_input(
                        "Varianti per prompt", 1, 10, 1,
                        help="Genera N immagini per ogni prompt del file (con lo stesso testo).",
                    )
                    selected = jobs[: int(limit)] if limit else jobs
                    est = len(selected) * int(variants)
                    show_cost_estimate(est)
                    if st.button(f"🚀 Genera batch ({est} immagini)", type="primary"):
                        final_jobs = [
                            PromptJob(
                                name=j.name if variants == 1 else f"{j.name}_{v + 1}",
                                prompt=compose_prompt(j.prompt),
                                source_row=j.source_row,
                            )
                            for j in selected
                            for v in range(int(variants))
                        ]
                        execute(final_jobs, "batch")
            else:
                st.info("Seleziona almeno una colonna prompt per continuare.")


# ---------------------------------------------------------------------------
# Risultati
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Storico delle generazioni")
folders = sorted((p for p in OUTPUT_DIR.iterdir() if p.is_dir()), reverse=True) if OUTPUT_DIR.exists() else []
entries = {}
for folder in folders:
    if (folder / 'run.json').exists():
        try:
            entry = load_run(folder)
            completed = sum(j['status'] == 'completed' for j in entry['jobs'])
            entries[folder.name] = f"{folder.name} · {entry['model']['label']} · {completed}/{len(entry['jobs'])} completate"
        except ValueError:
            entries[folder.name] = f"{folder.name} · manifesto non leggibile"
    else:
        entries[folder.name] = f"{folder.name} · archivio precedente"
if st.session_state.get('history_choice') not in entries:
    st.session_state.history_choice = ''
choice = st.selectbox("Apri una run", [''] + list(entries),
                      format_func=lambda key: entries.get(key, 'Seleziona una run'), key='history_choice')


def close_history():
    st.session_state.history_choice = ''


def resume_history(folder, **options):
    progress = st.progress(0.0)
    try:
        run_saved(folder, API_KEYS, on_progress=lambda done, total, result: progress.progress(done / total), **options)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
    else:
        st.rerun()
    finally:
        progress.empty()


if choice:
    st.button("Chiudi visualizzazione", on_click=close_history)
    folder = OUTPUT_DIR / choice
    images = []
    if not (folder / 'run.json').exists():
        st.info("Archivio precedente: immagini consultabili, ma prompt e impostazioni non sono disponibili per la ripresa.")
        images = [p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}]
    else:
        try:
            data = load_run(folder)
            active = is_running(folder)
            states = {j['id']: j['status'] if active else effective_status(folder, j) for j in data['jobs']}
        except (OSError, ValueError) as exc:
            st.error(str(exc))
            data = None
        if data:
            if active:
                st.info('Run in esecuzione in un’altra sessione. La ripresa è bloccata fino al termine.')
                st.button('Aggiorna avanzamento')
            labels = {'pending': 'In attesa', 'running': 'In esecuzione', 'recoverable': 'Recuperabile',
                      'saving': 'Salvataggio', 'completed': 'Completato', 'failed': 'Fallito', 'uncertain': 'Esito incerto'}
            st.caption(f"Creata: {data['created_at']} · Modello: {data['model']['label']}")
            st.write(' · '.join(f"{label}: {sum(s == status for s in states.values())}"
                                for status, label in labels.items() if status in states.values()))
            with st.expander('Impostazioni originali e riferimenti'):
                st.json(data['params'])
                st.caption(f"Parallelismo: {data['concurrency']} · Tentativi per fase: {data['attempts_per_phase']}")
                for ref in data['references']:
                    path = local_path(folder, ref['path'])
                    try:
                        st.image(path.read_bytes(), width=150)
                    except (OSError, ValueError):
                        st.warning(f"Riferimento mancante o non leggibile: {ref['path']}")
            with st.expander('Prompt, avanzamento ed errori', expanded=True):
                st.dataframe([{'nome': j['name'], 'prompt': j['prompt'], 'stato': labels[states[j['id']]],
                               'esecuzioni': j['executions'], 'tentativi per fase': str(j.get('phase_attempts', {})), 'errore': j.get('error') or ''}
                              for j in data['jobs']], width="stretch")
                st.caption("Le esecuzioni contano gli avvii/riprese del job; ogni fase può effettuare i retry configurati.")
            with st.expander('Registro degli eventi'):
                st.json([{'nome': j['name'], 'eventi': j['events']} for j in data['jobs']])
            pending = [j for j in data['jobs'] if states[j['id']] in {'pending', 'recoverable'}]
            if pending and not active:
                new_count = sum(not j.get('remote') for j in pending)
                if st.button(f"Riprendi batch ({len(pending)} job, {new_count} nuove generazioni)"):
                    resume_history(folder)
            failed = [j for j in data['jobs'] if states[j['id']] == 'failed']
            if failed and not active:
                chosen = st.multiselect('Job falliti da riprovare', [j['id'] for j in failed],
                                        format_func=lambda i: next(j['name'] for j in failed if j['id'] == i), key=f'retry_{choice}')
                new_count = sum(not j.get('remote') for j in failed if j['id'] in chosen)
                if st.button(f"Riprova i falliti ({new_count} nuove generazioni)", disabled=not chosen):
                    resume_history(folder, selected=chosen)
            uncertain = [j for j in data['jobs'] if states[j['id']] in {'uncertain', 'recoverable', 'failed'}]
            if uncertain and not active:
                with st.expander('Rigenera esplicitamente job incerti o non recuperabili'):
                    st.warning("Questa azione invia nuove richieste a pagamento. Il servizio potrebbe aver già completato le richieste precedenti. Per BFL abbandona il recupero della richiesta originale.")
                    chosen = st.multiselect('Job da rigenerare', [j['id'] for j in uncertain],
                                            format_func=lambda i: next(j['name'] for j in uncertain if j['id'] == i), key=f'regenerate_{choice}')
                    if st.button(f"Rigenera con nuove richieste ({len(chosen)} immagini)", disabled=not chosen):
                        resume_history(folder, selected=chosen, regenerate=True)
            images = [local_path(folder, j['path']) for j in data['jobs'] if states[j['id']] == 'completed']
    if images:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as archive:
            for path in images:
                archive.write(path, arcname=path.name)
        st.download_button('Scarica immagini (zip)', buffer.getvalue(), file_name=f'{choice}.zip', mime='application/zip')
        columns = st.columns(3)
        for i, path in enumerate(images):
            columns[i % 3].image(str(path), caption=path.stem, width="stretch")
