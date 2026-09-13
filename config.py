"""Registro dei modelli supportati e delle loro capacità.

Ogni modello dichiara qui i parametri che la UI può esporre: la UI si adatta
automaticamente (es. guidance/steps solo per flux-2-flex, image_size solo per
Gemini). Per aggiungere un modello basta aggiungere una voce a MODELS.
"""
from dataclasses import dataclass, field


GEMINI = "gemini"
BFL = "bfl"
OPENAI = "openai"

# Aspect ratio supportati da Gemini (fonte: ai.google.dev/gemini-api/docs/image-generation)
GEMINI_ASPECT_RATIOS = [
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
    "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1",
]

# Per BFL non esiste un parametro aspect_ratio: si passano width/height in pixel.
# La UI mostra comunque un aspect ratio + un target in megapixel e calcola le
# dimensioni (vedi providers/bfl_provider.py: dims_for_aspect).
BFL_ASPECT_RATIOS = [
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
]

# Anche OpenAI non ha un parametro aspect_ratio: si passa `size` in pixel, con
# vincoli precisi (lati multipli di 16, lato max 3840, rapporto lungo/corto <= 3:1,
# pixel totali fra 655.360 e 8.294.400). La UI espone aspect ratio + megapixel e
# calcola la size (vedi providers/openai_provider.py: dims_for_aspect).
OPENAI_ASPECT_RATIOS = [
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
]

PROMPT_HINT_GEMINI = (
    "**Nano Banana (Gemini) ragiona sul linguaggio naturale**: descrivi la scena "
    "in modo narrativo e discorsivo, come racconteresti una fotografia o "
    "un'illustrazione a un'altra persona. Funzionano bene frasi complete che "
    "descrivono soggetto, ambientazione, atmosfera, luce e stile. Puoi dare "
    "istruzioni esplicite sulle immagini di riferimento (es. \"usa l'immagine "
    "fornita solo come riferimento stilistico\"). Il modello gestisce bene anche "
    "testo da renderizzare dentro l'immagine se lo metti tra virgolette."
)

PROMPT_HINT_FLUX = (
    "**FLUX.2 preferisce prompt strutturati e diretti**: parti dal soggetto "
    "principale, poi composizione/inquadratura, poi stile, luce e dettagli. "
    "L'ordine delle parole conta: le prime frasi pesano di più. Evita giri di "
    "parole; elenca gli elementi in modo concreto (\"a red vintage bicycle "
    "leaning against a brick wall, golden hour light, 35mm photo\"). I prompt in "
    "inglese tendono a dare risultati più fedeli. Con `prompt_upsampling` attivo "
    "l'API arricchisce automaticamente il prompt prima della generazione. Le "
    "immagini di riferimento guidano stile/soggetto: descrivi nel prompt che "
    "cosa riprendere da ciascuna."
)


PROMPT_HINT_OPENAI = (
    "**gpt-image-2 segue le istruzioni alla lettera**: descrivi il soggetto e poi "
    "dai indicazioni esplicite su composizione, inquadratura, palette, luce e "
    "stile. È il modello più affidabile per il **testo dentro l'immagine**: "
    "scrivilo fra virgolette e indica dove posizionarlo. Puoi chiedere sfondo "
    "trasparente (usa formato `png` o `webp`). Se carichi immagini di "
    "riferimento la richiesta passa automaticamente all'endpoint di editing: "
    "in quel caso spiega nel prompt che cosa modificare o che cosa riprendere "
    "da ciascuna immagine."
)


@dataclass(frozen=True)
class ModelSpec:
    key: str                      # chiave interna univoca
    label: str                    # nome mostrato nella UI
    provider: str                 # GEMINI | BFL | OPENAI
    model_id: str                 # model id (Gemini/OpenAI) o endpoint path (BFL)
    aspect_ratios: list[str]
    resolutions: list[str]        # Gemini: image_size; BFL: target megapixel
    default_resolution: str
    max_reference_images: int
    response_modalities: tuple[str, ...] = ()   # solo Gemini
    supports_guidance_steps: bool = False        # solo flux-2-flex
    qualities: tuple[str, ...] = ()              # solo OpenAI
    prompting_hint: str = ""
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec(
            key="nano-banana-2-flash",
            label="Nano Banana 2 Flash (Gemini 3.1 Flash Image)",
            provider=GEMINI,
            model_id="gemini-3.1-flash-image-preview",
            aspect_ratios=GEMINI_ASPECT_RATIOS,
            resolutions=["512", "1K", "2K", "4K"],
            default_resolution="2K",
            max_reference_images=14,
            response_modalities=("Image",),
            prompting_hint=PROMPT_HINT_GEMINI,
            notes="Veloce ed economico, adatto a batch numerosi.",
        ),
        ModelSpec(
            key="nano-banana-pro",
            label="Nano Banana Pro (Gemini 3 Pro Image)",
            provider=GEMINI,
            model_id="gemini-3-pro-image-preview",
            aspect_ratios=GEMINI_ASPECT_RATIOS,
            resolutions=["1K", "2K", "4K"],
            default_resolution="2K",
            max_reference_images=14,
            response_modalities=("Text", "Image"),
            prompting_hint=PROMPT_HINT_GEMINI,
            notes="Ragionamento avanzato, più lento e costoso del Flash.",
        ),
        ModelSpec(
            key="flux-2-max",
            label="FLUX.2 [max] (Black Forest Labs)",
            provider=BFL,
            model_id="flux-2-max",
            aspect_ratios=BFL_ASPECT_RATIOS,
            resolutions=["0.5 MP", "1 MP", "2 MP", "4 MP"],
            default_resolution="1 MP",
            max_reference_images=8,
            prompting_hint=PROMPT_HINT_FLUX,
            notes="Modello FLUX.2 di fascia massima; qualità più alta e costo per immagine maggiore.",
        ),
        ModelSpec(
            key="flux-2-pro",
            label="FLUX.2 [pro] (Black Forest Labs)",
            provider=BFL,
            model_id="flux-2-pro",
            aspect_ratios=BFL_ASPECT_RATIOS,
            resolutions=["0.5 MP", "1 MP", "2 MP", "4 MP"],
            default_resolution="1 MP",
            max_reference_images=8,
            prompting_hint=PROMPT_HINT_FLUX,
            notes="Modello di punta BFL per la generazione production-grade.",
        ),
        ModelSpec(
            key="flux-2-flex",
            label="FLUX.2 [flex] (Black Forest Labs)",
            provider=BFL,
            model_id="flux-2-flex",
            aspect_ratios=BFL_ASPECT_RATIOS,
            resolutions=["0.5 MP", "1 MP", "2 MP", "4 MP"],
            default_resolution="1 MP",
            max_reference_images=8,
            supports_guidance_steps=True,
            prompting_hint=PROMPT_HINT_FLUX,
            notes="Specializzato in tipografia e dettagli fini; espone guidance e steps.",
        ),
        ModelSpec(
            key="flux-2-klein-9b",
            label="FLUX.2 [klein] 9B (Black Forest Labs)",
            provider=BFL,
            model_id="flux-2-klein-9b",
            aspect_ratios=BFL_ASPECT_RATIOS,
            resolutions=["0.5 MP", "1 MP", "2 MP"],
            default_resolution="1 MP",
            max_reference_images=8,
            prompting_hint=PROMPT_HINT_FLUX,
            notes="Inferenza quasi istantanea, qualità inferiore a pro/flex.",
        ),
        ModelSpec(
            key="gpt-image-2",
            label="GPT Image 2 (OpenAI)",
            provider=OPENAI,
            model_id="gpt-image-2",
            aspect_ratios=OPENAI_ASPECT_RATIOS,
            resolutions=["1 MP", "2 MP", "4 MP", "8 MP"],
            default_resolution="1 MP",
            max_reference_images=10,
            qualities=("auto", "low", "medium", "high"),
            prompting_hint=PROMPT_HINT_OPENAI,
            notes=(
                "Ottimo su testo e istruzioni precise. Con immagini di riferimento "
                "usa l'endpoint di editing (/v1/images/edits)."
            ),
        ),
    ]
}
