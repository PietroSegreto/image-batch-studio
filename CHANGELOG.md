# Changelog

## 0.2.0

- Add Nano Banana 2.1 (`gemini-nano-banana-2.1`, new default) and Nano Banana 2 Lite
  (`gemini-3.1-flash-lite-image`).
- Move Nano Banana 2 and Pro from the retired `-preview` IDs to the GA
  `gemini-3.1-flash-image` and `gemini-3-pro-image`.
- Add an optional thinking level for Gemini models that support it.
- Restrict Nano Banana Pro to the 10 aspect ratios it supports.
- Move application modules into the `image_batch_studio/` package; `app.py` stays
  the entry point.
- Update dependencies (google-genai 2.33, Streamlit 1.66, pandas 3.1, Pillow 13)
  and replace the deprecated `use_container_width` argument.
- Add GPT Image 2.5 Sunburst and Flare through the existing OpenAI Image API provider.
- Add xhigh/max quality settings and pre-generation image output cost estimates
  for manual variants and file batches, with an indicative range for auto quality.
- Preserve API usage details and reported-token output costs in history metadata.
- Validate unsupported quality and transparent JPEG before sending OpenAI requests.


## 0.1.0

- Streamlit interface for Gemini, FLUX.2 and OpenAI image generation.
- Manual prompts and DOCX, XLSX or CSV batches with reference images.
- Concurrent execution, collision-safe filenames and retries per phase.
- Persistent run history, selective retries and recovery of existing BFL jobs.
- Local regression tests using simulated providers; no API credentials required.
