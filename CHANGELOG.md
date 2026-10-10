# Changelog

## 1.2.0

### New
- **Export for sharing** — a new Export tab finds a public download link for every model in the workflow (or lets you paste one; "Find by hash" identifies local files the name lookup misses). Apply adds a "Model links" note in the style of ComfyUI's official templates, writes the links into the loader nodes so ComfyUI's own missing-models downloader can use them, and adds a Model Linker node. Tokens are stripped from every link.
- **All models tab** — browse every model in the workflow grouped by folder and swap any of them from a searchable picker; one swap updates every node using that model (#4).
- **Locate node** — jump to the node using a model, including inside (nested) subgraphs (#9).
- **Multi-LoRA loaders** — LoRAs in rgthree Power Lora Loader and LoraManager loaders are detected, linked and swappable, with strength and on/off shown.
- **API tokens** — Settings (⚙) stores a HuggingFace token and CivitAI API key on the ComfyUI server, with a Check button for each; `HF_TOKEN` / `CIVITAI_API_TOKEN` also work. Gated and login-only downloads now work, and a missing or rejected token is reported clearly.
- **More download sources** — a bundled index of every model file in the Comfy-Org and Kijai HuggingFace repos (refreshed weekly); lookups by SHA-256 on CivitAI and CivArchive when a workflow carries a model's hash (also finds models deleted from CivitAI); HuggingFace search now sees files in subfolders and retries without precision/quant suffixes.

### Improved
- **Smarter matching** — names that differ only by precision or quantization (`fp8`, `bf16`, `e4m3fn`, `Q4_K_M`, …) score as the same model; a different parameter count or generation (4B vs 8B, Qwen2 vs Qwen3) stays below the suggestion threshold; word order no longer matters.
- **Custom loader detection** — model fields of third-party loader nodes are found automatically by tracing which folder each input reads from, so it works even when that folder is empty (#5). GGUF models are now detected.
- **Wrong-folder warning** — a file in a folder the node can't load from is flagged instead of failing at prompt time, and never auto-linked (#3).
- **Cross-platform paths** — workflows made on Windows match on Linux and vice versa (#8).
- Auto-Link updates every node that references a model, not just the first.
- Download progress stays visible while models download; downloads can no longer leave corrupt partial files.
- Analysis and source lookups no longer block ComfyUI while they run.
- Roomier UI with green highlights on cards that are one click from resolved.

### Fixed
- Nodes disappearing after applying links on newer ComfyUI frontends — links are applied to the live graph in place (thanks @gontz, PR #14).
- CivitAI API keys are sent as a header instead of being embedded in download URLs.

### Other
- Added the MIT LICENSE file (#15).
- New dependency: `rapidfuzz` (optional — matching falls back to Python's difflib without it).
- The pack now registers one node, **Model Linker**; restart ComfyUI after updating.
