# ComfyUI Model Linker

A ComfyUI extension that relinks missing models in shared workflows: it finds the closest matches among your local files using fuzzy matching, and can download what you don't have from HuggingFace or CivitAI.

![Model Linker Interface](model-linker.png)

## Features

- **Workflow scanning** — finds model references in all nodes, including nested subgraphs
- **Custom node support** — detects model fields of third-party loader nodes automatically by tracing which model folder each input reads from (works even when that folder is empty); no hardcoded node list
- **Multi-LoRA loaders** — rgthree Power Lora Loader and LoraManager LoRAs are detected, linked and swappable like any other model, with strength and on/off shown
- **Fuzzy matching** — intelligent similarity scoring finds files despite renames, different separators, or capitalization; knows that `fp8`/`bf16`/`Q4_K_M` variants are the same model but a 4B is not an 8B; 100% matches shown first, otherwise best matches ≥70% confidence
- **Cross-platform paths** — workflows authored on Windows match on Linux and vice versa (`\` vs `/` is handled)
- **Folder-aware suggestions** — a file sitting in a folder the node can't load from is flagged with a "wrong folder" warning instead of silently failing at prompt time
- **Auto-resolve** — one click links every perfect match; also available directly from ComfyUI's native Missing Models popup
- **Browse & swap** — the All models tab lists every model in the workflow grouped by category; swap any of them for another local file without hunting through the node tree, applied to all referencing nodes at once
- **Downloads** — fetches missing models from HuggingFace/CivitAI (URLs from the workflow, a model database, a bundled index of every Comfy-Org and Kijai file, or online search), with progress, speed display, bulk download, and cancel
- **Hash lookups** — when a workflow carries a model's SHA-256, it's identified on HuggingFace, CivitAI or CivArchive even if the file was renamed or deleted from CivitAI
- **API tokens** — add a HuggingFace token and CivitAI API key in Settings (⚙) for gated and login-only models; stored on the ComfyUI server, never sent back to the browser (or set `HF_TOKEN` / `CIVITAI_API_TOKEN`)
- **Export for sharing** — the Export tab finds a download link for every model (or lets you paste one), then adds a "Model links" note like ComfyUI's official templates, writes the links into the loader nodes so ComfyUI's own downloader can use them, and adds a Model Linker node
- **Safe updates** — resolved models are applied to the live graph in place (no canvas rebuild); you save the workflow yourself when ready

## Installation

1. Clone or download this repository
2. Place it in your ComfyUI `custom_nodes/` directory
3. Restart ComfyUI

## Usage

1. Open a workflow with missing models
2. Open the Model Linker via the button in ComfyUI's top menu bar, the `Ctrl+Shift+L` shortcut, or the button injected into ComfyUI's Missing Models popup
3. Review missing models and their suggested matches
4. Link individual matches, use **Auto-Link** for all 100% matches, or **Download All Missing** for models with known sources
5. Save your workflow when ready

## HuggingFace downloads

Files from huggingface.co are downloaded with `huggingface_hub` (multiple connections via `hf_xet` where available), which is much faster than a single stream on fast links. CivitAI and other URLs use the built-in downloader. If `huggingface_hub` is missing or older than 1.1.0 (for example with transformers 4.x), HuggingFace downloads use the built-in downloader too.

Gated models need a HuggingFace token: add one in the Settings tab (⚙), or set `HF_TOKEN` in the environment ComfyUI runs in. A **Read**-scoped token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) is enough. You also need to accept the model's terms on its HuggingFace page once. Downloads never use a token cached by `hf auth login`.

Files are staged in a hidden folder next to the destination and moved into place when complete. `hf_xet` wrote no chunk cache to `HF_HOME` in testing, only about 60 KB of logs per run. If ComfyUI is killed mid-download, the hidden `.model_linker_tmp_*` folder in the destination can keep the partial file (up to the full model size) and is safe to delete.

## License

[MIT](LICENSE)
