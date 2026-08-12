# ComfyUI Model Linker

A ComfyUI extension that relinks missing models in shared workflows: it finds the closest matches among your local files using fuzzy matching, and can download what you don't have from HuggingFace or CivitAI.

![Model Linker Interface](model-linker.png)

## Features

- **Workflow scanning** — finds model references in all nodes, including nested subgraphs
- **Custom node support** — detects model fields of third-party loader nodes automatically by introspecting their input definitions; no hardcoded node list
- **Fuzzy matching** — intelligent similarity scoring finds files despite renames, different separators, or capitalization; 100% matches shown first, otherwise best matches ≥70% confidence
- **Cross-platform paths** — workflows authored on Windows match on Linux and vice versa (`\` vs `/` is handled)
- **Folder-aware suggestions** — a file sitting in a folder the node can't load from is flagged with a "wrong folder" warning instead of silently failing at prompt time
- **Auto-resolve** — one click links every perfect match; also available directly from ComfyUI's native Missing Models popup
- **Browse & swap** — the All models tab lists every model in the workflow grouped by category; swap any of them for another local file without hunting through the node tree, applied to all referencing nodes at once
- **Downloads** — fetches missing models from HuggingFace/CivitAI (URLs from the workflow, a model database, or online search), with progress, speed display, bulk download, and cancel
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

## HuggingFace authentication

Downloads from HuggingFace are **anonymous by default** — a token is only ever sent if a model turns out to be gated (license click-through required) and one is available, or if you explicitly opt in. 

**"Always send HF token" checkbox**: off by default. Turning it on sends your token on every HuggingFace download from the start instead of only when a model is actually gated. Not required for correctness — anonymous downloads work fine for public models — but authenticated requests can be more resilient to throttling under load, so it's potentially worth enabling if you're pulling a lot of models on a fast connection and want to avoid anonymous rate limits. The checkbox shows **(detected)** in green or **(not detected)** in red next to it, so you can tell at a glance whether a token is actually available to send before you bother checking it.

### Providing a HF_TOKEN

Set either of these in the environment ComfyUI's process runs in (an env var exported in your shell before launching it won't reach ComfyUI unless ComfyUI itself was started from that same shell — see your launch script):

- `HF_TOKEN` (preferred)
- `HUGGING_FACE_HUB_TOKEN` (older name, kept for compatibility with other tools)

For example, added to whatever script launches ComfyUI:
```bat
:: Windows batch
set "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
```bash
# Linux/macOS shell
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Don't have a token yet? Generate one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — a **Read**-scoped token is sufficient, no write access needed. You'll also need to visit the gated model's page on HuggingFace at least once and accept its license terms before a token grants access to it.

## License

[MIT](LICENSE)
