# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit app that lets a user chat with a Vision-Language Model about an image. LLM access goes through [OpenRouter](https://openrouter.ai) using `langchain_openai.ChatOpenAI` pointed at OpenRouter's OpenAI-compatible endpoint. Default model is `meta-llama/llama-3.2-11b-vision-instruct:free`. Deployed at `https://image-qna.streamlit.app/`.

## Run

```bash
pip install -r requirements.txt

# Single-page (original) app — one model, sidebar image upload
streamlit run streamlit_app.py

# Multi-page entry — single-VLM page + Compare VLMs page
streamlit run st_entry.py

# Compare VLMs page directly (side-by-side, multi-image multimodal chat input)
streamlit run compare_vlm.py
```

The OpenRouter key is read from `st.secrets["OpenRouter_key"]` (`.streamlit/secrets.toml`); if missing or the user toggles "use your own API key", the sidebar prompts for one.

There are no tests, no lint config, and no build step.

## Architecture

There are three Streamlit entry points and one helper module, with significant duplication that is important to be aware of before editing:

- `streamlit_app.py` — single-VLM page. Image is uploaded once in the sidebar (URL or file), stored as base64 in `st.session_state["image"]`, and re-attached to every turn. Uses `st.chat_input` (text only). Imports helpers (`build_llm`, `get_openrouter_api_key`, `get_llm_icon`, `image_to_base64`, `is_valid_url`) from `compare_vlm.py` at the repo root.
- `compare_vlm.py` (root) — the Compare-VLMs page **and** the shared helper module imported by `streamlit_app.py`. Supports two models side-by-side, multi-image input via `st_multimodal_chatinput` or `file_chat_input` (toggle), and stores chat history with per-model role names (`messages[i]["role"]` is the model name for assistant turns) so each column filters its own history in `generate_response`.
- `pages/compare_vlm.py` — a near-identical copy of root `compare_vlm.py`, present because Streamlit's classic multipage routing picks pages out of a `pages/` directory. Keep the two in sync when changing the Compare-VLMs UI, or migrate fully to `st_entry.py`'s `st.navigation` approach.
- `st_entry.py` — newer multipage entry using `st.navigation([st.Page("./streamlit_app.py"), st.Page("compare_vlm.py")])`. With this entry, the `pages/` directory is not used.
- `streamlit_app_dev.py` — scratch/dev variant of `streamlit_app.py` using `file_chat_input`; not wired into any entry point.
- `st-file-chat-input.py` — local scratch file (note the hyphenated, non-importable name).

### Message shape

History lives in `st.session_state.messages` as OpenAI-style dicts. User turns with images use the content list form: `[{"type":"text","text":...}, {"type":"image_url","image_url": "<data:image/jpeg;base64,...>"}]`. In `compare_vlm.py`, assistant `role` is set to the model name; `generate_response` deepcopies the list, filters to `{"user", "assistant", <model_name>}`, then normalizes the model name back to `"assistant"` before calling `llm.invoke`. Editing the role-naming convention will break the per-column history filter.

### LLM construction

`build_llm` is `@st.cache_resource`-decorated, keyed on `(api_key, model_name, temperature)`. It sets `default_headers={"HTTP-Referer": ..., "X-Title": "Image Q&A"}` for OpenRouter app attribution — these must be passed via `default_headers`, not `model_kwargs["headers"]` (the commented-out form does not work with current `langchain_openai`).

### Theme / icons

`.streamlit/config.toml` pins `theme.base = "dark"` and hides the sidebar nav (`showSidebarNavigation = false`). `streamlit_theme.st_theme()` is read at runtime to pick a light/dark variant of model icons from `unpkg.com/@lobehub/icons-static-png`. `st_multimodal_chatinput`'s text is only legible on the dark theme — a known limitation surfaced as a toast.


## Guidelines

- Create tests in `tests/` and update CLAUDE.md for each new feature
- use Google-style docstring for new functions and add a doctest compatible unit test if possible
- keep code modular to ensure ease in future refactoring
- Prefer native Streamlit features over custom CSS
- Keep custom CSS minimal
