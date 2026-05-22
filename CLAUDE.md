# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit app that lets a user chat with a Vision-Language Model about an image. LLM access goes through [OpenRouter](https://openrouter.ai) using `langchain_openai.ChatOpenAI` pointed at OpenRouter's OpenAI-compatible endpoint. Default model is `meta-llama/llama-3.2-11b-vision-instruct:free`. Deployed at `https://image-qna.streamlit.app/`.

## Run

```bash
uv sync   # or: pip install -e .

# Multi-page app — Image Q&A (default) + Compare VLMs + Image Generation, selected via top nav
streamlit run streamlit_app.py
```

Requires `streamlit>=1.56.0` (for `st.chat_input(accept_file=...)`).

The OpenRouter key is read from `st.secrets["OpenRouter_key"]` (`.streamlit/secrets.toml`); if missing or the user toggles "use your own API key", the sidebar prompts for one.

There are no tests, no lint config, and no build step.

## Architecture

One entry point + three pages + a shared helpers module — all unified in `streamlit_app.py`:

- `streamlit_app.py` — the canonical entry. Holds the shared utilities (`image_to_base64`, `is_valid_url`, `get_openrouter_api_key`, `get_llm_icon`, `build_llm`) at module top-level, and inside an `if __name__ == "__main__":` guard runs `st.navigation([st.Page("pages/image-vqa.py", default=True), st.Page("pages/compare_vlm.py"), st.Page("pages/image_gen.py")]).run()`. The guard is load-bearing: each page script does `from streamlit_app import ...`, which re-imports the module as `streamlit_app` (not `__main__`), so the helpers get defined but `pg.run()` is not re-triggered.
- `pages/image-vqa.py` — single-VLM page. Image is uploaded once in the sidebar (URL or file), stored as base64 in `st.session_state["image"]`, and re-attached to every turn. Uses `st.chat_input` (text only). Calls `Main()` unconditionally at module bottom.
- `pages/compare_vlm.py` — Compare-VLMs page. Supports two models side-by-side. Multi-image input uses native `st.chat_input(accept_file="multiple", file_type=[...])` (Streamlit ≥1.56.0); the returned `ChatInputValue` is converted to OpenAI-style multimodal messages in `chatinput2msg`. The chat input is wrapped in a `streamlit_float`-pinned container because native bottom-docking is not always reliable in practice. Chat history uses per-model role names (`messages[i]["role"]` is the model name for assistant turns) so each column filters its own history in `generate_response`.
- `pages/image_gen.py` — Image generation page (Google Nano Banana family). Bypasses `build_llm`/langchain and POSTs directly to OpenRouter's `/api/v1/chat/completions` via `httpx` with `modalities: ["text", "image"]` and an `image_config` block (`aspect_ratio`, `image_size`). The generated image arrives as `choices[0].message.images[0].image_url.url` (a `data:image/png;base64,...` URL), which langchain doesn't surface — hence the direct call. Session-state keys are namespaced `gen_*` (`gen_carry_images`, `gen_last_output`) to avoid colliding with the chat pages' `messages`/`image` state. The "Use as input for next generation" button appends the last output's data URL to `gen_carry_images`, which is then prepended to the user's content list on the next request.
- `streamlit_app_dev.py`, `st-file-chat-input.py` — legacy scratch files that imported the now-removed `st_multimodal_chatinput` / `file_chat_input` components. **They will fail at import** under the current deps; retained only for historical reference and not wired into any entry point.

### Message shape

History lives in `st.session_state.messages` as OpenAI-style dicts. User turns with images use the content list form: `[{"type":"text","text":...}, {"type":"image_url","image_url": "<data:image/jpeg;base64,...>"}]`. In `pages/compare_vlm.py`, assistant `role` is set to the model name; `generate_response` deepcopies the list, filters to `{"user", "assistant", <model_name>}`, then normalizes the model name back to `"assistant"` before calling `llm.invoke`. Editing the role-naming convention will break the per-column history filter.

### LLM construction

`build_llm` (defined in `streamlit_app.py`) is `@st.cache_resource`-decorated, keyed on `(api_key, model_name, temperature)`. It sets `default_headers={"HTTP-Referer": ..., "X-Title": "Image Q&A"}` for OpenRouter app attribution — these must be passed via `default_headers`, not `model_kwargs["headers"]` (current `langchain_openai` requires the former).

### Theme / icons

`.streamlit/config.toml` pins `theme.base = "dark"` and hides the sidebar nav (`showSidebarNavigation = false`). `streamlit_theme.st_theme()` is read at runtime to pick a light/dark variant of model icons from `unpkg.com/@lobehub/icons-static-png`.


## Guidelines

- Create tests in `tests/` and update CLAUDE.md for each new feature
- use Google-style docstring for new functions and add a doctest compatible unit test if possible
- keep code modular to ensure ease in future refactoring
- Prefer native Streamlit features over custom CSS
- Keep custom CSS minimal
