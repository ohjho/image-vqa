import base64
import httpx
import streamlit as st
from io import BytesIO
from PIL import Image
from retrying import retry
from streamlit_theme import st_theme

from streamlit_app import (
    get_llm_icon,
    get_openrouter_api_key,
    image_to_base64,
)

MODELS = [
    "google/gemini-2.5-flash-image",
    "google/gemini-3-pro-image-preview",
]
ASPECT_RATIOS = ["Auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
RESOLUTIONS = ["1K", "2K", "4K"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@retry(
    stop_max_attempt_number=5,
    retry_on_result=lambda result: result is None,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    wrap_exception=False,
)
def generate_image(
    api_key: str,
    model: str,
    prompt: str,
    input_images: list,
    aspect_ratio: str | None,
    image_size: str | None,
):
    """Call OpenRouter chat/completions with image modality.

    Args:
        api_key: OpenRouter API key.
        model: OpenRouter model id.
        prompt: Text prompt describing the image.
        input_images: List of data-URL strings (``data:image/...;base64,...``).
        aspect_ratio: e.g. ``"1:1"`` or ``None`` to omit.
        image_size: ``"1K"`` | ``"2K"`` | ``"4K"`` or ``None`` to omit.

    Returns:
        Data-URL string of the first generated image, or ``None`` on failure
        (the ``@retry`` decorator will retry on ``None``).
    """
    content = [{"type": "text", "text": prompt}]
    for im in input_images:
        content.append({"type": "image_url", "image_url": im})

    body = {
        "model": model,
        "modalities": ["text", "image"],
        "messages": [{"role": "user", "content": content}],
    }
    image_config = {}
    if aspect_ratio:
        image_config["aspect_ratio"] = aspect_ratio
    if image_size:
        image_config["image_size"] = image_size
    if image_config:
        body["image_config"] = image_config

    try:
        r = httpx.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://image-qna.streamlit.app/",
                "X-Title": "Image Q&A",
            },
            json=body,
            timeout=120.0,
        )
        r.raise_for_status()
        data = r.json()
        images = data["choices"][0]["message"].get("images") or []
        if not images:
            st.toast(f"No image returned: {data['choices'][0]['message']}")
            return None
        return images[0]["image_url"]["url"]
    except Exception as e:
        st.toast(f"Image generation error: {e}")
        return None


def data_url_to_bytes(data_url: str) -> bytes:
    """Decode a ``data:<mime>;base64,...`` URL to raw bytes."""
    return base64.b64decode(data_url.split(",", 1)[1])


def uploaded_file_to_data_url(uploaded_file) -> str:
    return image_to_base64(Image.open(uploaded_file))


def reset_state():
    st.session_state.pop("gen_carry_images", None)
    st.session_state.pop("gen_last_output", None)


def Main():
    st.set_page_config(
        page_title="Image Generation",
        layout="wide",
        page_icon="https://seekingvega.github.io/sv-journal/assets/images/sv-favicon-v2.png",
    )
    st.sidebar.title("Image Generation")
    page_theme = st_theme()["base"]
    st.logo(get_llm_icon("openrouter", page_theme))

    api_key = get_openrouter_api_key()
    if not api_key:
        st.warning(":point_left: Provide your OpenRouter API key to get started")
        return None

    if st.sidebar.button("Clear"):
        reset_state()
        st.rerun()

    st.session_state.setdefault("gen_carry_images", [])
    st.session_state.setdefault("gen_last_output", None)

    l_col, r_col = st.columns(2)

    with l_col:
        # Reused-from-prior-generation images
        if st.session_state["gen_carry_images"]:
            # st.caption("Reused from prior generation (will be sent as input):")
            reuse_im_container = st.expander(
                "Reused from prior generation (will be sent as input):", expanded=True
            )
            cols = reuse_im_container.columns(
                min(len(st.session_state["gen_carry_images"]), 4)
            )
            for i, im in enumerate(list(st.session_state["gen_carry_images"])):
                with cols[i % len(cols)]:
                    st.image(im, width="stretch")
                    if st.button("✕ remove", key=f"rm_carry_{i}"):
                        st.session_state["gen_carry_images"].pop(i)
                        st.rerun()

        uploaded = st.file_uploader(
            "Upload one or more reference images (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
        if uploaded:
            thumb_cols = st.columns(min(len(uploaded), 4))
            for i, f in enumerate(uploaded):
                with thumb_cols[i % len(thumb_cols)]:
                    st.image(f, caption=f.name, use_container_width=True)

        prompt = st.text_area("Prompt", placeholder="Describe the image to generate...")

        c1, c2, c3 = st.columns((2, 1, 1))
        with c1:
            model = st.selectbox("Model", MODELS, index=0)
        with c2:
            aspect = st.selectbox("Aspect ratio", ASPECT_RATIOS, index=0)
        with c3:
            resolution = st.selectbox("Resolution", RESOLUTIONS, index=0)

        # Validate 4K vs model
        image_size = resolution
        if resolution == "4K" and model != "google/gemini-3-pro-image-preview":
            st.warning("4K is only supported by gemini-3-pro-image-preview — using 2K.")
            image_size = "2K"

        if st.button("Generate", type="primary", disabled=not prompt.strip()):
            input_images = list(st.session_state["gen_carry_images"])
            for f in uploaded or []:
                try:
                    input_images.append(uploaded_file_to_data_url(f))
                except Exception as e:
                    st.error(f"Could not read {f.name}: {e}")
                    return None
            with st.spinner("Generating..."):
                out = generate_image(
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    input_images=input_images,
                    aspect_ratio=None if aspect == "Auto" else aspect,
                    image_size=image_size,
                )
            if out:
                st.session_state["gen_last_output"] = out

    with r_col:
        if st.session_state["gen_last_output"]:
            st.subheader("Output")
            out = st.session_state["gen_last_output"]
            st.image(out)
            dl_col, reuse_col = st.columns(2)
            with dl_col:
                st.download_button(
                    "Download PNG",
                    data=data_url_to_bytes(out),
                    file_name="generated.png",
                    mime="image/png",
                )
            with reuse_col:
                if st.button("Use as input for next generation"):
                    st.session_state["gen_carry_images"].append(out)
                    st.session_state["gen_last_output"] = None
                    st.rerun()


Main()
