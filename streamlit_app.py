import streamlit as st
import base64
import httpx
from io import BytesIO


def image_to_base64(pil_im):
    if pil_im.mode != "RGB":
        pil_im = pil_im.convert("RGB")
    buffered = BytesIO()
    pil_im.save(buffered, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"


def is_valid_url(url_string):
    try:
        url = httpx.URL(url_string)
        return url.scheme and url.host
    except Exception:
        return False


def get_openrouter_api_key():
    # OpenReuter Credentials
    api_key = None
    with st.sidebar.expander("OpenRouter config", expanded=False):
        if "OpenRouter_key" in st.secrets and not st.toggle(
            "use your own API key",
            help="see free API key limits [here](https://openrouter.ai/docs/api-reference/limits#rate-limits-and-credits-remaining)",
        ):
            st.success("OpenRouter API key already provided!", icon="✅")
            api_key = st.secrets["OpenRouter_key"]
        else:
            api_key = st.text_input("Enter API Key:", type="password")
            if not api_key:
                st.warning("Please enter your credentials!", icon="⚠️")
            else:
                st.success("Proceed to entering your prompt message!", icon="👉")
        st.markdown(
            "📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!"
        )
    return api_key


@st.cache_data
def get_llm_icon(model_name: str, theme: str = "dark"):
    """return an LLM Icon thanks to lobe-icons
    ref: https://github.com/lobehub/lobe-icons#-cdn-usage
    """
    icon_slug = model_name.split("/")[0]
    icon_slug = icon_slug.split("-")[0] if "-" in icon_slug else icon_slug
    icon_url = (
        f"https://unpkg.com/@lobehub/icons-static-png@latest/{theme}/{icon_slug}.png"
    )
    return icon_url if is_valid_url(icon_url) else None


@st.cache_resource
def build_llm(
    api_key: str,
    model_name: str = "meta-llama/llama-3.2-11b-vision-instruct:free",
    temperature: float = 0.7,
):
    """creating an OpenRouter LLM
    free model list: https://openrouter.ai/models?order=pricing-low-to-high&max_price=0
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model_name,
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "https://image-qna.streamlit.app/",  # Optional, for including app on Openrouter's ranking
            "X-Title": "Image Q&A",
        },
    )


if __name__ == "__main__":
    pg = st.navigation(
        [
            st.Page("pages/image-vqa.py", title="Image Q&A", default=True),
            st.Page("pages/compare_vlm.py", title="Compare VLMs"),
        ],
        position="top",
    )
    pg.run()
