import streamlit as st
import os, sys, base64, httpx
from io import BytesIO
from PIL import Image


### Helper functions ###
def image_to_base64(pil_im):
    buffered = BytesIO()
    pil_im.save(buffered, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"


def is_valid_url(url_string):
    try:
        url = httpx.URL(url_string)
        return url.scheme and url.host
    except Exception:
        return False


### end of Helper functions ###


def clear_st_messages():
    """clear chat history"""
    st.toast("chat history cleared!")
    return st.session_state.pop("messages", None)


def get_init_user_query():
    assert "messages" in st.session_state, f"messages not in st.session_state"
    assert (
        len(st.session_state.messages) > 1
    ), f"chat history is not long enough, only {len(st.session_state.messages)} messages found."
    user_queries = [q for q in st.session_state.messages if q["role"] == "user"]
    return user_queries[0]["content"]


def get_user_image(clear_context: bool = True, force_b64: bool = True):
    with st.sidebar.form("image_upload_form"):
        im_url = st.text_input("Image URL")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
        if st.form_submit_button("upload image") and any([im_url, uploaded_file]):
            if im_url and uploaded_file:
                st.error(f"cannot provide both image file and image url")
                return None
            if im_url:
                if not is_valid_url(im_url):
                    st.warning(f"Your URL is not valid")
                    return None
                else:
                    im = (
                        image_to_base64(Image.open(BytesIO(httpx.get(im_url).content)))
                        if force_b64
                        else im_url
                    )
            elif uploaded_file:
                im = Image.open(uploaded_file)
                im = image_to_base64(im)
            if clear_context:
                clear_st_messages()
            st.session_state["image"] = im
    return st.session_state["image"] if "image" in st.session_state else None


def get_openrouter_api_key():
    # OpenReuter Credentials
    api_key = None
    with st.sidebar.expander("OpenRouter config", expanded=True):
        if "OpenRouter_key" in st.secrets:
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


@st.cache_resource
def build_llm(
    api_key: str, model_name: str = "meta-llama/llama-3.2-11b-vision-instruct:free"
):
    """creating an OpenRouter LLM
    free model list: https://openrouter.ai/models?order=pricing-low-to-high&max_price=0
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model_name,
        # model_kwargs={
        #   "headers": {
        #     "HTTP-Referer": getenv("APP_URL"), # Optional, for including app on Openrouter's ranking
        #     "X-Title": getenv("APP_TITLE"),
        #   }
        # },
    )


def generate_response(llm, query, im, use_context: bool = True):
    """Function for generating LLM response"""
    add_context = len(st.session_state.messages) > 1 and use_context
    init_query = get_init_user_query() if add_context else query

    message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": init_query},
                {"type": "image_url", "image_url": im},
            ],
        }
    ]
    if add_context:
        message += st.session_state.messages[2:]
    msg = st.toast(f"message context: {len(message)}")
    try:
        r = llm.invoke(message)
        return r.content
    except Exception as e:
        msg.toast(f"LLM invoke error: {e}")
        return None


def Main():
    # App title
    st.set_page_config(page_title="🤗💬 Image Q&A")
    st.sidebar.title("🤗💬 Image Q&A")
    api_key = get_openrouter_api_key()

    # build LLM and RAG Chain
    if api_key:
        model_name = st.sidebar.text_input(
            "model name",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            help="[list of free LVLM available at OpenRouter](https://openrouter.ai/models?max_price=0&order=pricing-low-to-high&modality=text%2Bimage-%3Etext)",
        )
        llm = build_llm(api_key, model_name=model_name)
        msg = st.toast(f"LLM {model_name} loaded from OpenRouter")

    # Get User's Image
    im = get_user_image()
    if not im:
        st.warning(f":point_left: Provide your image to get started")
        return None
    else:
        st.sidebar.image(im, caption="uploaded image")

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "What Question do you have about your image?",
            }
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input(disabled=not (api_key and im)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(llm, query=prompt, im=im)
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    Main()
