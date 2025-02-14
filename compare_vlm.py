import streamlit as st
import os, sys, base64, httpx
from io import BytesIO
from PIL import Image
from copy import deepcopy
from st_multimodal_chatinput import multimodal_chatinput
from file_chat_input import file_chat_input
from streamlit_float import float_init
from streamlit_theme import st_theme


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


def clear_context():
    """clear chat history"""
    st.session_state.pop("messages", None)
    st.rerun()
    st.toast("chat history cleared!")


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
        # model_kwargs={
        #     "headers": {
        #         "HTTP-Referer": "https://image-qna.streamlit.app/",  # Optional, for including app on Openrouter's ranking
        #         "X-Title": "Image Q&A",
        #     }
        # },
    )


def generate_response(llm, model_name: str):
    """Function for generating LLM response"""
    messages = deepcopy(st.session_state.messages)
    messages = [m for m in messages if m["role"] in [model_name, "assistant", "user"]]
    for m in messages:
        m["role"] = "assistant" if m["role"] == model_name else m["role"]
    # msg = st.toast(f"message context: {len(message)}")
    try:
        r = llm.invoke(messages)
        return r.content
    except Exception as e:
        st.toast(f"LLM invoke error: {e}")
        return None


def chatinput2msg(chatinput):
    if not chatinput:
        return None
    if not chatinput["message"]:  # ["text"]:
        st.toast("Your input must contain text")
        return None

    im_msg = []
    if chatinput["files"]:  # ["images"]:
        for im in chatinput["files"]:  # ["images"]:
            if im["content"].startswith("data:image"):
                im_msg.append({"type": "image_url", "image_url": im["content"]})
            else:
                st.toast(f"Problem loading {im} from your input")
                return None

    if im_msg:
        return {
            "role": "user",
            "content": [
                # {"type": "text", "text": chatinput["text"]},
                {"type": "text", "text": chatinput["message"]},
            ]
            + im_msg,
        }
    else:
        # return {"role": "user", "content": chatinput["text"]}
        return {"role": "user", "content": chatinput["message"]}


def get_mminput(st_container):
    msg = None
    # chatinput = multimodal_chatinput(
    #     default=None,
    #     disabled=False,  # placeholder="Ask me anything about images..."
    # )
    with st_container:
        chatinput = file_chat_input("Ask me anything about an image...")
    if chatinput:
        msg = chatinput2msg(chatinput)
    return msg


def write_message(msg, st_container):
    with st_container:
        if type(msg["content"]) == str:
            st.write(msg["content"])
        elif type(msg["content"]) == list:
            for m in msg["content"]:
                if m["type"] == "text":
                    st.write(m["text"])
                elif m["type"] == "image_url":
                    st.image(m["image_url"])
        else:
            return None


@st.fragment
def show_chat_interface(
    chatinput, st_container, llm, model_name: str, page_theme: str = "dark"
):
    # Display Previous chat messages
    msg_history = [
        m
        for m in st.session_state.messages
        if m["role"] in ["assistant", "user", model_name]
    ]
    for message in msg_history:
        write_message(
            message,
            st_container=st_container.chat_message(
                message["role"],
                avatar=(
                    get_llm_icon(model_name, theme=page_theme)
                    if message["role"] != "user"
                    else None
                ),
            ),
        )

    # Chat Layout management
    st_msg_board = st_container.empty()
    st_thinking_placeholder = st_container.empty()

    # User-provided prompt
    if chatinput:
        if chatinput != msg_history[-1]:
            st.session_state.messages.append(chatinput)
            write_message(chatinput, st_container=st_msg_board.chat_message("user"))

    # Generate a new response if last message is from user
    if not st.session_state.messages[-1]["role"] in [model_name, "assistant"]:
        with st_thinking_placeholder.chat_message(
            "assistant", avatar=get_llm_icon(model_name, theme=page_theme)
        ):
            with st.spinner("Thinking..."):
                response = generate_response(llm, model_name=model_name)
                st.write(response)
        message = {"role": model_name, "content": response}
        st.session_state.messages.append(message)


def Main():
    # App title
    st.set_page_config(
        page_title="Image Q&A",
        layout="wide",
        page_icon="https://seekingvega.github.io/sv-journal/assets/images/sv-favicon-v2.png",
    )
    st.sidebar.title("Image Q&A")
    float_init()
    page_theme = st_theme()["base"]
    st.logo(get_llm_icon("openrouter", page_theme))
    api_key = get_openrouter_api_key()

    if not api_key:
        return None

    # build LLM and RAG Chain
    model_name_1 = st.sidebar.text_input(
        "model 1",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        help="[list of free LVLM available at OpenRouter](https://openrouter.ai/models?max_price=0&order=pricing-low-to-high&modality=text%2Bimage-%3Etext)",
    )
    model_name_2 = st.sidebar.text_input(
        "model 2 (optional)",
        "",
        help="""
        Add a second model to compare responds side-by-side

        [list of free LVLM available at OpenRouter](https://openrouter.ai/models?max_price=0&order=pricing-low-to-high&modality=text%2Bimage-%3Etext)
        """,
    )
    temperature = st.sidebar.slider(
        "temperature",
        value=0.0,
        min_value=0.0,
        max_value=1.0,
        help="lower temperature's responses are more deterministic, higher temperature's more creative",
    )
    dual_model = model_name_2 and model_name_1
    llm1 = build_llm(api_key, model_name=model_name_1, temperature=temperature)
    msg = st.toast(f"LLM {model_name_1} loaded from OpenRouter")
    llm2 = (
        build_llm(api_key, model_name=model_name_2, temperature=temperature)
        if dual_model
        else None
    )
    if llm2:
        msg.toast(f"LLM {model_name_2} also loaded from OpenRouter")

    if st.sidebar.button(f"Clear Context"):
        clear_context()

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "What Question do you have about your image?",
            }
        ]

    # Chat Layout management
    msg_container = st.container()
    user_input = get_mminput(msg_container)
    msg_container.float("bottom: 0")
    if dual_model:
        cols = st.columns(2)
        chat_container1 = cols[0]
        chat_container2 = cols[1]
    else:
        chat_container1 = st
        chat_container2 = None

    show_chat_interface(
        chatinput=user_input,
        st_container=chat_container1,
        llm=llm1,
        model_name=model_name_1,
        page_theme=page_theme,
    )
    if dual_model:
        show_chat_interface(
            chatinput=user_input,
            st_container=chat_container2,
            llm=llm2,
            model_name=model_name_2,
            page_theme=page_theme,
        )


if __name__ == "__main__":
    Main()
