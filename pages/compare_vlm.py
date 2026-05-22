import streamlit as st
from PIL import Image
from copy import deepcopy
from retrying import retry
from streamlit_float import float_init
from streamlit_theme import st_theme

from streamlit_app import (
    build_llm,
    get_llm_icon,
    get_openrouter_api_key,
    image_to_base64,
    is_valid_url,
)


def clear_context():
    """clear chat history"""
    st.session_state.pop("messages", None)
    st.rerun()
    st.toast("chat history cleared!")


@retry(
    stop_max_attempt_number=5,
    retry_on_result=lambda result: result is None,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    wrap_exception=False,
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
    """return formatted message for LLM to use from a native st.chat_input value.

    Expects a ChatInputValue with ``.text`` and ``.files`` (a list of UploadedFile),
    as returned by ``st.chat_input(accept_file="multiple", ...)``.
    """
    if not chatinput:
        return None
    text = chatinput.text
    if not text:
        st.toast("Your input must contain text")
        return None

    im_msg = []
    for f in chatinput.files or []:
        try:
            im_url = image_to_base64(Image.open(f))
        except Exception as e:
            st.toast(f"Problem loading {getattr(f, 'name', f)}: {e}")
            return None
        im_msg.append({"type": "image_url", "image_url": im_url})

    if im_msg:
        return {
            "role": "user",
            "content": [{"type": "text", "text": text}] + im_msg,
        }
    return {"role": "user", "content": text}


def get_mminput(st_container):
    """get formatted multimodal input from user using native st.chat_input.

    Requires streamlit>=1.56.0 for ``accept_file`` support.
    """
    with st_container:
        chatinput = st.chat_input(
            "Ask me anything about an image...",
            accept_file="multiple",
            file_type=["jpg", "jpeg", "png", "webp"],
        )
    return chatinput2msg(chatinput) if chatinput else None


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
        "google/gemma-4-26b-a4b-it:free",
        help="[list of free LVLM available at OpenRouter](https://openrouter.ai/models?order=pricing-low-to-high&modality=text%2Bimage-%3Etext&input_modalities=text,image&max_price=0&output_modalities=text)",
    )
    model_name_2 = st.sidebar.text_input(
        "model 2 (optional)",
        "nvidia/nemotron-nano-12b-v2-vl:free",
        help="""
        Add a second model to compare responds side-by-side

        [list of free LVLM available at OpenRouter](https://openrouter.ai/models?order=pricing-low-to-high&modality=text%2Bimage-%3Etext&input_modalities=text,image&max_price=0&output_modalities=text)
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


Main()
