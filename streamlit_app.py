import streamlit as st
import os, sys, base64, httpx
from io import BytesIO
from PIL import Image
from st_multimodal_chatinput import multimodal_chatinput
from file_chat_input import file_chat_input
from streamlit_float import float_init

float_init()


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


def generate_response(llm):
    """Function for generating LLM response"""
    message = st.session_state.messages
    msg = st.toast(f"message context: {len(message)}")
    try:
        r = llm.invoke(message)
        return r.content
    except Exception as e:
        msg.toast(f"LLM invoke error: {e}")
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


def Main():
    # App title
    # st.set_page_config(page_title="🤗💬 Image Q&A")
    st.sidebar.title("🤗💬 Image Q&A")
    api_key = get_openrouter_api_key()

    # build LLM and RAG Chain
    if api_key:
        model_name = st.sidebar.text_input(
            "model name",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            help="[list of free LVLM available at OpenRouter](https://openrouter.ai/models?max_price=0&order=pricing-low-to-high&modality=text%2Bimage-%3Etext)",
        )
        temperature = st.sidebar.slider(
            "temperature",
            value=0.7,
            min_value=0.0,
            max_value=1.0,
            help="lower temperature's responses are more deterministic, higher temperature's more creative",
        )
        llm = build_llm(api_key, model_name=model_name, temperature=temperature)
        msg = st.toast(f"LLM {model_name} loaded from OpenRouter")

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

    # Display chat messages
    for message in st.session_state.messages:
        write_message(message, st_container=st.chat_message(message["role"]))
        # with st.chat_message(message["role"]):
        #     st.write(message["content"])

    # Chat Layout management
    st_msg_board = st.empty()
    st_thinking_placeholder = st.empty()
    msg_container = st.container()
    msg_container.float("bottom: 0")

    # User-provided prompt
    user_input = get_mminput(msg_container)
    if user_input:
        st.session_state.messages.append(user_input)
        write_message(user_input, st_container=st_msg_board.chat_message("user"))

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st_thinking_placeholder.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(llm)  # , query=prompt, im=im)
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    Main()
