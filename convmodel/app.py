import streamlit as st
from convmodel import ConversationModel


def show_conversation(placeholder, context):
    markdown = ""
    icon = ["🙂", "🤖"]
    for idx, item in enumerate(context):
        markdown += f"* {icon[idx % 2]} {item}\n"
    placeholder.markdown(markdown)


@st.cache
def load_model(model_dir):
    model = ConversationModel.from_pretrained(model_dir)
    return model


# Prepare context
if "context" not in st.session_state:
    st.session_state.context = []
# Prepare key for text input
if "text_input_key" not in st.session_state:
    st.session_state.text_input_key = 0


#
# Set sidebar
#
model_dir = st.sidebar.text_input('Model folder', "model")
top_p = st.sidebar.number_input('top_p', value=0.90)
top_k = st.sidebar.number_input('top_k', value=30)
max_length = st.sidebar.number_input('Max input length to the model', value=1024)

#
# Set main layout
#
st.markdown("# Try your conversation model")

# Load model
model = load_model(model_dir)
st.success(f'Loading model from {model_dir} succeeded 😉')

col_left, col_right = st.columns(2)

# Left side
input_area = col_left.empty()
reset_button = col_left.button('Reset conversation')

if reset_button:
    st.session_state.context = []
    st.session_state.text_input_key += 1

user_input = input_area.text_input('User input', "", key=str(st.session_state.text_input_key))

# Right side
conversation_area = col_right.empty()
if reset_button:
    conversation_area.write("")

if (not reset_button) and user_input:
    # If user inputted a text, clea input text
    # This technique is explained below
    # https://github.com/streamlit/streamlit/issues/623#issuecomment-551755236
    st.session_state.text_input_key += 1
    input_area.text_input('User input', "", key=str(st.session_state.text_input_key))

    st.session_state.context.append(user_input)
    show_conversation(conversation_area, st.session_state.context)

    with st.spinner(text="(・ω・ ).oO(thinking...)"):
        gen = model.generate(
            context=st.session_state.context,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            max_length=max_length,
            bad_words_ids=[[model.hf_tokenizer.unk_token_id]],
        )
        res = gen.responses[0]
        st.session_state.context.append(res)

    show_conversation(conversation_area, st.session_state.context)
