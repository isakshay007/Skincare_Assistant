import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Skin Care Assistant👩‍⚕️")
st.markdown("Welcome to our Skin Care Assistant powered by Lyzr Automata! Get tailored skincare recommendations by entering your skin type and any concerns you want to tackle.")

input = st.text_input("Enter your skin type and any particular concerns you'd like to address:",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def skincare_generation(input):
    generator_agent = Agent(
        role="SKINCARE CONSULTANT expert",
        prompt_persona=f"Your task is to RECOMMEND a CUSTOMIZED SKINCARE REGIMEN tailored to user entered skin types"
    )

    prompt = f"""
You are an Expert SKINCARE CONSULTANT. Always introduce yourself. Your task is to RECOMMEND a CUSTOMIZED SKINCARE REGIMEN tailored to user entered skin types—be it OILY, SENSITIVE, COMBINATION, NORMAL, or DRY and also based on SPECIFIC CONCERNS or requirements mentioned by the user.

Here's your step-by-step guide:

1. FIRST, IDENTIFY the user's skin type based on the information they provide.

2. SELECT a SUITABLE CLEANSER that is specifically FORMULATED for the identified skin type.

3. CHOOSE a MOISTURIZER that will HYDRATE the skin without causing irritation or imbalance.

4. RECOMMEND a SUNSCREEN that offers adequate protection and is compatible with their skin type.

5. ADD additional products such as toners, serums, exfoliants, or masks that ADDRESS SPECIFIC CONCERNS or requirements mentioned by the user.

6. ENSURE that all recommended products are NON-COMEDOGENIC for oily or acne-prone skin types and FRAGRANCE-FREE for sensitive skin when necessary.

7. PROVIDE USAGE INSTRUCTIONS for each product to OPTIMIZE their effectiveness and ensure proper application.

You MUST always consider the user's unique requirements and preferences when curating this skincare routine.

      """

    generator_agent_task = Task(
        name="skincare Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Generate!"):
    solution = skincare_generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent Optimize your code. For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)