import gradio as gr
from dotenv import load_dotenv
from typing import List, Tuple, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel
from typing import List, Tuple
from tools import search_tool, wiki_tool, save_tool, calculator_tool, content_generator_tool, unit_converter_tool, time_tool, file_reader_tool, code_execution_tool, translation_tool

from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time
import re

load_dotenv()

# Define the output model
class TaskResponse(BaseModel):
    query: str
    response: str
    tools_used: list[str]
    sources: list[str] = []

# Define the LLM
llm = ChatOpenAI(
    api_key="none-needed",
    model="qwen/qwen3-4b-thinking-2507",
    base_url="http://127.0.0.1:1234/v1",
    temperature=0.5,
    max_tokens=1000
)

# Define the parser
parser = PydanticOutputParser(pydantic_object=TaskResponse)

# Define the prompt with chat history
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a versatile AI assistant capable of handling a wide range of tasks, including answering questions, performing calculations, generating content, conducting research, unit conversions, time zone queries, reading files, executing Python code, and translating text.
            Use the provided tools when necessary and tailor your response to the user's query.
            - For file reading, use the FileReader tool with the provided file path.
            - For code execution, use the CodeExecutor tool to run Python code and return results or errors.
            - For translation, use the Translator tool to translate text into English or a specified language (e.g., 'translate to Spanish').
            For research tasks, include sources if applicable.
            Use the conversation history to provide context-aware responses.
            **Output only a valid JSON object in the format specified below. Do not include any additional text, comments, or tags like <think> outside the JSON.**
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Debug: Verify prompt type
print(f"Prompt type: {type(prompt)}")
try:
    print(f"Input variables: {prompt.input_variables}")
except AttributeError:
    print("Input variables not directly accessible; relying on automatic inference.")

# List of all available tools
all_tools = [
    search_tool,
    wiki_tool,
    save_tool,
    calculator_tool,
    content_generator_tool,
    unit_converter_tool,
    time_tool,
    file_reader_tool,
    code_execution_tool,
    translation_tool
]
tool_names = ["Search", "Wiki", "Save", "Calculator", "Content Generator", "Unit Converter", "Time Zone", "File Reader", "Code Executor", "Translator"]

# Initialize agent
llm = llm.bind_tools(all_tools)
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=all_tools)
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True, return_intermediate_steps=True)

# Function to format intermediate steps
def format_intermediate_steps(steps):
    if not steps:
        return "No intermediate steps taken."
    formatted = ""
    for i, step in enumerate(steps, 1):
        action = step[0]
        observation = step[1]
        formatted += f"Step {i}:\nAction: {action.tool}\nInput: {action.tool_input}\nObservation: {observation}\n\n"
    return formatted

def extract_json(text):
    # Strip out <think> and similar artifacts
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Find JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No valid JSON found in model output")


from langchain_core.messages import HumanMessage, AIMessage

def format_chat_history(history):
    formatted = []
    for role, message in history:
        if role == "human":
            formatted.append(HumanMessage(content=message))
        elif role == "assistant":
            formatted.append(AIMessage(content=message))
    return formatted

def run_agent(prompt_input, search_cb, wiki_cb, save_cb, calculator_cb, content_generator_cb, unit_converter_cb, time_cb, file_reader_cb, code_executor_cb, translator_cb, uploaded_file, history):
    selected_tools = []
    if search_cb:
        selected_tools.append("Search")
    if wiki_cb:
        selected_tools.append("Wiki")
    if save_cb:
        selected_tools.append("Save")
    if calculator_cb:
        selected_tools.append("Calculator")
    if content_generator_cb:
        selected_tools.append("Content Generator")
    if unit_converter_cb:
        selected_tools.append("Unit Converter")
    if time_cb:
        selected_tools.append("Time Zone")
    if file_reader_cb:
        selected_tools.append("File Reader")
    if code_executor_cb:
        selected_tools.append("Code Executor")
    if translator_cb:
        selected_tools.append("Translator")

    # Handle file upload for FileReader
    file_path = None
    if file_reader_cb and uploaded_file:
        file_path = uploaded_file
        prompt_input = f"{prompt_input}\nFile path: {file_path}"

    formatted_history = format_chat_history(history)
    try:
        response = agent_executor.invoke({"query": prompt_input, "chat_history": formatted_history})
        thoughts = format_intermediate_steps(response["intermediate_steps"])
        json_output = extract_json(response["output"])
        structured_response = parser.parse(json_output)
        summary = f"Response: {structured_response.response}\n\nTools Used: {', '.join(structured_response.tools_used) or 'None'}\n\nSources: {', '.join(structured_response.sources) or 'None'}"
        print(f"Debug - Thoughts: {thoughts[:100]}...")
        print(f"Debug - Summary: {summary[:100]}...")
        updated_history = history + [("human", prompt_input), ("assistant", structured_response.response)]
        return thoughts, summary, updated_history
    except Exception as e:
        error_msg = f"Error during agent execution: {str(e)}\nRaw output: {response['output']}"
        print(error_msg)
        return error_msg, "No summary available due to error.", history

def clear_inputs():
    return "", False, False, False, False, False, False, False, False, False, False, None, []

# Define custom Gradio themeß
theme = gr.themes.Soft(
    primary_hue="yellow",
    secondary_hue="green",
    neutral_hue="sky",
    text_size="lg",
    spacing_size="lg",
    radius_size="xxl",
    font="IBM Plex Sans, sans-serif",
    font_mono="IBM Plex Mono, monospace",
).set(
    shadow_drop='*button_primary_shadow_hover'
)


# Define the Gradio interface
with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        """
        # GPA-Qwen Enhanced
        A versatile AI assistant for research, calculations, content generation, file reading, code execution, translation, and more. Enter your query, select tools, and upload files if needed.
        """
    )
    history = gr.State(value=[])
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Enter Your Query",
                placeholder="e.g., 'Read my file' or 'Execute this Python code: print(2+2)' or 'Translate hello to Spanish'",
                lines=5,
                show_label=True
            )
            with gr.Accordion("Research and Creativity", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_cb = gr.Checkbox(label="Search", value=False, info="Search the web for information")
                            wiki_cb = gr.Checkbox(label="Wiki", value=False, info="Query Wikipedia for details")
                            save_cb = gr.Checkbox(label="Save", value=False, info="Save output to a text file")
                            content_generator_cb = gr.Checkbox(label="Content Generator", value=False, info="Generate creative content")
                                
            with gr.Accordion("Utilities", open=False):
                    with gr.Row():
                            calculator_cb = gr.Checkbox(label="Calculator", value=False, info="Perform mathematical calculations")
                            unit_converter_cb = gr.Checkbox(label="Unit Converter", value=False, info="Convert between units")
                            time_cb = gr.Checkbox(label="Time Zone", value=False, info="Get time in a specific timezone")
                            file_reader_cb = gr.Checkbox(label="File Reader", value=False, info="Read contents of an uploaded file")
                            code_executor_cb = gr.Checkbox(label="Code Executor", value=False, info="Execute Python code")
                            translator_cb = gr.Checkbox(label="Translator", value=False, info="Translate text to English or specified language")
            with gr.Row():
                    file_upload = gr.File(label="Upload File for File Reader", file_types=[".txt"], visible=True)
            
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Summary"):
                    summary_output = gr.Textbox(label="Summary", lines=10, placeholder="Main output goes here...", show_label=False)
                with gr.Tab("Thoughts"):
                    thoughts_output = gr.Textbox(label="Thoughts", lines=10, placeholder="Thoughts go here...", show_label=False)
    
    gr.Markdown(
        """
        ---
        *Powered by GPA-Qwen Enhanced | Version 1.0 *
        """,
        elem_classes=["footer"]
    )

    submit_btn.click(
        run_agent,
        inputs=[prompt_input, search_cb, wiki_cb, save_cb, calculator_cb, content_generator_cb, unit_converter_cb, time_cb, file_reader_cb, code_executor_cb, translator_cb, file_upload, history],
        outputs=[thoughts_output, summary_output, history]
    )
    clear_btn.click(
        clear_inputs,
        inputs=[],
        outputs=[prompt_input, search_cb, wiki_cb, save_cb, calculator_cb, content_generator_cb, unit_converter_cb, time_cb, file_reader_cb, code_executor_cb, translator_cb, file_upload, history]
    )

# Launch the app
demo.launch()