# Tkinter-based version of the AgenticPi application
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from tools import (
    search_tool, wiki_tool, save_tool, calculator_tool,
    content_generator_tool, unit_converter_tool, time_tool,
    file_reader_tool, code_execution_tool, translation_tool
)
import time, re

load_dotenv()
# --- LLM and agent setup (same as original) ---
class TaskResponse(BaseModel):
    query: str
    response: str
    tools_used: list[str]
    sources: list[str] = []

llm = ChatOpenAI(api_key="none-needed", model="qwen3:0.6b",
                 base_url="http://127.0.0.1:11434/v1",
                 temperature=0.5, max_tokens=1000)
parser = PydanticOutputParser(pydantic_object=TaskResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are a versatile AI assistant capable of handling a wide range of tasks...
            **Output only a valid JSON object in the format specified below.**
            {format_instructions}
            """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

all_tools = [
    search_tool, wiki_tool, save_tool, calculator_tool, content_generator_tool,
    unit_converter_tool, time_tool, file_reader_tool, code_execution_tool,
    translation_tool
]
llm = llm.bind_tools(all_tools)
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=all_tools)
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True, return_intermediate_steps=True)

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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("No valid JSON found in model output")

def format_chat_history(history):
    formatted = []
    for role, message in history:
        if role == "human":
            formatted.append(HumanMessage(content=message))
        elif role == "assistant":
            formatted.append(AIMessage(content=message))
    return formatted

def run_agent(prompt_input, search_cb, wiki_cb, save_cb, calculator_cb,
              content_generator_cb, unit_converter_cb, time_cb,
              file_reader_cb, code_executor_cb, translator_cb,
              uploaded_file, history):
    selected_tools = []
    if search_cb: selected_tools.append("Search")
    if wiki_cb: selected_tools.append("Wiki")
    if save_cb: selected_tools.append("Save")
    if calculator_cb: selected_tools.append("Calculator")
    if content_generator_cb: selected_tools.append("Content Generator")
    if unit_converter_cb: selected_tools.append("Unit Converter")
    if time_cb: selected_tools.append("Time Zone")
    if file_reader_cb: selected_tools.append("File Reader")
    if code_executor_cb: selected_tools.append("Code Executor")
    if translator_cb: selected_tools.append("Translator")

    file_path = None
    if file_reader_cb and uploaded_file:
        file_path = uploaded_file
        prompt_input = f"{prompt_input}\nFile path: {file_path}"

    formatted_history = format_chat_history(history)
    try:
        response = agent_executor.invoke({"query": prompt_input, "chat_history": formatted_history})
        thoughts = format_intermediate_steps(response["intermediate_steps"])
        json_output = extract_json(response["output"])
        structured = parser.parse(json_output)
        summary = (f"Response: {structured.response}\n\n"
                   f"Tools Used: {', '.join(structured.tools_used) or 'None'}\n\n"
                   f"Sources: {', '.join(structured.sources) or 'None'}")
        updated_history = history + [("human", prompt_input), ("assistant", structured.response)]
        return thoughts, summary, updated_history
    except Exception as e:
        return str(e), "No summary available due to error.", history

# --- Tkinter UI Setup ---
root = tk.Tk()
root.title("AgenticPi")

history = []

left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

tk.Label(left_frame, text="Enter Your Query:").pack(anchor="w")
query_input = tk.Text(left_frame, height=5, width=50)
query_input.pack()

rc_frame = tk.LabelFrame(left_frame, text="Research and Creativity")
rc_frame.pack(fill="both", padx=5, pady=5)
search_var = tk.BooleanVar()
wiki_var = tk.BooleanVar()
save_var = tk.BooleanVar()
content_var = tk.BooleanVar()
tk.Checkbutton(rc_frame, text="Search", variable=search_var).pack(anchor="w")
tk.Checkbutton(rc_frame, text="Wiki", variable=wiki_var).pack(anchor="w")
tk.Checkbutton(rc_frame, text="Save", variable=save_var).pack(anchor="w")
tk.Checkbutton(rc_frame, text="Content Generator", variable=content_var).pack(anchor="w")

util_frame = tk.LabelFrame(left_frame, text="Utilities")
util_frame.pack(fill="both", padx=5, pady=5)
calculator_var = tk.BooleanVar()
unit_var = tk.BooleanVar()
time_var = tk.BooleanVar()
file_var = tk.BooleanVar()
code_var = tk.BooleanVar()
translator_var = tk.BooleanVar()
tk.Checkbutton(util_frame, text="Calculator", variable=calculator_var).pack(anchor="w")
tk.Checkbutton(util_frame, text="Unit Converter", variable=unit_var).pack(anchor="w")
tk.Checkbutton(util_frame, text="Time Zone", variable=time_var).pack(anchor="w")
tk.Checkbutton(util_frame, text="File Reader", variable=file_var).pack(anchor="w")
tk.Checkbutton(util_frame, text="Code Executor", variable=code_var).pack(anchor="w")
tk.Checkbutton(util_frame, text="Translator", variable=translator_var).pack(anchor="w")

tk.Label(left_frame, text="Upload File for File Reader:").pack(anchor="w")
file_path_var = tk.StringVar()
file_entry = tk.Entry(left_frame, textvariable=file_path_var, width=40)
file_entry.pack(side="left", padx=(0,5))
def browse_file():
    fname = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if fname: file_path_var.set(fname)
tk.Button(left_frame, text="Browse", command=browse_file).pack(side="left")

button_frame = tk.Frame(left_frame)
button_frame.pack(pady=5)
submit_btn = tk.Button(button_frame, text="Submit", width=10)
clear_btn = tk.Button(button_frame, text="Clear", width=10)
submit_btn.pack(side="left", padx=5)
clear_btn.pack(side="left", padx=5)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
notebook = ttk.Notebook(right_frame)
summary_tab = ttk.Frame(notebook)
thoughts_tab = ttk.Frame(notebook)
notebook.add(summary_tab, text="Summary")
notebook.add(thoughts_tab, text="Thoughts")
notebook.pack(fill="both", expand=True)

summary_output = ScrolledText(summary_tab, height=20, width=80)
summary_output.pack(fill="both", expand=True)
thoughts_output = ScrolledText(thoughts_tab, height=20, width=80)
thoughts_output.pack(fill="both", expand=True)

def on_submit():
    global history
    prompt_text = query_input.get("1.0", tk.END).strip()
    thoughts, summary, new_history = run_agent(
        prompt_text,
        search_var.get(), wiki_var.get(), save_var.get(), calculator_var.get(),
        content_var.get(), unit_var.get(), time_var.get(),
        file_var.get(), code_var.get(), translator_var.get(),
        file_path_var.get() or None,
        history
    )
    thoughts_output.delete("1.0", tk.END)
    thoughts_output.insert(tk.END, thoughts)
    summary_output.delete("1.0", tk.END)
    summary_output.insert(tk.END, summary)
    history = new_history
    
    
def on_clear():
    query_input.delete("1.0", tk.END)
    search_var.set(False); wiki_var.set(False); save_var.set(False)
    calculator_var.set(False); content_var.set(False); unit_var.set(False)
    time_var.set(False); file_var.set(False); code_var.set(False)
    translator_var.set(False)
    file_path_var.set("")
    global history
    history = []

submit_btn.config(command=on_submit)
clear_btn.config(command=on_clear)

root.mainloop()
