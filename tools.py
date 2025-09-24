from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import math
import os
import traceback
import pandas as pd
import json
from googletrans import Translator
from sympy import sympify
import pint
import pytz


# 1. File Reader Tool
def read_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        return f"File not found: {filepath}"
    
    try:
        file_extension = os.path.splitext(filepath)[1].lower()
        if file_extension == ".txt" or file_extension == ".md" or file_extension == ".py" or file_extension == ".html":
            with open(filepath, "r", encoding="utf-8") as file:
                return file.read()
        elif file_extension == ".csv":
            df = pd.read_csv(filepath)
            return df.to_string()
        elif file_extension == ".json":
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
                return json.dumps(data, indent=2)
        else:
            return f"Unsupported file type: {file_extension}. Supported types: .txt, .csv, .json, .md, .py, .html"
    except Exception as e:
        return f"Error reading file: {str(e)}"

file_reader_tool = Tool(
    name="FileReader",
    func=read_file,
    description="Reads and returns the contents of a text-based file (.txt, .csv, .json, .md, .py, .html). Input should be a valid file path."
)

# 4. Code Execution Tool
def run_python_code(code: str) -> str:
    local_vars = {}
    try:
        exec(code, {}, local_vars)
        return str(local_vars) if local_vars else "Code executed successfully."
    except Exception:
        return f"Error:\n{traceback.format_exc()}"

code_execution_tool = Tool(
    name="CodeExecutor",
    func=run_python_code,
    description="Executes a block of Python code and returns output variables or errors."
)


# 6. Translation Tool
translator = Translator()

def translate_text(text: str, target_language: str = "en") -> str:
    try:
        result = translator.translate(text, dest=target_language)
        return result.text
    except Exception as e:
        return f"Translation error: {str(e)}"

translation_tool = Tool(
    name="Translator",
    func=lambda x: translate_text(x),
    description="Translates text into English or a specified language (e.g., 'es' for Spanish). Input format: 'text' or 'text, target_language'."
)

# Initialize pint unit registry
ureg = pint.UnitRegistry()

def save_to_txt(data: str, filename: str = "output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Task Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

def calculate(expression: str) -> str:
    try:
        # Use SymPy for advanced mathematical calculations
        result = sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def generate_content(prompt: str) -> str:
    # Placeholder for content generation; in practice, this could use the LLM directly
    return f"Generated content based on prompt: {prompt}"

def convert_units(input_str: str) -> str:
    try:
        parts = input_str.split()
        value = float(parts[0])
        from_unit = parts[1]
        to_unit = parts[3]  # Assumes format "value from_unit to to_unit"
        quantity = value * ureg(from_unit)
        converted = quantity.to(to_unit)
        return f"{converted.magnitude} {to_unit}"
    except Exception as e:
        return f"Error in unit conversion: {str(e)}"

def get_time(timezone: str) -> str:
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"Error getting time: {str(e)}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves task output to a text file."
)

search_tool = Tool(
    name="search",
    func=DuckDuckGoSearchRun().run,
    description="Search the web for information"
)

wiki_tool = Tool(
    name="wiki",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100000)).run,
    description="Query Wikipedia for information"
)

calculator_tool = Tool(
    name="calculator",
    func=calculate,
    description="Perform mathematical calculations including symbolic math using SymPy, e.g., 'solve x^2 - 4 = 0' or 'diff(x^2, x)'"
)

content_generator_tool = Tool(
    name="content_generator",
    func=generate_content,
    description="Generate creative content based on a given prompt"
)

unit_converter_tool = Tool(
    name="unit_converter",
    func=convert_units,
    description="Convert between units. Provide input like '10 km to miles'."
)

time_tool = Tool(
    name="time_zone",
    func=get_time,
    description="Get current time in a specified time zone, e.g., 'America/New_York'."
)

import asyncio
from functools import wraps

async def async_search(query):
    return await asyncio.to_thread(search_tool.func, query)

async def async_wiki(query):
    return await asyncio.to_thread(wiki_tool.func, query)