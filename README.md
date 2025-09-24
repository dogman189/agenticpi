# General Purpose Agent(GPA) Qwen

## Overview

This project builds an AI-powered multitalented agent that leverages LangChain to assist users in exploring topics and generating structured  outputs. It allows users to input research queries, and the agent will dynamically use various tools (search engine, Wikipedia, calculator, creativity and a text-saving functionality) to explore the topic and provide a structured response containing a summary, sources, and a record of tools used.  The output can also be saved to a text file.  The UI is built with Gradio.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/dogman189/gpa-qwen.git
    cd gpa-qwen
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Environment Variables:**  The code uses environment variables for API keys.

   *   **`API_KEY` (OpenAI):**  While currently configured to use Ollama, you might need to configure this depending on your OpenAI usage.  It may need to be changed to an OpenAI key if using a different LLM that requires it.
    *   **Ensure that an instance of `qwen3:1.7b` is running locally with your LLM server, or is available.  In this example, the code defaults to `http://localhost:11434/v1`

## Usage

1.  **Run the Gradio Interface:**
    ```bash
    python app.py
    ```

2.  **Access the Interface:**  Open your web browser and go to the address printed by the `gradio_interface`.  Typically, it will be something like `http://localhost:7860`.

3.  **Enter a Research Query:**  Type your research question or topic into the text box, select your tools and click the "Submit" or similar button.

4.  **Review the Output:** The agent will generate a structured response containing:
    *   **Topic:**  The researched topic.
    *   **Summary:** A concise summary of the topic.
    *   **Sources:** A list of sources used for research.
    *   **Tools Used:** A record of the tools employed (e.g., search engine, Wikipedia).

5. **Saving Research:**  The agent automatically saves the detailed research result to a file named `research_output.txt`. Each new result will be appended to the file with a timestamp.

## Code Structure

The project is organized into the following modules:

*   **`app.py`**: This file contains the Gradio interface code for interacting with the research agent. It takes user input, passes it to the agent executor, and displays the output in a Gradio textbox.
*   **`main.py`**:  This is the core logic of the research agent. It defines the research response schema, loads the LLM, sets up the prompt, and creates the agent executor.
*   **`tools.py`**:  This module provides the tools used by the agent:
    *   `save_text_to_file`: Saves the research output to a text file.
    *   `search`:  Uses DuckDuckGo to search the web.
    *   `wiki_tool`: Queries Wikipedia using Langchain.
    *   `calculator_tool`: Uses a calculator algorithm to solve math problems.
    *   `content_generator`: Makes creative content instead of structured responses.

### Data Structures

*   **`ResearchResponse` (in `main.py`):** A Pydantic model that defines the structure of the research output.

## Contributing

1.  **Fork the Repository:**  Create a fork of the repository on GitHub.
2.  **Create a Branch:**  Create a new branch for your feature or bug fix.
3.  **Make Changes:**  Commit your changes to the branch.
4.  **Submit a Pull Request:**  Create a pull request to the main repository.

## Dependencies

The project relies on the following libraries:

*   `gradio`
*   `python-dotenv`
*   `langchain`
*   `langchain-openai`
*    `langchain-anthropic`
*   `pydantic`
*   `duckduckgo-search`

See the `requirements.txt` file for the exact versions.

## License

[Choose an appropriate license, e.g., MIT License]

---
