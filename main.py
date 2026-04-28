import typer
from rich.console import Console
import requests
import json
import re
import os
from typing import Dict, Any, List

app = typer.Typer()
console = Console()

# --- AI Model Communication ---

def call_ai(prompt: str, is_json: bool = False):
    """
    Generic function to call the local AI model.
    Provides a structured prompt and handles potential JSON output.
    """
    # Enhanced prompt structure for better model guidance
    structured_prompt = f"""
    You are an expert AI assistant. Please follow the instructions precisely.
    Respond in the format requested.

    ---
    INSTRUCTIONS:
    {prompt}
    ---
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3-coder:latest",
                "prompt": structured_prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        raw_response = response.json().get("response", "")

        if is_json:
            # The model might still wrap the JSON in markdown, so we clean it.
            match = re.search(r"```json\n(.*)\n```", raw_response, re.DOTALL)
            if match:
                clean_response = match.group(1)
            else:
                clean_response = raw_response
            
            try:
                return json.loads(clean_response)
            except json.JSONDecodeError:
                console.print("[bold red]AI did not return valid JSON. Aborting.[/bold red]")
                console.print(f"Received: {clean_response}")
                return None
        return raw_response

    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to Ollama: {e}[/bold red]")
        return None
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        return None

# --- Tutorial Type Classification ---

def get_topic_type(topic: str) -> str:
    """Uses the AI to classify the topic as 'code' or 'text'."""
    console.print(f"[bold blue]Analyzing your request to determine output format...[/bold blue]")
    prompt = f"""
    Analyze the following user request: '{topic}'
    
    Is the user asking primarily for a specific programming language, framework, or software tool where code examples would be the main focus? Or are they asking for theoretical, scientific, or general knowledge content?

    Respond with a single word: 'code' for programming/coding topics, and 'text' for all others.
    """
    response = call_ai(prompt)
    topic_type = response.strip().lower() if response else 'text'

    if 'code' in topic_type:
        console.print("[bold green]Format identified as 'code'. Will generate source files and a README.[/bold green]")
        return 'code'
    else:
        console.print("[bold green]Format identified as 'text'. Will generate detailed markdown documents.[/bold green]")
        return 'text'

# --- Text-Based Tutorial Generation (Recursive) ---

def generate_text_outline_recursive(topic_path: List[str], user_request: str, level: int = 0) -> List[Dict[str, Any]]:
    """Recursively generates a hierarchical outline for a text tutorial."""
    if level >= 3:  # Max recursion depth to prevent infinite loops
        return []

    current_topic = " -> ".join(topic_path)
    console.print(f"{'  ' * level}[bold cyan]Generating sub-outline for: '{current_topic}'...[/bold cyan]")
    
    prompt = f"""
    The user has requested: "{user_request}"
    
    We are currently detailing the section: '{current_topic}'.

    Break this section down into a list of more detailed sub-sections that would help the user understand this part thoroughly.
    Return a JSON array of strings. Each string is a sub-section title.
    If this topic is fundamental and cannot be broken down further, return an empty array [].

    Example for a machine learning section:
    ["Gradient Descent Mechanics", "Optimization Algorithms", "Convergence Analysis"]
    """
    sub_sections = call_ai(prompt, is_json=True)

    if not sub_sections:
        return []

    outline = []
    for title in sub_sections:
        new_topic_path = topic_path + [title]
        children = generate_text_outline_recursive(new_topic_path, user_request, level + 1)
        outline.append({"title": title, "path": new_topic_path, "children": children})
    
    return outline


def generate_text_tutorial(topic: str, output_dir: str):
    """Generates a hierarchical, long-form Markdown tutorial with numbered files."""
    os.makedirs(output_dir, exist_ok=True)
    plan_file = os.path.join(output_dir, "_plan.json")

    if os.path.exists(plan_file):
        console.print(f"[bold yellow]Found existing plan file '{plan_file}'. Resuming generation.[/bold yellow]")
        with open(plan_file, "r") as f:
            outline = json.load(f)
    else:
        console.print(f"[bold green]Generating tutorial based on your request...[/bold green]")
        outline = generate_text_outline_recursive([topic], topic)
        with open(plan_file, "w") as f:
            json.dump(outline, f, indent=2)

    # Generate content with hierarchical numbering
    def process_section(section_data: Dict[str, Any], numbers: List[int]):
        title = section_data['title']
        path = section_data['path']
        
        # Generate numbering like "1.1", "1.2", "2.1", etc.
        file_number = ".".join(map(str, numbers))
        
        console.print(f"[bold blue]Generating content for: '{' -> '.join(path)}' (file: {file_number}.md)...[/bold blue]")
        
        content_prompt = f"""
        The user requested: "{topic}"
        
        You are writing content for section: '{' -> '.join(path)}'.
        Specifically, write detailed content for '{title}'.

        - Assume the reader is learning from scratch.
        - Explain concepts thoroughly and clearly.
        - Use mathematical notation where appropriate.
        - Provide concrete examples and intuitions.
        - Use Markdown for formatting.
        - Start with a title "# {title}" and then provide the content.
        """
        content = call_ai(content_prompt)
        
        if content:
            file_path = os.path.join(output_dir, f"{file_number}.md")
            with open(file_path, "w") as f:
                f.write(content)
        else:
            console.print(f"[bold yellow]Warning: Could not generate content for section: '{title}'.[/bold yellow]")

        for i, child in enumerate(section_data.get("children", []), 1):
            process_section(child, numbers + [i])

    for i, section in enumerate(outline, 1):
        process_section(section, [i])
    
    # Clean up the plan file after successful completion
    if os.path.exists(plan_file):
        os.remove(plan_file)


# --- Code-Based Tutorial Generation ---

def generate_code_tutorial(topic: str, output_dir: str):
    """Generates a code-based tutorial with source files and a README."""
    console.print(f"[bold green]Generating code tutorial plan based on your request...[/bold green]")
    
    src_dir = os.path.join(output_dir, "src")
    os.makedirs(src_dir, exist_ok=True)

    outline_prompt = f"""
    User request: "{topic}"
    
    Create a logical, step-by-step plan of small code examples that a beginner can follow to learn from this request.
    
    Return a JSON array of objects. Each object must have a 'filename' and a 'description'.
    - 'filename' should be valid for the language (e.g., 01_basics.py, 02_functions.cpp).
    - 'description' should clearly explain what concept/skill this file teaches.

    Example:
    [
        {{"filename": "01_basics.py", "description": "Basic setup and Hello World example."}},
        {{"filename": "02_variables.py", "description": "Working with variables and data types."}}
    ]
    """
    plan = call_ai(outline_prompt, is_json=True)
    if not plan:
        console.print("[bold red]Failed to generate tutorial plan. Aborting.[/bold red]")
        return

    readme_content = f"# Tutorial\n\nThis tutorial teaches the following through a series of code examples. Follow them in order.\n\n"
    
    for i, item in enumerate(plan):
        filename = item.get("filename")
        description = item.get("description")
        if not filename or not description:
            continue

        console.print(f"[bold blue]Generating code for file {i+1}/{len(plan)}: '{filename}'...[/bold blue]")
        code_prompt = f"""
        User request: "{topic}"
        
        Write a single, clean code file for this tutorial.
        Filename: '{filename}'
        Purpose: '{description}'

        - Write clear, well-commented, and runnable code for this specific concept.
        - **IMPORTANT**: Output ONLY the raw source code for the file. Do not add any surrounding text, explanations, or markdown formatting like ```.
        """
        code_content = call_ai(code_prompt)
        
        if code_content:
            # The model sometimes still adds markdown, so we clean it just in case.
            code_content = re.sub(r"^\s*```[a-zA-Z]*\n", "", code_content)
            code_content = re.sub(r"\n```\s*$", "", code_content)
            
            with open(os.path.join(src_dir, filename), "w") as f:
                f.write(code_content)
            readme_content += f"### {i+1}. `{filename}`\n\n*   **Concept:** {description}\n\n"
        else:
            console.print(f"[bold yellow]Warning: Could not generate code for '{filename}'.[/bold yellow]")

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)

# --- Main CLI Command ---

@app.callback()
def main():
    """
    A CLI to generate long-form tutorials using AI.
    """
    pass

@app.command()
def version():
    """
    Display the version of the Tutorial CLI.
    """
    console.print("[bold green]Tutorial CLI version 0.1.0[/bold green]")

@app.command()
def create(
    topic: str = typer.Option(..., "--topic", "-t", help="What you want to learn. Can be a topic or a detailed request (e.g., 'I want to learn machine learning theory mathematically')."),
):
    """
    Generates a comprehensive, long-form tutorial based on your request.

    The tool automatically determines the best format:
    - For programming topics, it creates a directory with commented code files and a README.
    - For other topics, it creates a directory with numbered markdown files (1.md, 1.1.md, 1.2.md, etc.).
    """
    topic_type = get_topic_type(topic)
    output_dir = topic.lower().replace(" ", "_").replace(".", "")[:30] + "_tutorial"
    
    if topic_type == 'text':
        generate_text_tutorial(topic, output_dir)
    elif topic_type == 'code':
        generate_code_tutorial(topic, output_dir)

    console.print(f"\n[bold green]Tutorial generation complete! Your tutorial is available at: '{output_dir}'[/bold green]")

if __name__ == "__main__":
    app()
