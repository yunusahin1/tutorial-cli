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
                "model": "gpt-oss:20b",
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
    console.print(f"[bold blue]Analyzing topic: '{topic}' to determine output format...[/bold blue]")
    prompt = f"""
    Analyze the topic '{topic}'. Is it primarily about a specific programming language, framework, or software tool that involves writing code? Or is it a theoretical, scientific, or general knowledge subject?

    Respond with a single word: 'code' for programming topics, and 'text' for all others.
    """
    response = call_ai(prompt)
    topic_type = response.strip().lower() if response else 'text'

    if 'code' in topic_type:
        console.print("[bold green]Topic identified as 'code'. Will generate source files and a README.[/bold green]")
        return 'code'
    else:
        console.print("[bold green]Topic identified as 'text'. Will generate a Markdown document.[/bold green]")
        return 'text'

# --- Text-Based Tutorial Generation (Recursive) ---

def generate_text_outline_recursive(topic_path: List[str], level: int = 0) -> List[Dict[str, Any]]:
    """Recursively generates a hierarchical outline for a text tutorial."""
    if level >= 3:  # Max recursion depth to prevent infinite loops
        return []

    current_topic = " -> ".join(topic_path)
    console.print(f"{'  ' * level}[bold cyan]Generating sub-outline for: '{current_topic}'...[/bold cyan]")
    
    prompt = f"""
    You are a curriculum designer creating a tutorial on '{topic_path[0]}'.
    We are currently detailing the section: '{current_topic}'.

    Break this section down into a list of more detailed sub-sections.
    Return a JSON array of strings. Each string is a sub-section title.
    If this topic is fundamental and cannot be broken down further, return an empty array [].

    Example for 'Calculus -> Derivatives':
    ["Definition of the Derivative", "The Power Rule", "The Product Rule", "The Chain Rule"]
    """
    sub_sections = call_ai(prompt, is_json=True)

    if not sub_sections:
        return []

    outline = []
    for title in sub_sections:
        new_topic_path = topic_path + [title]
        children = generate_text_outline_recursive(new_topic_path, level + 1)
        outline.append({"title": title, "path": new_topic_path, "children": children})
    
    return outline


def generate_text_tutorial(topic: str, output_file: str):
    """Generates a hierarchical, long-form Markdown tutorial."""
    plan_file = f"{os.path.splitext(output_file)[0]}_plan.json"

    if os.path.exists(plan_file):
        console.print(f"[bold yellow]Found existing plan file '{plan_file}'. Resuming generation.[/bold yellow]")
        with open(plan_file, "r") as f:
            outline = json.load(f)
    else:
        console.print(f"[bold green]Generating hierarchical tutorial outline for '{topic}'...[/bold green]")
        outline = generate_text_outline_recursive([topic])
        with open(plan_file, "w") as f:
            json.dump(outline, f, indent=2)

    with open(output_file, "w") as f:
        f.write(f"# Comprehensive Tutorial: {topic}\\n\\n")

    # Flatten the outline and generate content
    def process_section(section_data: Dict[str, Any], level: int):
        title = section_data['title']
        path = section_data['path']
        
        console.print(f"{'  ' * level}[bold blue]Generating content for: '{' -> '.join(path)}'...[/bold blue]")
        
        content_prompt = f"""
        You are an expert technical writer and educator. Your task is to write a single, detailed section for a comprehensive tutorial on '{topic}'.
        
        The full path to the current section is: '{' -> '.join(path)}'.
        
        Write the content for the final part of that path ('{title}').
        - Assume the reader is a beginner.
        - Explain the concepts clearly and thoroughly.
        - Provide examples where helpful.
        - Use Markdown for formatting.
        - **IMPORTANT**: Do NOT write a title. The title is handled externally. Start directly with the content for '{title}'.
        """
        content = call_ai(content_prompt)
        
        if content:
            with open(output_file, "a") as f:
                f.write(f"{'#' * (level + 2)} {title}\\n\\n{content}\\n\\n")
        else:
            console.print(f"[bold yellow]Warning: Could not generate content for section: '{title}'.[/bold yellow]")

        for child in section_data.get("children", []):
            process_section(child, level + 1)

    for section in outline:
        process_section(section, 0)
    
    # Clean up the plan file after successful completion
    os.remove(plan_file)


# --- Code-Based Tutorial Generation ---

def generate_code_tutorial(topic: str, output_dir: str):
    """Generates a code-based tutorial with source files and a README."""
    console.print(f"[bold green]Generating code tutorial plan for '{topic}'...[/bold green]")
    
    src_dir = os.path.join(output_dir, "src")
    os.makedirs(src_dir, exist_ok=True)

    outline_prompt = f"""
    You are a senior software engineer designing a tutorial for '{topic}'.
    Create a logical, step-by-step plan of small code examples that a beginner can follow.
    
    Return a JSON array of objects. Each object must have a 'filename' and a 'description'.
    - 'filename' should be valid for the language (e.g., 01_basics.py, 02_functions.cpp).
    - 'description' should clearly explain the concept this file will teach.

    Example for 'Python Functions':
    [
        {{"filename": "01_simple_function.py", "description": "How to define and call a basic function."}},
        {{"filename": "02_arguments.py", "description": "Passing arguments and using default values."}},
        {{"filename": "03_return_values.py", "description": "Returning values from a function."}}
    ]
    """
    plan = call_ai(outline_prompt, is_json=True)
    if not plan:
        console.print("[bold red]Failed to generate tutorial plan. Aborting.[/bold red]")
        return

    readme_content = f"# Tutorial: {topic}\\n\\nThis tutorial teaches {topic} through a series of code examples. It's recommended to follow them in order.\\n\\n"
    
    for i, item in enumerate(plan):
        filename = item.get("filename")
        description = item.get("description")
        if not filename or not description:
            continue

        console.print(f"[bold blue]Generating code for file {i+1}/{len(plan)}: '{filename}'...[/bold blue]")
        code_prompt = f"""
        You are a programmer writing a single, clean code file for a tutorial on '{topic}'.
        The file is named '{filename}' and its purpose is: '{description}'.

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
    topic: str = typer.Option(..., "--topic", "-t", help="The topic for the tutorial."),
    output: str = typer.Option(None, "--output", "-o", help="The name of the output file or directory."),
):
    """
    Generates a comprehensive, long-form tutorial for a given topic.

    The tool automatically determines the best format:
    - For programming topics, it creates a directory with commented code files and a README.
    - For other topics (like 'Calculus'), it creates a single, detailed Markdown document,
      recursively building the content to handle very large subjects without hitting limits.
    """
    topic_type = get_topic_type(topic)
    
    final_output_path = ""
    if topic_type == 'text':
        final_output_path = output if output else topic.lower().replace(" ", "_") + ".md"
        generate_text_tutorial(topic, final_output_path)
    elif topic_type == 'code':
        final_output_path = output if output else topic.lower().replace(" ", "_") + "_tutorial"
        generate_code_tutorial(topic, final_output_path)

    console.print(f"\n[bold green]Tutorial generation complete! Your tutorial is available at: '{final_output_path}'[/bold green]")

if __name__ == "__main__":
    app()
