import os
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError
from openai.types.responses import FunctionToolParam

from argparse import ArgumentParser
from rich.console import Console
import tiktoken

from wikipedia import summary

MANDATORY_TOOL_INSTRUCTIONS = (
    "\n\nCRITICAL SYSTEM INSTRUCTION: \n"
    "You have access to specific tools defined in your schema. \n"
    "1. For ANY mathematical calculation, you MUST use the 'calculate' tool \n"
    "2. For ANY request to explain a topic, you MUST use the 'explain' tool. \n"
    "3. For ANY request to search information, you MUST use the 'search_wikipedia' tool. \n"
    "4. For ANY request to create quizzes, you MUST use the 'generate_quiz' tool to create prompt. \n"
    "Do not attempt to answer these requests directly without calling the appropriate tool first. \n"
)

tools: list[FunctionToolParam] = [
    {
        "type": "function",
        "name": "calculate",
        "description": "Calculate a result of a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "type": "function",
        "name": "explain",
        "description": "Explains a selected topic",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Selected topic"
                }
            },
            "required": ["topic"]
        }
    },
    {
        "type": "function",
        "name": "search_wikipedia",
        "description": "Search Wikipedia for a topic summary",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "type": "function",
        "name": "generate_quiz",
        "description": "Create a quiz on a given theme",
        "parameters": {
            "type": "object",
            "properties": {
                "theme": {
                    "type": "string",
                    "description": "Quiz theme"
                },
                "difficulty": {
                    "type": "string",
                    "description": "Quiz difficulty"
                }
            },
            "required": ["theme", "difficulty"]
        }
    }
]

console = Console()
load_dotenv()

class ChatSession:
    def __init__(self, system_prompt: str) -> None:
        full_system_prompt = f"{system_prompt}{MANDATORY_TOOL_INSTRUCTIONS}"

        self.context: list[dict[str, Any]] = [{"role": "system", "content": full_system_prompt}]
        self.total_tokens = 0
        self.client = self._init_client()
        self.model_name = self._init_model_name()
        self.encoder = tiktoken.encoding_for_model(self.model_name)

        self.function_logs: list[dict[str, Any]] = []

    @staticmethod
    def calculate(expression: str) -> str:
        return eval(expression)

    @staticmethod
    def explain(topic: str) -> str:
        return f"Explain a {topic} topic in simple terms"

    @staticmethod
    def generate_quiz(theme: str, difficulty: str) -> str:
        return f"Create a 3-question quiz on {theme} with {difficulty} difficulty"

    @staticmethod
    def search_wikipedia(query: str) -> str:
        try:
            return summary(query, sentences=2)
        except Exception as e:
            return f"Error searching Wikipedia: {e}"

    @staticmethod
    def _init_client() -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        return OpenAI(api_key=api_key)

    @staticmethod
    def _init_model_name() -> str:
        model_name = os.getenv("MODEL_NAME")
        if not model_name:
            raise ValueError("MODEL_NAME not found in .env")
        return model_name

    def call_function(self, name: str, args: dict) -> str:
        console.print(f"\n[dim]Calling function: {name} with args: {args}[/dim]")

        func = getattr(self, name)
        result = func(**args) if func else f"Unknown function: {name}"

        self.function_logs.append({
            "function_name": name,
            "arguments": args,
            "result": result,
            "completed_at": datetime.now().isoformat(),
        })

        return result

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def log_message(self, text: str, source: Literal["user", "assistant"]) -> None:
        tokens = self.count_tokens(text)
        self.context.append({"role": source, "content": text})
        role_name = source.capitalize()
        console.print(f"[dim][{role_name} tokens: {tokens}][/dim]")

    def make_response(self) -> None:
        console.print("[yellow]Assistant is typing...[/yellow]\n")

        try:
            response = self.client.responses.create(
                model=self.model_name,
                tools=tools,
                input=self.context,
            )
            self.context += response.output

            for tool_call in response.output:
                if tool_call.type != "function_call":
                    continue

                name = tool_call.name
                try:
                    args = json.loads(tool_call.arguments)
                except json.JSONDecodeError as e:
                    console.print(f"[red]Bad tool arguments: {e}[/red]")
                    continue
                result = self.call_function(name, args)

                self.context.append({
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": str(result)
                })

            final_response = self.client.responses.create(
                model=self.model_name,
                input=self.context,
                tools=tools
            )

            if final_response.output_text:
                output_text = final_response.output_text
                console.print(output_text)
                self.log_message(output_text, "assistant")

            if final_response.usage:
                response_tokens = final_response.usage.total_tokens
                self.total_tokens += response_tokens
                console.print(
                    f"[dim][Tokens used: {response_tokens} | Total so far: {self.total_tokens}][/dim]"
                )

        except RateLimitError:
            console.print("[bold red]Rate limit exceeded (429). Please wait and try again.[/bold red]")
        except APITimeoutError:
            console.print("[bold red]Request timed out. Check your network connection.[/bold red]")
        except APIError as e:
            console.print(f"[bold red]OpenAI API Error: {e}[/bold red]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

    def save_log(self) -> None:
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)

        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.json")
        path = log_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "total_tokens": self.total_tokens,
            "function_calls_count": len(self.function_logs),
            "function_logs": self.function_logs
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]Function logs saved to {path}[/green]")


def main() -> None:
    parser = ArgumentParser(description="CLI Chat")
    parser.add_argument("--prompt", type=str, default="You are a helpful assistant.")
    args = parser.parse_args()

    session = ChatSession(args.prompt)

    try:
        while True:
            console.print("\n[bold cyan]You:[/bold cyan] ", end="")
            user_input = input()

            if user_input.lower() in ["exit", "quit"]:
                break

            session.log_message(user_input, "user")
            session.make_response()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    finally:
        session.save_log()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {e}")