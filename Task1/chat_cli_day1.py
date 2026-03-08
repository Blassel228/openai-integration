import os
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIError, Timeout

from argparse import ArgumentParser
from rich.console import Console
import tiktoken

console = Console()


class ChatSession:
    def __init__(self, system_prompt: str) -> None:
        self.context = [{"role": "system", "content": system_prompt}]
        self.total_tokens = 0
        self.client = self._init_client()
        self.model_name = os.getenv("MODEL_NAME")
        if not self.model_name:
            raise ValueError("MODEL_NAME not found in .env")
        self.encoder = tiktoken.encoding_for_model(self.model_name)

    def _init_client(self) -> OpenAI:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        return OpenAI(api_key=api_key)

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def log_message(self, text: str, source: Literal["user", "assistant"]) -> None:
        tokens = self.count_tokens(text)
        self.context.append({"role": source, "content": text})
        console.print(f"[dim][Assistant tokens: {tokens}][/dim]")

    def stream_response(self) -> None:
        console.print("[yellow]Assistant is typing...[/yellow]\n")

        try:
            reply = ""
            response_total_tokens = 0

            with self.client.responses.create(
                model=self.model_name,
                input=self.context,
                stream=True,
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        console.print(event.delta, end="")
                        reply += event.delta

                    elif event.type == "response.completed":
                        if event.response.usage:
                            response_total_tokens = event.response.usage.total_tokens
                            self.total_tokens += response_total_tokens

            console.print()
            self.log_message(reply, "assistant")
            console.print(
                f"[dim][Tokens used: {response_total_tokens} | Total so far: {self.total_tokens}][/dim]"
            )

        except RateLimitError:
            console.print(
                "[bold red]Rate limit exceeded (429). Please wait and try again.[/bold red]"
            )
        except Timeout:
            console.print(
                "[bold red]Request timed out. Check your network connection.[/bold red]"
            )
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
            "total_tokens": self.total_tokens,
            "context": self.context,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]Conversation saved to {path}[/green]")


def main() -> None:
    parser = ArgumentParser(description="CLI Chat")

    parser.add_argument(
        "--prompt",
        type=str,
        default="You are a helpful assistant.",
    )

    args = parser.parse_args()
    system_prompt = args.prompt
    session = ChatSession(system_prompt)

    while True:
        try:
            console.print("\n[bold cyan]You:[/bold cyan] ", end="")
            user_input = input()

            if user_input.lower() in ["exit", "quit"]:
                session.save_log()
                break

            session.log_message(user_input, "user")
            session.stream_response()

        except KeyboardInterrupt:
            session.save_log()
            break


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {e}")
