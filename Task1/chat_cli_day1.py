import os
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIError, Timeout

from argparse import ArgumentParser
from rich.console import Console
import tiktoken

console = Console()

MODEL_NAME = "gpt-4o"


class ChatSession:
    def __init__(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.total_tokens = 0
        self.client = self._init_client()
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _init_client(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        return OpenAI(api_key=api_key)

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def add_user_message(self, text: str):
        tokens = self.count_tokens(text)
        self.messages.append({"role": "user", "content": text})
        console.print(f"[dim][User tokens: {tokens}][/dim]")

    def add_assistant_message(self, text: str):
        tokens = self.count_tokens(text)
        self.messages.append({"role": "assistant", "content": text})
        console.print(f"[dim][Assistant tokens: {tokens}][/dim]")

    def stream_response(self):
        console.print("[yellow]Assistant is typing...[/yellow]\n")

        try:
            reply = ""
            response_total_tokens = 0
            messages_for_chunk = []

            with self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
                stream_options={"include_usage": True},
                stream=True
            ) as stream:
                for chunk in stream:
                    delta = None
                    if chunk.choices:
                        delta = chunk.choices[0].delta

                    if chunk.usage:
                        response_total_tokens += chunk.usage.total_tokens
                        self.total_tokens += chunk.usage.total_tokens

                    if delta and delta.content:
                        messages_for_chunk.append(delta.content)
                        console.print(delta.content, end="")
                        reply += delta.content

            console.print()
            self.add_assistant_message(reply)
            console.print(
                f"[dim][Tokens used: {response_total_tokens} | Total so far: {self.total_tokens}][/dim]"
            )
            return reply

        except RateLimitError:
            console.print("[bold red]Rate limit exceeded (429). Please wait and try again.[/bold red]")
        except Timeout:
            console.print("[bold red]Request timed out. Check your network connection.[/bold red]")
        except APIError as e:
            console.print(f"[bold red]OpenAI API Error: {e}[/bold red]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

        return ""

    def save_log(self):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.json")
        path = log_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "total_tokens": self.total_tokens,
            "messages": self.messages
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]Conversation saved to {path}[/green]")


def main():
    parser = ArgumentParser(description="CLI Chat")

    parser.add_argument(
        "-prompt",
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

            session.add_user_message(user_input)
            session.stream_response()

        except KeyboardInterrupt:
            session.save_log()
            break


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        console.print("\n[red]Session interrupted.[/red]")

    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {e}")