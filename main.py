import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(user_request: str) -> None:
    # Import here so .env is loaded first
    from src import content_graph, ContentState

    print("\n" + "═" * 70)
    print(f"  REQUEST  : {user_request}")
    print("═" * 70 + "\n")

    initial_state = ContentState(user_request=user_request)
    result = content_graph.invoke(initial_state)

    final_state = ContentState(**result) if isinstance(result, dict) else result

    score = final_state.evaluation_score or 0.0
    print("\n" + "═" * 70)
    print(f"  ITERATIONS : {final_state.iteration}")
    print(f"  SCORE      : {score:.1f}/10")
    print(f"  APPROVED   : {final_state.is_approved}")
    print("═" * 70)
    print("\n📄  FINAL CONTENT\n")
    print(final_state.final_content or "[No content produced]")
    print("\n" + "─" * 70)

    if final_state.error:
        print(f"\n⚠  Error captured: {final_state.error}")


def main() -> None:
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
    else:
        print("LangGraph Content Generator (Gemini)")
        print("─" * 40)
        user_request = input("Enter your content request: ").strip()
        if not user_request:
            print("No request provided. Exiting.")
            sys.exit(1)

    run(user_request)


if __name__ == "__main__":
    main()