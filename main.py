# main.py
from graph import build_graph


def get_user_query():
    print("Welcome to the Multi-Step Research Agent (MSRA)")
    return input("Enter your research question: ").strip()


if __name__ == "__main__":
    graph = build_graph()
    user_query = get_user_query()

    final_state = graph.invoke({"user_query": user_query})

    print("\n========== FINAL OUTPUT ==========\n")

    answer = (final_state.get("final_answer") or "").strip()
    if answer:
        print(answer)
    else:
        print("No answer produced.")

    refs = final_state.get("references") or []
    if refs:
        print("\nReferences:")
        for i, url in enumerate(refs, start=1):
            print(f"[{i}] {url}")

    print("\n========== END ==========")

