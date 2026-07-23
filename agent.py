"""
agent.py — the agentic layer: let the MODEL decide when to search.

Day 5 was a fixed pipeline: every question always retrieved, then always
answered. The CODE was in control. An AGENT flips that: we hand the model a
TOOL (search_documents) and let IT decide whether to search, what to search
for, whether to search again, and when it's ready to answer. The MODEL is in
control of the flow.

The heart of this file is the AGENT LOOP:

    1. Send the question + the available tools to the model.
    2. The model replies with EITHER:
         (a) a final answer  -> we stop and return it, or
         (b) a request to call a tool -> we run the tool, hand back the
             result, and loop again.
    3. Repeat until the model answers (with a max-steps cap so a confused
       model can never loop forever).

This "who controls the flow" distinction (code vs model) is the single most
important idea to be able to explain about this project.
"""

import json
import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_client = Groq()
MODEL = "llama-3.3-70b-versatile"

# The agent's rules. Note the difference from Day 5's prompt: we now TELL the
# model it HAS a search tool and that deciding when to use it is ITS job.
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a set of "
    "documents. You have a `search_documents` tool. When a question might be "
    "answered by the documents, call the tool to fetch relevant passages, then "
    "answer using ONLY those passages and cite them like [1], [2]. If a "
    "question needs several distinct facts, you may call the tool more than "
    "once with different queries. If the documents don't contain the answer, "
    "say you don't know. For simple greetings or small talk, just reply "
    "directly without searching."
)

# TOOL DEFINITION (OpenAI/Groq "function calling" format). This is a
# DESCRIPTION of the tool we give the model — its name, what it's for, and
# what arguments it takes (as a JSON Schema). The model reads this to decide
# when and how to call it. The `description` fields are effectively a prompt:
# the better they are, the better the model's tool decisions.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the uploaded documents for passages relevant to a "
                "query. Returns numbered passages [1], [2], .... Use this "
                "whenever the user asks something the documents might answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, in natural language.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def _run_search_tool(store, query: str, k: int = 3) -> str:
    """
    The ACTUAL implementation of the search_documents tool.

    The model can only REQUEST a tool call — it can't run code. WE run the
    real work (our Day 4 retrieval) and hand the text back. This is the
    security boundary of tool use: the model proposes, our code disposes.
    """
    results = store.search(query, k=k)
    lines = [f"[{n}] {chunk['text']}" for n, (score, chunk) in enumerate(results, start=1)]
    return "\n\n".join(lines) if lines else "No relevant passages found."


def run_agent(question: str, store, max_steps: int = 5, verbose: bool = False) -> str:
    """
    Run the agent loop until the model produces a final answer.

    Args:
        question:  the user's question.
        store:     a VectorStore with documents already added.
        max_steps: safety cap on tool-call rounds (prevents infinite loops).
        verbose:   if True, print what the agent decides at each step.

    Returns:
        The model's final answer text.
    """
    # The running transcript. The API is STATELESS — each call must resend the
    # whole conversation so far, which is exactly how the model "remembers" the
    # tool results from earlier steps.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        # Ask the model what to do next. `tools=TOOLS` tells it what's available;
        # tool_choice="auto" lets IT decide whether to call one or just answer.
        response = _client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0,
            tools=TOOLS,
            tool_choice="auto",
            messages=messages,
        )
        msg = response.choices[0].message

        # CASE (a): no tool call requested -> this is the final answer. Stop.
        if not msg.tool_calls:
            if verbose:
                print(f"  [step {step}] agent answered directly.")
            return msg.content

        # CASE (b): the model wants to call one or more tools.
        # We MUST append the assistant's tool-request message before the tool
        # results, or the API can't match results to requests.
        messages.append(msg)

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            # The model sends arguments as a JSON *string*; parse it to a dict.
            args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"  [step {step}] agent calls {name}(query={args.get('query')!r})")

            if name == "search_documents":
                result = _run_search_tool(store, args["query"])
            else:
                # Defensive: model hallucinated a tool we don't have.
                result = f"Error: unknown tool {name!r}."

            # Feed the tool's output back in. role="tool" + tool_call_id is how
            # the API links this result to the exact request that asked for it.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
        # loop continues: the model now sees the tool results and decides again

    # If we fall out of the loop, the model kept calling tools past the cap.
    return "Sorry — I couldn't finish that within the step limit."


# Manual test: watch the agent DECIDE across different kinds of questions.
# Run with:  python agent.py
if __name__ == "__main__":
    from ingest import extract_text_from_pdf, chunk_text
    from store import VectorStore

    text = extract_text_from_pdf("sample.pdf")
    store = VectorStore()
    store.add(chunk_text(text), source="sample.pdf")

    questions = [
        "What problem does RAG solve?",       # should search once, then answer
        "Hello!",                             # should answer directly, NO search
        "What is chunking and why is overlap used?",  # may search (maybe twice)
    ]

    for q in questions:
        print(f"\nQ: {q}")
        final = run_agent(q, store, verbose=True)
        print(f"A: {final}")
