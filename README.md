# Dev Team

A small multi-agent pipeline that turns a programming task into a Python function, runs generated tests against it, and retries when the tests fail.

The workflow is a [LangGraph](https://github.com/langchain-ai/langgraph) state graph. Each node is an agent that reads and updates a shared `TeamState`. The language model is Groq’s `llama-3.3-70b-versatile`, loaded through `langchain-groq`.

## Agents

| Agent | Role |
| --- | --- |
| **Planner** | Writes a pseudocode blueprint. On a later pass it revises that blueprint when the verifier sends the failure back. |
| **Coder** | Turns the blueprint into a Python function named `solution`. On a later pass it patches the code using the last error. |
| **Tester** | Writes three `assert` statements for `solution`, then `exec`s the code and the tests together. |
| **Verifier** | After a failure, decides whether the fault is algorithmic (`PLANNER`) or an implementation bug (`CODER`) and routes the next attempt. |
| **Write file** | Runs only after the tests pass. Saves the code and a markdown log of every agent step. |

## Flow

```text
planner → coder → tester
                     │
                     ├─ tests pass ──► write_file ──► END
                     │
                     └─ tests fail ──► verifier
                                          │
                                          ├─ iterations < 4 ──► planner or coder
                                          └─ iterations ≥ 4 ──► END
```

The loop stops after four planner iterations. Each planner visit increments `iterations`.

## State

`TeamState` is the object passed between nodes:

| Field | Meaning |
| --- | --- |
| `task` | The original problem statement. |
| `pseudocode` | Latest blueprint from the planner. |
| `code` | Latest `solution` implementation from the coder. |
| `test_cases` | Assertion suite from the tester. |
| `errors` | Traceback from the last failed run, or an empty string. |
| `iterations` | How many times the planner has run. |
| `success` | Whether the latest test execution passed. |
| `route_back_to` | `"planner"` or `"coder"`, set by the verifier. |
| `execution_log` | Markdown snippets appended by each agent. |

## Default task

Running the module invokes the graph on a built-in task: write `solution(num)` that returns `(highest_prime, total_primes)` for integers from 1 through `num`, in O(n) time and O(1) extra space.

Edit the `task` string in the `if __name__ == "__main__"` block to solve a different problem. The coder is instructed to name the function `solution`.

## Setup

Python 3.10+ and a Groq API key.

```bash
pip install langchain langchain-core langchain-groq langgraph python-dotenv
```

Create a `.env` file in this directory:

```text
GROQ_API_KEY=your_key_here
```

`load_dotenv()` reads that file before the model client is created.

## Run

```bash
python multiagent.py
```

The script prints a short status line as each agent runs. When the tests pass, it writes two files in the working directory:

- `verified_solution.py` — the generated `solution` function.
- `workflow_log.md` — the task, iteration count, and the planner, coder, tester, and verifier notes from the run.

If the iteration cap is hit first, those files are not written and the process exits after the verifier routes to `END`.

## Notes

The tester executes model-generated code with `exec` in the same process. Run it only on tasks you are willing to execute locally.

Several names and call sites in `multiagent.py` do not match the libraries they import (`StatsGraph` vs `StateGraph`, `from_+message`, `invode`, `utf-5`, and a malformed conditional-edge map). The graph described above is what the file is written to do; those spots need to be corrected before `python multiagent.py` will start.
