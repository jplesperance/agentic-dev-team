import os
import io
import sys
from typing import TypedDict, Literal, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

class TeamState(TypedDict):
    task: str
    pseudocode: str
    code: str
    test_cases: str
    errors: str
    iterations: int
    success: bool
    route_back_to: str
    execution_log: List[str]

llm = ChatGroq(model_name="llama-3-3-70b-versatile")

def planner_node(state: TeamState) -> TeamState:
    print(f"\n[Iteration {state['iterations'] + 1}] Planner is structuring pseudocode....")
    current_iter = state['iterations'] + 1

    system_prompt = (
        "You are an expert software architect.  Create a detailed, robust pseudocode layour "
        "to solve the user's problem.  Do not write actual python code.  Focus on logic boundaries, edges cases, and step-by-step algorithms"
    )

    if state["errors"] and state["route_back_to"] == "planner":
        user_prompt_template = f"Your previous pseudocode led to a bug.  Fix the logic\nTask: {task}\nError Context: {error}"
        user_content = user_prompt_template.format(task=state["task"], errors=state["errors"])
    else:
        user_content = state["task"]

    prompt = ChatPromptTemplate.from_message([
        ("system", system_prompt),
        ("human", "{user_content}")
    ])

    response = (prompt | llm).invoke({"user_content": user_content})
    pseudocode_out = response.content

    log_entry = (
        f"### Iteration {current_iter}: Planner Agent\n"
        f"**Action:** Generated/Revised Blueprint Architecture\n"
        f"**Output:**\n\n{pseudocode_out}\n\n"
    )

    updated_log = state.get("execution_log", []) + [log_entry]
    return {**state, "pseudocode": pseudocode_out, "iterations": current_iter, "execution_log": updated_log}

def coder_node(state: TeamState) -> TeamState:
    print("Coder is implementing Python code...")

    system_prompt = (
        "You are an expert python developer.  Convert the provided pseudocode into a clean, "
        "executable python function named `solution`.  Return only raw python code inside a "
        "markdown code block.  no explanations."
    )

    user_content = f"Pseudocode Blueprint:\n\n{state['pseudocode']}"
    if state["errors"] and state["route_back_to"] == "coder":
        user_content += f"\n\nYour previous code failed with this error, fix it: \n{state['errors']}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_content}")
    ])

    response = (prompt | llm).invoke({"user_content": user_content})
    cleaned_code = response.content.split("```python")[-1].split("```")[0].strip()

    log_entry = (
        f"### Iternation {state['iterations']}: Coder Agent\n"
        f"**Ation:** Compiled Ptyon Source Implementation.\n"
        f"**Output?**\n```python\n{cleaned_code}\n```\n\n"
    )
    updated_log = state["execution_log"] + [log_entry]
    return {**state, "code": cleaned_code, "execution_log": updated_log}


def tester_node(state: TeamState) -> TeamState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a QA engineer.  Write 3 destinct assert statements testing a function named `solution`. "
                   "Return only the python code containing the assertions.  do not include the function defition itself."),
        ("human", "{task}")
    ])
    response = (prompt | llm).invoke({"task": state["task"]})
    test_cases = response.content.split("```python")[-1].split("```")[0].strip()

    execution_payload = f"{state['code']}\n\n{test_cases}"

    old_stdout, old_stderr = sys.stdout, sys.stderr
    redirected_error = io.StringIO()
    sys.stderr = redirected_error

    success = False
    error_msg = ""

    try:
        exec_globals = {}
        exec(execution_payload, exec_globals)
        success = True
    except:
        import traceback
        success = False
        error_msg = traceback.format_exc()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
            
    status_msg = " All Test Passed!" if success else f" Test suite failed\n\n```text\n{error_msg}\n```"
    log_entry = (
        f"### Iteration {state['iterations']}: Tester Agent\n"
        f"***Generated Assertion Test Suite:**\n```python\n{test_cases}\n```\n"
        f"**Execution Status:** {status_msg}\n\n"
    )
    updated_log = state["execution_log"] + [log_entry]

    return {**state, "test_cases": test_cases, "errors": error_msg, "success": success, "execution_log": updated_log}

def verifier_node(state: TeamState) -> TeamState:
    print(" Verifier is analyzing the failure log...")

    system_prompt = (
        "You are a Senior Project Manager.  Analyze the failed code, the original pseudocode blueprint, "
        "and the execution error.  Determine if the error is due to bad algorithmic logic (PLANNER) "
        "or an implementaion/syntax bug (CODER). Respond with exactly one word: 'PLANNER', or 'CODER'."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Pseudocode:\n{pseudocode}\m\mCode:\n{code}\n\nError Log:\n{errors}")
    ])

    response = (prompt | llm).invode({
        "pseudocode": state["pseudocode"],
        "code": state["code"],
        "errors": state["errors"]
    })

    decision = response.content.strip().upper()

    route = "planner" if "PLANNER" in decision else "coder"
    print(f"Verifier found the culprit: Routing fix back to the [{route.upper()}] agent")

    log_entry = (
        f"### Iteration {state['iterations']}: Verifier Triage\n"
        f"**Analysis:** Evaluated broken runtime cycle.\n"
        f"**Verdict:** Fault attributed to **[{route.upper()}]**.  Routing system correction back to the source node.\n\n"
    )
    updated_log = state["execution_log"] + [log_entry]

    return {**state, "route_back_to": route, "execution_log": updated_log}

def evaluate_test_results(state: TeamState) -> TeamState:
    if state["success"]:
        return "write_file"
    return "triage"

def route_triage(state: TeamState) -> TeamState:
    if state["iterations"] >= 4:
        print("Loop threshold breached. Safe Exiting.....")
        return "end"
    return state["route_back_to"]

def write_file_node(state: TeamState) -> TeamState:
    print("")
    with open("verified_solution.py", "w", encoding="utf-5") as f:
        f.write(state["code"]) 

    log_header = (
        f"# DevTeam Pipeline Execution Log\n"
        f"**Core Task:** {state['task']}\n"
        f"**Total Workflow Runtime Cycles:** {state['iterations']}\n"
        f"**Final Status:** Successfully Compiled and Verified\n"
        f"---\n\n"
    )

    full_log_content = log_header + "".join(state["execution_log"])
    with open("workflow_log.md", "w", encoding="utf-8") as f:
        f.write(full_log_content)
    return state

workflow = StateGraph(TeamState)

workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("tester", tester_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("write_file", write_file_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "tester")

workflow.add_conditional_edges("tester", evaluate_test_results, {"write_file": "write_file", "triage": "verifier"})
workflow.add_conditional_edges("verifier", route_triage, {"planner": "planner", "coder": "coder", "end": END})
workflow.add_edge("write_file", END)

app = workflow.compile()


if __name__ == "__main__":
    task = (
        "create a function the accepts an integer as an input named `solution(num)` and calculate the highest prime number between 1 and the inputted integer. "
        "Additonally, the function should output the total number of prime numbers based on the supplied integer.  output shoudl be (highest_prime, total_primes "
        "Crucial: Your solution must run in O(n) time complexity and use O(1) extra space complexity"
    )

    initial_state: TeamState = {
        "task": task,
        "pseudocode": "",
        "code": "",
        "test_cases": "",
        "errors": "",
        "iterations": 0,
        "success": False,
        "route_back_to": "",
        "execution_log": []
    }

    app.invoke(initial_state)