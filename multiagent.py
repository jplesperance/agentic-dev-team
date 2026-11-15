import os
import signal
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import TypedDict, Literal, List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# Use temperature=0 for reliable tool calling and JSON/YAML generation
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, max_retries=2, timeout=90)

def kill_process_tree(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


# =====================================================================
# 1. WORKSPACE-JAILED TOOLS (Replaces single-string code & exec())
# =====================================================================
def create_workspace_tools(workspace_dir: str):
    root = Path(workspace_dir).resolve()

    def _safe_path(rel_path: str) -> Path:
        target = (root / rel_path).resolve()
        if not str(target).startswith(str(root)):
            raise ValueError(f"Blocked path outside workspace: {rel_path}")
        return target

    @tool
    def list_files(subdir: str = ".") -> str:
        """List all project files relative to the workspace root."""
        base = _safe_path(subdir)
        ignored = {"node_modules", ".next", ".git", "__pycache__", ".venv"}
        files = [
            str(p.relative_to(root))
            for p in base.rglob("*")
            if p.is_file() and not any(part in ignored for part in p.parts)
        ]
        return "\n".join(sorted(files)) or "(empty)"

    @tool
    def read_file(rel_path: str) -> str:
        """Read a file inside the project workspace."""
        target = _safe_path(rel_path)
        return target.read_text(encoding="utf-8") if target.exists() else f"Not found: {rel_path}"

    @tool
    def write_file(rel_path: str, content: str) -> str:
        """Create or overwrite a file inside the project workspace."""
        target = _safe_path(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Saved {rel_path} ({len(content)} chars)"

    @tool
    def run_command(command: str, cwd_subdir: str = ".") -> str:
        """Run a shell command (pytest, npm run build, npx playwright test) inside the workspace."""
        work_dir = _safe_path(cwd_subdir)
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = f"EXIT_CODE: {res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            return output[-5000:]  # Truncate massive build logs to protect context window
        except subprocess.TimeoutExpired:
            return "ERROR: Command timed out after 120s."

    tools = [list_files, read_file, write_file, run_command]
    return tools, {t.name: t for t in tools}


def run_tool_agent(
    tools: list,
    tools_by_name: dict,
    system_prompt: str,
    user_prompt: str,
    max_steps: int = 40,
    require_tools: bool = True,
) -> str:
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    used_write_or_run = False

    for step in range(1, max_steps + 1):
        if step == max_steps - 3:
            messages.append(
                HumanMessage(
                    content="SYSTEM NOTICE: Only 3 tool calls remain. Complete any final file write or build check now and summarize."
                )
            )
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            if require_tools and not used_write_or_run:
                messages.append(
                    HumanMessage(
                        content="You did not call `write_file` or `run_command`. Do NOT paste code in chat—use your tools to edit the files on disk and run the build/tests now."
                    )
                )
                continue
            return response.content or "Completed."

        for tc in response.tool_calls:
            if tc["name"] in {"write_file", "run_command"}:
                used_write_or_run = True
            print(f"   [step {step}/{max_steps}] -> {tc['name']}({list(tc['args'].keys())})")
            fn = tools_by_name.get(tc["name"])
            result = fn.invoke(tc["args"]) if fn else f"Unknown tool: {tc['name']}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return f"Stopped after reaching max tool steps ({max_steps})."


def wait_for_port(port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


# =====================================================================
# 2. GRAPH STATE (Adapted from your TeamState)
# =====================================================================
class TeamState(TypedDict):
    project_name: str
    workspace_dir: str
    task: str
    bugs: List[dict]
    iterations: int
    success: bool
    route_back_to: str
    execution_log: List[str]


# =====================================================================
# 3. GRAPH FACTORY (Builds a workspace-bound 4-agent team per project)
# =====================================================================
def build_project_graph(workspace_dir: str):
    tools, tools_by_name = create_workspace_tools(workspace_dir)

    # --- AGENT 1: PRODUCT AGENT (Replaces planner_node) ---
    def product_node(state: TeamState) -> TeamState:
        print(f"\n[{state['project_name']}] Product Agent writing specs...")
        sys_prompt = (
            "You are a Principal Product Manager. Inspect the starter template with `list_files`, then write:\n"
            "1. `docs/PRD.md`: Concise user stories, UI routes, SQLite schema, numbered acceptance criteria, "
            "and a **UI Test Selector Contract** table listing exact `data-testid` attributes for every page, "
            "form input, `<select>` dropdown, submit button, and list/summary element.\n"
            "2. `docs/openapi.yaml`: Complete, compact OpenAPI 3.1 contract defining every FastAPI endpoint, "
            "request body, query parameter, and JSON response schema.\n"
            "SCOPE RULE: Build ONLY what the user asked for. Do NOT add user registration, login, or authentication "
            "unless the user's prompt explicitly requests user accounts or authentication."
        )
        summary = run_tool_agent(
            tools, tools_by_name, sys_prompt, f"Build specifications for:\n{state['task']}", max_steps=20
        )
        log_entry = f"### Product Agent\n**Summary:** {summary}\n\n"
        return {**state, "execution_log": state["execution_log"] + [log_entry]}

    # --- AGENT 2: BACKEND AGENT (FastAPI + Pytest) ---
    def backend_node(state: TeamState) -> TeamState:
        print(f"[{state['project_name']} | Iter {state['iterations']}] Backend Agent building FastAPI...")

        sys_prompt = (
            "You are a Senior FastAPI Engineer. Read `docs/PRD.md`, `docs/openapi.yaml`, and `docs/bugs.json` "
            "(if present). Implement the complete backend inside `backend/app/` using FastAPI, Pydantic v2, "
            "and SQLite (SQLAlchemy) with automatic table creation on startup (`Base.metadata.create_all`), "
            "and CORSMiddleware allowing `http://localhost:3000` and `http://127.0.0.1:3000`. "
            "Do NOT leave stub or TODO endpoints. Write thorough tests in `backend/tests/test_api.py` covering "
            "every endpoint in `openapi.yaml`, then run `.venv/bin/pytest` in `backend` via `run_command` until it passes."
        )
        bug_msg = f"\nFix these reported bugs:\n{json.dumps(state['bugs'], indent=2)}" if state["bugs"] else ""
        summary = run_tool_agent(tools, tools_by_name, sys_prompt, f"Implement or fix the FastAPI backend.{bug_msg}")

        log_entry = f"### Iteration {state['iterations']}: Backend Agent\n**Summary:** {summary}\n\n"
        return {**state, "execution_log": state["execution_log"] + [log_entry]}

    # --- AGENT 3: FRONTEND AGENT (Next.js App Router + TypeScript) ---
    def frontend_node(state: TeamState) -> TeamState:
        iter_num = state["iterations"] + 1
        print(f"[{state['project_name']} | Iter {iter_num}] Frontend Agent building Next.js...")
        sys_prompt = (
            "You are a Senior Next.js (App Router + TypeScript + Tailwind) Engineer. Always use `write_file` and "
            "`run_command`—never output code only in chat.\n"
            "1. If `docs/bugs.json` has bugs, your VERY FIRST tool call MUST be `read_file('docs/bugs.json')`, "
            "followed by `read_file('frontend/tests/e2e.spec.ts')`.\n"
            "2. Read `docs/PRD.md`, `docs/openapi.yaml`, and inspect `backend/app/` to confirm exact JSON field names.\n"
            "3. Build the UI in `frontend/src/app/` and a typed fetch client in `frontend/src/lib/api.ts` targeting "
            "`http://127.0.0.1:8000`.\n"
            "CRITICAL RULES:\n"
            "- Apply every `data-testid` attribute defined in `docs/PRD.md`.\n"
            "- NEVER render an empty container with 0 height: every list/table container with a `data-testid` MUST "
            "have `min-h-[40px]` and render an empty-state message inside it (e.g., `<p>No items found.</p>`) when the array is empty.\n"
            "- Pass `{ cache: 'no-store' }` to all `fetch` calls and include visible navigation links (`<Link>`) between `/` and all subpages.\n"
            "- Wrap every `fetch` in `try { ... } catch (err) { setError(...) } finally { setLoading(false) }` and render errors visibly.\n"
            "- Run `npm run build` in `frontend` via `run_command` and fix all build errors before finishing."
        )
        bug_msg = f"\nFix these reported bugs:\n{json.dumps(state['bugs'], indent=2)}" if state["bugs"] else ""
        summary = run_tool_agent(
            tools, tools_by_name, sys_prompt, f"Implement or fix the Next.js frontend.{bug_msg}", max_steps=40
        )
        log_entry = f"### Iteration {iter_num}: Frontend Agent\n**Summary:** {summary}\n\n"
        return {**state, "execution_log": state["execution_log"] + [log_entry]}

    # --- AGENT 4: QA AGENT (Replaces tester_node + verifier_node) ---
    def qa_node(state: TeamState) -> TeamState:
        current_iter = state["iterations"] + 1
        print(f"[{state['project_name']} | Iter {current_iter}] QA Agent testing live full-stack app...")
        ws = Path(workspace_dir)

        # Reset SQLite databases so each QA iteration starts with clean state
        for db_file in (ws / "backend").glob("*.db"):
            db_file.unlink()

        backend_proc = subprocess.Popen(
            [".venv/bin/uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ws / "backend",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev", "--", "-p", "3000"],
            cwd=ws / "frontend",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        try:
            be_up = wait_for_port(8000, timeout=15)
            fe_up = wait_for_port(3000, timeout=20)

            sys_prompt = (
                "You are a QA Automation Engineer. FastAPI is live on http://127.0.0.1:8000 and Next.js is live on "
                "http://127.0.0.1:3000. Read `docs/PRD.md` and `docs/openapi.yaml`.\n"
                "Write `frontend/tests/e2e.spec.ts` with TWO test groups:\n"
                "1. API Smoke Tests using Playwright's `request` fixture against `http://127.0.0.1:8000` verifying "
                "every endpoint in `openapi.yaml`.\n"
                "2. Browser E2E Tests against `http://127.0.0.1:3000` using `page.getByTestId(...)` from `docs/PRD.md`.\n"
                "PLAYWRIGHT RULES:\n"
                "- Remember each Playwright `test()` block starts with empty `localStorage` and cookies. Either seed "
                "required prerequisites per test or use `test.describe.serial` with a shared flow.\n"
                "- Attach `page.on('pageerror', ...)` and `page.on('response', async res => { if (res.status() >= 400) ... })` "
                "to capture HTTP status codes, response bodies, and runtime JS errors.\n"
                "- For `<select>` dropdowns, use `await select.selectOption(...)` and assert `toHaveValue(...)`—NEVER "
                "call `.toBeVisible()` on an `<option>` element.\n"
                "- Run `npx playwright test --reporter=list` in `frontend` via `run_command`.\n"
                "Write `docs/bugs.json` as a JSON array:\n"
                '[{"layer": "backend" | "frontend", "route": "...", "description": "...", "trace": "..."}]\n'
                "Include failed HTTP response bodies and page text inside `trace`. Tag 404/422/500 API errors as "
                "`\"layer\": \"backend\"`. If all tests pass, write `[]` to `docs/bugs.json`."
            )
            prompt = f"Servers ready (backend={be_up}, frontend={fe_up}). Run E2E tests and file bugs in docs/bugs.json."
            summary = run_tool_agent(tools, tools_by_name, sys_prompt, prompt, max_steps=25)
        finally:
            kill_process_tree(frontend_proc)
            kill_process_tree(backend_proc)

        bugs_file = ws / "docs" / "bugs.json"
        bugs = json.loads(bugs_file.read_text(encoding="utf-8")) if bugs_file.exists() else []
        success = len(bugs) == 0

        # Triage routing (replaces verifier_node)
        if any(b.get("layer") == "backend" for b in bugs):
            route_back = "backend"
        elif bugs:
            route_back = "frontend"
        else:
            route_back = "finalize"

        status_msg = "All E2E tests passed!" if success else f"{len(bugs)} bug(s) filed -> routing to {route_back}"
        log_entry = f"### Iteration {current_iter}: QA Agent\n**Status:** {status_msg}\n**Notes:** {summary}\n\n"

        return {
            **state,
            "bugs": bugs,
            "success": success,
            "route_back_to": route_back,
            "iterations": current_iter,
            "execution_log": state["execution_log"] + [log_entry],
        }

    # --- FINALIZE NODE (Replaces write_file_node) ---
    def finalize_node(state: TeamState) -> TeamState:
        ws = Path(workspace_dir)
        status = "Successfully Verified" if state["success"] else f"Stopped at Max Iterations ({len(state['bugs'])} bugs remaining)"
        log_header = (
            f"# DevTeam Pipeline Log: {state['project_name']}\n"
            f"**Task:** {state['task']}\n"
            f"**Status:** {status}\n---\n\n"
        )
        (ws / "docs" / "workflow_log.md").write_text(
            log_header + "".join(state["execution_log"]),
            encoding="utf-8",
        )
        print(f"[{state['project_name']}] Finished -> {status}")
        return state

    def route_after_qa(state: TeamState) -> Literal["backend", "frontend", "finalize"]:
        if state["success"] or state["iterations"] >= 4:
            return "finalize"
        return state["route_back_to"]  # type: ignore

    def route_after_backend(state: TeamState) -> Literal["frontend", "qa"]:
        if state["iterations"] == 0 or any(b.get("layer") == "frontend" for b in state["bugs"]):
            return "frontend"
        return "qa"

    workflow = StateGraph(TeamState)
    workflow.add_node("product", product_node)
    workflow.add_node("backend", backend_node)
    workflow.add_node("frontend", frontend_node)
    workflow.add_node("qa", qa_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("product")
    workflow.add_edge("product", "backend")
    workflow.add_conditional_edges(
        "backend",
        route_after_backend,
        {"frontend": "frontend", "qa": "qa"},
    )
    workflow.add_edge("frontend", "qa")
    workflow.add_conditional_edges(
        "qa",
        route_after_qa,
        {"backend": "backend", "frontend": "frontend", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)

    return workflow.compile()


# =====================================================================
# 4. CONSECUTIVE PROJECT RUNNER
# =====================================================================
def run_project_queue(projects: List[dict], template_dir: str = "./starter-template"):
    for project in projects:
        slug = project["slug"]
        ws_path = Path(f"./workspaces/{slug}").resolve()

        # Clean slate for each consecutive project
        if ws_path.exists():
            shutil.rmtree(ws_path)
        shutil.copytree(template_dir, ws_path, symlinks=True)

        app = build_project_graph(str(ws_path))

        initial_state: TeamState = {
            "project_name": slug,
            "workspace_dir": str(ws_path),
            "task": project["task"],
            "bugs": [],
            "iterations": 0,
            "success": False,
            "route_back_to": "",
            "execution_log": [],
        }

        app.invoke(initial_state)


if __name__ == "__main__":
    project_queue = [
        {
            "slug": "invoice-tracker",
            "task": "Build an invoice manager where users can create invoices, filter by Paid/Pending status, and view total revenue.",
        },
        {
            "slug": "habit-streaks",
            "task": "Build a daily habit tracker where users can add habits, toggle daily completions, and see active streak counts.",
        },
    ]

    run_project_queue(project_queue)