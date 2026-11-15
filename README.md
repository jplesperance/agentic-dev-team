# Dev Team

A LangGraph pipeline that builds small full-stack apps one after another. For each project it copies `starter-template` into `workspaces/<slug>`, then a four-agent team writes the spec, the API, the UI, and Playwright coverage against a live pair of servers.

The model client is `ChatOpenAI` (`gpt-4o`, `temperature=0`, `max_retries=2`) from `langchain-openai`. `load_dotenv()` loads `.env` before the client is created. The OpenAI client reads `OPENAI_API_KEY`.

## What a run does

`run_project_queue` walks the queue in order. For each project it:

1. Deletes `workspaces/<slug>` if that directory already exists.
2. Copies `starter-template` into `workspaces/<slug>` (`shutil.copytree` with `symlinks=True`).
3. Compiles a new graph bound to that workspace and invokes it with the project task.

The default queue is:

| Slug | Task |
| --- | --- |
| `invoice-tracker` | Invoice manager: create invoices, filter by Paid/Pending, show total revenue. |
| `habit-streaks` | Habit tracker: add habits, toggle daily completions, show active streaks. |

Change `project_queue` in the `if __name__ == "__main__"` block to add or replace projects. Each entry needs a `slug` and a `task`.

## Agents

Every agent except finalize works through `run_tool_agent`: up to 15 model turns, with tools bound to the workspace. The loop ends when the model stops calling tools.

| Node | Role |
| --- | --- |
| **Product** | Lists the template, then writes `docs/PRD.md` (user stories, pages, SQLite models, numbered acceptance criteria) and `docs/openapi.yaml` (OpenAPI 3.1 for the FastAPI contract). |
| **Backend** | Reads the PRD, OpenAPI file, and `docs/bugs.json` when it exists. Implements `backend/app/` with FastAPI, Pydantic v2, and SQLite via SQLAlchemy. CORS allows `http://localhost:3000`. Writes `backend/tests/test_api.py` and runs `.venv/bin/pytest` in `backend` until the tests pass. |
| **Frontend** | Reads the same docs. Builds `frontend/src/app/` with Next.js App Router, TypeScript, and Tailwind, plus `frontend/src/lib/api.ts` calling `http://127.0.0.1:8000` from the OpenAPI contract. Runs `npm run build` in `frontend` and fixes type and build errors. |
| **QA** | Starts the API and the Next dev server, writes `frontend/tests/e2e.spec.ts` for every acceptance criterion, runs `npx playwright test` in `frontend`, and writes `docs/bugs.json`. |
| **Finalize** | Writes `docs/workflow_log.md` in the workspace and prints the final status. |

QA starts both processes and always terminates them in a `finally` block:

- Backend: `.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000` with working directory `backend`.
- Frontend: `npm run dev -- -p 3000` with working directory `frontend`.

It waits up to 15 seconds for port 8000 and 20 seconds for port 3000, then tells the QA model whether each port accepted a connection.

## Retry routing

```text
product → backend → frontend → qa
                                 │
                                 ├─ no bugs, or iterations ≥ 4 ──► finalize ──► END
                                 ├─ any bug with layer "backend" ──► backend → frontend → qa
                                 └─ frontend-only bugs ──► frontend → qa
```

`docs/bugs.json` is a JSON array. An empty array means success. Each bug is shaped like:

```json
{"layer": "backend", "route": "...", "description": "...", "trace": "..."}
```

`layer` is `"backend"` or `"frontend"`. If any bug is tagged `backend`, the next hop is the backend agent, even when frontend bugs are also present. Frontend-only bugs go back to the frontend agent.

`iterations` starts at 0. The product node adds 1, and every QA pass adds 1. `route_after_qa` sends the graph to finalize when `success` is true or `iterations` is 4 or more. A failing run therefore gets the product pass plus QA passes until the counter hits 4, then stops with the remaining bugs recorded in the log.

## Workspace tools

`create_workspace_tools` closes over the project directory. Paths are resolved and rejected when they fall outside that root.

| Tool | Behavior |
| --- | --- |
| `list_files` | Recurses from an optional subdirectory. Skips `node_modules`, `.next`, `.git`, `__pycache__`, and `.venv`. |
| `read_file` | Returns the UTF-8 text of a workspace file, or `Not found: ...`. |
| `write_file` | Creates parent directories and overwrites the file. |
| `run_command` | Runs a shell command in a workspace subdirectory, 120 second timeout, and returns only the last 5000 characters of the combined exit code, stdout, and stderr. |

`run_command` uses `shell=True` inside the copied project. The model chooses the command string.

## State

| Field | Meaning |
| --- | --- |
| `project_name` | Queue slug, also used in log lines. |
| `workspace_dir` | Absolute path of `workspaces/<slug>`. |
| `task` | Product prompt for this project. |
| `bugs` | Parsed contents of `docs/bugs.json` after QA. |
| `iterations` | Product pass plus each QA pass. |
| `success` | True when the bug list is empty. |
| `route_back_to` | `"backend"`, `"frontend"`, or `"finalize"`. |
| `execution_log` | Markdown sections appended by each node. |

## Starter template

`starter-template` is the skeleton copied into every workspace.

```text
starter-template/
  docs/bugs.json              # []
  backend/app/main.py         # FastAPI app, CORS for port 3000, GET /health
  backend/tests/test_api.py   # health-check test
  backend/requirements.txt    # fastapi, uvicorn, sqlalchemy, pydantic, pytest, httpx, pyyaml
  frontend/                       # Next.js 15, React 19, Tailwind 4, Playwright
    src/app/page.tsx              # placeholder home page
    tests/e2e.spec.ts
    playwright.config.ts          # Chromium, baseURL http://127.0.0.1:3000
```

The backend agent is told to run `.venv/bin/pytest`, and QA launches `.venv/bin/uvicorn`. Those are POSIX virtualenv paths. Create `backend/.venv` from `backend/requirements.txt`, install the frontend dependencies (`npm install` in `frontend`), and install the Playwright browser before a run. The queue runner copies the template as it exists; it does not create the virtualenv or `node_modules` itself. A virtualenv or `node_modules` placed inside `starter-template` is copied along with the rest of the tree.

## Setup

Python 3.10 or newer, plus Node.js for the Next.js app.

```bash
pip install langchain-core langchain-openai langgraph python-dotenv
```

`.env` in the project root:

```text
OPENAI_API_KEY=your_key_here
```

Prepare the template runtime once, then each workspace inherits it if those directories live inside `starter-template`:

```bash
python -m venv starter-template/backend/.venv
starter-template/backend/.venv/bin/pip install -r starter-template/backend/requirements.txt
npm install --prefix starter-template/frontend
npx --prefix starter-template/frontend playwright install chromium
```

On Windows, create the virtualenv the same way and point the pytest and uvicorn paths in `multiagent.py` at `Scripts\pytest.exe` and `Scripts\uvicorn.exe`. The file currently calls `.venv/bin/pytest` and `.venv/bin/uvicorn`.

## Run

```bash
python multiagent.py
```

Projects run back to back. Each finished workspace keeps its app and docs:

- `docs/PRD.md` and `docs/openapi.yaml` from the product agent
- `docs/bugs.json` from the QA agent
- `docs/workflow_log.md` from finalize, with the task, status, and every agent summary
- `backend/` and `frontend/` as the agents left them

Finalize marks the log `Successfully Verified` when the last bug list is empty, and `Stopped at Max Iterations` with the remaining bug count otherwise.

## Before the first run

`multiagent.py` imports `ChatOpen` and then constructs `ChatOpenAI`. Those names need to match or the process exits at import time.
