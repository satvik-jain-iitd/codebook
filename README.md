# Codebase-X

**The Intelligent AI Assistant for Your Entire Codebase.**

Codebase-X is a production-grade, offline platform designed to give you total control over your code. By building a deep, persistent Knowledge Graph of your repository, Codebase-X enables advanced architectural analysis, atomic task decomposition, and prompt engineering—all while keeping your data 100% private and local.

---

## 🗺️ User Journey

1.  **`codebookx analyze`**: Scan your repository to build a Knowledge Graph and architectural teardown.
2.  **`codebookx view`**: Launch the interactive Knowledge Graph browser to explore your code visually.
3.  **`codebookx ask "..."`**: Ask technical questions about your codebase, powered by the Knowledge Graph context.

---

## 🚀 Core Capabilities

### 🏛️ Codebase Teardown (Analyze)
Stop guessing how your system works. Codebase-X scans your entire repository to generate a high-level architectural blueprint.
*   **Elevator Pitch:** Get an instant summary of any project.
*   **Architecture Blueprint:** Automatically mapped folder roles and purposes.
*   **Data Flow:** Understand how the core components of your system interact.
```bash
codebookx analyze
```

### 🔨 Atomic Task Decomposition (Decompose)
Break complex features into small, manageable pieces. Codebase-X analyzes your feature request against your actual codebase to produce a series of atomic tasks.
*   **AI-Ready:** Each task includes a "copy-paste" prompt specifically designed for AI coding assistants.
*   **Context-Aware:** Tasks reference the exact files and symbols needed.
```bash
codebookx decompose "Add a user settings page"
```

### ✨ Prompt Enhancement (Enhance)
Stop writing vague prompts. The Prompt Enhancer uses the Knowledge Graph to inject relevant file paths, function signatures, and dependencies into your request, creating a high-context prompt that gets it right the first time.
```bash
codebookx enhance "Refactor the authentication flow"
```

### 🕸️ Knowledge Graph Browser (View)
Explore your code visually. Launch an interactive local web UI to traverse your project's dependency graph, find symbols, and manage your AI tasks.
*   **Visual Map:** Interactive D3.js force-directed graph available at `/graph`.
*   **Relations:** Track CALLS, IMPORTS, and CONTAINS relationships cross-file.
```bash
codebookx view
```

### ❓ Technical Q&A (Ask)
Knowledge Graph-backed Q&A for your codebase. Answers are saved automatically.
```bash
# Single question
codebookx ask "How is the authentication handled in this project?"

# Interactive chat mode (multi-turn, context carry-over)
codebookx ask -c "Walk me through the data flow"

# Save Q&A logs to a custom directory
codebookx ask "What does run_ask do?" --dir ~/my-notes
```
*   **Auto-save:** Every Q&A and chat session is saved to `ask_history/` by default (`--dir` to override, or set `CODEBOOK_ASK_DIR` env var).
*   **Chat mode (`-c`):** Multi-turn conversation with full context carry-over. Each turn is saved incrementally — no history lost on crash.

---

## 🛠️ Getting Started

**1. Install**
```bash
pip install .
```

**2. Local LLM Setup**
Codebase-X is designed to work with **LM Studio**, **Ollama**, or any OpenAI-compatible local server. Ensure your server is running.

**3. Initial Analysis**
Scan your repository to build the Knowledge Graph:
```bash
codebookx analyze
```

---

## 🔒 Privacy & Performance
*   **100% Offline:** Your code never leaves your machine. No cloud APIs, no subscriptions.
*   **Persistent Indexing:** Uses a lightning-fast SQLite-backed Knowledge Graph with hash-based caching.
*   **Multi-Language:** Deep support for Python, TypeScript, JavaScript, and more.

---

## 📄 License
Codebase-X is released under the MIT License.
