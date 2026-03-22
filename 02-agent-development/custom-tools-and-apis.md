# Building Custom Tools, APIs, and Functions

How to build, connect, and manage external tools that agents use to interact with the real world.

---

## Why Agents Need Tools

LLMs can reason but they can't do things on their own. They can't search the web, query a database, send an email, or check the weather. Tools give agents the ability to act.

Without tools: "What's the weather?" → LLM guesses (often wrong)
With tools: "What's the weather?" → LLM calls `get_weather("Paris")` → gets real data → answers accurately

Tools are the bridge between reasoning and action.

---

## Tool Design Patterns

### The Tool Interface

At its core, every tool is a function with a name, description, input schema, and implementation:

```python
# Minimal tool structure
{
    "name": "search_database",
    "description": "Search the company knowledge base for relevant documents",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {"type": "integer", "description": "Maximum results to return", "default": 5}
        },
        "required": ["query"]
    }
}
```

The LLM sees the name, description, and parameters. Based on this info, it decides when to call the tool and what arguments to pass. Good descriptions are critical. If the description is vague, the LLM won't know when to use the tool.

### OpenAI Function Calling Format

This is the standard format used by most frameworks (LangChain, AutoGen, NVIDIA NIM):

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol. Use this when the user asks about stock prices or market data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'NVDA', 'AAPL')"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

# LLM call with tools
response = client.chat.completions.create(
    model="nvidia/llama-3.1-70b-instruct",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # LLM decides when to use tools
)
```

### Tool Choice Options

| Option                                                        | Behavior                                                     |
| ------------------------------------------------------------- | ------------------------------------------------------------ |
| `"auto"`                                                      | LLM decides whether to use a tool or respond directly        |
| `"none"`                                                      | LLM must respond without tools (even if tools are available) |
| `"required"`                                                  | LLM must use at least one tool                               |
| `{"type": "function", "function": {"name": "specific_tool"}}` | Force a specific tool                                        |

---

## Best Practices for Tool Design

### 1. Clear, Specific Descriptions

Bad:

```python
description = "Gets data"  # Too vague, LLM won't know when to use it
```

Good:

```python
description = "Search the internal HR knowledge base for employee policies, benefits information, and company procedures. Use when the user asks about HR-related topics like PTO, insurance, or company policies."
```

The description should answer: **when** should the LLM use this tool and **what** does it return?

### 2. Descriptive Parameter Names and Descriptions

Bad:

```python
"properties": {"q": {"type": "string"}}
```

Good:

```python
"properties": {
    "query": {
        "type": "string",
        "description": "Natural language search query describing what information to find"
    },
    "department": {
        "type": "string",
        "enum": ["engineering", "sales", "hr", "finance"],
        "description": "Filter results to a specific department"
    }
}
```

Use `enum` when there's a fixed set of valid values. This constrains the LLM to valid inputs.

### 3. Keep Tools Focused

Each tool should do one thing well. Don't build a mega-tool.

Bad: One tool that searches, filters, sorts, and formats results.

Good: Separate tools for search, filter, and format. Let the agent compose them.

### 4. Return Structured Output

Return data the LLM can easily parse:

```python
def search_products(query: str, category: str = None) -> dict:
    results = db.search(query, category)
    return {
        "status": "success",
        "count": len(results),
        "products": [
            {"name": r.name, "price": r.price, "in_stock": r.stock > 0}
            for r in results
        ]
    }
```

Include a status field so the agent knows if the call succeeded.

### 5. Handle Errors Gracefully

Tools will fail. Return clear error messages the agent can act on:

```python
def query_database(sql: str) -> dict:
    try:
        results = db.execute(sql)
        return {"status": "success", "data": results}
    except ConnectionError:
        return {"status": "error", "message": "Database connection failed. Try again in a moment."}
    except InvalidQueryError as e:
        return {"status": "error", "message": f"Invalid query: {e}. Please fix the SQL syntax."}
```

The agent can then reason about the error: "The database is down, let me tell the user" or "My query was wrong, let me fix it and retry."

---

## API Integration Patterns

### REST API Integration

Most external services expose REST APIs. Wrap them as tools:

```python
import httpx

def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city from the weather API."""
    response = httpx.get(
        "https://api.weather.com/v1/current",
        params={"city": city, "units": units},
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    if response.status_code == 200:
        data = response.json()
        return {
            "status": "success",
            "temperature": data["temp"],
            "condition": data["condition"],
            "humidity": data["humidity"]
        }
    else:
        return {"status": "error", "message": f"API returned {response.status_code}"}
```

Key considerations:

- **Authentication**: API keys, OAuth tokens. Store securely, never in prompts.
- **Rate limiting**: External APIs have rate limits. Implement backoff or caching.
- **Timeouts**: Set reasonable timeouts. Don't let a slow API hang the agent.
- **Retries**: Transient failures happen. Implement retry with exponential backoff (links to error-handling.md in Domain 01).

### Database Tools

Agents that query databases need careful design:

```python
def query_customer_data(customer_id: str) -> dict:
    """Look up customer information by customer ID.
    Returns name, email, subscription tier, and account status."""
    # Use parameterized queries to prevent SQL injection!
    result = db.execute(
        "SELECT name, email, tier, status FROM customers WHERE id = %s",
        (customer_id,)
    )
    if result:
        return {"status": "success", "customer": dict(result)}
    return {"status": "error", "message": "Customer not found"}
```

Never let the agent write raw SQL against production databases unless you have read-only access and query validation.

### GraphQL Integration

For APIs that use GraphQL:

```python
def search_issues(project: str, status: str = "open") -> dict:
    """Search for project issues in the issue tracker."""
    query = """
    query($project: String!, $status: String!) {
        issues(project: $project, status: $status) {
            id
            title
            assignee
            priority
        }
    }
    """
    response = httpx.post(
        "https://api.issuetracker.com/graphql",
        json={"query": query, "variables": {"project": project, "status": status}}
    )
    return response.json()["data"]
```

---

## Cross-Framework Tool Patterns

Different frameworks define tools differently, but the concept is the same.

### LangChain Tools

```python
from langchain.tools import tool

@tool
def search_documents(query: str) -> str:
    """Search the document store for relevant information.
    Use this when the user asks about company policies or procedures."""
    results = vector_store.similarity_search(query, k=3)
    return "\n".join([r.page_content for r in results])

# Use in agent
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools=[search_documents], prompt=prompt)
```

LangChain uses the `@tool` decorator. The docstring becomes the tool description.

### AutoGen Functions

```python
from autogen import register_function

def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate the discounted price given original price and discount percentage."""
    return price * (1 - discount_percent / 100)

# Register: caller decides when to use, executor runs it
register_function(
    calculate_discount,
    caller=assistant_agent,
    executor=user_proxy_agent,
    description="Calculate discounted price"
)
```

AutoGen separates the caller (LLM that decides to use the tool) from the executor (agent that runs it). This is unique to AutoGen.

### CrewAI Tools

```python
from crewai_tools import tool

@tool("Search Knowledge Base")
def search_kb(query: str) -> str:
    """Search the company knowledge base. Use for any questions about
    internal processes, policies, or technical documentation."""
    return knowledge_base.search(query)

# Assign to specific agent
researcher = Agent(
    role="Research Analyst",
    tools=[search_kb]
)
```

CrewAI assigns tools to specific agents based on their role.

### NVIDIA AIQ Toolkit Tools

```python
from aiq.tool import tool

@tool
def nim_inference(prompt: str, model: str = "llama-3.1-70b") -> str:
    """Run inference on an NVIDIA NIM endpoint."""
    response = nim_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### Comparison

| Aspect             | LangChain                         | AutoGen                               | CrewAI                         |
| ------------------ | --------------------------------- | ------------------------------------- | ------------------------------ |
| Tool definition    | `@tool` decorator                 | Python function + `register_function` | `@tool` decorator              |
| Description source | Docstring                         | `description` parameter               | Decorator argument + docstring |
| Tool assignment    | Global (all agents see all tools) | Per-agent (caller + executor)         | Per-agent (role-based)         |
| Execution          | Direct                            | Separated (caller vs executor)        | Direct                         |

---

## Tool Validation and Safety

### Input Validation

Always validate tool inputs before execution:

```python
def transfer_funds(from_account: str, to_account: str, amount: float) -> dict:
    # Validate inputs
    if amount <= 0:
        return {"status": "error", "message": "Amount must be positive"}
    if amount > 10000:
        return {"status": "error", "message": "Amount exceeds single transaction limit. Requires approval."}
    if from_account == to_account:
        return {"status": "error", "message": "Cannot transfer to same account"}

    # Execute transfer
    result = banking_api.transfer(from_account, to_account, amount)
    return {"status": "success", "transaction_id": result.id}
```

### Permission Levels

Not all tools are equal. Some are read-only, some modify data:

```python
# Safe: read-only tools
SAFE_TOOLS = [search_documents, get_weather, lookup_customer]

# Dangerous: tools that modify state
REQUIRES_CONFIRMATION = [send_email, update_record, delete_file]

# Very dangerous: irreversible actions
REQUIRES_HUMAN_APPROVAL = [transfer_funds, delete_account, publish_content]
```

Implement human-in-the-loop for destructive tools. The agent proposes the action, a human approves.

### Rate Limiting

Prevent agents from hammering external APIs:

```python
from functools import lru_cache
import time

class RateLimitedTool:
    def __init__(self, func, max_calls_per_minute=10):
        self.func = func
        self.max_calls = max_calls_per_minute
        self.calls = []

    def __call__(self, *args, **kwargs):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < 60]
        if len(self.calls) >= self.max_calls:
            return {"status": "error", "message": "Rate limit reached. Try again in a moment."}
        self.calls.append(now)
        return self.func(*args, **kwargs)
```

---

## Tool Composition

Agents often need to chain multiple tools. This happens naturally through ReAct:

```
Thought: User wants to know NVIDIA's revenue growth. I need to get financial data.
Action: get_stock_info(ticker="NVDA")
Observation: {revenue: 60.9B, prev_year: 27.0B}
Thought: Now I need to calculate the growth rate.
Action: calculate(expression="(60.9 - 27.0) / 27.0 * 100")
Observation: 125.6%
Thought: I have the answer.
Action: respond("NVIDIA's revenue grew approximately 125.6% year-over-year")
```

The agent figured out on its own that it needed two tools and what order to call them in. Good tool descriptions enable this.

---

## Exam-Style Questions

**Q1: An agent has access to a `send_email` tool and a `search_documents` tool. The user asks "find the latest report and email it to my manager." What tool design principle should be applied to `send_email`?**

- A) Make it fully autonomous with no restrictions
- B) Require human confirmation before executing (destructive action)
- C) Disable it entirely
- D) Only allow it during business hours

**Answer: B** - Sending email is a destructive action (can't unsend). It should require human confirmation before the agent actually sends.

**Q2: What is the most important factor in helping an LLM correctly decide when to use a tool?**

- A) The tool's implementation code
- B) The tool's name and description
- C) The number of parameters
- D) The tool's execution speed

**Answer: B** - The LLM only sees the tool's name, description, and parameter schema. A clear, specific description determines whether the LLM uses the tool correctly.

**Q3: In AutoGen, what distinguishes tool handling from LangChain?**

- A) AutoGen doesn't support tools
- B) AutoGen separates the caller (decides to use tool) from the executor (runs the tool)
- C) AutoGen only supports Python tools
- D) AutoGen requires all tools to be async

**Answer: B** - AutoGen has a unique caller/executor separation. The AssistantAgent decides to call a tool, but the UserProxyAgent executes it.

**Q4: An agent calls an external API tool that returns a 503 error. What is the best tool design response?**

- A) Crash the agent
- B) Return a clear error message so the agent can decide to retry or inform the user
- C) Silently return empty results
- D) Retry indefinitely

**Answer: B** - Return structured error info (status, message). The agent can then reason about whether to retry, try a different approach, or inform the user.

---

## Key Takeaways for NVIDIA Certification

1. Tools follow a standard structure: name, description, parameters, implementation
2. OpenAI function calling format is the de facto standard (used by NIM, LangChain, AutoGen)
3. Good tool descriptions are critical. The LLM relies on them to know when and how to use tools
4. Always validate inputs and handle errors gracefully with clear messages
5. Destructive tools (send, delete, modify) need human-in-the-loop confirmation
6. LangChain uses `@tool` decorator, AutoGen separates caller/executor, CrewAI assigns tools to roles
7. Tools should be focused (one thing well), return structured output, and include status fields
8. Rate limiting, timeouts, and retries are essential for API-based tools
