# Real-Time Knowledge Access and Reasoning Patterns

**Enable real-time access and reasoning over structured and unstructured knowledge**

Modern agents need to reason across both structured (databases, APIs) and unstructured (documents, text) knowledge in real time. This guide covers patterns for hybrid reasoning, caching, and real-time inference.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│         Agent (LLM) with Tools                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │
│   │ SQL Agent    │  │ RAG Tool     │  │ API Tool│ │
│   └──────────────┘  └──────────────┘  └─────────┘ │
│          │                 │                │      │
└──────────┼─────────────────┼────────────────┼──────┘
           │                 │                │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  Databases  │   │ Vector DB + │   │ External    │
    │  (SQL/NoSQL)│   │ Document    │   │ APIs        │
    │             │   │ Store       │   │             │
    └─────────────┘   └─────────────┘   └─────────────┘
           │                 │                │
           └─────────────────┼────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Knowledge Cache │
                    │ (Redis/Memcached)
                    └─────────────────┘
```

## Pattern 1: SQL Agent for Structured Data

SQL agents translate natural language to database queries, enabling agents to reason over structured data.

```python
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI

# Setup
database = SQLDatabase.from_uri(
    "postgresql://user:pass@localhost/analytics"
)
toolkit = SQLDatabaseToolkit(db=database, llm=OpenAI())

sql_agent = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_executor_kwargs={"max_iterations": 10}
)

# Agent can now answer questions about the database
response = sql_agent.run(
    "What are the top 5 products by revenue in Q1 2024?"
)
```

### Advanced: Text-to-SQL with LLM

```python
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate

class TextToSQLEngine:
    """Convert natural language to SQL queries"""

    def __init__(self, database, llm):
        self.database = database
        self.llm = llm

    def generate_sql(self, question: str) -> str:
        """Generate SQL from natural language"""

        # Create schema context
        schema_info = self.database.get_table_info()

        prompt = PromptTemplate(
            input_variables=["question", "schema"],
            template="""Given the database schema:

{schema}

Answer the following question by generating SQL:

Question: {question}

SQL Query:"""
        )

        chain = create_sql_query_chain(self.llm, self.database)
        sql_query = chain.run(question)

        return sql_query

    def execute_with_fallback(self, question: str) -> dict:
        """Execute query with validation and fallback"""

        sql = self.generate_sql(question)

        try:
            result = self.database.run(sql)
            return {
                "status": "success",
                "sql": sql,
                "result": result
            }
        except Exception as e:
            # Fallback: use LLM to fix query
            fixed_sql = self.llm.predict(
                f"Fix this SQL error: {str(e)}\nOriginal: {sql}"
            )

            try:
                result = self.database.run(fixed_sql)
                return {
                    "status": "success_with_fix",
                    "original_sql": sql,
                    "fixed_sql": fixed_sql,
                    "result": result
                }
            except:
                return {
                    "status": "failed",
                    "error": str(e),
                    "sql": sql
                }
```

### Schema Optimization for SQL Agents

```python
class SchemaOptimizer:
    """Prepare database schema for agent reasoning"""

    @staticmethod
    def add_helpful_views(database):
        """Create materialized views that help agents reason"""

        views = [
            # Sales analysis
            """
            CREATE MATERIALIZED VIEW product_performance AS
            SELECT
                p.product_id,
                p.name,
                COUNT(*) as total_sales,
                SUM(o.amount) as revenue,
                AVG(o.amount) as avg_order_value,
                DATE_TRUNC('month', o.created_at)::date as month
            FROM products p
            JOIN orders o ON p.product_id = o.product_id
            GROUP BY p.product_id, p.name, DATE_TRUNC('month', o.created_at)
            """,

            # Customer segments
            """
            CREATE MATERIALIZED VIEW customer_segments AS
            SELECT
                c.customer_id,
                c.email,
                COUNT(*) as purchase_count,
                SUM(o.amount) as lifetime_value,
                CASE
                    WHEN SUM(o.amount) > 10000 THEN 'VIP'
                    WHEN SUM(o.amount) > 1000 THEN 'Regular'
                    ELSE 'New'
                END as segment
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.customer_id, c.email
            """
        ]

        for view in views:
            database.run(view)

    @staticmethod
    def add_helpful_indexes(database):
        """Create indexes to speed up agent queries"""

        indexes = [
            "CREATE INDEX idx_orders_customer ON orders(customer_id)",
            "CREATE INDEX idx_orders_date ON orders(created_at)",
            "CREATE INDEX idx_products_category ON products(category_id)",
        ]

        for index in indexes:
            database.run(index)
```

## Pattern 2: API Agents for Real-Time Data

API agents fetch fresh data from external sources during reasoning.

```python
from typing import Optional
from datetime import datetime
import httpx

class APIAgent:
    """Agent tool for real-time API access"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}
        )

    async def get_real_time_data(self, endpoint: str, params: dict) -> dict:
        """Fetch fresh data from API"""

        url = f"{self.base_url}/{endpoint}"

        try:
            response = await self.client.get(
                url,
                params=params,
                timeout=5.0
            )
            response.raise_for_status()

            return {
                "status": "success",
                "data": response.json(),
                "timestamp": datetime.utcnow().isoformat(),
                "source": endpoint
            }
        except httpx.TimeoutException:
            return {"status": "timeout", "endpoint": endpoint}
        except httpx.HTTPError as e:
            return {"status": "error", "error": str(e)}

    async def get_stock_price(self, symbol: str) -> dict:
        """Example: Get real-time stock price"""
        return await self.get_real_time_data(
            "quote",
            {"symbol": symbol}
        )

    async def get_weather(self, location: str) -> dict:
        """Example: Get current weather"""
        return await self.get_real_time_data(
            "forecast/current",
            {"location": location}
        )

# Register with agent
api_tools = [
    APIAgent("https://api.example.com", "key").get_stock_price,
    APIAgent("https://api.example.com", "key").get_weather,
]
```

## Pattern 3: Hybrid Reasoning (SQL + RAG + API)

Combine multiple data sources in a single agent reasoning loop.

```python
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.memory import ConversationBufferMemory

class HybridReasoningAgent:
    """Agent that reasons across SQL, RAG, and API data"""

    def __init__(self, llm, database, vector_store, api_agent):
        self.llm = llm
        self.database = database
        self.vector_store = vector_store
        self.api_agent = api_agent
        self.memory = ConversationBufferMemory()

    def create_tools(self):
        """Define tools available to agent"""

        tools = [
            Tool(
                name="QueryDatabase",
                func=self._query_database,
                description="Query the SQL database for structured data like sales, customers, products"
            ),
            Tool(
                name="SearchDocuments",
                func=self._search_documents,
                description="Search company documents, policies, and unstructured knowledge"
            ),
            Tool(
                name="FetchRealTimeData",
                func=self._fetch_real_time,
                description="Get current market data, weather, or other real-time information"
            ),
            Tool(
                name="AnalyzeTrends",
                func=self._analyze_trends,
                description="Analyze trends by combining historical DB data with real-time API data"
            )
        ]

        return tools

    def _query_database(self, question: str) -> str:
        """Query structured database"""
        result = TextToSQLEngine(
            self.database, self.llm
        ).execute_with_fallback(question)
        return str(result)

    def _search_documents(self, query: str) -> str:
        """Search vector database"""
        results = self.vector_store.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in results])

    async def _fetch_real_time(self, request: str) -> str:
        """Fetch real-time data from APIs"""
        # Parse request to determine which API to call
        if "stock" in request.lower():
            symbol = request.split()[-1]
            data = await self.api_agent.get_stock_price(symbol)
        elif "weather" in request.lower():
            location = request.split("in")[-1].strip()
            data = await self.api_agent.get_weather(location)
        else:
            return "Unknown real-time data request"

        return str(data)

    def _analyze_trends(self, analysis_request: str) -> str:
        """Combine multiple data sources for analysis"""

        # Extract components from request
        # Example: "Compare Q1 sales trends with current market conditions"

        # Get historical data
        historical = self._query_database(
            f"Get {analysis_request.split()[2]} data"
        )

        # Get real-time context
        real_time = asyncio.run(
            self._fetch_real_time(analysis_request)
        )

        # Let LLM synthesize
        synthesis_prompt = f"""
        Historical Data: {historical}
        Real-Time Data: {real_time}

        Analyze: {analysis_request}
        """

        return self.llm.predict(synthesis_prompt)

    def run(self, user_query: str) -> str:
        """Execute hybrid reasoning"""

        tools = self.create_tools()

        agent = initialize_agent(
            tools,
            self.llm,
            agent="zero-shot-react-description",
            memory=self.memory,
            verbose=True
        )

        return agent.run(user_query)
```

## Pattern 4: Streaming Data Integration

Process live data feeds while maintaining consistency with agent reasoning.

```python
from datetime import datetime
from collections import deque
import asyncio

class StreamingDataBuffer:
    """Buffer streaming data for agent consumption"""

    def __init__(self, max_window_size: int = 1000):
        self.buffer = deque(maxlen=max_window_size)
        self.aggregates = {}
        self.last_update = None

    async def ingest_stream(self, stream_source: str):
        """Ingest streaming data"""

        async for event in stream_source:
            self.buffer.append({
                "timestamp": datetime.utcnow(),
                "data": event,
                "source": stream_source
            })

            # Update aggregates
            await self.update_aggregates(event)

            self.last_update = datetime.utcnow()

    async def update_aggregates(self, event: dict):
        """Compute running aggregates"""

        # Example: compute moving averages, counts
        if "price" in event:
            prices = [e["data"].get("price") for e in self.buffer if "price" in e["data"]]

            if prices:
                self.aggregates["avg_price"] = sum(prices) / len(prices)
                self.aggregates["max_price"] = max(prices)
                self.aggregates["min_price"] = min(prices)

    def get_current_state(self) -> dict:
        """Get current state for agent"""

        return {
            "buffer_size": len(self.buffer),
            "last_update": self.last_update,
            "aggregates": self.aggregates,
            "recent_events": list(self.buffer)[-10:]  # Last 10 events
        }

# Usage: Real-time stock price monitoring
class StockStreamAgent:
    def __init__(self, llm, database):
        self.llm = llm
        self.db = database
        self.stream_buffer = StreamingDataBuffer()

    async def monitor_and_react(self, stock_symbol: str):
        """Monitor stock stream and react to events"""

        # Start ingesting price stream
        async for price_event in stream_stock_prices(stock_symbol):
            state = self.stream_buffer.get_current_state()

            # Check if current price warrants action
            if state["aggregates"]["max_price"] > threshold:
                # Use LLM to decide action
                action = self.llm.predict(f"""
                    Current stock state: {state}
                    Historical context: [Query DB for previous behavior]
                    Should we alert or take action?
                """)

                if "alert" in action.lower():
                    await self.execute_alert(stock_symbol, state)
```

## Caching Strategies for Real-Time Performance

### Query Result Caching

```python
from functools import wraps
import redis
from datetime import timedelta
import json

class CacheManager:
    """Smart caching for frequently accessed data"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)

    def cache_key(self, *args, **kwargs):
        """Generate deterministic cache key"""
        key_data = json.dumps({
            "args": args,
            "kwargs": sorted(kwargs.items())
        }, default=str)
        return f"query:{hash(key_data)}"

    def cached_query(self, ttl_seconds: int = 300):
        """Decorator for caching database queries"""

        def decorator(func):
            def wrapper(*args, **kwargs):
                cache_key = self.cache_key(*args, **kwargs)

                # Check cache
                cached = self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)

                # Execute query
                result = func(*args, **kwargs)

                # Store in cache with TTL
                self.redis.setex(
                    cache_key,
                    timedelta(seconds=ttl_seconds),
                    json.dumps(result, default=str)
                )

                return result

            return wrapper

        return decorator

    # Usage
    @cached_query(ttl_seconds=3600)  # 1 hour cache
    def get_customer_profile(customer_id: str):
        """Cache customer data for 1 hour"""
        return database.query(f"SELECT * FROM customers WHERE id = {customer_id}")

    @cached_query(ttl_seconds=60)  # 60 second cache
    def get_real_time_metrics(metric_name: str):
        """Cache metrics with shorter TTL"""
        return api.fetch_metrics(metric_name)
```

### Predictive Caching

```python
class PredictiveCache:
    """Precompute likely-to-be-needed queries"""

    def __init__(self, cache_manager: CacheManager, llm):
        self.cache = cache_manager
        self.llm = llm

    async def warm_cache(self, context: dict):
        """Predict and precompute needed queries"""

        # Use LLM to predict what user might ask
        predictions = self.llm.predict(f"""
            Given context: {context}
            What 5 queries might the user ask next?
            Return as JSON list of queries.
        """)

        # Precompute predictions
        for query in json.loads(predictions):
            try:
                # These get cached automatically
                await execute_query(query)
            except:
                pass  # Ignore failed predictions
```

## Real-Time Performance with NVIDIA NIM

NVIDIA NIM provides optimized inference endpoints for real-time reasoning.

```python
from openai import AsyncOpenAI

class NIMAcceleratedAgent:
    """Use NVIDIA NIM for sub-second inference"""

    def __init__(self, nim_endpoint: str = "http://localhost:8000/v1"):
        self.client = AsyncOpenAI(
            api_key="nim-key",
            base_url=nim_endpoint
        )

    async def reason_with_tools(self, user_query: str, available_tools: list):
        """Real-time reasoning with NIM acceleration"""

        messages = [
            {
                "role": "system",
                "content": f"You are an agent with access to these tools: {[t.name for t in available_tools]}"
            },
            {"role": "user", "content": user_query}
        ]

        response = await self.client.chat.completions.create(
            model="mistral-7b",  # NIM-optimized model
            messages=messages,
            temperature=0,
            max_tokens=500
        )

        return response.choices[0].message.content

    async def parallel_reasoning(self, queries: list) -> list:
        """Parallel tool calls for throughput"""

        tasks = [
            self.reason_with_tools(q, [])
            for q in queries
        ]

        results = await asyncio.gather(*tasks)

        return results
```

## Integration Example: Full Real-Time Agent

```python
class RealTimeKnowledgeAgent:
    """Production agent combining all patterns"""

    def __init__(self, config: dict):
        self.db = SQLDatabase.from_uri(config["db_uri"])
        self.vector_store = Milvus(uri=config["milvus_uri"])
        self.api_agent = APIAgent(config["api_url"], config["api_key"])
        self.cache = CacheManager(config["redis_url"])
        self.stream_buffer = StreamingDataBuffer()
        self.llm = OpenAI(model="gpt-4")

    def handle_query(self, user_query: str) -> str:
        """Process user query using all available knowledge"""

        # 1. Check cache first
        cache_key = f"user_query:{hash(user_query)}"
        cached_result = self.cache.redis.get(cache_key)
        if cached_result:
            return json.loads(cached_result)

        # 2. Get relevant context
        structured_context = self._query_database(user_query)
        unstructured_context = self._search_documents(user_query)
        real_time_context = asyncio.run(
            self._fetch_real_time_context(user_query)
        )

        # 3. Prepare prompt
        full_context = f"""
        Structured Data: {structured_context}
        Documents: {unstructured_context}
        Real-Time: {real_time_context}
        """

        # 4. Reasoning with NIM
        response = self.llm.predict(f"{full_context}\n\nQuery: {user_query}")

        # 5. Cache result
        self.cache.redis.setex(
            cache_key,
            timedelta(seconds=300),
            json.dumps(response)
        )

        return response
```

## Study Questions

**Q1:** An agent needs to answer "What were Q4 sales AND how do they compare to market conditions today?" Which patterns should it use?

A) SQL Agent only - database has all historical data
B) API Agent only - use market data API
C) Hybrid: SQL Agent (historical) + API Agent (real-time) + LLM synthesis
D) RAG only - embed all data and search

**Answer: C** - Hybrid reasoning across SQL (historical sales), API (current market), and LLM synthesis best answers comparative questions.

---

**Q2:** Your agent gets 10K queries/second but 70% are repeated within 5 minutes. What's the best optimization?

A) Increase database connection pool size
B) Implement query result caching with 5-minute TTL
C) Switch to vector search instead of SQL
D) Use larger batch sizes

**Answer: B** - Query caching at 5-minute TTL for 70% hit rate reduces database load by 70% immediately.

---

**Q3:** You need real-time stock price updates integrated with historical analysis. What's the best architecture?

A) Poll API every second and update vector DB
B) Streaming buffer (deque) + agent tool that taps buffer for recent data
C) Cache all stock history in Redis
D) Generate embeddings for each price tick

**Answer: B** - Streaming buffer efficiently maintains recent window. Agent can call tool to get current aggregate state without polling database.

---

**Q4:** Your agent needs sub-100ms response time for user queries. Which is most important?

A) Caching strategy for frequent queries
B) Using NVIDIA NIM for optimized inference
C) Implementing predictive warming of cache
D) All of above equally

**Answer: A** - Caching has highest ROI for response time (can drop from seconds to milliseconds). B and C compound the benefit but A is necessary first.
