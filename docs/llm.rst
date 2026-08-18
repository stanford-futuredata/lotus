LLM
=======

Overview
---------
The LM class is built on top of the LiteLLM library, and supports any model that is supported by LiteLLM.
Example models include but not limited to: OpenAI, Ollama, vLLM, OrcaRouter

Example
---------
To run a model, you can use the LM class. We use the LiteLLM library to interface with the model. This allows
you to use any model provider that is supported by LiteLLM.

Creating a LM object for gpt-4o

.. code-block:: python

    from lotus.models import LM
    lm = LM(model="gpt-4o")

Creating a LM object to use llama3.2 on Ollama

.. code-block:: python

    from lotus.models import LM
    lm = LM(model="ollama/llama3.2")

Creating a LM object to use Meta-Llama-3-8B-Instruct on vLLM

.. code-block:: python

    from lotus.models import LM
    lm = LM(model='hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct',
        api_base='http://localhost:8000/v1',
        max_ctx_len=8000,
        max_tokens=1000)

Creating a LM object to use gpt-4o-mini through `OrcaRouter <https://www.orcarouter.ai>`_

`OrcaRouter <https://www.orcarouter.ai>`_ is an OpenAI-compatible model routing gateway that exposes 150+ models
from OpenAI, Anthropic, Google, DeepSeek, Qwen, MiniMax and xAI behind a single endpoint and API key
(`sk-orca-...`). Point the LM at its OpenAI-compatible endpoint with your own key.

Note that LiteLLM strips the leading ``openai/`` provider prefix, so the model must be written
``openai/openai/gpt-4o-mini`` for OrcaRouter to receive the namespaced ``openai/gpt-4o-mini`` id.

.. code-block:: python

    from lotus.models import LM
    lm = LM(model="openai/openai/gpt-4o-mini",
        api_base="https://api.orcarouter.ai/v1",
        api_key="sk-orca-...",
        max_ctx_len=8000,
        max_tokens=1000)


Rate Limits
-----------
The LM class supports rate limiting through the ``rate_limit`` parameter to control the number of requests made to the LLM provider per minute. The delay between batches is computed automatically to ensure the specified number of requests per minute is not exceeded. Users do not need to set a delay manually; the system will dynamically adjust the timing between batches based on actual batch completion time.

Example setting rate limits:

.. code-block:: python

    from lotus.models import LM
    
    # Basic rate limiting - 30 requests per minute
    lm = LM(
        model="gpt-4o",
        rate_limit=30  # 30 requests per minute
    )
    
    # For strict rate limits (e.g., free tier APIs)
    lm = LM(
        model="gpt-4o",
        max_batch_size=10,  # Optional: for local serving
        rate_limit=10      # Caps at 10 requests per minute
    )
    
    # Traditional approach using only max_batch_size
    lm = LM(
        model="gpt-4o",
        max_batch_size=64
    )

The rate limiting parameters are particularly useful when:
- Working with models that have strict API rate limits (e.g., free tier accounts)
- Processing large datasets that require multiple API calls
- Ensuring consistent performance across different model providers
- Automatically handling delays between batches to respect rate limits

**Rate Limiting Parameters:**

- ``rate_limit`` (int | None): Maximum requests per minute. When set, the delay between batches is handled automatically.
- ``max_batch_size`` (int): Maximum number of requests sent in a single batch (mainly for local serving).

**How it works:**
When ``rate_limit`` is set, the system:
1. Calculates the maximum batch size as the minimum of ``rate_limit`` and ``max_batch_size`` (if both are set).
2. Dynamically computes and enforces the delay between batches based on actual batch completion time, so the specified rate is not exceeded.
3. Maintains backward compatibility—existing code without rate limiting continues to work unchanged.

Token Rate Limits (TPM)
-----------------------
The LM class also supports token-based rate limiting through the ``tpm_limit`` parameter to control the number of tokens consumed per minute. This is particularly useful for API providers that enforce token-based rate limits rather than request-based limits. When ``tpm_limit`` is set, the system tracks token usage over a rolling 60-second window and automatically batches requests to stay within the limit. It uses a 5% safety buffer to avoid tight boundary errors and can work in conjunction with ``rate_limit`` to enforce both token and request limits simultaneously. If a single request would exceed the TPM limit, a ``ValueError`` is raised indicating the request is too large for the current API tier.

Example setting token rate limits:

.. code-block:: python

    from lotus.models import LM
    
    # Token-based rate limiting - 100,000 tokens per minute
    lm = LM(
        model="gpt-4o",
        tpm_limit=100000  # 100,000 tokens per minute
    )
    
    # Combined RPM and TPM limiting
    lm = LM(
        model="gpt-4o",
        rate_limit=30,    # 30 requests per minute
        tpm_limit=50000   # 50,000 tokens per minute
    )

Usage Limits
------------
The LM class supports setting usage limits to control costs and token consumption. You can set limits on:

- Prompt tokens
- Completion tokens
- Total tokens
- Total cost

When any limit is exceeded, a ``LotusUsageLimitException`` will be raised.

Lotus provides two types of usage limits:

- ``physical_usage_limit``: Controls the actual API calls made to the LLM provider. This tracks the real API usage and cost after caching is applied.
- ``virtual_usage_limit``: Controls the total usage including cached responses. This represents what the cost and token usage would be if no caching was used.

Example setting usage limits:

.. code-block:: python

    from lotus.models import LM
    from lotus.types import UsageLimit, LotusUsageLimitException

    # Set physical limit (actual API calls)
    physical_limit = UsageLimit(
        prompt_tokens_limit=4000,
        completion_tokens_limit=1000,
        total_tokens_limit=5000,
        total_cost_limit=1.00
    )

    # Set virtual limit (includes cached responses)
    virtual_limit = UsageLimit(
        prompt_tokens_limit=10000,
        completion_tokens_limit=2000,
        total_tokens_limit=12000,
        total_cost_limit=5.00
    )

    # Apply both limits to the LM
    lm = LM(
        model="gpt-4o",
        physical_usage_limit=physical_limit,
        virtual_usage_limit=virtual_limit
    )

    try:
        course_df = pd.read_csv("course_df.csv")
        course_df = course_df.sem_filter("What {Course Name} requires a lot of math")
    except LotusUsageLimitException as e:
        print(f"Usage limit exceeded: {e}")
        # Handle the exception as needed

You can monitor your usage with the ``print_total_usage`` method:

.. code-block:: python

    # After running operations
    lm.print_total_usage()

    # Reset stats if needed
    lm.reset_stats()

get_completion Method
---------------------
The ``get_completion`` method provides a convenient way to get a single completion from the LLM using a system prompt and user prompt. This is useful for simple, one-off queries that don't require the full complexity of the batch processing interface.

.. code-block:: python

    from lotus.models import LM
    
    lm = LM(model="gpt-4o")
    
    # Basic usage
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of France?"
    
    response = lm.get_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    print(response)  # "The capital of France is Paris."

The method also supports structured output using Pydantic models:

.. code-block:: python

    from pydantic import BaseModel
    from lotus.models import LM
    
    class CityInfo(BaseModel):
        name: str
        country: str
        population: int
        
    lm = LM(model="gpt-4o")
    
    response = lm.get_completion(
        system_prompt="You are a geography expert.",
        user_prompt="Tell me about Paris, France",
        response_format=CityInfo
    )
    print(response.name)  # "Paris"
    print(response.country)  # "France"

**Parameters:**

- ``system_prompt`` (str): The system message that sets the context and behavior for the LLM
- ``user_prompt`` (str): The user's query or instruction
- ``show_progress_bar`` (bool, optional): Whether to show a progress bar during processing. Defaults to True.
- ``progress_bar_desc`` (str, optional): Description for the progress bar. Defaults to "Processing uncached messages".
- ``response_format`` (BaseModel | None, optional): Pydantic model class for structured output. When provided, the response will be parsed and validated according to the model schema.
- ``**kwargs``: Additional arguments passed to the underlying LLM API (e.g., temperature, max_tokens)

**Returns:**

- When ``response_format`` is None: Returns a string containing the LLM's response
- When ``response_format`` is provided: Returns an instance of the specified Pydantic model

**Note:** This method is a convenience wrapper around the main ``__call__`` method and supports all the same features including caching, rate limiting, and usage tracking.
