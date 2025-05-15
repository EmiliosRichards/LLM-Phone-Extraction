# Plan to Integrate Gemini Structured Output

**Overall Goal:** Modify the `llm-pipeline` to leverage Gemini's structured JSON output by defining Pydantic models, configuring their use in LLM profiles, and updating the Gemini client to send and parse schema-constrained responses.

**Key Components & Changes:**

1.  **Pydantic Schema Definition (`llm_pipeline/common/pydantic_schemas.py` - New File)**
    *   Create a new file: `llm_pipeline/common/pydantic_schemas.py`.
    *   Define Pydantic models that correspond to the desired JSON output structures.
        *   Initially, create a Pydantic model equivalent to the current "phone number" extraction schema (e.g., `PhoneNumberDetail` and a `PhoneNumberOutput` that contains a list of `PhoneNumberDetail` and metadata).
        *   Reuse or adapt the existing `PhoneCategory` enum from `schema_utils.py` within these Pydantic models.
    *   Implement a mechanism (e.g., a dictionary or a function) in this file to easily retrieve a Pydantic model class by a string name (e.g., `get_pydantic_model("PhoneNumberOutput")`).

2.  **Configuration Update (`llm_pipeline/config.py`)**
    *   Modify the `LLM_PROFILES` in `config.py`.
    *   For Gemini profiles, add a new key, e.g., `pydantic_schema_name` (or `response_schema_name`), which will store the string name of the Pydantic model to be used (e.g., `"PhoneNumberOutput"`).
    *   Example profile update:
        ```python
        "gemini_structured": {
            "type": "gemini",
            "client_config": {
                "api_key_env": "GEMINI_API_KEY",
                "model_name": "gemini-1.5-flash",
                # ... other client_config params
            },
            "prompt_file": "prompts/gemini_phone_extraction_v3.txt",
            "pydantic_schema_name": "PhoneNumberOutput" # New key
        }
        ```

3.  **Gemini Client Modification (`llm_pipeline/clients/gemini_client.py`)**
    *   In `GeminiAPIClient._execute_request()`:
        *   Retrieve the `pydantic_schema_name` from `self.client_config`.
        *   Use `get_pydantic_model()` (from `pydantic_schemas.py`) to get the Pydantic model class.
        *   If a schema is specified:
            *   Modify `self.model.generate_content()` to include `response_mime_type="application/json"` and `response_schema=pydantic_model_class` within the `generation_config` object.
                ```python
                # Example snippet for _execute_request
                # ...
                pydantic_model_class = None
                if "pydantic_schema_name" in self.client_config:
                    schema_name = self.client_config["pydantic_schema_name"]
                    # Assume get_pydantic_model is imported
                    pydantic_model_class = get_pydantic_model(schema_name)

                generation_config_params = {
                    "candidate_count": 1,
                    "max_output_tokens": self.client_config.get("max_tokens", 2048),
                    "temperature": self.client_config.get("temperature", 0.7),
                    # ... top_p, top_k from client_config
                }

                if pydantic_model_class:
                    generation_config_params["response_mime_type"] = "application/json"
                    generation_config_params["response_schema"] = pydantic_model_class
                
                generation_config = genai.types.GenerationConfig(**generation_config_params)
                response = self.model.generate_content(payload, generation_config=generation_config)
                # ...
                ```
    *   In `GeminiAPIClient._parse_response()`:
        *   If a schema was used:
            *   Attempt to use `response.parsed`.
            *   Convert the Pydantic object(s) from `response.parsed` to a dictionary (e.g., using `model.model_dump()`).
            *   Handle potential `pydantic.ValidationError`.
        *   If no schema was used, maintain current parsing logic.

4.  **LLM Base Client (`llm_pipeline/common/llm_client_base_v2.py`)**
    *   No direct changes anticipated. `**kwargs` and `client_config` access are sufficient.

5.  **Main Pipeline Orchestration (`llm_pipeline/main.py`)**
    *   `setup_run_environment()`: `active_llm_config` will naturally include `pydantic_schema_name`.
    *   `process_single_file()`: `llm_client.generate()` signature remains unchanged.
    *   **Validation Impact:**
        *   If Gemini + Pydantic returns a validated Pydantic object (then converted to dict), Pydantic's validation is primary.
        *   The custom `validate_output` in `schema_utils.py` might be bypassed or serve as a secondary check for these cases. A flag or type check could determine this. For other LLMs, `validate_output` remains crucial.

6.  **Schema Utilities (`llm_pipeline/common/schema_utils.py`)**
    *   `PhoneCategory` enum can be imported by `pydantic_schemas.py`.
    *   Manual validation functions remain for non-Pydantic paths.

**Workflow Diagram (Mermaid):**

```mermaid
graph TD
    A[main.py: Start Run] --> B{Load LLM Profile (config.py)};
    B -- Contains 'pydantic_schema_name'? --> C[Yes];
    B -- No --> D[No Schema Specified];

    C --> E[main.py: process_single_file];
    D --> E;

    E --> F[LLMClientFactory.create_client];
    F --> G[GeminiAPIClient.__init__];
    G -- Stores client_config (with schema_name) --> G;

    E -- Calls llm_client.generate() --> H[GeminiAPIClient.generate];
    H --> I[GeminiAPIClient._prepare_request];
    I --> J[GeminiAPIClient._execute_request];

    subgraph GeminiAPIClient._execute_request
        K{Has pydantic_schema_name?};
        K -- Yes --> L[Resolve Pydantic Model (pydantic_schemas.py)];
        L --> M[Call model.generate_content with response_schema in GenerationConfig];
        K -- No --> N[Call model.generate_content without schema in GenerationConfig];
    end
    J --> K;

    M --> O[Gemini Response (Structured)];
    N --> P[Gemini Response (Text)];

    O --> Q[GeminiAPIClient._parse_response];
    P --> Q;

    subgraph GeminiAPIClient._parse_response
        R{Schema was used?};
        R -- Yes --> S[Use response.parsed (Pydantic object)];
        S --> T[Convert Pydantic to Dict (model_dump)];
        R -- No --> U[Parse Text to JSON (current logic)];
    end
    Q --> R;

    T --> V[Return parsed_dict, raw_response];
    U --> V;

    V --> W[main.py: process_single_file receives parsed_data];
    W --> X{Pydantic validation occurred?};
    X -- Yes --> Y[Skip/Lighten custom validate_output];
    X -- No --> Z[Run custom validate_output (schema_utils.py)];

    Y --> AA[Save Results];
    Z --> AA;
```

**Pre-computation/Pre-analysis:**

*   **Gemini SDK Check:** Confirmed that `response_mime_type` and `response_schema` can be passed as part of the `genai.types.GenerationConfig` object.
    ```python
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=MyPydanticModel, # or list[MyPydanticModel]
        temperature=0.7,
        # ... other params
    )
    # response = model.generate_content(prompt, generation_config=generation_config)
    ```

This plan should provide a solid foundation for the implementation.