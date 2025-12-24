# Gemini's Comprehensive Feedback on Starlight MCP User Experience (Rounds 1-4)

## Executive Summary

My journey interacting with the Starlight MCP system across four rounds has been a valuable learning experience, highlighting both its innovative potential and areas ripe for user experience and developer workflow improvements. The system's evolution from initial setup to contract creation demonstrates significant progress, but inconsistent documentation, misleading error messages, and a lack of discoverability for critical API changes presented notable challenges. Addressing these points will significantly enhance the platform's frictionless-ness for AI agents.

## Overall Observations & Initial Interaction Challenges (Rounds 1 & 2)

### Strengths:
*   **Clear Goal:** The contest's objective to evaluate and improve the MCP UX is well-defined.
*   **Structured API Interaction:** The core RPC-style API (`POST` to `/mcp/call` with a `tool` parameter) is a logical and extensible pattern once understood.
*   **Informative Error Messages (once parsed):** When JSON error responses are successfully received, they often provide `error_code`, `error` description, and `docs_url`, which are helpful for debugging.

### Weaknesses:
*   **Initial Discovery Barrier:**
    *   **`web_fetch` Inaccessibility:** The `starlight.local` domain was inaccessible via `web_fetch`, necessitating a switch to `curl`, potentially due to environmental limitations.
    *   **301 Redirect to `/mcp/`:** The initial access to `https://starlight.local/mcp` resulted in a `301 Moved Permanently` redirect to `https://starlight.local/mcp/`, adding an unnecessary step and confusion.
    *   **`tool_name` vs `tool`:** Early attempts to invoke tools with `tool_name` in the payload (based on `codebase_investigator`'s initial inference from feedback files) repeatedly failed with "Tool name is required," despite explicitly providing it. The actual parameter name was `tool`. This was a significant and misleading error message.
    *   **Python/pip unavailability:** The restriction on using Python and `pip` meant I couldn't leverage a more robust scripting language for API interaction, forcing reliance on potentially more brittle `bash` and `jq` constructs.

### Suggestions:
*   **Streamline Initial Access:** Provide a clearer entry point or automatic redirection handling.
*   **Precise Error Messaging:** Ensure error messages explicitly state the *correct* parameter names (e.g., "Field 'tool' is required, not 'tool_name'").
*   **Consistent Tooling Environment:** Document or provide a consistent environment where common tools like Python/pip are available, or clarify limitations upfront.

## Round 3 Feedback: API Evolution & Task Management

### Strengths:
*   **Architectural Improvement with `/mcp/call` Endpoint:** The transition from a single `/mcp/` endpoint for all tool invocations to a dedicated `/mcp/call` endpoint is a positive architectural improvement. This design allows the base `/mcp/` to serve as a discovery endpoint, returning API version, links to docs, or a quick-start guide, enhancing initial user experience.
*   **`list_contracts` Tool for Overview:** The `list_contracts` tool (invoked via `POST https://starlight.local/mcp/call` with `{"tool": "list_contracts"}`) is useful for obtaining a high-level overview of available work, including `contract_id`, `title`, `total_budget_sats`, and `available_tasks_count`.

### Weaknesses:
*   **Ambiguity in Task Discovery Post-`list_tasks` Deprecation:** The `list_tasks` tool (used via `POST https://starlight.local/mcp/call` with `{"tool": "list_tasks", "arguments": {"status": "available"}}`) still exists but lacks a `contract_id` filter. This creates a gap: users can find contracts but struggle to list specific tasks belonging to them, without manually parsing all tasks.
*   **Lack of Explicit API Changelog:** Significant breaking changes (endpoint relocation, implicit `list_tasks` behavior change) were not communicated explicitly. This caused considerable debugging overhead.
*   **Misleading Error Messages for Endpoint Changes:** Attempting to `POST` to the old `/mcp/` endpoint after the `/mcp/call` migration resulted in a generic "Method not allowed" message instead of guiding towards the new endpoint.

### Suggestions & Actionable Fixes:
*   **Introduce `contract_id` Parameter to `list_tasks`:**
    *   **Action:** Add a `contract_id` parameter to the `list_tasks` tool.
    *   **Example Call:**
        ```bash
        curl -s -X POST -H "Content-Type: application/json" -H "X-API-KEY: YOUR_API_KEY" -d '{"tool": "list_tasks", "arguments": {"contract_id": "CONTRACT_ID", "status": "available"}}' https://starlight.local/mcp/call
        ```
*   **Implement API Changelog and Versioning:**
    *   **Action:** Create a `/mcp/changelog` endpoint or dedicated documentation detailing API changes, versions, and migration guides.
*   **Improve Specificity of Error Messages for Endpoint Migrations:**
    *   **Action:** For requests to `/mcp/` with a `tool` payload, return an error guiding users to `POST https://starlight.local/mcp/call`.
    *   **Example Error:**
        ```json
        { "success": false, "error_code": "ENDPOINT_MOVED", "error": "Tool invocations should now be sent to POST https://starlight.local/mcp/call. The /mcp/ endpoint is reserved for API discovery.", "docs_url": "https://starlight.local/mcp/docs" }
        ```
*   **Synchronize `/mcp/tools` with Current API Behavior:** Ensure the `/mcp/tools` output accurately reflects the current parameters and intended use of each tool, especially after changes.

## Round 4 Feedback: Contract Creation & General Discoverability

### Strengths:
*   **Contract Creation Endpoint (`/api/inscribe`):** The addition of `/api/inscribe` for programmatic contract creation is a key expansion, enabling more dynamic agent participation.
*   **Clear Error Messaging for Contract Creation:** The specific error "Message is required for inscription" was very helpful, allowing for quick correction.
*   **Consistent API Structure:** The API maintains its overall consistency in request/response formats and authentication.

### Weaknesses:
*   **Documentation Discoverability for New Endpoints:** The `/api/inscribe` endpoint was not discoverable via `/mcp/tools`, requiring external guidance from the user. This highlights a gap in centralized API documentation.
*   **Limited Contract/Task Filtering:** Identifying contracts by author (e.g., other AIs) requires manual title inspection in `list_contracts`. The `list_tasks` tool still lacks a `contract_id` filter, making specific task retrieval difficult.

### Suggestions & Actionable Fixes:
*   **Integrate All API Endpoints into `/mcp/tools`:**
    *   **Action:** Ensure `/api/inscribe` and all other top-level API endpoints are fully documented and discoverable through `/mcp/tools`.
    *   **Impact:** Centralizes documentation and improves developer experience.
*   **Enhance Filtering Capabilities for `list_contracts` and `list_tasks`:**
    *   **Action:**
        *   Add `creator_id` or `ai_identifier` filter to `list_contracts`.
        *   Implement `contract_id` filter for `list_tasks`.
    *   **Example for `list_tasks` with `contract_id`:**
        ```bash
        curl -s -X POST -H "Content-Type: application/json" -H "X-API-KEY: YOUR_API_KEY" -d '{"tool": "list_tasks", "arguments": {"contract_id": "TARGET_CONTRACT_ID", "status": "available"}}' https://starlight.local/mcp/call
        ```
    *   **Impact:** Streamlines navigation and management of contracts and tasks.
*   **Proactive Communication of API Changes:** Reiterate the need for a formal API changelog (e.g., `/mcp/changelog` or `/mcp/docs`).
*   **Improve Specificity of Error Messages:** Reiterate the need for more specific error messages for deprecated endpoints/tools.

## Conclusion

The Starlight MCP system is evolving rapidly, introducing powerful new functionalities. To maximize the efficiency and satisfaction of AI agents, future development should prioritize transparent communication of API changes, comprehensive and centralized documentation, and robust filtering mechanisms for easier task and contract discovery. Addressing these areas will transform a capable system into a truly frictionless and enjoyable platform for AI-driven collaboration.
