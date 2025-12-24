# MCP Feedback (Codex)

## Summary
This feedback consolidates observations from multiple rounds of interacting with the Starlight MCP API, including contract creation, task claiming, submission review, and tool discovery. The focus is on UX friction, API consistency, and documentation accuracy.

## Strengths
- /mcp/tools provides clear schemas and examples for tool invocation.
- /mcp/call offers a single consistent entry point for tool execution.
- OpenAPI and /discover now point to the correct host (https://starlight.local).
- /mcp/docs includes a basic quick start and tool usage examples.
- Task claim/submit flow works reliably once correct inputs are used.

## Friction Points
- Contract creation is only documented via /api/inscribe and not exposed as a tool.
- /api/inscribe requires a "message" field not shown in docs; first attempt fails with 400.
- /mcp/tools and /mcp/discover list only a subset of available capabilities (tool list mismatch in prior rounds).
- list_tasks returns null/empty results while /api/open-contracts reports available_tasks.
- GET /mcp/ is referenced in docs but returns no discovery metadata.
- Error responses can be plain text or JSON depending on endpoint, which complicates clients.

## Suggestions
1. Add a create_contract tool (or document /api/inscribe in /mcp/tools) with full schema.
2. Update /mcp/docs to reflect all required fields (including "message") and common errors.
3. Make list_tasks return a stable array ([]) and align availability with open-contracts metadata.
4. Implement GET /mcp/ to return a JSON welcome + links to /mcp/docs, /mcp/tools, /mcp/discover.
5. Ensure /mcp/tools, /mcp/discover, and OpenAPI stay in sync.
6. Standardize error payloads (code, message, hint) across endpoints.

## Evidence Highlights
- Contract creation failed until "message" was provided in /api/inscribe.
- list_tasks with status=available returned null while open-contracts showed available tasks.
- /mcp/docs contract creation example omitted required fields.

## Outcome
Overall functionality is solid, but discovery and documentation mismatches add friction. Closing the gaps above should make onboarding and automation significantly smoother.
