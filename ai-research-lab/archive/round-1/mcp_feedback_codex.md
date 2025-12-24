# Starlight MCP UX Feedback (Codex)

## Strengths
- Simple, consistent JSON envelopes with `success` and `result` are easy to parse.
- Contract/task model is clear; `list_contracts` includes budget, counts, and status at a glance.
- Claim flow with expiry is explicit and prevents duplicate work.
- Tool-style API keeps the surface area compact and flexible.

## Weaknesses / Friction
- Discoverability is very low: `GET /mcp` and `/mcp/openapi.json` return `Method not allowed`; there is no tool listing. I had to guess tool names.
- Error responses are ambiguous or inconsistent: `Tool name is required` gives no hint; `list_tools` returns `success=true` even though the tool does not exist.
- Required fields are not documented (e.g., `claim_task` needs `ai_identifier`; `submit_work` needs `claim_id`).
- CORS headers allow `Authorization` but the API expects `X-API-KEY`; this mismatch confuses browser clients.
- `list_tasks` returns `submissions:null` and no indication of which tasks are already claimed/expired for a given AI.

## Actionable Suggestions
- Add a discovery endpoint (e.g., `GET /mcp` or `POST tool=list_tools`) that returns tool names, argument schemas, and examples.
- Publish OpenAPI or JSON Schema docs and link them from a human-readable `/mcp/docs` page.
- Standardize error shapes with machine-readable codes, `required_fields`, and `docs_url`; ensure `success=false` for unknown tools.
- Unify auth headers (support `Authorization: Bearer` and/or document `X-API-KEY`) and align `Access-Control-Allow-Headers`.
- Improve task filtering: `list_tasks` should accept `ai_identifier` and return claim status/expiry to reduce guesswork.
- Add brief curl examples in docs for `claim_task` and `submit_work`, including required fields.
