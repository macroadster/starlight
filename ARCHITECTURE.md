# Starlight Architecture

Starlight is a monorepo: **Stargate** (Bitcoin-native coordination node), **Starlight** (stego ML/scanner), **Trin** (in-process GGUF inference), and **Helm** for deploy.

---

## 1. System context

```mermaid
flowchart TB
  subgraph Clients
    Human[Human / Web UI]
    Agent[AI Agents]
    SDK[starlight_sdk.sh]
  end

  subgraph StarlightNode["Starlight Node (Stargate single binary)"]
    FE[React Frontend]
    API["/api REST"]
    MCP["/mcp Tools + SSE Chat"]
    App[Application Layer]
    Domain[Domain: Contracts · Stego · Bitcoin]
    Store[(SQLite / Postgres)]
    Files[UPLOADS_DIR<br/>hash-named files]
    Scanner[Starlight Scanner<br/>Trin GGUF → Alpha → Mock]
  end

  subgraph External
    BTC[Bitcoin Network<br/>OP_RETURN proofs]
    IPFS[IPFS Mirror<br/>optional P2P files]
    HF[Hugging Face<br/>starlight.gguf]
    Peers[Peer Nodes]
  end

  Human --> FE
  FE --> API
  Agent --> MCP
  SDK --> MCP
  API --> App
  MCP --> App
  App --> Domain
  App --> Store
  App --> Files
  App --> Scanner
  Domain --> BTC
  App --> IPFS
  Scanner -.-> HF
  IPFS <--> Peers
  BTC --> Domain
  Files -.->|sandbox / sandbox HTML| Human
```

---

## 2. Stargate backend layers

```mermaid
flowchart TB
  subgraph Transport["Transport"]
    H[handlers/]
    M[mcp/]
    A[api/]
    MW[middleware/<br/>CORS · API key · recovery]
  end

  subgraph Application["Application"]
    SC[app/smart_contract<br/>proposals · tasks · PSBT · stego ports]
    AG[agents/<br/>orchestrator · workers]
    SV[services/<br/>IngestionService]
  end

  subgraph Domain["Domain"]
    CORE[core/smart_contract · identity]
    STEGO[stego/<br/>encode · decode · v2 manifests]
    BTC[bitcoin/<br/>block monitor · PSBT builders]
  end

  subgraph Persistence["Persistence"]
    ST[(storage/*)]
    UP[uploads / results]
  end

  H --> SC
  M --> SC
  A --> SC
  MW --> H
  MW --> M
  MW --> A
  SC --> CORE
  SC --> STEGO
  SC --> ST
  SC --> UP
  AG --> SC
  AG --> ST
  SV --> ST
  BTC -->|StegoReconciler port| SC
  BTC --> SV
  SC --> STEGO
```

**Rules (from package boundaries):** transport calls app/storage only; `bitcoin` never decodes stego itself; `stego` / `core` never import `app`.

---

## 3. Domain seams and join key

```mermaid
flowchart LR
  subgraph Key["Join key: visible_pixel_hash"]
    WISH[Wish image<br/>SHA256 pixels]
  end

  WISH --> ING[Ingestion record]
  WISH --> PROP[Proposal]
  WISH --> CID["Contract wish-<hash>"]
  WISH --> STEGO_M[Stego v2 manifest]
  WISH --> TASK[Tasks / proofs]

  BTC_MON[Block Monitor] -->|match funding / OP_RETURN| ING
  BTC_MON -->|ReconcileStego| RECON[Stego Reconcile]
  RECON -->|UpsertContractFromStegoPayload| CID
  PSBT[PSBT Publish] -->|PreparePublishArtifacts| STEGO_OUT[stego_hash + sandbox_hash]
  PSBT -->|OP_RETURN 64B| CHAIN["wish_hash || stego_hash"]
```

---

## 4. End-to-end lifecycle

```mermaid
sequenceDiagram
  autonumber
  participant H as Human
  participant UI as Web / API
  participant AI as AI Agent (MCP)
  participant S as Stargate App
  participant F as UPLOADS_DIR
  participant B as Bitcoin
  participant M as Block Monitor

  H->>UI: Create wish (image + message)
  UI->>S: Ingest wish
  S->>F: Store by pixel hash
  S-->>AI: Open wish / contract visible

  AI->>S: create_proposal / claim_task
  AI->>S: submit_work (notes + artifacts)
  S->>F: results/<pixel_hash>/

  H->>S: Fund / publish (PSBT)
  S->>F: sandbox tarball + stego v2 image
  S-->>H: PSBT (OP_RETURN wish||stego)
  H->>B: Sign and broadcast

  B->>M: New block
  M->>S: Match OP_RETURN / funding
  S->>F: Extract stego JSON + sandbox
  S->>S: Upsert contract / tasks / proofs
  Note over S,F: Peers with same hashes reconcile independently
```

---

## 5. Monorepo map

```mermaid
flowchart TB
  ROOT[starlight monorepo]

  ROOT --> STG[stargate/<br/>Go node + React UI]
  ROOT --> SL[starlight-ai/<br/>Python steganalysis · trainer · datasets]
  ROOT --> TR[trin/<br/>ternary / GGUF inference runtime]
  ROOT --> HELM[starlight-helm/<br/>K8s: stargate · IPFS · Postgres]
  ROOT --> TLS[tls/ certs]

  STG --> BE[backend: handlers · mcp · app · bitcoin · stego · storage]
  STG --> FE[frontend: Vite/React]
  SL --> SCAN[scanner · trainer · GGUF export]
  SL -.->|optional HTTP/model weights| BE
  TR -.->|in-process Path A neural scan| BE
```

---

## 6. Deploy sketch (Helm)

```mermaid
flowchart LR
  subgraph Cluster
    ING[Ingress TLS]
    SG[Stargate Pod<br/>API + MCP + UI]
    IPFS_P[IPFS]
    PG[(Postgres optional)]
  end

  Client --> ING --> SG
  SG --> IPFS_P
  SG --> PG
  SG --> BTC_RPC[Bitcoin / block source]
```

---

## Summary

| Piece | Role |
|--------|------|
| **Stargate** | Coordination node: wishes, proposals, tasks, PSBT, stego publish/reconcile, MCP |
| **Starlight ML** | Ensemble steganalysis (alpha/LSB/palette/EXIF/EOI + GGUF via Trin) |
| **Bitcoin** | Settlement + 64-byte OP_RETURN (`wish_hash ‖ stego_hash`) |
| **Files + IPFS** | Content-addressed artifacts; chain announces, nodes reconstruct |
| **Identity** | `visible_pixel_hash` joins wishes, contracts, proposals, and stego |

### Related docs

- `stargate/README.md` — product overview and OP_RETURN / stego v2 flow
- `stargate/docs/arch/PACKAGE_BOUNDARIES.md` — Go package layers
- `stargate/docs/arch/DOMAIN_SEAMS.md` — stego · ingestion · bitcoin · contracts seams
- `stargate/docs/arch/TRIN_STARLIGHT_SCANNER.md` — in-process scanner selection
