# LegacyLens — Product Requirements Document

## Building RAG Systems for Legacy Enterprise Codebases

| Field | Value |
|-------|-------|
| Track | G4 |
| Version | 1.0 |
| Status | Final |

---

## 1. Product Overview

### 1.1 Problem Statement

Enterprise systems running on COBOL, Fortran, and other legacy languages power critical infrastructure — banking transactions, insurance claims, government services, and scientific computing. These codebases contain decades of accumulated business logic, yet few engineers possess the domain knowledge to understand, navigate, or maintain them.

The primary barrier is **discoverability**: developers cannot efficiently locate specific logic, understand data flows, identify dependencies, or extract business rules from codebases spanning hundreds of thousands of lines across hundreds of files.

### 1.2 Product Vision

**LegacyLens** is a RAG-powered (Retrieval-Augmented Generation) system that makes large legacy codebases queryable and understandable through natural language. Developers can ask questions about the codebase in plain English and receive accurate, cited answers with direct references to the source code.

### 1.3 Target Users

- Software engineers tasked with maintaining or migrating legacy systems
- Technical leads conducting codebase audits or impact analysis
- Developers onboarding to unfamiliar enterprise codebases
- Teams modernizing COBOL or Fortran systems to modern languages

### 1.4 Target Codebase

The system must be built against one primary legacy codebase meeting the following minimum requirements:

| Project | Language | Description | Minimum Size |
|---------|----------|-------------|--------------|
| OpenCOBOL Contrib | COBOL | Sample COBOL programs and utilities | 10,000+ LOC, 50+ files |
| GnuCOBOL | COBOL | Open source COBOL compiler | 10,000+ LOC, 50+ files |
| GNU Fortran (gfortran) | Fortran | Fortran compiler in GCC | 10,000+ LOC, 50+ files |
| LAPACK | Fortran | Linear algebra library | 10,000+ LOC, 50+ files |
| BLAS | Fortran | Basic linear algebra subprograms | 10,000+ LOC, 50+ files |
| Custom Proposal | Any legacy | Requires prior approval | 10,000+ LOC, 50+ files |

---

## 2. MVP Requirements

The following requirements constitute the **hard gate for MVP**. All items are mandatory. A simple RAG pipeline with accurate retrieval is preferable to a complex system with poor retrieval quality.

### 2.1 Codebase Ingestion

- [ ] Ingest at least one complete legacy codebase (COBOL, Fortran, or approved alternative)
  - All files in the target repository must be discovered and processed
  - Minimum codebase size: 10,000+ lines of code across 50+ files
  - File types to include: `.cbl`, `.cob`, `.cpy` for COBOL; `.f`, `.f90`, `.for` for Fortran

### 2.2 Syntax-Aware Chunking

- [ ] Chunk all code files using syntax-aware splitting — not fixed-size character splitting
  - COBOL: split at PARAGRAPH and SECTION boundaries
  - Fallback to fixed-size (500 tokens) with overlap for unstructured sections
  - Each chunk must preserve its structural context (parent division, section name)

### 2.3 Embedding Generation

- [ ] Generate vector embeddings for every chunk in the codebase
  - Use the same embedding model for both ingestion and query phases
  - Embedding dimensions must remain consistent across the entire pipeline
  - Batch processing required — embeddings must not be generated one-by-one

### 2.4 Vector Database Storage

- [ ] Store all embeddings in a vector database with associated metadata
  - Metadata must include: file path, line range, chunk type, parent section
  - Storage must be persistent — embeddings must survive application restarts
  - All chunks must be retrievable by similarity search

### 2.5 Semantic Search

- [ ] Implement semantic similarity search across the full indexed codebase
  - Return top-k most similar chunks for any natural language query
  - Search must operate across all indexed files, not a subset

### 2.6 Natural Language Query Interface

- [ ] Provide a natural language query interface — CLI or web-based
  - Accept free-text questions about the codebase
  - Interface must be functional and usable for evaluation

### 2.7 File and Line References

- [ ] Return relevant code snippets with accurate file path and line number references for every result
  - File path must be the exact relative path within the repository
  - Line numbers must correspond to the original source file

### 2.8 Answer Generation

- [ ] Generate a natural language answer using the retrieved code context
  - LLM must synthesize retrieved chunks into a coherent explanation
  - Answer must reference the specific code that supports the response

### 2.9 Public Deployment

- [ ] Deploy the application to a publicly accessible URL before MVP submission
  - Application must be reachable from any browser without authentication
  - Deployment environment must support persistent vector storage
  - All API keys must be managed via environment variables — never hardcoded
  - A setup guide must be included in the GitHub repository

---

## 3. Ingestion Pipeline Requirements

The ingestion pipeline is the foundation of the system. All downstream retrieval quality depends on the correctness and completeness of ingestion. This section must be completed before any retrieval or interface work begins.

### 3.1 File Discovery

- Recursively scan the entire target codebase directory
- Filter files by relevant extension: `.cbl`, `.cob`, `.cpy` (COBOL); `.f`, `.f90`, `.for` (Fortran)
- Track total file count and lines of code to confirm minimum coverage requirements are met
- Log all discovered files with their relative paths for audit purposes

### 3.2 Preprocessing

- Handle encoding issues including EBCDIC legacy artifacts and non-UTF-8 characters
- Normalize whitespace — remove redundant blank lines, standardize line endings
- **COBOL-specific:** Strip columns 1-6 (sequence numbers) and columns 73-80 (identification area) per the COBOL 72-column fixed-format rule before embedding
- Extract inline comments separately from executable code — preserve for metadata but weight lower during embedding to prevent comment-bloat
- Identify and flag dead code sections (commented-out blocks) for reduced embedding weight

### 3.3 Chunking Strategy

Legacy code requires specialized chunking. The following strategies must be implemented in priority order:

| Strategy | Use Case | Priority |
|----------|----------|----------|
| Paragraph-level (COBOL) | Split at COBOL PARAGRAPH boundaries — primary strategy | Primary |
| Section-level (COBOL) | Split at DIVISION/SECTION boundaries for large sections | Secondary |
| Function/Subroutine-level | Each function or subroutine as a chunk for Fortran | Primary (Fortran) |
| Fixed-size + overlap | Fallback for unstructured or non-standard sections (500 tokens) | Fallback |
| Hierarchical | Multiple granularities: file → section → paragraph | Enhancement |

- Each chunk must include its parent SECTION header for surrounding context
- Chunk boundaries must align with logical code units — never split mid-paragraph or mid-function
- Overlap strategy: include last paragraph of preceding section when splitting at section boundaries

### 3.4 Metadata Extraction

Every chunk must be stored with the following metadata schema:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| file_path | str | Relative path to source file within repository | Yes |
| line_range | [int, int] | Start and end line numbers in the original source file | Yes |
| type | str | Division type: PROCEDURE or DATA | Yes |
| parent_section | str | Name of the parent SECTION or DIVISION | Yes |
| paragraph_name | str | Name of the COBOL PARAGRAPH or function name | Yes |
| dependencies | list[str] | List of CALL, COPY, and USING references extracted from chunk | Yes |

Recommended Security & Reliability Updates

| Field | Type | Purpose for Final Submission |
|-------|------|------------------------------|
file_hash | str | Stores a SHA-256 hash to ensure the answer matches the exact version of the code. |
| security_flag | bool | Indicates if PII or sensitive literals were detected/redacted during preprocessing. |

- Dependencies must be populated by the Reference Scraper (see Section 3.5)
- All metadata fields are mandatory — chunks with missing metadata must not be stored

### 3.5 Dependency Reference Scraper

A static analysis component must scan each file for COBOL dependency keywords during ingestion. This operates independently of semantic search and provides structural dependency data:

- **CALL** — identifies external program or module invocations
- **COPY** — identifies copybook (`.cpy`) file inclusions
- **USING** — identifies parameter passing between modules

Extracted dependencies must be stored in the metadata schema under the `dependencies` field and must be queryable to answer questions such as *"What are the dependencies of MODULE-X?"*

### 3.6 Embedding Generation

- Generate vector embeddings for every chunk after preprocessing and chunking
- Use a code-optimized embedding model — general-purpose text models are insufficient for code retrieval tasks
- Batch all embedding API calls — process chunks in batches, not individually
- Cache all generated embeddings to persistent storage — re-embedding must not be required on application restart
- **Ingestion throughput requirement:** 10,000+ lines of code must be fully ingested in under 5 minutes

### 3.7 Vector Storage

- Insert all chunks with their embeddings and metadata into the selected vector database
- Verify successful insertion — log chunk count and confirm all files are represented
- Storage must support metadata filtering by `file_path`, `type` (PROCEDURE/DATA), and `paragraph_name`
- Index must support hybrid search (vector similarity + keyword) for fallback retrieval

---

## 4. Chunking Strategy Requirements

Chunking quality is the single most important factor in retrieval precision. This section defines the detailed requirements for COBOL-specific chunking that differentiates LegacyLens from generic RAG implementations.

### 4.1 COBOL Structure Awareness

- The chunker must understand and respect COBOL's four DIVISION structure: IDENTIFICATION, ENVIRONMENT, DATA, PROCEDURE
- PROCEDURE DIVISION chunks must be kept separate from DATA DIVISION chunks — they serve different retrieval purposes
- DATA DIVISION chunks provide variable definitions; PROCEDURE DIVISION chunks provide executable logic
- Cross-referencing between DATA and PROCEDURE divisions is handled at retrieval time, not at chunking time

### 4.2 Boundary Detection

- **PARAGRAPH boundaries:** detected by lines ending with a period following a paragraph header
- **SECTION boundaries:** detected by SECTION keyword in columns 8-11
- **DIVISION boundaries:** detected by DIVISION keyword
- **Copybook boundaries:** `.cpy` files are chunked independently and linked to parent programs via COPY references

### 4.3 Context Preservation

- Every chunk must carry the name of its parent PARAGRAPH, SECTION, and DIVISION in metadata
- When a chunk is retrieved, the system must be able to reconstruct its full structural path (DIVISION > SECTION > PARAGRAPH)
- **Copybook context injection:** if a retrieved chunk contains a COPY statement, the referenced `.cpy` file content must be appended to the LLM context to provide variable definitions

### 4.4 Chunking Validation

- After chunking, verify that 100% of source lines are represented across chunks — no lines may be dropped
- Verify that no chunk exceeds the embedding model's maximum token limit
- Log chunk size distribution (min, max, average) for evaluation purposes

---

## 5. Embedding & Vector Database Requirements

### 5.1 Embedding Model Requirements

The embedding model must meet the following criteria:

- Must be **code-optimized** — trained or fine-tuned on source code, not general text corpora
- Minimum 1024 dimensions for sufficient semantic resolution on code
- Must support batch processing for efficient ingestion
- The exact same model instance and version must be used for both ingestion and query embedding

| Model | Dimensions | Type | Suitability |
|-------|-------------|------|-------------|
| Voyage Code 2 | 1536 | API | Optimized for code — recommended |
| OpenAI text-embedding-3-small | 1536 | API | General purpose — acceptable |
| OpenAI text-embedding-3-large | 3072 | API | Higher quality, higher cost |
| sentence-transformers | varies | Local | Free, no code-specific variant |

### 5.2 Vector Database Requirements

The vector database must meet the following criteria:

- Must support metadata storage and filtering alongside vector embeddings
- Must support top-k similarity search with configurable k value
- Must support hybrid search (vector + keyword/BM25) for fallback scenarios
- Must provide persistent storage — data must survive application restarts and redeployment
- Must be deployable to the chosen hosting environment without manual infrastructure provisioning

| Database | Hosting | Metadata Support | Hybrid Search | Suitability |
|----------|---------|-------------------|----------------|-------------|
| ChromaDB | Embedded/self-host | Yes | Limited | Recommended for prototype scale |
| Pinecone | Managed cloud | Yes | Yes | Recommended for 10K+ users |
| Qdrant | Self-host or cloud | Yes | Yes | Strong filtering capability |
| Weaviate | Self-host or cloud | Yes | Yes | GraphQL API |
| pgvector | PostgreSQL ext. | Yes | Yes | Requires existing Postgres |

---

## 6. Retrieval Pipeline Requirements

The retrieval pipeline transforms a natural language query into a set of relevant code chunks. All components must operate end-to-end within the 3-second latency target.

### 6.1 Query Processing

- Accept free-text natural language queries about the codebase
- Normalize query text before embedding — handle COBOL-specific terminology and abbreviations
- Support query expansion for ambiguous terms — broaden query scope when initial results are sparse
- Extract intent and entities from the query to guide metadata filtering (e.g., identify if query is about a specific paragraph name, file, or pattern)

### 6.2 Query Embedding

- Convert the processed query to a vector using the same embedding model used during ingestion
- Embedding dimensions must match ingestion embeddings exactly
- Query embedding must be generated at request time — not cached across different queries

### 6.3 Similarity Search

- Execute top-k similarity search against the full vector index — default k=5
- Apply metadata filters where extractable from query (e.g., filter by PROCEDURE type for logic queries, DATA type for variable queries)
- **Fallback:** if similarity search returns fewer than 3 relevant results, execute BM25 keyword search as a secondary retrieval pass

### 6.4 Re-ranking

- Re-rank retrieved chunks by relevance score after initial similarity search
- Prioritize chunks whose `paragraph_name` or `parent_section` directly matches terms in the query
- Deprioritize chunks from DATA DIVISION when query is clearly asking about executable logic

### 6.5 Context Assembly

Before passing retrieved chunks to the LLM, the system must assemble full context:

- Include the retrieved paragraph chunk as the primary context
- Append the parent SECTION content for surrounding context
- **Cross-reference DATA DIVISION:** if retrieved chunk references variables (e.g., `WS-TRANS-AMT`), look up and append the variable definition from the DATA DIVISION
- **Copybook injection:** if retrieved chunk contains a COPY statement, fetch and append the referenced `.cpy` file content
- **Dependency context:** append the dependency list from metadata to inform the LLM of module relationships
- Respect the LLM context window limit — truncate appended context from least-relevant to most-relevant if limit is approached

### 6.6 Failure Handling

- If similarity search returns no results above minimum threshold: fall back to BM25 keyword search
- If BM25 also returns no results: return a structured "no results found" response with suggested query reformulations
- If a referenced copybook file cannot be located: log the missing reference and continue without it
- If LLM API call fails: return retrieved chunks directly without generated explanation
- All API failures must implement exponential backoff with a maximum of 3 retries

---

## 7. Code Understanding Features

A **minimum of 4 features** from the list below must be implemented. Features are listed in recommended implementation priority order based on retrieval coverage of the 6 mandatory test scenarios.

| Feature | Description | Test Scenario Addressed | Priority |
|---------|-------------|-------------------------|----------|
| Code Explanation | Explain what a COBOL paragraph or section does in plain English, including its business purpose | Explain what CALCULATE-INTEREST does | Required |
| Dependency Mapping | Show what programs, copybooks, and modules a given paragraph calls or depends on, using static analysis data from the Reference Scraper | What are the dependencies of MODULE-X? | Required |
| Business Logic Extract | Identify and explain business rules embedded in PROCEDURE DIVISION logic, including conditions, thresholds, and data transformations | Find all file I/O operations | Required |
| Documentation Generation | Auto-generate inline documentation for undocumented paragraphs and sections in a structured format | General documentation coverage | Required |
| Pattern Detection | Find similar code patterns across multiple files in the codebase | Show error handling patterns | Optional |
| Impact Analysis | Identify what other modules or paragraphs would be affected if a specific chunk is changed | General analysis | Optional |
| Translation Hints | Suggest equivalent constructs in modern languages (Java, Python, Go) for COBOL logic | General modernization | Optional |
| Bug Pattern Search | Identify potential issues based on known COBOL anti-patterns and legacy coding issues | General quality | Optional |

### 7.1 Code Explanation — Requirements

- Must explain the purpose and behavior of any COBOL PARAGRAPH in plain English
- Explanation must reference the specific variables and conditions present in the retrieved code
- Must not hallucinate variable names, operations, or logic not present in the retrieved context
- Must include the source paragraph name and file:line reference in the response

### 7.2 Dependency Mapping — Requirements

- Must surface all CALL, COPY, and USING references for a queried module
- Must use static analysis data from the Reference Scraper, not only semantic search
- Must indicate whether each dependency is an internal module or an external copybook
- Must be answerable for any module present in the indexed codebase

### 7.3 Business Logic Extract — Requirements

- Must identify business rules: conditions, thresholds, data transformations, and control flow decisions
- Must distinguish between infrastructure code (file I/O, memory management) and business logic
- Must present extracted rules in structured, human-readable format

### 7.4 Documentation Generation — Requirements

- Must generate documentation for any paragraph or section that lacks inline comments
- Generated documentation must follow a consistent structure: purpose, inputs, outputs, side effects
- Must not generate documentation that contradicts the actual code logic

---

## 8. Answer Generation Requirements

### 8.1 LLM Selection

- Must use a capable LLM for synthesis — GPT-4, Claude, or equivalent open-source model (Llama, Mistral)
- Model must be capable of reasoning about code structure and business logic
- Model selection must be documented with rationale in the RAG Architecture Document

### 8.2 Prompt Design

- Prompt must include all retrieved context chunks assembled in Section 6.5
- Prompt must explicitly instruct the LLM to only reference variables, logic, and constructs present in the provided context
- Prompt must instruct the LLM to always include file path and line number citations in the response
- Prompt must instruct the LLM not to suggest modern syntax equivalents unless the Translation Hints feature is explicitly invoked

### 8.3 Response Format

- Every response must include: natural language explanation, relevant code snippet, file path, and line number reference
- File and line references must be formatted as clickable deep links to the exact line in the GitHub repository
- Confidence or relevance score must be included for each retrieved chunk
- Streaming responses must be supported for improved perceived latency in the UI

### 8.4 Hallucination Prevention

- LLM must be constrained to only reference content present in the retrieved context
- Any variable name, function name, or business rule not present in retrieved chunks must not appear in the generated answer
- If retrieved context is insufficient to answer the query, the system must state this explicitly rather than generating a speculative answer

---

## 9. Query Interface Requirements

The query interface is the primary user-facing component of LegacyLens. It must be functional, usable, and demonstrable within the evaluation environment.

### 9.1 Input

- Natural language text input field for submitting questions about the codebase
- Input must accept free-form questions of any length
- Submit action available via button click and keyboard shortcut

### 9.2 Results Display

- Display retrieved code snippets with syntax highlighting appropriate to the source language (COBOL or Fortran)
- Show file path and line number for each retrieved result
- Show confidence or relevance score for each retrieved chunk
- Display generated natural language explanation or answer from the LLM
- Provide clickable deep links from each result directly to the exact line in the GitHub repository

### 9.3 Drill-Down

- Allow the user to expand any result to view the full file context surrounding the retrieved chunk
- Full file view must highlight the specific retrieved lines within the broader file

### 9.4 Interface Standards

- Interface must be publicly accessible without authentication
- Interface must be responsive and functional on desktop browsers
- Query latency must be visually communicated — loading state must be shown during retrieval and generation

---

## 10. Performance Requirements

All performance targets below are **mandatory**. Results must be measured and documented in the final submission.

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Query latency | < 3 seconds end-to-end | Measured from query submission to full response rendered in UI |
| Retrieval precision | > 70% relevant chunks in top-5 | Evaluated against 20-query benchmark suite anchored by 6 mandatory test scenarios |
| Codebase coverage | 100% of files indexed | Verified by comparing indexed file count to total discovered file count |
| Ingestion throughput | 10,000+ LOC ingested in < 5 minutes | Measured from ingestion start to ChromaDB storage completion |
| Answer accuracy | Correct file/line references in every response | Manually verified against source repository for each benchmark query |

### 10.1 Evaluation Methodology

Performance must be measured using a structured iterative evaluation loop:

1. **Baseline:** BM25 keyword search only — establishes minimum retrieval performance floor
2. **Stage 2:** Fixed-size chunking (500 tokens) + selected embedding model — measures embedding gain over keyword baseline
3. **Stage 3:** Syntax-aware paragraph chunking + selected embedding model — quantifies Context Gain from COBOL-specific splitting
4. **Stage 4:** Full pipeline with copybook injection and dependency metadata — measures multi-file reasoning uplift

Each stage must be documented with precision metrics to demonstrate the contribution of each architectural decision.


### 10.2 Benchmark Suite

The evaluation ground truth dataset must consist of **20 targeted Q&A pairs**, anchored by the following 6 mandatory test scenarios:

1. "Where is the main entry point of this program?"
2. "What functions modify the CUSTOMER-RECORD?"
3. "Explain what the CALCULATE-INTEREST paragraph does"
4. "Find all file I/O operations"
5. "What are the dependencies of MODULE-X?"
6. "Show me error handling patterns in this codebase"

The remaining 14 queries must extend coverage to edge cases including dead code references, copybook variable lookups, and cross-file dependency chains.

10.1.1 Advanced Testing Tiers * Edge-Case Stress Testing: We will test the chunker against "spaghetti code" modules containing non-standard GOTO jumps and unstructured paragraphs to verify context integrity.
* Reference Integrity Check: Every 10th query during development will undergo manual verification where generated line numbers are cross-referenced against the raw GitHub source for 100% accuracy.
* Cold-Start & Missing Data Simulation: We will intentionally drop specific .cpy (copybook) files from the index to test the system’s ability to gracefully degrade and inform the user of missing context.

---

## 11. AI Cost Analysis Requirements

A cost analysis is a **required submission deliverable**. It must cover both actual development spend and projected production costs.

### 11.1 Development & Testing Costs

Track and report all actual API spend incurred during development:

- Embedding API costs: total tokens embedded across all ingestion runs
- LLM API costs: total tokens consumed for answer generation during development and testing
- Vector database costs: any hosting or usage fees incurred
- Total development spend breakdown by component

### 11.2 Production Cost Projections

Estimate monthly costs at each of the following user scales. All assumptions must be explicitly stated.

| Scale | Est. Monthly Cost | Primary Cost Driver |
|-------|-------------------|----------------------|
| 100 users | $___/month | LLM answer generation |
| 1,000 users | $___/month | LLM answer generation |
| 10,000 users | $___/month | LLM + Vector DB hosting upgrade |
| 100,000 users | $___/month | LLM at scale + managed vector DB |

### 11.3 Required Assumptions

- Queries per user per day
- Average input tokens per query (context + retrieved chunks)
- Average output tokens per response
- Embedding costs for incremental codebase additions (new code per month)
- Vector database storage costs at each scale
- Caching rate assumption — percentage of queries served from cache vs fresh retrieval

---

## 12. RAG Architecture Documentation Requirements

A **1-2 page RAG Architecture Document** is a required submission deliverable. It must cover all sections below using **actual results from the built system** — not pre-build projections.

| Section | Required Content |
|---------|------------------|
| Vector DB Selection | Why this database was chosen, what alternatives were considered, specific tradeoffs that informed the decision |
| Embedding Strategy | Model chosen, why it is appropriate for code understanding, dimension size rationale |
| Chunking Approach | How legacy code was split, boundary detection logic, COBOL-specific decisions made |
| Retrieval Pipeline | Full query flow, re-ranking approach, context assembly logic |
| Failure Modes | What does not work well, edge cases discovered during testing, known limitations |
| Performance Results | Actual measured latency, actual retrieval precision metrics, representative query examples with results |

12.3 Security & Data Privacy Standards * PII Redaction: The ingestion pipeline includes a regex-based masking layer to redact sensitive hardcoded literals (e.g., IP addresses, passwords) from the DATA DIVISION before they are sent to external embedding APIs.
* Zero-Retention Policy: We utilize Enterprise-tier API configurations for Voyage and OpenAI to ensure that no code snippets are stored or used for further model training.
* Transport Security: All communication between the ingestion pipeline, the vector database, and the LLM is encrypted via TLS 1.3.

---

## 13. Submission Deliverables

All deliverables below are **required for final submission**. Missing any deliverable constitutes an incomplete submission.

| Deliverable | Requirements |
|-------------|---------------|
| GitHub Repository | Public repository with: complete source code, setup/installation guide, architecture overview, and link to deployed application |
| Demo Video (3-5 min) | Screen recording demonstrating: natural language query submission, retrieval results with file/line references, answer generation, at least 2 code understanding features |
| Pre-Search Document | Completed Phase 1-3 Pre-Search checklist saved as a reference document |
| RAG Architecture Doc | 1-2 page document covering all 6 sections defined in Section 12 of this PRD |
| AI Cost Analysis | Development spend breakdown + production cost projections for 100, 1K, 10K, and 100K users with stated assumptions |
| Deployed Application | Publicly accessible query interface at a stable URL — must be live at time of submission |
| Social Post | Post on X or LinkedIn including: project description, key features, demo screenshot or video, tag @GauntletAI |

---

## 14. Interview Preparation Requirements

Technical and behavioral interviews are **required for Austin admission**. The following topics must be prepared:

### 14.1 Technical Topics

- Rationale for vector database selection — specific tradeoffs considered
- Chunking strategy tradeoffs — why paragraph-level over fixed-size for COBOL
- Embedding model selection rationale — why code-optimized over general-purpose
- Retrieval failure handling — what happens when no relevant results are found
- Performance optimization decisions — what was measured, what was changed

### 14.2 Mindset & Growth Topics

- How ambiguity in the project requirements was approached
- Instances where the approach was pivoted based on failure or poor results
- What was learned about the technology and about personal working style
- How pressure and uncertainty were managed during the sprint
