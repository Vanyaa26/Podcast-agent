# Controlled Agentic RAG Podcast Demo

## Core Idea

Research podcast generation should not be a black-box "PDF in, audio out" task.

The useful interaction is closer to how people already work with AI tools:

```text
User gives a paper
  -> system understands the paper
  -> proposes a plan
  -> user steers the focus
  -> system retrieves evidence
  -> generates an audio overview
  -> listener can interrupt with questions
```

This prototype explores that direction for SARAL-style paper-to-podcast generation.

## What This Prototype Is Today

This is a **controlled agentic RAG pipeline**.

It is not a fully autonomous podcast where two agents freely talk forever. The design is deliberately hybrid:

- **Deterministic spine:** fixed pipeline stages, approval gate, artifact storage, bounded tool calls.
- **Agentic layers:** planning, outline revision, evidence retrieval, script generation, and listener Q&A.
- **RAG grounding:** the paper is indexed and searched instead of relying only on a truncated prompt.

```text
PDF upload
  -> parse and chunk
  -> embed and index paper KB
  -> create paper brief
  -> create episode outline
  -> user reviews / revises / approves
  -> generate podcast beats from retrieved evidence
  -> synthesize audio + cue track
  -> user can ask voice questions during playback
```

## Why The Approval Gate Matters

The system stops before audio generation.

The user can say:

```text
Focus only on the xyz section.
Go deep on setup, removed components, task-specific result drops, and takeaway.
```

Then the planner revises the outline using the paper knowledge base, instead of just paraphrasing the user's prompt.

This matters because audio is expensive and hard to quickly inspect. The user should know the direction before the system starts speaking.

## Current Demo Flow

1. Open the podcast UI beside the AutoScientist paper.
2. Upload the paper.
3. Show the pipeline stages:
  - reading paper
  - building knowledge base
  - summarizing
  - planning episode
4. At the outline review step, ask for a specific focus:

```text
Focus on the Ablations of AUTOSCIENTISTS section.
Explain what ablation means, the setup, the four removed components,
the task-specific result drops, and the final takeaway.
```

1. Show that the system updates the outline around the requested focus.
2. Generate the podcast.
3. Play a short part.
4. Ask an interrupt question:

```text
What does ablation mean in this paper?
```

1. Show that the answer is grounded in the paper and the trace shows tool use.

## Why This Is Not Just "Retrieve Chunks And Summarize"

The next stronger version should make paper understanding explicit.

Instead of:

```text
retrieve chunks -> summarize
```

the system should move toward:

```text
PDF
  -> parse sections
  -> identify abstract, introduction, method, experiments, results, limitations
  -> extract key claims
  -> attach evidence to each claim
  -> rank claims by importance
  -> build episode plan
  -> generate podcast from selected claims and evidence
```

For research papers, importance is not random:

- **Abstract:** core claim
- **Introduction:** problem and gap
- **Contributions:** novelty
- **Method:** actual mechanism
- **Experiments:** setup and baselines
- **Results tables:** concrete evidence
- **Limitations:** caveats
- **Conclusion:** author framing

## Current System vs Future System


| Area                | Current Prototype                    | Better Next Version                                              |
| ------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| Paper parsing       | Text chunks                          | Section-aware parsing                                            |
| Paper understanding | Brief + outline                      | Claim graph with evidence                                        |
| Retrieval           | Query chunks per beat/question       | Retrieve by claim, section, and evidence type                    |
| User steering       | Revise outline with natural language | Revise selected claims and episode goals                         |
| Script generation   | Beat-by-beat RAG writer              | Generate from ranked claims + evidence                           |
| Interrupts          | Voice Q&A with paper/web tools       | Conversational splice back into episode flow                     |
| Observability       | Tool trace in UI                     | Full trace across planning, retrieval, claims, script, and audio |


## Observability And Traceability

As this becomes more agentic, observability becomes necessary.

Suggested addition: **Langfuse** or a similar tracing layer.

Track:

- user instruction
- planner prompt and output
- tool calls and retrieval queries
- retrieved chunks
- extracted claims
- evidence linked to claims
- outline changes
- script turns
- source chunk IDs per turn
- interrupt questions
- answer traces
- token cost and latency per stage

This helps answer:

```text
Why did the agent include this point?
Which paper passage supports it?
Which tool call found it?
Where did the podcast drift or become vague?
```

## Why Not Fully Agentic Immediately?

A fully autonomous two-agent podcast could be impressive, but it is also fragile:

- agents can drift away from the paper
- hallucinations sound confident in audio
- long conversations can lose structure
- debugging becomes difficult
- user has less control over what gets generated

The safer direction is:

```text
controlled pipeline + agentic decision points
```

Agents should be used where judgment helps:

- choosing what to retrieve
- revising the plan from user intent
- extracting paper claims
- deciding what evidence matters
- answering interruptions
- bridging back to the episode

The product should still control:

- stage order
- approval
- evidence boundaries
- tool budgets
- final artifact generation

## Proposed Roadmap

### Phase 1: Current Prototype

- upload PDF
- build paper KB
- plan episode
- user review and revise
- generate grounded podcast
- support voice interruptions
- show tool trace

### Phase 2: Better Paper Understanding

- section-aware parsing
- claim extraction
- evidence linking
- importance ranking
- outline from selected claims

### Phase 3: Better Interactive Experience

- conversational splice instead of isolated Q&A
- pass current topic, recent turns, and resume target
- answer the question and bridge back:

```text
"That sets us up for the results section..."
```

### Phase 4: Observability And Evaluation

- Langfuse tracing
- claim-to-evidence audit trail
- retrieval quality metrics
- hallucination checks
- cost and latency dashboard

### Phase 5: Production Integration

- use SARAL's PyMuPDF parser
- improve table/figure extraction
- switch voice stack as needed
- support longer papers and domain-specific templates

## Final Position

This prototype is not claiming that full autonomy is solved.

The claim is:

> A research podcast should be controlled, grounded, inspectable, and user-steerable. Agentic behavior should appear at the decision points where it improves understanding, while deterministic orchestration should protect reliability.

