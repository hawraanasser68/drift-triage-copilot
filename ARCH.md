# Architecture

## System Diagram

```
                        ┌─────────────────────────────────────────┐
                        │           Model Platform                 │
                        │                                         │
  [CSV Data] ──────► [train.py] ──► [MLflow Registry]            │
                                          │                       │
                                    pipeline.pkl                  │
                                    schema.json                   │
                                    model_card.json               │
                                    reference_stats.json          │
                                          │                       │
                                          ▼                       │
                              [FastAPI Model Service :8000]       │
                              ├── POST /predict                   │
                              ├── GET  /drift-report              │
                              └── POST /promote/:version          │
                                          │                       │
                              [Drift Engine: PSI + chi²]          │
                                          │ severity changed      │
                                          ▼                       │
                              POST /drift-event ──────────────────┼───►
                        └─────────────────────────────────────────┘    │
                                                                        │
                        ┌───────────────────────────────────────────────┘
                        │           Triage Agent
                        │
                        │   [LangGraph Supervisor]
                        │         │
                        │    ┌────┴──────────────────┐
                        │    ▼                       ▼
                        │ [Triage]              [Action] ◄── HIL interrupt
                        │    │                      │            │
                        │    └──────────┬───────────┘           │
                        │               ▼                        │
                        │           [Comms]              [Streamlit Dashboard]
                        │               │                        │
                        │          [Postgres]◄───────────────────┘
                        │        (checkpoints +              HIL Inbox
                        │        investigations)
                        │               │
                        │    [Redis Queue] ──► [Worker]
                        │    queue:jobs            ├── replay_test
                        │    queue:dlq             ├── retrain ──► MLflow
                        │                          └── rollback ──► MLflow
                        └──────────────────────────────────────────────
```

## Services

| Service        | Port | Technology          | Role                                     |
|----------------|------|---------------------|------------------------------------------|
| model-service  | 8000 | FastAPI             | Serve predictions, compute drift, gate promotion |
| mlflow         | 5000 | MLflow              | Model registry and artifact store        |
| agent          | 8001 | FastAPI + LangGraph | Receive drift webhooks, run investigations |
| worker         | —    | Python              | Execute slow jobs from Redis queue       |
| dashboard      | 8501 | Streamlit           | Registry, investigations, HIL inbox      |
| postgres       | 5432 | PostgreSQL 16       | LangGraph checkpoints, predictions, investigations |
| redis          | 6379 | Redis 7             | Job queue and dead-letter queue          |

## Data Flow

1. `train.py` trains a GradientBoosting pipeline, tunes the operating threshold on the validation set (recall ≥ 0.75), and registers four artifacts to MLflow: `pipeline.pkl`, `schema.json`, `model_card.json`, `reference_stats.json`.

2. `model-service` loads the Production model from MLflow at startup. Every `POST /predict` call logs the input and probability to Postgres.

3. `GET /drift-report` reads the last 500 predictions, computes PSI (numerics) and chi² (categoricals) against the training reference distribution, and emits `POST /drift-event` to the agent if severity changes.

4. The agent receives the drift event, starts a LangGraph investigation: triage → (HIL if Production-touching) → action → comms. State is checkpointed in Postgres after every node.

5. If the action requires human approval, the graph pauses at the `action` node. The dashboard shows the pending approval in the HIL inbox. The human clicks Approve or Reject, which calls `POST /approve/:id` on the agent.

6. After approval, the action sub-agent enqueues a job to Redis with an idempotency key. The worker picks it up, executes it (replay / retrain / rollback), and marks the key as processed.

## HTTP Contract (Platform ↔ Agent)

**Platform → Agent** (drift alert):
```json
POST /drift-event
{
  "severity": "critical",
  "window_size": 500,
  "output_drift": 0.31
}
```

**Agent → Platform** (after human approves a promotion):
```json
POST /promote/{version}
→ 200 { "version": 2, "stage": "Production", "message": "..." }
```

Schema changes to this contract are breaking and must be versioned.
