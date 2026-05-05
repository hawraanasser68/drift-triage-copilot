# Design Decisions

## Model: GradientBoostingClassifier

**Why:** Compared four models (Logistic Regression, Random Forest, GBM, SVC) on the validation set with recall ≥ 0.75 as the hard constraint. GBM achieved the best ROC-AUC (0.795) and average precision (0.487) while meeting the recall floor — better discrimination than logistic regression and more stable than SVC at low thresholds.

**Trade-off:** Slower to train than logistic regression, but this is a one-time cost and the model is registered in MLflow. Inference latency is negligible for a single-row prediction.

## Operating Threshold: 0.06

**Why:** The highest threshold where recall ≥ 0.75 on the validation set. The business context is a phone campaign — missing a potential subscriber (false negative) costs more than calling a non-subscriber (false positive). Maximising the threshold at the recall floor keeps precision as high as possible while protecting recall.

## Drift Detection: Webhook (not polling)

**Why:** The agent subscribes to drift webhooks from the platform rather than polling on an interval. Webhooks fire immediately when severity changes, so the agent responds to real events rather than checking on a schedule that may miss fast-moving drift or waste resources when nothing changes.

**Trade-off:** If the agent is down when a webhook fires, the event is lost. Mitigation: the model service only emits on severity *change*, not on every report, so drift is still captured on the next `GET /drift-report` call.

## Supervisor Topology (not a chain)

**Why:** The brief explicitly requires a supervisor topology — three sub-agents (triage, action, comms) routed by a central supervisor node, not wired sequentially. This means the supervisor can skip nodes (e.g. skip action for "monitor" severity) and the routing logic is testable in isolation.

## LangGraph Postgres Checkpointer

**Why:** The agent must survive restarts mid-investigation. LangGraph's `PostgresSaver` persists the full graph state (including the HIL interrupt payload) to Postgres after every node. On restart, `graph.get_state(config)` restores exactly where the investigation was paused.

## HIL Interrupt: `interrupt_before=["action"]`

**Why:** The graph always checkpoints before the action node. Inside the action node, a runtime check decides whether human approval is actually required (only for Production-touching actions). This means the checkpoint is always taken, which simplifies recovery — we never need to figure out "did the interrupt happen before or after the checkpoint."

## Redis Queue: Idempotency Keys

**Why:** The worker checks a `queue:processed_keys` Redis set before executing any job. The idempotency key is `{investigation_id}-{job_type}`. This guarantees that two retries of the same retrain (e.g. after a crash mid-execution) never kick off two trainings — only the first one runs.

## Single Dockerfile for All Services

**Why:** All Python services share the same codebase and dependency set. One Dockerfile keeps the image build surface minimal and avoids dependency drift between services. The `CMD` is overridden per-service in docker-compose.

## `unknown` Treated as a Real Category

**Why:** The UCI dataset uses `unknown` to represent genuinely unknown information (e.g. a client who declined to answer). It is informative — clients who answer `unknown` may behave differently from those who answer `yes` or `no`. Replacing it with a mode imputation would destroy that signal. OneHotEncoder creates a dedicated `unknown` column.

## `duration` Dropped

**Why:** `duration` is the length of the phone call. It is only known after the call ends — it would not be available at the time of scoring a new lead. Including it would cause data leakage and inflate all metrics artificially.
