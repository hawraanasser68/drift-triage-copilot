# Runbook

## First-time setup

```bash
git clone <your-repo-url>
cd <repo>
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Start the full stack

```bash
docker-compose up --build
```

This will:
1. Start Postgres and Redis
2. Start the MLflow tracking server
3. Run `train.py` (trainer service) — registers the model, then exits
4. Start the model service, agent, worker, and dashboard

First run takes ~5 minutes while training completes. Subsequent runs skip training if the model is already registered (trainer re-runs but exits quickly).

**Service URLs after startup:**
| Service        | URL                        |
|----------------|----------------------------|
| Model service  | http://localhost:8000/docs |
| MLflow UI      | http://localhost:5000      |
| Agent          | http://localhost:8001/docs |
| Dashboard      | http://localhost:8501      |

## Run tests locally

```bash
# Train first to generate artifacts/
MLFLOW_TRACKING_URI=sqlite:///mlflow.db python train.py

# Run all tests
pytest tests/ -v
```

## Demo: trigger drift and watch the agent respond

### 1. Send a normal prediction (baseline)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "job": "admin.", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "mon",
    "campaign": 1, "pdays": 999, "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1, "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4, "euribor3m": 4.857,
    "nr.employed": 5191.0
  }'
```

### 2. Inject drifted data (shift euribor3m and change contact type)

Send ~50 predictions with `euribor3m` shifted to `0.5` (from ~4.8) and `contact` changed to `"telephone"`. This shifts both a numeric and a categorical feature enough to trigger drift.

```bash
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "age": 40, "job": "blue-collar", "marital": "single",
      "education": "basic.9y", "default": "unknown",
      "housing": "no", "loan": "no", "contact": "telephone",
      "month": "nov", "day_of_week": "fri",
      "campaign": 5, "pdays": 999, "previous": 0,
      "poutcome": "nonexistent",
      "emp.var.rate": -3.0, "cons.price.idx": 90.0,
      "cons.conf.idx": -50.0, "euribor3m": 0.5,
      "nr.employed": 4900.0
    }' > /dev/null
done
```

### 3. Poll the drift report to trigger the webhook

```bash
curl http://localhost:8000/drift-report
```

When severity changes to `warning` or `critical`, the model service fires a webhook to the agent automatically.

### 4. Watch the agent open an investigation

```bash
curl http://localhost:8001/investigations
```

### 5. Approve the action in the dashboard

Open http://localhost:8501 → **HIL Inbox** tab → click **Approve**.

Or via API:
```bash
curl -X POST http://localhost:8001/approve/<investigation_id> \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'
```

### 6. Watch the queue dispatch the job

```bash
# Check queue depth
docker-compose exec redis redis-cli LLEN queue:jobs

# Check DLQ
docker-compose exec redis redis-cli LLEN queue:dlq
```

## Promote a model version to Production

```bash
curl -X POST http://localhost:8000/promote/2
```

The promotion gate checks: AUC ≥ 0.75, recall ≥ 0.75, dataset hash present, not already in Production.

## Rollback

Rollback is triggered through the agent (via the HIL approval flow) or directly:
```bash
curl -X POST http://localhost:8001/approve/<investigation_id> \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'
```
The worker's `rollback.py` promotes the previous version to Production.

## Teardown

```bash
docker-compose down          # stop containers, keep volumes
docker-compose down -v       # stop containers and delete all data
```
