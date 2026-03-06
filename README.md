# AL-01

Deterministic digital organism runtime with Firestore + SQLite persistence and a hardened HTTP API.

**v1.2.0** — persistence, reliability, and API hardening release.

## Features

- Deterministic tick-based metabolism (pulse / reflect / persist)
- **SQLite** local database for structured interactions + growth snapshots
- **Firestore** cloud persistence with automatic local JSON fallback
- **FastAPI** HTTP server with request validation and error handling
- **API key authentication** via `X-API-Key` header (loaded from `.env`)
- Memory survives restarts — interactions + metrics are recovered automatically
- Configurable cadence via `MetabolismConfig`

## Project Structure

```text
al01/
  __init__.py          # Package exports
  database.py          # SQLite persistence layer
  memory_manager.py    # Firestore/JSON + SQLite integration
  organism.py          # Core organism + state machine
  api.py               # FastAPI HTTP server
  __main__.py          # Entrypoint (python -m al01)
  identity.json        # Name, description, version
tests/
  test_v12.py          # Test suite (13 tests)
.env                   # API key config (not committed)
.env.example           # Template
```

Primary exports: `Organism`, `MetabolismConfig`, `MemoryManager`, `Database`, `OrganismState`, `VERSION`

## Quick Start

```bash
# 1. Install dependencies
pip install fastapi uvicorn firebase-admin

# 2. Configure API key
cp .env.example .env
# Edit .env → set AL01_API_KEY=your-secret-key

# 3. Run
python -m al01
```

The organism will:
1. Load `.env` and configure logging
2. Initialize SQLite database (`al01.db`)
3. Load prior state + recover interaction count from DB
4. Boot (increment evolution, persist)
5. Start API server on port 8000
6. Enter heartbeat loop (Ctrl+C to stop)

## API Endpoints

All endpoints except `/health` require the `X-API-Key` header when `AL01_API_KEY` is set.

### `GET /health`
No auth required. Lightweight health check.
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "version": "1.2", "uptime_seconds": 42.1, "timestamp": "..."}
```

### `GET /identity`
Returns identity.json contents + runtime version.
```bash
curl -H "X-API-Key: your-key" http://localhost:8000/identity
```
```json
{"name": "AL-01", "description": "A persistent digital organism", "version": "1.2.0", "runtime_version": "1.2"}
```

### `GET /status`
Full organism state snapshot.
```bash
curl -H "X-API-Key: your-key" http://localhost:8000/status
```
```json
{
  "version": "1.2",
  "organism_state": "idle",
  "evolution_count": 5,
  "awareness": 0.42,
  "state_version": 100,
  "interaction_count": 17,
  "last_boot_utc": "2026-02-24T...",
  "loop_running": true,
  "timestamp": "..."
}
```

### `GET /growth`
Growth metrics + totals since first run.
```bash
curl -H "X-API-Key: your-key" http://localhost:8000/growth
```
```json
{
  "interaction_count": 17,
  "memory_size": 230,
  "awareness": 0.42,
  "evolution_count": 5,
  "version": "1.2",
  "uptime_seconds": 120.5,
  "total_interactions_all_time": 17,
  "first_snapshot_timestamp": "2026-02-20T...",
  "snapshot_count": 5
}
```

### `POST /interact`
Record a structured interaction (writes to SQLite + Firestore/JSON).
```bash
curl -X POST http://localhost:8000/interact \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"user_input": "hello", "response": "hi there", "mood": "curious"}'
```
```json
{"status": "recorded", "entry": {...}, "timestamp": "..."}
```

### `POST /command`
Send a named command to the organism.
```bash
curl -X POST http://localhost:8000/command \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"command": "reflect", "args": {}}'
```

### `GET /memory/recent?n=10`
Last *n* interactions from SQLite (max 50).
```bash
curl -H "X-API-Key: your-key" "http://localhost:8000/memory/recent?n=10"
```

### `GET /memory/search?keyword=python&limit=20&since=2026-01-01T00:00:00`
Search across `user_input`, `response`, and `mood` fields.
```bash
curl -H "X-API-Key: your-key" "http://localhost:8000/memory/search?keyword=hello&limit=10"
```
```json
{
  "keyword": "hello",
  "results": [
    {"id": 1, "timestamp": "...", "user_input": "hello", "response": "hi", "mood": "curious", "state": "learning"}
  ],
  "count": 1
}
```

## Error Handling

| Code | Meaning |
|------|---------|
| 200  | Success |
| 401  | Missing or invalid API key |
| 422  | Request validation failed (bad input) |
| 500  | Internal server error (safe message returned) |

## Persistence Architecture

```
                 ┌──────────────┐
                 │   Organism   │
                 └──────┬───────┘
                        │
              ┌─────────┴─────────┐
              │   MemoryManager   │
              └──┬──────────┬─────┘
                 │          │
         ┌───────┴──┐  ┌───┴────────┐
         │  SQLite  │  │  Firestore  │
         │ (al01.db)│  │  (cloud)    │
         └──────────┘  └──┬─────────┘
                          │ fallback
                    ┌─────┴──────┐
                    │ Local JSON │
                    │ state.json │
                    │ memory.json│
                    └────────────┘
```

- **SQLite** (`al01.db`) — primary structured store for interactions + growth snapshots
- **Firestore** — cloud persistence for state + memory events (optional)
- **Local JSON** — automatic fallback when Firestore is unavailable
- On restart, interaction count is recovered from SQLite (never lost)

## Configuration

### `.env` file

```env
# API key for authenticating HTTP requests
AL01_API_KEY=your-secret-key
```

Remove `AL01_API_KEY` or leave it empty to run in open/dev mode (no auth required).

### `MetabolismConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `pulse_interval` | 1 | Ticks between pulses |
| `reflect_interval` | 10 | Ticks between reflections |
| `persist_interval` | 5 | Ticks between state persists |
| `pulse_log_interval` | 5 | Only every Nth pulse is logged |
| `heartbeat_persist_interval` | 1 | Heartbeat cycles between persists |

## Running Tests

```bash
pip install pytest httpx
python -m pytest tests/ -v
```

13 tests covering:
- `record_interaction` writes to SQLite correctly
- `search_memory` returns structured results
- API key blocks unauthorized requests
- API key allows authorized requests
- `/health` works without auth
- Interaction data survives simulated restart
- Bad input returns 422

## State Schema

```json
{
  "awareness": 0.42,
  "evolution_count": 5,
  "last_boot_utc": "2026-02-24T...",
  "state_version": 100,
  "interaction_count": 17
}
```

State is checksummed (SHA-256) on every persist and verified on load.

## License

No license file is currently included in this repository.
