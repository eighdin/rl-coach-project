"""
web/server.py — FastAPI server for the RL Coach web app.

Run from the project root:
    .venv/bin/uvicorn web.server:app --reload
"""

import json
import os
import sys
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze import analyze, _load_cached_report
from database import CoachingReport, GameSession, PlayerMode, Player, GameStats, create_tables, get_engine

app = FastAPI(title="RL Coach")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.post("/api/analyze")
async def analyze_replay(file: UploadFile, player_name: str = ""):
    if not file.filename or not file.filename.endswith(".replay"):
        raise HTTPException(status_code=400, detail="File must be a .replay file.")

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".replay", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        report = await analyze(tmp_path, player_name.strip() or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    if not report:
        raise HTTPException(status_code=500, detail="AI coach returned an empty report.")

    return JSONResponse(report)


@app.get("/api/report/{replay_hash}")
async def get_report(replay_hash: str):
    report = _load_cached_report(replay_hash)
    if not report:
        raise HTTPException(status_code=404, detail="No cached report for this replay.")
    return JSONResponse(report)


@app.get("/api/history")
async def get_history():
    engine = get_engine()
    create_tables(engine)
    with Session(engine) as db:
        sessions = db.exec(select(GameSession).order_by(GameSession.id.desc())).all()  # type: ignore[arg-type]
        rows = []
        for s in sessions:
            mode = db.exec(select(PlayerMode).where(PlayerMode.id == s.player_mode_id)).first()
            player = db.exec(select(Player).where(Player.id == mode.player_id)).first() if mode else None
            report = db.exec(
                select(CoachingReport).where(CoachingReport.session_id == s.id)
            ).first()
            stats = db.exec(
                select(GameStats).where(GameStats.session_id == s.id)
            ).first()
            rows.append({
                "replay_hash":   s.replay_hash,
                "player_name":   player.name if player else "Unknown",
                "playlist_name": mode.playlist_name if mode else "Unknown",
                "outcome":       s.outcome,
                "team_score":    s.team_score,
                "opponent_score": s.opponent_score,
                "duration_s":    round(s.duration_s),
                "played_at":     s.played_at.isoformat() if s.played_at else None,
                "total_events":  stats.total_events_flagged if stats else 0,
                "has_report":    bool(report and report.game_summary and json.loads(report.coaching_points_json)),
            })
        return JSONResponse(rows)
