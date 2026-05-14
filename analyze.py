"""
analyze.py — end-to-end replay coaching pipeline.

Usage:
    python analyze.py <replay_path> [player_name]

Loads the replay, detects coaching events, calls the AI coach, saves everything
to the SQLite database, and prints the coaching report.
"""

import asyncio
import json
import math
import os
import re
import sys

from sqlmodel import Session, select

from agent_analysis.replay_coach import rl_coach
from database import (
    PLAYLIST_NAMES,
    CoachingEvent,
    CoachingReport,
    GameSession,
    GameStats,
    Player,
    PlayerMode,
    create_tables,
    get_engine,
)
from replay_processing import (
    build_aggregate_stats,
    build_coaching_input,
    detect_events,
    identify_uploader,
    load_replay,
)


def _get_or_create_player(db: Session, platform_id: str, name: str) -> Player:
    player = db.exec(select(Player).where(Player.platform_id == platform_id)).first()
    if not player:
        player = Player(platform_id=platform_id, name=name)
        db.add(player)
        db.flush()
    elif player.name != name:
        player.name = name
        db.flush()
    return player


def _get_or_create_mode(db: Session, player: Player, playlist: int) -> PlayerMode:
    mode = db.exec(
        select(PlayerMode)
        .where(PlayerMode.player_id == player.id)
        .where(PlayerMode.playlist == playlist)
    ).first()
    if not mode:
        mode = PlayerMode(
            player_id=player.id,
            playlist=playlist,
            playlist_name=PLAYLIST_NAMES.get(playlist, f"Playlist {playlist}"),
        )
        db.add(mode)
        db.flush()
    return mode


def _nan_to_none(v):
    if v is None:
        return None
    try:
        return None if math.isnan(v) else v
    except TypeError:
        return v


def _save_to_db(
    db: Session,
    replay_hash: str,
    played_at,
    summary: dict,
    events_df,
    report: dict,
    player_proto,
) -> GameSession:
    existing = db.exec(
        select(GameSession).where(GameSession.replay_hash == replay_hash)
    ).first()
    if existing:
        print(f"Replay already in database (session id={existing.id}), skipping save.")
        return existing

    player = _get_or_create_player(db, player_proto.id.id, summary["player_name"])
    mode = _get_or_create_mode(db, player, summary["playlist"])

    session_row = GameSession(
        player_mode_id=mode.id,
        replay_hash=replay_hash,
        played_at=played_at,
        duration_s=summary["duration_s"],
        outcome=summary["outcome"],
        team_score=summary["team_score"],
        opponent_score=summary["opponent_score"],
    )
    db.add(session_row)
    db.flush()

    s = summary
    stats_row = GameStats(
        session_id=session_row.id,
        goals=s["goals"],
        assists=s["assists"],
        saves=s["saves"],
        shots=s["shots"],
        score=s["score"],
        goals_conceded=s["goals_conceded"],
        avg_boost_pct=s["avg_boost_pct"],
        time_no_boost_s=s["time_no_boost_s"],
        time_low_boost_s=s["time_low_boost_s"],
        time_full_boost_s=s["time_full_boost_s"],
        boost_tanks_consumed=s["boost_tanks_consumed"],
        boost_wasted_usage_pct=s["boost_wasted_usage_pct"],
        boost_wasted_collect_pct=s["boost_wasted_collect_pct"],
        num_large_boosts=s["num_large_boosts"],
        num_small_boosts=s["num_small_boosts"],
        num_stolen_boosts=s["num_stolen_boosts"],
        time_defensive_third_s=s["time_defensive_third_s"],
        time_neutral_third_s=s["time_neutral_third_s"],
        time_offensive_third_s=s["time_offensive_third_s"],
        time_defensive_half_s=s["time_defensive_half_s"],
        time_offensive_half_s=s["time_offensive_half_s"],
        time_behind_ball_s=s["time_behind_ball_s"],
        time_in_front_ball_s=s["time_in_front_ball_s"],
        time_on_ground_s=s["time_on_ground_s"],
        time_low_air_s=s["time_low_air_s"],
        time_high_air_s=s["time_high_air_s"],
        time_on_wall_s=s["time_on_wall_s"],
        avg_speed_uu=s["avg_speed_uu"],
        time_supersonic_s=s["time_supersonic_s"],
        time_slow_speed_s=s["time_slow_speed_s"],
        time_boost_speed_s=s["time_boost_speed_s"],
        avg_hit_distance_uu=s["avg_hit_distance_uu"],
        total_kickoffs=s["total_kickoffs"],
        avg_kickoff_boost_used=_nan_to_none(s.get("avg_kickoff_boost_used")),
        kickoffs_first_touch=s["kickoffs_first_touch"],
        kickoffs_go_to_ball=s["kickoffs_go_to_ball"],
        avg_possession_duration_s=s["avg_possession_duration_s"],
        avg_hits_per_possession=s["avg_hits_per_possession"],
        total_possessions=s["total_possessions"],
        total_carries=s["total_carries"],
        avg_carry_duration_s=s["avg_carry_duration_s"],
        total_flicks=s["total_flicks"],
        n_double_commits=s.get("n_double_commits", 0),
        n_overextensions=s.get("n_overextensions", 0),
        n_low_boost_defense=s.get("n_low_boost_defense", 0),
        n_boost_starvation=s.get("n_boost_starvation", 0),
        n_overpursuit=s.get("n_overpursuit", 0),
        n_rotation_gaps=s.get("n_rotation_gaps", 0),
        n_missed_aerials=s.get("n_missed_aerials", 0),
        total_events_flagged=s.get("total_events_flagged", 0),
    )
    db.add(stats_row)

    if not events_df.empty:
        for _, row in events_df.iterrows():
            db.add(CoachingEvent(
                session_id=session_row.id,
                event_type=row["event_type"],
                timestamp_s=_nan_to_none(row.get("timestamp")) or 0.0,
                seconds_remaining=_nan_to_none(row.get("seconds_remaining")),
                description=row.get("description", ""),
                player_pos_x=_nan_to_none(row.get("player_pos_x")),
                player_pos_y=_nan_to_none(row.get("player_pos_y")),
                player_pos_z=_nan_to_none(row.get("player_pos_z")),
                player_boost_pct=_nan_to_none(row.get("player_boost_pct")),
                player_speed_uu=_nan_to_none(row.get("player_speed")),
                ball_pos_x=_nan_to_none(row.get("ball_pos_x")),
                ball_pos_y=_nan_to_none(row.get("ball_pos_y")),
                ball_pos_z=_nan_to_none(row.get("ball_pos_z")),
                ball_speed_uu=_nan_to_none(row.get("ball_speed")),
                nearest_big_pad_dist=_nan_to_none(row.get("nearest_big_pad_dist")),
            ))

    db.add(CoachingReport(
        session_id=session_row.id,
        game_summary=report.get("coaching_text") or report.get("game_summary", ""),
        coaching_points_json=json.dumps(report.get("coaching_points", [])),
    ))

    db.commit()
    return session_row


def _extract_coaching_text(raw_text: str) -> str:
    # Strip <think>...</think> reasoning traces from qwen3 models
    return re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()


def _print_report(report: dict) -> None:
    print("\n" + "═" * 60)
    if report.get("coaching_text"):
        print(report["coaching_text"])
    else:
        print(f"GAME SUMMARY: {report.get('game_summary', '')}")
        print("═" * 60)
        for pt in report.get("coaching_points", []):
            print(
                f"\n[{pt['rank']}] {pt['label']}"
                f"  ({pt['impact']} impact / {pt['category']})"
            )
            print(f"  Observation:    {pt['observation']}")
            print(f"  Why it matters: {pt['why_it_matters']}")
            print(f"  Fix:            {pt['fix']}")


def _load_cached_report(replay_hash: str) -> dict | None:
    engine = get_engine()
    create_tables(engine)
    with Session(engine) as db:
        session_row = db.exec(
            select(GameSession).where(GameSession.replay_hash == replay_hash)
        ).first()
        if not session_row:
            return None
        report_row = db.exec(
            select(CoachingReport).where(CoachingReport.session_id == session_row.id)
        ).first()
        if not report_row or not report_row.game_summary:
            return None
        coaching_points = json.loads(report_row.coaching_points_json)
        if coaching_points:
            return {"game_summary": report_row.game_summary, "coaching_points": coaching_points}
        return {"coaching_text": report_row.game_summary, "coaching_points": []}


async def analyze(replay_path: str, player_name: str | None = None) -> dict:
    replay_hash = os.path.splitext(os.path.basename(replay_path))[0]

    cached = _load_cached_report(replay_hash)
    if cached:
        print(f"Returning cached coaching report for {replay_hash}.")
        _print_report(cached)
        return cached

    am, proto, df = load_replay(replay_path)
    player_proto = identify_uploader(proto, player_name)
    played_at = getattr(am.game, "datetime", None)

    print(f"Analyzing: {player_proto.name!r} ({'orange' if player_proto.is_orange else 'blue'} team)")

    events_df = detect_events(player_proto, proto, df)
    summary = build_aggregate_stats(player_proto, proto, df)

    if not events_df.empty:
        counts = events_df.groupby("event_type").size().to_dict()
        summary["n_double_commits"]    = counts.get("double_commit", 0)
        summary["n_overextensions"]    = counts.get("overextension", 0)
        summary["n_low_boost_defense"] = counts.get("low_boost_defense", 0)
        summary["n_boost_starvation"]  = counts.get("boost_starvation", 0)
        summary["n_overpursuit"]       = counts.get("overpursuit", 0)
        summary["n_rotation_gaps"]     = counts.get("rotation_gap", 0)
        summary["n_missed_aerials"]    = counts.get("missed_aerial", 0)
        summary["total_events_flagged"] = len(events_df)

    coaching_input = build_coaching_input(summary, events_df)

    if not events_df.empty:
        counts = events_df.groupby("event_type").size().sort_values(ascending=False)
        print(f"Detected {len(events_df)} events across {len(counts)} types:")
        for etype, n in counts.items():
            print(f"  {etype}: {n}")
    else:
        print("No events detected.")

    print("\nQuerying AI coach...")
    response = await rl_coach.run(coaching_input)
    coaching_text = _extract_coaching_text(response.text)

    if not coaching_text:
        print("\nAI coach returned an empty response.")
        raise ValueError("AI coach returned an empty report — try again.")

    report = {"coaching_text": coaching_text, "coaching_points": []}

    engine = get_engine()
    create_tables(engine)
    with Session(engine) as db:
        game_session = _save_to_db(
            db, replay_hash, played_at, summary, events_df, report, player_proto
        )
        print(f"Saved to database (session id={game_session.id})")

    _print_report(report)

    return report


if __name__ == "__main__":
    replay = sys.argv[1] if len(sys.argv) > 1 else "replays/084163F1425422E7E8B3DAAE74629EAA.replay"
    player = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(analyze(replay, player))
