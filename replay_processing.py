"""
RL Coach — Replay Processing

Parses a Rocket League replay with carball (sprocket-rl-parser) and produces:
  - events_df   : DataFrame of timestamped coaching-relevant events
  - summary_stats: dict of aggregate stats for the analyzed player
  - coaching_input: JSON string ready to pass to the rl_coach AI agent
"""

import warnings
import json
from typing import Dict, List, Optional, Tuple

import carball
import numpy as np
import pandas as pd


# ── Arena geometry (Unreal Units) ──────────────────────────────────────────────
MAP_Y = 10_280
MAP_THIRD = MAP_Y / 6   # ≈ 1 713 UU from midfield — third boundary

# ── Boost ──────────────────────────────────────────────────────────────────────
BOOST_MAX = 255.0
BOOST_LOW = BOOST_MAX * 0.20       # 51  — flagged as "low boost"
BOOST_CRITICAL = BOOST_MAX * 0.10  # 25.5 — flagged as "critical / starved"

BIG_PAD_XY = np.array([
    (3072, -4096), (-3072, -4096),
    (3584,     0), (-3584,     0),
    (3072,  4096), (-3072,  4096),
], dtype=float)

# ── Detection thresholds ───────────────────────────────────────────────────────
STARVATION_SECS = 3.0      # consecutive seconds below BOOST_CRITICAL → starvation
AERIAL_BALL_Z = 300.0      # ball height (UU) for aerial events
AERIAL_PLAYER_Z = 150.0    # player height for aerial events
AERIAL_MAX_DIST = 700.0    # max ball-to-player distance for "attempted aerial"

# carball stores velocities at ~10× real UU/s (verified against internal thresholds:
# supersonic ≥ 22000, boost speed > 14100, slow ≤ 7000 in carball source)
CARBALL_VEL_SCALE = 10.0   # divide raw vel by this for display as approximate UU/s
BALL_CLEARING_VEL = 4_000  # raw carball units (~400 UU/s) — ball heading toward own goal
FAST_PLAYER_VEL = 10_000   # raw carball units (~1000 UU/s) — player committed to a run

# Min seconds between flags of the same event type (per-type debounce)
DEBOUNCE: Dict[str, float] = {
    "double_commit":     4.0,
    "low_boost_defense": 5.0,
    "overextension":     5.0,
    "boost_starvation": 10.0,
    "overpursuit":       4.0,
    "rotation_gap":      5.0,
    "missed_aerial":     3.0,
}

CHALLENGE_DIST = 500.0   # UU — both players within this range of ball = challenge
TOWARD_THRESH  = 5_000.0 # raw carball vel units — minimum velocity component toward ball


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _speed(df_slice: pd.DataFrame) -> pd.Series:
    return np.sqrt(df_slice.vel_x**2 + df_slice.vel_y**2 + df_slice.vel_z**2)


def _nearest_big_pad(px: float, py: float) -> float:
    dists = np.sqrt((BIG_PAD_XY[:, 0] - px)**2 + (BIG_PAD_XY[:, 1] - py)**2)
    return float(dists.min())


def _pid(player_proto) -> str:
    return player_proto.id.id


def _debounced_frames(boolean_mask: pd.Series, times: pd.Series,
                      min_gap: float) -> pd.Index:
    """From a boolean Series, return frame indices spaced at least min_gap apart."""
    selected = []
    last_t = -min_gap - 1.0
    for frame in boolean_mask[boolean_mask].index:
        t = times.loc[frame]
        if pd.isna(t):
            continue
        if t - last_t >= min_gap:
            selected.append(frame)
            last_t = t
    return pd.Index(selected)


def _event_row(frame, event_type, p_row, b_row, g_row, description) -> dict:
    boost = float(p_row.get("boost", np.nan))
    pvx = float(p_row.get("vel_x", 0))
    pvy = float(p_row.get("vel_y", 0))
    pvz = float(p_row.get("vel_z", 0))
    bvx = float(b_row.get("vel_x", 0))
    bvy = float(b_row.get("vel_y", 0))
    bvz = float(b_row.get("vel_z", 0))
    px, py, pz = float(p_row.get("pos_x", np.nan)), float(p_row.get("pos_y", np.nan)), float(p_row.get("pos_z", np.nan))
    return {
        "frame_number":          frame,
        "timestamp":             float(g_row.get("time", np.nan)),
        "seconds_remaining":     float(g_row.get("seconds_remaining", np.nan)),
        "event_type":            event_type,
        "description":           description,
        "player_pos_x":          px,
        "player_pos_y":          py,
        "player_pos_z":          pz,
        "player_boost_pct":      boost / BOOST_MAX * 100 if not np.isnan(boost) else np.nan,
        # Divide raw carball velocity by CARBALL_VEL_SCALE for approximate real-world UU/s
        "player_speed":          float(np.sqrt(pvx**2 + pvy**2 + pvz**2)) / CARBALL_VEL_SCALE,
        "ball_pos_x":            float(b_row.get("pos_x", np.nan)),
        "ball_pos_y":            float(b_row.get("pos_y", np.nan)),
        "ball_pos_z":            float(b_row.get("pos_z", np.nan)),
        "ball_speed":            float(np.sqrt(bvx**2 + bvy**2 + bvz**2)) / CARBALL_VEL_SCALE,
        "nearest_big_pad_dist":  _nearest_big_pad(px, py) if not np.isnan(px) else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load & identify
# ─────────────────────────────────────────────────────────────────────────────

def load_replay(path: str):
    """Return (analysis_manager, proto, dataframe) for a replay file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = carball.analyze_replay_file(path)
    return am, am.get_protobuf_data(), am.get_data_frame()


def identify_uploader(proto, player_name: Optional[str] = None):
    """
    Return the player proto for the person who uploaded the replay.
    If player_name is given, match by name (case-insensitive).
    Defaults to the first blue (is_orange=0) player.
    """
    if player_name:
        for p in proto.players:
            if p.name.lower() == player_name.lower():
                return p
        raise ValueError(
            f"Player {player_name!r} not found. "
            f"Available: {[p.name for p in proto.players]}"
        )
    for p in proto.players:
        if not p.is_orange:
            return p
    return proto.players[0]


# ─────────────────────────────────────────────────────────────────────────────
# Individual event detectors
# ─────────────────────────────────────────────────────────────────────────────

def _detect_double_commits(player_proto, proto, df) -> List[dict]:
    """
    Both the analyzed player and a teammate simultaneously challenge the same ball —
    a double commit that leaves the team's rotation broken.
    Only meaningful in 2v2+; returns [] for 1v1.
    """
    is_orange = player_proto.is_orange
    teammates = [p for p in proto.players
                 if p.is_orange == is_orange and p.name != player_proto.name]
    if not teammates:
        return []

    name = player_proto.name
    pdf  = df[name]
    bdf  = df["ball"]
    gdf  = df["game"]
    active = gdf["goal_number"].notnull()
    frames = pdf.index.intersection(gdf.index).intersection(bdf.index)
    frames = frames[active.reindex(frames, fill_value=False)]

    p = pdf.loc[frames]
    b = bdf.loc[frames]
    g = gdf.loc[frames]
    times = g["time"]

    # Distance and velocity-toward-ball for the analyzed player
    p_to_ball = np.sqrt((p.pos_x - b.pos_x)**2 + (p.pos_y - b.pos_y)**2 + (p.pos_z - b.pos_z)**2)
    p_dot = ((b.pos_x - p.pos_x) * p.vel_x
           + (b.pos_y - p.pos_y) * p.vel_y
           + (b.pos_z - p.pos_z) * p.vel_z) / p_to_ball.clip(lower=1)
    p_challenging = (p_to_ball <= CHALLENGE_DIST) & (p_dot > TOWARD_THRESH)

    events: List[dict] = []
    for teammate in teammates:
        if teammate.name not in df.columns.get_level_values(0):
            continue
        tdf = df[teammate.name].reindex(frames)
        t_to_ball = np.sqrt(
            (tdf.pos_x - b.pos_x)**2 + (tdf.pos_y - b.pos_y)**2 + (tdf.pos_z - b.pos_z)**2
        )
        t_dot = ((b.pos_x - tdf.pos_x) * tdf.vel_x
               + (b.pos_y - tdf.pos_y) * tdf.vel_y
               + (b.pos_z - tdf.pos_z) * tdf.vel_z) / t_to_ball.clip(lower=1)
        t_challenging = (t_to_ball <= CHALLENGE_DIST) & (t_dot > TOWARD_THRESH)

        both_committing = p_challenging & t_challenging
        selected = _debounced_frames(both_committing, times, DEBOUNCE["double_commit"])
        for frame in selected:
            p_dist = float(p_to_ball.loc[frame])
            t_dist = float(t_to_ball.loc[frame])
            events.append(_event_row(
                frame, "double_commit",
                p.loc[frame], b.loc[frame], g.loc[frame],
                f"You and {teammate.name} both committed to the same ball "
                f"({p_dist:.0f} UU and {t_dist:.0f} UU away) — rotation broken.",
            ))
    return events


def _detect_frame_events(player_proto, proto, df) -> List[dict]:
    """
    Vectorized detection of positioning, boost, and aerial events.
    All coordinates are normalized so that positive effective-Y = attacking direction
    for both blue and orange players.
    """
    name = player_proto.name
    is_orange = player_proto.is_orange
    sign = -1 if is_orange else 1   # flip Y for orange so +eff_y always = attacking

    pdf = df[name]
    bdf = df["ball"]
    gdf = df["game"]

    # Only analyze frames during active play
    active = gdf["goal_number"].notnull()
    frames = pdf.index.intersection(gdf.index).intersection(bdf.index)
    frames = frames[active.reindex(frames, fill_value=False)]

    p = pdf.loc[frames]
    b = bdf.loc[frames]
    g = gdf.loc[frames]

    eff_py  = p.pos_y * sign
    eff_by  = b.pos_y * sign
    eff_bvy = b.vel_y * sign   # positive = ball moving toward opponent goal

    boost_raw = p["boost"]
    # Boost tracking is broken when all recorded boost values are 0
    boost_tracking_ok = bool((boost_raw.dropna() > 0).any())
    # Fill missing frames with neutral value for mask comparisons
    boost   = boost_raw.ffill().fillna(0)
    p_speed = _speed(p)
    b_speed = _speed(b)
    times   = g["time"]

    # ── low_boost_defense ────────────────────────────────────────────────────
    # Player in defensive third, < 20% boost, ball not yet in attacking third.
    if boost_tracking_ok:
        lbd_mask = (
            (eff_py < -MAP_THIRD) &
            (boost < BOOST_LOW) &
            (eff_by < MAP_THIRD)
        )
    else:
        lbd_mask = pd.Series(False, index=frames)
    lbd_frames = _debounced_frames(lbd_mask, times, DEBOUNCE["low_boost_defense"])

    # ── overextension ────────────────────────────────────────────────────────
    # In attacking third, < 15% boost, ball clearing toward own goal — can't get back.
    if boost_tracking_ok:
        ovx_mask = (
            (eff_py > MAP_THIRD) &
            (boost < BOOST_CRITICAL * 1.5) &
            (eff_by < 0) &
            (eff_bvy < -BALL_CLEARING_VEL)
        )
    else:
        ovx_mask = pd.Series(False, index=frames)
    ovx_frames = _debounced_frames(ovx_mask, times, DEBOUNCE["overextension"])

    # ── overpursuit ──────────────────────────────────────────────────────────
    # Player sprinting in own half, ball clearing hard toward own goal — committed
    # past the point of recovery.
    opu_mask = (
        (eff_py > 0) &
        (eff_bvy < -BALL_CLEARING_VEL) &
        (p_speed > FAST_PLAYER_VEL)
    )
    opu_frames = _debounced_frames(opu_mask, times, DEBOUNCE["overpursuit"])

    # ── missed_aerial ────────────────────────────────────────────────────────
    pid = _pid(player_proto)
    player_hit_frames = frozenset(
        h.frame_number for h in proto.game_stats.hits if h.player_id.id == pid
    )
    dist_to_ball = np.sqrt(
        (p.pos_x - b.pos_x)**2 + (p.pos_y - b.pos_y)**2 + (p.pos_z - b.pos_z)**2
    )
    # jump_active can be False/NaN/int (frame counter) — coerce to boolean
    def _col_active(col_name):
        col = p[col_name] if col_name in p.columns else pd.Series(False, index=frames)
        return col.fillna(False).astype(bool)

    airborne_attempt = (
        (p.pos_z > AERIAL_PLAYER_Z) &
        (b.pos_z > AERIAL_BALL_Z) &
        (_col_active("dodge_active") | _col_active("double_jump_active")) &
        (dist_to_ball < AERIAL_MAX_DIST)
    )

    # Filter to frames where no hit follows within ~20 frames (~0.5 s at 30 fps)
    def no_hit_follows(frame):
        return not any(frame < hf <= frame + 20 for hf in player_hit_frames)

    aerial_candidates = airborne_attempt[airborne_attempt].index
    aerial_mask_filtered = pd.Series(
        {f: no_hit_follows(f) for f in aerial_candidates},
        dtype=bool
    ).reindex(frames, fill_value=False)
    aer_frames = _debounced_frames(aerial_mask_filtered, times, DEBOUNCE["missed_aerial"])

    # ── boost_starvation ─────────────────────────────────────────────────────
    # Identify runs of frames where boost < 10% lasting >= STARVATION_SECS.
    # We flag at the moment the run first reaches the threshold duration.
    starved = (boost < BOOST_CRITICAL) if boost_tracking_ok else pd.Series(False, index=frames)
    run_id = (~starved).cumsum()   # increments each time we exit a starved run
    delta = g["delta"].fillna(0)

    run_duration = delta[starved].groupby(run_id[starved]).cumsum()
    # First frame in each run where cumulative time >= threshold
    crossed = run_duration[run_duration >= STARVATION_SECS]
    stv_candidate_frames: List[int] = []
    seen_runs: set = set()
    for frame, rid in run_id[starved].items():
        if rid in seen_runs:
            continue
        if frame in crossed.index:
            stv_candidate_frames.append(frame)
            seen_runs.add(rid)

    stv_mask = pd.Series(False, index=frames)
    stv_mask.loc[[f for f in stv_candidate_frames if f in frames]] = True
    stv_frames = _debounced_frames(stv_mask, times, DEBOUNCE["boost_starvation"])

    # ── Assemble events ───────────────────────────────────────────────────────
    events: List[dict] = []

    for frame in lbd_frames:
        boost_val = float(boost.loc[frame])
        events.append(_event_row(
            frame, "low_boost_defense",
            p.loc[frame], b.loc[frame], g.loc[frame],
            f"In defensive third with {boost_val/BOOST_MAX*100:.0f}% boost — limited recovery options.",
        ))

    for frame in ovx_frames:
        boost_val = float(boost.loc[frame])
        ball_vel = float(b.vel_y.loc[frame] * sign) / CARBALL_VEL_SCALE
        events.append(_event_row(
            frame, "overextension",
            p.loc[frame], b.loc[frame], g.loc[frame],
            f"In attacking third with {boost_val/BOOST_MAX*100:.0f}% boost while ball clears toward own goal at {abs(ball_vel):.0f} UU/s.",
        ))

    for frame in opu_frames:
        ps = float(p_speed.loc[frame]) / CARBALL_VEL_SCALE
        bv = float(b_speed.loc[frame]) / CARBALL_VEL_SCALE
        events.append(_event_row(
            frame, "overpursuit",
            p.loc[frame], b.loc[frame], g.loc[frame],
            f"Charging at {ps:.0f} UU/s while ball clears at {bv:.0f} UU/s — overcommitted past recovery point.",
        ))

    for frame in aer_frames:
        bz = float(b.pos_z.loc[frame])
        dist = float(np.sqrt(
            (p.pos_x.loc[frame] - b.pos_x.loc[frame])**2 +
            (p.pos_y.loc[frame] - b.pos_y.loc[frame])**2 +
            (p.pos_z.loc[frame] - b.pos_z.loc[frame])**2
        ))
        events.append(_event_row(
            frame, "missed_aerial",
            p.loc[frame], b.loc[frame], g.loc[frame],
            f"Aerial attempt toward ball at Z={bz:.0f} UU — no contact made (distance {dist:.0f} UU).",
        ))

    for frame in stv_frames:
        run_t = float(run_duration.loc[frame]) if frame in run_duration.index else STARVATION_SECS
        events.append(_event_row(
            frame, "boost_starvation",
            p.loc[frame], b.loc[frame], g.loc[frame],
            f"Below 10% boost for {run_t:.1f}s — boost-starved during live play.",
        ))

    return events


def _detect_rotation_gap(player_proto, proto, df) -> List[dict]:
    """
    Flag frames where the player and at least one teammate are both in the
    attacking third while the ball is in their defensive half — no backman.
    Only meaningful in 2v2+; returns [] for 1v1.
    """
    is_orange = player_proto.is_orange
    teammates = [p for p in proto.players
                 if p.is_orange == is_orange and p.name != player_proto.name]
    if not teammates:
        return []

    name = player_proto.name
    sign = -1 if is_orange else 1
    pdf  = df[name]
    gdf  = df["game"]
    bdf  = df["ball"]
    active = gdf["goal_number"].notnull()
    frames = pdf.index.intersection(gdf.index).intersection(bdf.index)
    frames = frames[active.reindex(frames, fill_value=False)]

    p      = pdf.loc[frames]
    g      = gdf.loc[frames]
    b      = bdf.loc[frames]
    eff_py = p.pos_y * sign
    eff_by = b.pos_y * sign
    times  = g["time"]

    events: List[dict] = []
    for teammate in teammates:
        if teammate.name not in df.columns.get_level_values(0):
            continue
        tdf    = df[teammate.name].reindex(frames)
        eff_ty = tdf.pos_y * sign
        both_up = (eff_py > MAP_THIRD) & (eff_ty > MAP_THIRD) & (eff_by < 0)
        selected = _debounced_frames(both_up, times, DEBOUNCE["rotation_gap"])
        for frame in selected:
            events.append(_event_row(
                frame, "rotation_gap",
                p.loc[frame], b.loc[frame], g.loc[frame],
                f"Both you and {teammate.name} are in the attacking third while the ball is in your defensive half.",
            ))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Public: detect all events
# ─────────────────────────────────────────────────────────────────────────────

def detect_events(player_proto, proto, df) -> pd.DataFrame:
    """
    Detect all coaching-relevant events for the given player.
    Returns a DataFrame sorted by timestamp, one row per event.
    """
    rows: List[dict] = []
    rows.extend(_detect_double_commits(player_proto, proto, df))
    rows.extend(_detect_frame_events(player_proto, proto, df))
    rows.extend(_detect_rotation_gap(player_proto, proto, df))

    if not rows:
        return pd.DataFrame()

    events_df = pd.DataFrame(rows)
    col_order = [
        "timestamp", "seconds_remaining", "frame_number", "event_type", "description",
        "player_pos_x", "player_pos_y", "player_pos_z",
        "player_boost_pct", "player_speed",
        "ball_pos_x", "ball_pos_y", "ball_pos_z", "ball_speed",
        "nearest_big_pad_dist",
    ]
    existing = [c for c in col_order if c in events_df.columns]
    events_df = events_df[existing].sort_values("timestamp").reset_index(drop=True)
    return events_df


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats
# ─────────────────────────────────────────────────────────────────────────────

def build_aggregate_stats(player_proto, proto, df=None) -> Dict:
    """Extract aggregate statistics from the proto for the analyzed player."""
    gm = proto.game_metadata
    is_orange = player_proto.is_orange

    team_score = opp_score = 0
    for team in proto.teams:
        if team.is_orange == is_orange:
            team_score = team.score
        else:
            opp_score = team.score

    if team_score > opp_score:
        outcome = "win"
    elif team_score < opp_score:
        outcome = "loss"
    else:
        outcome = "draw"

    # goals scored by opponent team (goal.player_id matches an opponent's proto id)
    opp_ids = frozenset(p.id.id for p in proto.players if p.is_orange != is_orange)
    goals_conceded = sum(1 for g in gm.goals if g.player_id.id in opp_ids)

    s   = player_proto.stats
    bst = s.boost
    pos = s.positional_tendencies
    spd = s.speed
    avg = s.averages
    ks  = s.kickoff_stats
    pp  = s.per_possession_stats
    bc  = s.ball_carries

    result = {
        # Identity / scoreline
        "player_name":          player_proto.name,
        "is_orange":            bool(is_orange),
        "outcome":              outcome,
        "team_score":           team_score,
        "opponent_score":       opp_score,
        "goals":                player_proto.goals,
        "assists":              player_proto.assists,
        "saves":                player_proto.saves,
        "shots":                player_proto.shots,
        "score":                player_proto.score,
        "goals_conceded":       goals_conceded,
        # Game context
        "duration_s":           gm.length,
        "playlist":             gm.playlist,
        # Boost management (avg_boost_pct=0 with time_no_boost≈duration means no boost data)
        "avg_boost_pct":            bst.average_boost_level,
        "time_no_boost_s":          bst.time_no_boost,
        "time_low_boost_s":         bst.time_low_boost,
        "time_full_boost_s":        bst.time_full_boost,
        "boost_tanks_consumed":     round(bst.boost_usage / BOOST_MAX, 2),
        "boost_wasted_usage_pct":   bst.wasted_usage,
        "boost_wasted_collect_pct": bst.wasted_collection,
        "num_large_boosts":         bst.num_large_boosts,
        "num_small_boosts":         bst.num_small_boosts,
        "num_stolen_boosts":        bst.num_stolen_boosts,
        # Positioning (seconds)
        "time_defensive_third_s":   pos.time_in_defending_third,
        "time_neutral_third_s":     pos.time_in_neutral_third,
        "time_offensive_third_s":   pos.time_in_attacking_third,
        "time_defensive_half_s":    pos.time_in_defending_half,
        "time_offensive_half_s":    pos.time_in_attacking_half,
        "time_behind_ball_s":       pos.time_behind_ball,
        "time_in_front_ball_s":     pos.time_in_front_ball,
        "time_on_ground_s":         pos.time_on_ground,
        "time_low_air_s":           pos.time_low_in_air,
        "time_high_air_s":          pos.time_high_in_air,
        "time_on_wall_s":           pos.time_on_wall,
        # Speed (carball internal raw speed divided by CARBALL_VEL_SCALE for real UU/s)
        "avg_speed_uu":             avg.average_speed / CARBALL_VEL_SCALE,
        "time_supersonic_s":        spd.time_at_super_sonic,
        "time_slow_speed_s":        spd.time_at_slow_speed,
        "time_boost_speed_s":       spd.time_at_boost_speed,
        # Hit quality
        "avg_hit_distance_uu":      avg.average_hit_distance,
        # Kickoffs
        "total_kickoffs":           ks.total_kickoffs,
        "avg_kickoff_boost_used":   ks.average_boost_used,
        "kickoffs_first_touch":     ks.num_time_first_touch,
        "kickoffs_go_to_ball":      ks.num_time_go_to_ball,
        # Per-possession
        "avg_possession_duration_s": pp.average_duration,
        "avg_hits_per_possession":   pp.average_hits,
        "total_possessions":         pp.count,
        # Ball carries / dribbles
        "total_carries":             bc.total_carries,
        "avg_carry_duration_s":      bc.average_carry_time,
        "total_flicks":              bc.total_flicks,
    }

    # Detect whether per-frame boost data is tracked (game_version 8 replays record all zeros).
    # Expose the flag so build_coaching_input can null out misleading zero-valued boost fields.
    if df is not None and player_proto.name in df.columns.get_level_values(0):
        boost_col = df[player_proto.name]["boost"]
        result["boost_data_available"] = bool((boost_col.dropna() > 0).any())
    else:
        result["boost_data_available"] = True
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Format for AI agent
# ─────────────────────────────────────────────────────────────────────────────

def build_coaching_input(
    summary_stats: Dict,
    events_df: pd.DataFrame,
    max_per_type: int = 6,
) -> str:
    """
    Combine aggregate stats and key event frames into a single JSON string
    matching the rl_coach agent's expected input format.
    """
    stats = dict(summary_stats)

    # When boost frame data isn't tracked, zero-valued boost stats mislead the model into
    # giving the same boost-related advice every game regardless of actual play.
    _BOOST_STAT_KEYS = {
        "avg_boost_pct", "time_no_boost_s", "time_low_boost_s", "time_full_boost_s",
        "boost_tanks_consumed", "boost_wasted_usage_pct", "boost_wasted_collect_pct",
        "num_large_boosts", "num_small_boosts", "num_stolen_boosts",
    }
    if not stats.get("boost_data_available", True):
        for k in _BOOST_STAT_KEYS:
            if k in stats:
                stats[k] = None

    if not events_df.empty:
        counts = events_df.groupby("event_type").size().to_dict()
        stats["n_double_commits"]     = counts.get("double_commit", 0)
        stats["n_overextensions"]     = counts.get("overextension", 0)
        stats["n_low_boost_defense"]  = counts.get("low_boost_defense", 0)
        stats["n_boost_starvation"]   = counts.get("boost_starvation", 0)
        stats["n_overpursuit"]        = counts.get("overpursuit", 0)
        stats["n_rotation_gaps"]      = counts.get("rotation_gap", 0)
        stats["n_missed_aerials"]     = counts.get("missed_aerial", 0)
        stats["total_events_flagged"] = len(events_df)

        # Boost tracking: if any frame has a non-zero boost value it's available
        boost_available = bool((events_df["player_boost_pct"].dropna() > 0).any())

        # High-signal types: keep all instances; low-signal: cap at max_per_type,
        # sampled evenly across the game timeline so coverage spans the whole match.
        HIGH_SIGNAL = {"double_commit", "overextension", "rotation_gap", "low_boost_defense"}
        parts = []
        for etype, group in events_df.groupby("event_type"):
            if etype in HIGH_SIGNAL or len(group) <= max_per_type:
                parts.append(group)
            else:
                idx = np.linspace(0, len(group) - 1, max_per_type, dtype=int)
                parts.append(group.iloc[idx])
        events_df = pd.concat(parts).sort_values("timestamp") if parts else events_df.iloc[:0]
    else:
        boost_available = False

    key_frames = []
    for _, row in events_df.iterrows():
        frame: Dict = {
            "time_remaining": f"{row.get('seconds_remaining', 0):.0f}s",
            "event_type":     row.get("event_type", "unknown"),
            "player_pos":     [round(row.get("player_pos_x", 0), 1),
                               round(row.get("player_pos_y", 0), 1),
                               round(row.get("player_pos_z", 0), 1)],
            "ball_pos":       [round(row.get("ball_pos_x", 0), 1),
                               round(row.get("ball_pos_y", 0), 1),
                               round(row.get("ball_pos_z", 0), 1)],
            "speed_uu":       int(round(row.get("player_speed", 0), 0)),
            "description":    row.get("description", ""),
        }
        if boost_available:
            frame["boost_pct"]   = round(row.get("player_boost_pct", 0), 1)
            frame["nearest_pad"] = int(round(row.get("nearest_big_pad_dist", 0), 0))
        key_frames.append(frame)

    def _clean(obj):
        if isinstance(obj, float):
            return None if (obj != obj) else round(obj, 1)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    return json.dumps(_clean({"summary_statistics": stats, "key_frames": key_frames}), indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_coaching_analysis(
    replay_path: str,
    player_name: Optional[str] = None,
) -> Tuple[Dict, pd.DataFrame, str]:
    """
    Full pipeline: load replay → detect events → build stats → format for AI agent.

    Args:
        replay_path:  Path to the .replay file.
        player_name:  Name of the player to analyse. Defaults to the first blue player.

    Returns:
        summary_stats   : dict of aggregate stats for the player
        events_df       : DataFrame of coaching-relevant events (one row per event)
        coaching_input  : JSON string ready to send to the rl_coach AI agent
    """
    _, proto, df = load_replay(replay_path)
    player = identify_uploader(proto, player_name)

    print(f"Analyzing: {player.name!r} ({'orange' if player.is_orange else 'blue'} team)")

    events_df    = detect_events(player, proto, df)
    summary_stats = build_aggregate_stats(player, proto, df)
    coaching_input = build_coaching_input(summary_stats, events_df)

    if not events_df.empty:
        counts = events_df.groupby("event_type").size().sort_values(ascending=False)
        print(f"Detected {len(events_df)} events across {len(counts)} types:")
        for etype, n in counts.items():
            print(f"  {etype}: {n}")
    else:
        print("No events detected.")

    return summary_stats, events_df, coaching_input


# ─────────────────────────────────────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    replay = sys.argv[1] if len(sys.argv) > 1 else "replays/084163F1425422E7E8B3DAAE74629EAA.replay"
    pname  = sys.argv[2] if len(sys.argv) > 2 else None

    stats, events, coaching_json = run_coaching_analysis(replay, pname)

    print("\n── Aggregate Stats ──")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    if not events.empty:
        print("\n── Key Events (first 10) ──")
        print(events.head(10).to_string(index=False))
