from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel, create_engine

PLAYLIST_NAMES: dict[int, str] = {
    1: "Casual 1v1",
    2: "Casual 2v2",
    3: "Casual 3v3",
    10: "Ranked 1v1",
    11: "Ranked 2v2",
    13: "Ranked 3v3",
    27: "Hoops",
    28: "Rumble",
    29: "Dropshot",
    30: "Snow Day",
}

DATABASE_URL = "sqlite:///rl_coach.db"


class Player(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    platform_id: str = Field(unique=True, index=True)
    name: str

    modes: List["PlayerMode"] = Relationship(back_populates="player")


class PlayerMode(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    player_id: int = Field(foreign_key="player.id")
    playlist: int
    playlist_name: str

    player: Player = Relationship(back_populates="modes")
    sessions: List["GameSession"] = Relationship(back_populates="player_mode")


class GameSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    player_mode_id: int = Field(foreign_key="playermode.id")
    replay_hash: str = Field(unique=True, index=True)
    played_at: Optional[datetime] = None
    duration_s: float
    outcome: str
    team_score: int
    opponent_score: int

    player_mode: Optional[PlayerMode] = Relationship(back_populates="sessions")
    stats: Optional["GameStats"] = Relationship(back_populates="session")
    events: List["CoachingEvent"] = Relationship(back_populates="session")
    report: Optional["CoachingReport"] = Relationship(back_populates="session")


class GameStats(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="gamesession.id")

    # Scoreline
    goals: int
    assists: int
    saves: int
    shots: int
    score: int
    goals_conceded: int

    # Boost
    avg_boost_pct: float
    time_no_boost_s: float
    time_low_boost_s: float
    time_full_boost_s: float
    boost_tanks_consumed: float
    boost_wasted_usage_pct: float
    boost_wasted_collect_pct: float
    num_large_boosts: int
    num_small_boosts: int
    num_stolen_boosts: int

    # Positioning
    time_defensive_third_s: float
    time_neutral_third_s: float
    time_offensive_third_s: float
    time_defensive_half_s: float
    time_offensive_half_s: float
    time_behind_ball_s: float
    time_in_front_ball_s: float
    time_on_ground_s: float
    time_low_air_s: float
    time_high_air_s: float
    time_on_wall_s: float

    # Speed
    avg_speed_uu: float
    time_supersonic_s: float
    time_slow_speed_s: float
    time_boost_speed_s: float

    # Hit quality
    avg_hit_distance_uu: float

    # Kickoffs
    total_kickoffs: int
    avg_kickoff_boost_used: Optional[float] = None
    kickoffs_first_touch: int
    kickoffs_go_to_ball: int

    # Per-possession
    avg_possession_duration_s: float
    avg_hits_per_possession: float
    total_possessions: int

    # Ball carries
    total_carries: int
    avg_carry_duration_s: float
    total_flicks: int

    # Event counts from detection layer
    n_double_commits: int = 0
    n_overextensions: int = 0
    n_low_boost_defense: int = 0
    n_boost_starvation: int = 0
    n_overpursuit: int = 0
    n_rotation_gaps: int = 0
    n_missed_aerials: int = 0
    total_events_flagged: int = 0

    session: Optional[GameSession] = Relationship(back_populates="stats")


class CoachingEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="gamesession.id")

    event_type: str
    timestamp_s: float
    seconds_remaining: Optional[float] = None
    description: str
    player_pos_x: Optional[float] = None
    player_pos_y: Optional[float] = None
    player_pos_z: Optional[float] = None
    player_boost_pct: Optional[float] = None
    player_speed_uu: Optional[float] = None
    ball_pos_x: Optional[float] = None
    ball_pos_y: Optional[float] = None
    ball_pos_z: Optional[float] = None
    ball_speed_uu: Optional[float] = None
    nearest_big_pad_dist: Optional[float] = None

    session: Optional[GameSession] = Relationship(back_populates="events")


class CoachingReport(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="gamesession.id")

    game_summary: str
    coaching_points_json: str

    session: Optional[GameSession] = Relationship(back_populates="report")


def get_engine():
    return create_engine(DATABASE_URL)


def create_tables(engine=None):
    if engine is None:
        engine = get_engine()
    SQLModel.metadata.create_all(engine)
