"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import "./WinProbabilityReplay.css";

// Real team colors, used only for the chart fill split and readout text.
// Covers all 32 teams so the live tracker can show any matchup, not just
// the 6 teams in the showcase replays. WAS/WSH both map to Washington -
// nflverse (historical showcase data) uses "WAS", ESPN (live feed) uses "WSH".
// Lightened from each team's true brand color so every one is readable as
// text/fills on our dark theme - many NFL primaries (navy, black) fail
// WCAG AA contrast here at full darkness (HSL lightness raised per-color
// until contrast >= 4.5:1 against the panel background, hue preserved).
const TEAM_COLORS = {
  ARI: "#da5e7b", ATL: "#e6566e", BAL: "#8a7be5", BUF: "#3d83ff",
  CAR: "#00a8ff", CHI: "#5d88d5", CIN: "#FB4F14", CLE: "#ff9700",
  DAL: "#3a88f3", DEN: "#0f87ff", DET: "#00a5ff", GB: "#53ac95",
  HOU: "#0fa3f0", IND: "#1f87ff", JAX: "#648cb4", KC: "#ec5068",
  LAC: "#00a5ff", LAR: "#3d83ff", LV: "#A5ACAF", MIA: "#00f0ff",
  MIN: "#a075d7", NE: "#0f87ff", NO: "#D3BC8D", NYG: "#5e83ed",
  NYJ: "#2cd39b", PHI: "#00e7ff", PIT: "#FFB612", SEA: "#69BE28",
  SF: "#ff3838", TB: "#f64646", TEN: "#4B92DB", WAS: "#dc6060", WSH: "#dc6060",
};

const TEAM_NAMES = {
  ARI: "Cardinals", ATL: "Falcons", BAL: "Ravens", BUF: "Bills",
  CAR: "Panthers", CHI: "Bears", CIN: "Bengals", CLE: "Browns",
  DAL: "Cowboys", DEN: "Broncos", DET: "Lions", GB: "Packers",
  HOU: "Texans", IND: "Colts", JAX: "Jaguars", KC: "Chiefs",
  LAC: "Chargers", LAR: "Rams", LV: "Raiders", MIA: "Dolphins",
  MIN: "Vikings", NE: "Patriots", NO: "Saints", NYG: "Giants",
  NYJ: "Jets", PHI: "Eagles", PIT: "Steelers", SEA: "Seahawks",
  SF: "49ers", TB: "Buccaneers", TEN: "Titans", WAS: "Commanders", WSH: "Commanders",
};

const GOLD = "#E8B94A";
const CHART_W = 1000;
const CHART_H = 320;
const MID = CHART_H / 2;

const DOWN_ORDINALS = { 1: "1st", 2: "2nd", 3: "3rd", 4: "4th" };
function formatDownDistance(down, ydstogo) {
  if (!down) return "";
  const ordinal = DOWN_ORDINALS[down] || `${down}th`;
  return ydstogo === null || ydstogo === undefined ? `${ordinal} down` : `${ordinal} & ${ydstogo}`;
}

// nflverse numbers periods past regulation sequentially (5 = OT, 6 = 2OT, ...)
// rather than resetting - "Q5" reads as a typo, so spell it out.
function formatQuarter(q) {
  return q <= 4 ? `Q${q}` : q === 5 ? "OT" : `${q - 4}OT`;
}

function formatClock(secsLeft) {
  // secsLeft counts down within the current period (max 900, i.e. 15:00).
  // secsLeft % 900 === 0 means the period just started (15:00 on the clock),
  // not that no time remains - `|| 900` corrects that edge case.
  const inQuarter = secsLeft % 900 || (secsLeft > 0 ? 900 : 0);
  const mm = Math.floor(inQuarter / 60);
  const ss = Math.floor(inQuarter % 60);
  return `${mm}:${ss.toString().padStart(2, "0")}`;
}

/**
 * Win probability replay chart.
 *
 * Expects `gamesData` shaped like:
 * {
 *   [gameId]: {
 *     label: string,       // e.g. "Vikings 39, Colts 36 (OT) — Dec 17, 2022 — ..."
 *     home: "MIN",
 *     away: "IND",
 *     plays: [ [qtr, secondsRemaining, homeWinProbPct, playDescription, down, ydstogo, fieldPosition, homeScore, awayScore], ... ]
 *     // down/ydstogo/fieldPosition/homeScore/awayScore are optional (older
 *     // data may omit them, or a play may have no meaningful down, e.g. a
 *     // kickoff) - falls back to hiding that part of the readout rather
 *     // than showing "undefined & undefined".
 *   }
 * }
 *
 * Fetch this from /public/data/games_data.json and pass it in, or fetch
 * inside a parent server component and pass down as a prop.
 *
 * Pass `liveUrl` to additionally poll a live games_data JSON endpoint
 * (written by data_pipeline/live_score.py) every `livePollMs` and merge it
 * in. Live entries also carry a `status` field (e.g. "STATUS_IN_PROGRESS",
 * "STATUS_FINAL") used to show the LIVE badge - static showcase entries
 * don't have this field and are treated as non-live.
 */
export default function WinProbabilityReplay({ gamesData, liveUrl, livePollMs = 20000 }) {
  const [liveGames, setLiveGames] = useState({});

  useEffect(() => {
    if (!liveUrl) return undefined;
    let cancelled = false;
    const fetchLive = async () => {
      try {
        const res = await fetch(liveUrl, { cache: "no-store" });
        if (!res.ok || cancelled) return;
        const data = await res.json();
        if (!cancelled) setLiveGames(data);
      } catch {
        // Transient network/poll errors are expected; just retry next tick.
      }
    };
    fetchLive();
    const id = setInterval(fetchLive, livePollMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [liveUrl, livePollMs]);

  const mergedGames = useMemo(() => ({ ...(gamesData || {}), ...liveGames }), [gamesData, liveGames]);
  const gameIds = useMemo(() => Object.keys(mergedGames), [mergedGames]);
  const [currentGameId, setCurrentGameId] = useState(gameIds[0] || null);
  const [playIndex, setPlayIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef(null);
  const svgRef = useRef(null);
  const atTipRef = useRef(true);
  const prevNRef = useRef(0);

  // If nothing was selected at mount (no static games passed in - e.g. a
  // live-only page) and a live game shows up, default to it.
  useEffect(() => {
    if (currentGameId) return;
    const liveIds = Object.keys(liveGames);
    if (liveIds.length) setCurrentGameId(liveIds[0]);
  }, [liveGames, currentGameId]);

  const game = currentGameId ? mergedGames[currentGameId] : null;
  const plays = game?.plays || [];
  const n = plays.length;
  const isLive = Boolean(game?.status) && game.status !== "STATUS_FINAL";

  useEffect(() => {
    atTipRef.current = playIndex >= n - 1;
  }, [playIndex, n]);

  // New plays arrived for the game currently on screen: only auto-advance
  // to the latest play if the viewer was already caught up, so scrubbing
  // back through history to review a play isn't yanked out from under them.
  useEffect(() => {
    if (n > prevNRef.current && atTipRef.current) {
      setPlayIndex(n - 1);
    }
    prevNRef.current = n;
  }, [n]);

  const points = useMemo(() => {
    if (n === 0) return [];
    return plays.map((p, i) => {
      const x = (i / (n - 1)) * CHART_W;
      const y = CHART_H - (p[2] / 100) * CHART_H;
      return [x, y];
    });
  }, [plays, n]);

  const linePath = useMemo(
    () => (points.length ? "M " + points.map((p) => p.join(",")).join(" L ") : ""),
    [points]
  );

  const areaPath = useMemo(() => {
    if (!points.length) return "";
    return (
      "M0," + MID + " L " + points.map((p) => p.join(",")).join(" L ") + " L " + CHART_W + "," + MID + " Z"
    );
  }, [points]);

  const stopPlay = useCallback(() => {
    clearInterval(timerRef.current);
    timerRef.current = null;
    setIsPlaying(false);
  }, []);

  const selectGame = useCallback(
    (id) => {
      stopPlay();
      setCurrentGameId(id);
      setPlayIndex(0);
      atTipRef.current = true;
      prevNRef.current = 0;
    },
    [stopPlay]
  );

  useEffect(() => {
    if (!isPlaying) return;
    timerRef.current = setInterval(() => {
      setPlayIndex((prev) => {
        if (prev >= n - 1) {
          stopPlay();
          return prev;
        }
        return prev + 1;
      });
    }, 90);
    return () => clearInterval(timerRef.current);
  }, [isPlaying, n, stopPlay]);

  const togglePlay = () => {
    if (isPlaying) {
      stopPlay();
    } else {
      if (playIndex >= n - 1) setPlayIndex(0);
      setIsPlaying(true);
    }
  };

  const step = useCallback(
    (delta) => {
      if (n === 0) return;
      stopPlay();
      setPlayIndex((prev) => Math.max(0, Math.min(n - 1, prev + delta)));
    },
    [n, stopPlay]
  );

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (n === 0) return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        step(1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        step(-1);
      } else if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [n, step, togglePlay]);

  const handleChartClick = (e) => {
    if (!svgRef.current || n === 0) return;
    const rect = svgRef.current.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    const idx = Math.round(frac * (n - 1));
    setPlayIndex(Math.max(0, Math.min(n - 1, idx)));
    stopPlay();
  };

  if (!game) return null;

  const current = plays[playIndex];
  const [qtr, secsLeft, homeWp, desc, down, ydstogo, fieldPosition, homeScore, awayScore] =
    current || [1, 3600, 50, ""];
  let downDistanceText = formatDownDistance(down, ydstogo);
  if (downDistanceText && fieldPosition) downDistanceText += ` at ${fieldPosition}`;
  const hasScore = homeScore !== undefined && awayScore !== undefined && homeScore !== null && awayScore !== null;
  const homeColor = TEAM_COLORS[game.home] || GOLD;
  const awayColor = TEAM_COLORS[game.away] || "#5FA8D3";
  const leadingTeam = homeWp >= 50 ? game.home : game.away;
  const leadingPct = homeWp >= 50 ? homeWp : 100 - homeWp;
  const leadingColor = TEAM_COLORS[leadingTeam] || GOLD;

  const [titlePart, ...contextParts] = (game.label || "").split(" — ");

  return (
    <div className="wp-replay">
      <div className="wp-picker" role="group" aria-label="Select a game to replay">
        {gameIds.map((id) => {
          const g = mergedGames[id];
          const [, ...ctx] = (g.label || "").split(" — ");
          const gIsLive = Boolean(g.status) && g.status !== "STATUS_FINAL";
          return (
            <button
              key={id}
              onClick={() => selectGame(id)}
              className={`wp-game-btn ${id === currentGameId ? "active" : ""}`}
              aria-pressed={id === currentGameId}
            >
              <div className="wp-matchup">
                {gIsLive && <span className="wp-live-dot" aria-hidden="true" />}
                {g.away} @ {g.home}
              </div>
              <div className="wp-matchup-sub">{ctx[ctx.length - 1] || ""}</div>
            </button>
          );
        })}
      </div>

      <div className="wp-panel">
        <div className="wp-panel-content" key={currentGameId}>
          <div className="wp-panel-header">
            <div className="wp-game-title">
              {isLive && <span className="wp-live-dot" aria-hidden="true" />}
              {titlePart}
            </div>
            <div className="wp-game-context">{contextParts.join(" — ")}</div>
          </div>

          <div className="wp-chart-area">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${CHART_W} ${CHART_H}`}
              preserveAspectRatio="none"
              onClick={handleChartClick}
              role="img"
              aria-label="Win probability chart for the selected play"
            >
              <line
                x1="0"
                y1={MID}
                x2={CHART_W}
                y2={MID}
                stroke="#232B3D"
                strokeWidth="2"
                strokeDasharray="4,4"
              />
              <clipPath id={`clipAbove-${currentGameId}`}>
                <rect x="0" y="0" width={CHART_W} height={MID} />
              </clipPath>
              <clipPath id={`clipBelow-${currentGameId}`}>
                <rect x="0" y={MID} width={CHART_W} height={MID} />
              </clipPath>
              <path
                d={areaPath}
                fill={homeColor}
                fillOpacity="0.35"
                clipPath={`url(#clipAbove-${currentGameId})`}
              />
              <path
                d={areaPath}
                fill={awayColor}
                fillOpacity="0.35"
                clipPath={`url(#clipBelow-${currentGameId})`}
              />
              <path d={linePath} fill="none" stroke={GOLD} strokeWidth="2.5" />
              {points[playIndex] && (
                <circle
                  className="wp-scrub-dot"
                  cx={points[playIndex][0]}
                  cy={points[playIndex][1]}
                  r="6"
                  fill={GOLD}
                  stroke="#0A0E14"
                  strokeWidth="2"
                />
              )}
            </svg>
          </div>

          <div className="wp-team-labels">
            <span>
              {TEAM_NAMES[game.away] || game.away} ({game.away})
            </span>
            <span>
              {TEAM_NAMES[game.home] || game.home} ({game.home})
            </span>
          </div>

          <div className="wp-controls">
            <button className="wp-play-btn wp-step-btn" onClick={() => step(-1)} aria-label="Previous play">
              ⏮
            </button>
            <button
              className="wp-play-btn"
              onClick={togglePlay}
              aria-label={isPlaying ? "Pause" : "Play"}
              aria-pressed={isPlaying}
            >
              {isPlaying ? "❚❚" : "▶"}
            </button>
            <button className="wp-play-btn wp-step-btn" onClick={() => step(1)} aria-label="Next play">
              ⏭
            </button>
            <input
              type="range"
              min="0"
              max={Math.max(n - 1, 0)}
              value={playIndex}
              aria-label="Play scrubber"
              onChange={(e) => {
                stopPlay();
                setPlayIndex(parseInt(e.target.value, 10));
              }}
            />
          </div>

          <div className="wp-readout" aria-live="polite">
            <div className="wp-readout-stat">
              <div className="wp-readout-label">Win probability</div>
              <div className="wp-readout-value" style={{ color: leadingColor }}>
                {leadingTeam} {leadingPct.toFixed(1)}%
              </div>
              <div className="wp-readout-time">
                {formatQuarter(qtr)} {formatClock(secsLeft)}
                {hasScore && ` · ${game.away} ${awayScore}–${game.home} ${homeScore}`}
              </div>
            </div>
            <div className="wp-readout-desc">
              {downDistanceText && <span className="wp-down-distance">{downDistanceText}</span>}
              {downDistanceText && " — "}
              {desc || "(No play description)"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
