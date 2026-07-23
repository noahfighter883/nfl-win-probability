"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import "./WinProbabilityReplay.css";

// Real team colors, used only for the chart fill split and readout text.
const TEAM_COLORS = {
  MIN: "#4F2683",
  IND: "#002C5F",
  KC: "#E31837",
  CIN: "#FB4F14",
  BUF: "#00338D",
  SF: "#AA0000",
};

const TEAM_NAMES = {
  MIN: "Vikings",
  IND: "Colts",
  KC: "Chiefs",
  CIN: "Bengals",
  BUF: "Bills",
  SF: "49ers",
};

const GOLD = "#E8B94A";
const CHART_W = 1000;
const CHART_H = 320;
const MID = CHART_H / 2;

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
 *     plays: [ [qtr, secondsRemaining, homeWinProbPct, playDescription], ... ]
 *   }
 * }
 *
 * Fetch this from /public/data/games_data.json and pass it in, or fetch
 * inside a parent server component and pass down as a prop.
 */
export default function WinProbabilityReplay({ gamesData }) {
  const gameIds = useMemo(() => Object.keys(gamesData || {}), [gamesData]);
  const [currentGameId, setCurrentGameId] = useState(gameIds[0] || null);
  const [playIndex, setPlayIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef(null);
  const svgRef = useRef(null);

  const game = currentGameId ? gamesData[currentGameId] : null;
  const plays = game?.plays || [];
  const n = plays.length;

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

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (n === 0) return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        stopPlay();
        setPlayIndex((prev) => Math.min(n - 1, prev + 1));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        stopPlay();
        setPlayIndex((prev) => Math.max(0, prev - 1));
      } else if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [n, stopPlay, togglePlay]);

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
  const [qtr, secsLeft, homeWp, desc] = current || [1, 3600, 50, ""];
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
          const g = gamesData[id];
          const [, ...ctx] = (g.label || "").split(" — ");
          return (
            <button
              key={id}
              onClick={() => selectGame(id)}
              className={`wp-game-btn ${id === currentGameId ? "active" : ""}`}
              aria-pressed={id === currentGameId}
            >
              <div className="wp-matchup">
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
            <div className="wp-game-title">{titlePart}</div>
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
            <button
              className="wp-play-btn"
              onClick={togglePlay}
              aria-label={isPlaying ? "Pause" : "Play"}
              aria-pressed={isPlaying}
            >
              {isPlaying ? "❚❚" : "▶"}
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
                Q{qtr} {formatClock(secsLeft)}
              </div>
            </div>
            <div className="wp-readout-desc">{desc || "(No play description)"}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
