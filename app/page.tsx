"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Line,
} from "recharts";

// -----------------------------
// Types
// -----------------------------
type PosteriorBundle = {
  CAP: number;
  age_mean: number;
  age_std: number;
  param_names: string[];
  draws: number[][];
};

type PredictInputs = {
  age: number;
  sex01: 0 | 1;         // 0=Male, 1=Female
  tMinutes: number;     // time in minutes
  useTime: boolean;     // true => m=1 (uses t); false => m=0 (ignores t)
};

type PredictResult = {
  mean: number;         // probability
  lo95: number;
  hi95: number;
};

// -----------------------------
// Helpers (math utilities)
// -----------------------------
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

const lnN = (x: number, mu: number, sig: number) => {
  const s = Math.max(sig, 1e-12);
  const z = (x - mu) / s;
  return -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * z * z;
};

const qtile = (arr: number[], q: number) => {
  if (arr.length === 0) return NaN;
  const a = [...arr].sort((x, y) => x - y);
  const pos = clamp((a.length - 1) * q, 0, a.length - 1);
  const base = Math.floor(pos);
  const frac = pos - base;
  if (base + 1 < a.length) return a[base] * (1 - frac) + a[base + 1] * frac;
  return a[base];
};

// -----------------------------
// Parameter indexing (robust to different exports)
// We support both 8-parameter and 10-parameter posterior.json
// -----------------------------
function makeIndex(names: string[]) {
  const idx = (k: string) => names.indexOf(k);
  const i = {
    rho0: idx("rho0"),
    rho1: idx("rho1"),
    mu0: idx("mu0"),
    mu1: idx("mu1"),
    s0: idx("sigma0"),
    s1: idx("sigma1"),
    bAge: idx("beta_age"),
    bAgeMiss: idx("beta_age_miss"),
    bSex: idx("beta_sex"),
    bSexMiss: idx("beta_sex_miss"),
  };
  return i;
}

function linCov(
  draw: number[],
  idx: ReturnType<typeof makeIndex>,
  ageStd: number,
  sex01: number,
  ageMiss = 0,
  sexMiss = 0
) {
  // Missing-indicator coefficients may not exist in older JSONs → treat as 0
  const bAge = idx.bAge >= 0 ? draw[idx.bAge] : 0;
  const bAgeMiss = idx.bAgeMiss >= 0 ? draw[idx.bAgeMiss] : 0;
  const bSex = idx.bSex >= 0 ? draw[idx.bSex] : 0;
  const bSexMiss = idx.bSexMiss >= 0 ? draw[idx.bSexMiss] : 0;
  return bAge * ageStd + bAgeMiss * ageMiss + bSex * sex01 + bSexMiss * sexMiss;
}

// Core prediction for one draw
function predictOne(
  bundle: PosteriorBundle,
  draw: number[],
  idx: ReturnType<typeof makeIndex>,
  inputs: PredictInputs
) {
  const CAP = bundle.CAP ?? 300;
  const t = clamp(inputs.tMinutes, 0, CAP);
  const ageStd =
    bundle.age_std > 0
      ? (inputs.age - bundle.age_mean) / bundle.age_std
      : inputs.age - bundle.age_mean;

  // Params (clip probabilities for stability)
  const rho0 = Math.min(Math.max(draw[idx.rho0], 1e-12), 1 - 1e-12);
  const rho1 = Math.min(Math.max(draw[idx.rho1], 1e-12), 1 - 1e-12);
  const mu0 = draw[idx.mu0];
  const mu1 = draw[idx.mu1];
  const s0 = Math.max(draw[idx.s0], 1e-12);
  const s1 = Math.max(draw[idx.s1], 1e-12);

  const linear = linCov(draw, idx, ageStd, inputs.sex01);

  let logit: number;
  if (inputs.useTime) {
    // m = 1 : logit = log(rho1) - log(rho0) + [lnN(t|mu1,s1) - lnN(t|mu0,s0)] + linear
    logit = Math.log(rho1) - Math.log(rho0) + (lnN(t, mu1, s1) - lnN(t, mu0, s0)) + linear;
  } else {
    // m = 0 : logit = log(1 - rho1) - log(1 - rho0) + linear
    logit = Math.log(1 - rho1) - Math.log(1 - rho0) + linear;
  }
  const p = 1 / (1 + Math.exp(-logit));
  return clamp(p, 1e-12, 1 - 1e-12);
}

function predictMany(bundle: PosteriorBundle, inputs: PredictInputs): PredictResult {
  const names = bundle.param_names ?? [];
  const idx = makeIndex(names);

  // Sanity check
  const required = [idx.rho0, idx.rho1, idx.mu0, idx.mu1, idx.s0, idx.s1];
  if (required.some((j) => j < 0)) {
    return { mean: NaN, lo95: NaN, hi95: NaN };
  }

  const probs: number[] = [];
  for (const d of bundle.draws) {
    probs.push(predictOne(bundle, d, idx, inputs));
  }
  const mean = probs.reduce((a, b) => a + b, 0) / Math.max(1, probs.length);
  const lo95 = qtile(probs, 0.025);
  const hi95 = qtile(probs, 0.975);
  return { mean, lo95, hi95 };
}

function curveOverTime(
  bundle: PosteriorBundle,
  inputs: Omit<PredictInputs, "tMinutes">,
  points = 120
) {
  // Only meaningful for useTime=true (m=1)
  const CAP = bundle.CAP ?? 300;
  const xs = Array.from({ length: points }, (_, i) => (i * CAP) / (points - 1));
  const names = bundle.param_names ?? [];
  const idx = makeIndex(names);
  const required = [idx.rho0, idx.rho1, idx.mu0, idx.mu1, idx.s0, idx.s1];
  if (required.some((j) => j < 0)) return [];

  // Precompute ageStd once
  const ageStd =
    bundle.age_std > 0
      ? (inputs.age - bundle.age_mean) / bundle.age_std
      : inputs.age - bundle.age_mean;

  // For speed, compute per-draw constants
  const draws = bundle.draws;
  const D = draws.length;
  const mean: number[] = [];
  const lo: number[] = [];
  const hi: number[] = [];

  for (const t of xs) {
    const probs: number[] = [];
    for (let d = 0; d < D; d++) {
      const row = draws[d];
      const rho0 = clamp(row[idx.rho0], 1e-12, 1 - 1e-12);
      const rho1 = clamp(row[idx.rho1], 1e-12, 1 - 1e-12);
      const mu0 = row[idx.mu0];
      const mu1 = row[idx.mu1];
      const s0 = Math.max(row[idx.s0], 1e-12);
      const s1 = Math.max(row[idx.s1], 1e-12);
      const linear = linCov(row, idx, ageStd, inputs.sex01);
      const logit =
        Math.log(rho1) - Math.log(rho0) + (lnN(t, mu1, s1) - lnN(t, mu0, s0)) + linear;
      probs.push(1 / (1 + Math.exp(-logit)));
    }
    mean.push(probs.reduce((a, b) => a + b, 0) / Math.max(1, D));
    lo.push(qtile(probs, 0.025));
    hi.push(qtile(probs, 0.975));
  }

  return xs.map((x, i) => ({
    t: Number(x.toFixed(2)),
    mean: mean[i],
    lo: lo[i],
    hi: hi[i],
  }));
}

// -----------------------------
// UI
// -----------------------------
export default function Page() {
  const [bundle, setBundle] = useState<PosteriorBundle | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  const [age, setAge] = useState(60);
  const [sex01, setSex01] = useState<0 | 1>(1);
  const [mode, setMode] = useState<"with-time" | "no-time">("with-time");
  const [tMinutes, setTMinutes] = useState(60);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        setLoadErr(null);
        const res = await fetch("/posterior.json", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as PosteriorBundle;
        if (!data || !Array.isArray(data.draws) || data.draws.length === 0) {
          throw new Error("posterior.json has no draws");
        }
        setBundle(data);
      } catch (e: any) {
        setLoadErr(e?.message ?? "Failed to load posterior.json");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const useTime = mode === "with-time";

  const prediction = useMemo(() => {
    if (!bundle) return null;
    return predictMany(bundle, { age, sex01, tMinutes, useTime });
  }, [bundle, age, sex01, tMinutes, useTime]);

  const chartData = useMemo(() => {
    if (!bundle || !useTime) return [];
    return curveOverTime(bundle, { age, sex01, useTime: true });
  }, [bundle, age, sex01, useTime]);

  const cap = bundle?.CAP ?? 300;

  return (
    <main className="min-h-dvh bg-white text-zinc-900">
      <div className="mx-auto max-w-5xl px-4 py-10">
        <h1 className="text-2xl md:text-3xl font-semibold mb-6">
          Bayesian Risk Predictor (Web)
        </h1>

        {/* Load state */}
        {loading && (
          <div className="rounded-2xl border p-6 mb-6">
            <p>Loading posterior…</p>
          </div>
        )}
        {loadErr && (
          <div className="rounded-2xl border border-red-300 bg-red-50 p-6 mb-6">
            <p className="font-medium">Failed to load posterior.json</p>
            <p className="text-sm mt-1">
              {loadErr}. Place <code>posterior.json</code> in <code>/public</code> and reload.
            </p>
          </div>
        )}

        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="rounded-2xl border p-4">
            <label className="block text-sm font-medium mb-2">Age</label>
            <input
              type="number"
              min={0}
              max={120}
              value={age}
              onChange={(e) => setAge(Number(e.target.value))}
              className="w-full rounded-xl border px-3 py-2"
            />
          </div>

          <div className="rounded-2xl border p-4">
            <label className="block text-sm font-medium mb-2">Sex</label>
            <div className="flex gap-3">
              <label className="inline-flex items-center gap-2">
                <input
                  type="radio"
                  name="sex"
                  checked={sex01 === 0}
                  onChange={() => setSex01(0)}
                />
                <span>Male</span>
              </label>
              <label className="inline-flex items-center gap-2">
                <input
                  type="radio"
                  name="sex"
                  checked={sex01 === 1}
                  onChange={() => setSex01(1)}
                />
                <span>Female</span>
              </label>
            </div>
          </div>

          <div className="rounded-2xl border p-4">
            <label className="block text-sm font-medium mb-2">Mode</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as any)}
              className="w-full rounded-xl border px-3 py-2"
            >
              <option value="with-time">m = 1 (uses time)</option>
              <option value="no-time">m = 0 (ignores time)</option>
            </select>
          </div>
        </div>

        {/* Time control (only when m=1) */}
        {useTime && (
          <div className="rounded-2xl border p-4 mb-6">
            <label className="block text-sm font-medium mb-2">
              Time (minutes) — capped at {cap}
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={0}
                max={cap}
                step={1}
                value={tMinutes}
                onChange={(e) => setTMinutes(Number(e.target.value))}
                className="w-full"
              />
              <input
                type="number"
                min={0}
                max={cap}
                step={1}
                value={tMinutes}
                onChange={(e) => setTMinutes(Number(e.target.value))}
                className="w-28 rounded-xl border px-3 py-2"
              />
            </div>
          </div>
        )}

        {/* Result */}
        <div className="rounded-2xl border p-5 mb-8">
          <h2 className="text-lg font-semibold mb-3">Prediction</h2>
          {!bundle ? (
            <p className="text-sm text-zinc-600">Waiting for posterior…</p>
          ) : prediction ? (
            <div className="flex flex-wrap items-end gap-6">
              <div>
                <div className="text-sm text-zinc-600">Estimated probability</div>
                <div className="text-3xl font-semibold">
                  {(prediction.mean * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-sm text-zinc-600">95% credible interval</div>
                <div className="text-xl">
                  {(prediction.lo95 * 100).toFixed(1)}% –{" "}
                  {(prediction.hi95 * 100).toFixed(1)}%
                </div>
              </div>
              <div className="text-sm text-zinc-600">
                Mode: <span className="font-medium">{useTime ? "m=1 (with time)" : "m=0 (no time)"}</span>
              </div>
            </div>
          ) : (
            <p className="text-sm text-zinc-600">Enter inputs to see prediction.</p>
          )}
        </div>

        {/* Chart (only m=1) */}
        {useTime && chartData.length > 0 && (
          <div className="rounded-2xl border p-5">
            <h3 className="text-lg font-semibold mb-3">
              Probability vs. Time (posterior mean & 95% band)
            </h3>
            <div className="h-80 w-full">
              <ResponsiveContainer>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="band" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#94a3b8" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="#94a3b8" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="t" tickFormatter={(v) => `${v}`} />
                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={(v) => `${Math.round(v * 100)}%`}
                  />
                  <Tooltip
                    formatter={(val: number, name) =>
                      name === "mean"
                        ? [`${(val * 100).toFixed(1)}%`, "Mean"]
                        : [`${(val * 100).toFixed(1)}%`, name === "hi" ? "Hi (95%)" : "Lo (95%)"]
                    }
                    labelFormatter={(lab) => `t = ${lab} min`}
                  />
                  {/* 95% band */}
                  <Area
                    type="monotone"
                    dataKey="hi"
                    stroke="none"
                    fill="url(#band)"
                    activeDot={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="lo"
                    stroke="none"
                    fill="#fff"
                    activeDot={false}
                  />
                  {/* Mean line */}
                  <Line type="monotone" dataKey="mean" stroke="#0f172a" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        <div className="text-xs text-zinc-500 mt-6">
          Tip: Put your <code>posterior.json</code> in the project’s{" "}
          <code>public/</code> folder. This page expects parameter names like{" "}
          <code>rho0, rho1, mu0, mu1, sigma0, sigma1</code> and will also use{" "}
          <code>beta_age</code>, <code>beta_sex</code> (and their <code>_miss</code> variants)
          if present.
        </div>
      </div>
    </main>
  );
}

