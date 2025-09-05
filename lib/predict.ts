// lib/predict.ts
export type PosteriorBundle = {
  CAP: number
  age_mean: number
  age_std: number
  param_names: string[]
  draws: number[][]
}

export type PredictInput = {
  age?: number | null        // 年齢（未入力可）
  sex01: 0 | 1               // 0=男性, 1=女性
  tMinutes?: number | null   // t（分; m=1時に使う）
  m: 0 | 1                   // 0 or 1
}

export type PredictResult = {
  mean: number
  lo: number
  hi: number
  draws: number[] // 各ドローの確率（必要ならグラフ用）
}

const LN2PI = Math.log(2 * Math.PI)

function lnN(x: number, mu: number, sig: number): number {
  const s = Math.max(sig, 1e-12)
  const z = (x - mu) / s
  return -0.5 * LN2PI - Math.log(s) - 0.5 * z * z
}

function safeLog(p: number): number {
  const q = Math.min(Math.max(p, 1e-12), 1 - 1e-12)
  return Math.log(q)
}

function safeLog1m(p: number): number {
  const q = Math.min(Math.max(1 - p, 1e-12), 1) // 1-p を安全に
  return Math.log(q)
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z)
    return 1 / (1 + e)
  } else {
    const e = Math.exp(z)
    return e / (1 + e)
  }
}

export async function loadPosterior(): Promise<PosteriorBundle> {
  const res = await fetch('/posterior.json', { cache: 'no-store' })
  if (!res.ok) throw new Error('posterior.json を読み込めませんでした')
  return res.json()
}

/** Swiftと同じ式で予測（事後ドローを全て使って平均と95%CI） */
export function predictFromPosterior(bundle: PosteriorBundle, input: PredictInput): PredictResult {
  const { CAP, age_mean, age_std, param_names, draws } = bundle

  // パラメータ名→列index
  const idx = (name: string) => Math.max(0, param_names.indexOf(name))
  const i_rho0 = idx('rho0'), i_rho1 = idx('rho1')
  const i_mu0  = idx('mu0'),  i_mu1  = idx('mu1')
  const i_s0   = idx('sigma0'),i_s1   = idx('sigma1')
  const i_ba   = param_names.indexOf('beta_age')
  const i_bam  = param_names.indexOf('beta_age_miss')
  const i_bs   = param_names.indexOf('beta_sex')
  const i_bsm  = param_names.indexOf('beta_sex_miss')

  const age = input.age ?? null
  const ageStd = (age == null) ? 0 : (age - age_mean) / (age_std > 0 ? age_std : 1)
  const ageMiss = (age == null) ? 1 : 0
  const sex01 = input.sex01
  const sexMiss = 0 // Web UIで必ず選ばせる前提

  // t は m=1 のときだけ使う。CAPでクリップ
  const t = Math.min(Math.max(input.tMinutes ?? 0, 0), CAP)
  const useTime = input.m === 1

  const probs: number[] = []

  for (const row of draws) {
    const rho0 = row[i_rho0], rho1 = row[i_rho1]
    const mu0  = row[i_mu0],  mu1  = row[i_mu1]
    const s0   = row[i_s0],   s1   = row[i_s1]

    // 係数は無い場合 0 扱い（古いJSONでも動く）
    const b_age      = (i_ba  >= 0) ? row[i_ba]  : 0
    const b_age_miss = (i_bam >= 0) ? row[i_bam] : 0
    const b_sex      = (i_bs  >= 0) ? row[i_bs]  : 0
    const b_sex_miss = (i_bsm >= 0) ? row[i_bsm] : 0

    const linear =
      b_age * ageStd +
      b_age_miss * ageMiss +
      b_sex * sex01 +
      b_sex_miss * sexMiss

    let logit: number
    if (useTime) {
      // m=1: log rho1 - log rho0 + logN1 - logN0 + linear
      logit =
        safeLog(rho1) - safeLog(rho0) +
        (lnN(t, mu1, s1) - lnN(t, mu0, s0)) +
        linear
    } else {
      // m=0: log(1-rho1) - log(1-rho0) + linear
      logit = safeLog1m(rho1) - safeLog1m(rho0) + linear
    }

    probs.push(sigmoid(logit))
  }

  probs.sort((a, b) => a - b)
  const mean = probs.reduce((s, v) => s + v, 0) / probs.length
  const lo   = probs[Math.floor(0.025 * (probs.length - 1))]
  const hi   = probs[Math.floor(0.975 * (probs.length - 1))]

  return { mean, lo, hi, draws: probs }
}

/** 曲線描画用：m=1想定でtを変化させたときの平均＆95%CIを返す */
export function predictCurve(
  bundle: PosteriorBundle,
  base: Omit<PredictInput, 'tMinutes' | 'm'> & { m?: 1 },
  tGrid: number[],
) {
  const rows = tGrid.map(t =>
    predictFromPosterior(bundle, { ...base, tMinutes: t, m: 1 })
  )
  return rows.map((r, i) => ({
    t: tGrid[i],
    mean: r.mean,
    lo: r.lo,
    hi: r.hi,
  }))
}

