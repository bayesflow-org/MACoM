"""Triton (GPU) simulators for DDM, RDM, CDM, LBA and aDDM.

Same update, boundary and output conventions as the numba serial kernels in ddm.py,
rdm.py and cdm.py (constant thresholds, i.e. decay == 0); only the random stream
differs. One lane per trial; each program advances BLOCK trials UNROLL steps at a time
and exits once all of its trials have terminated.

Randomness is Philox (``tl.randn``/``tl.rand``) keyed by ``seed + k`` with the trial
(and accumulator) index as the counter: keys 0-2 serve the per-trial draws, keys from
3 upward one per step (group), so every (trial, step, accumulator) draw is distinct
within a launch. A fresh 62-bit seed per launch keeps launches independent.

RTs are formed in float64 (``tau_i + k * dt``) and rounded once to float32, exactly
like the numba kernels (float64 time, float32 store). Forming them in float32 lands up
to 1 ulp off numba's value on the ``tau + k*dt`` lattice, which a KS test detects when
``s_tau = 0``.

Validated against numba (2026-09-28): see experiments/check_gpu_simulators.py.
"""
from __future__ import annotations

import os

import numpy as np

_BLOCK = 32     # one warp per program: a block exits when its slowest trial does
_UNROLL = 16

_kernels = None


def _device():
    rank = os.environ.get("LOCAL_RANK")
    return f"cuda:{rank}" if rank is not None else "cuda"


def _seed() -> int:
    # Drawn from numpy's global stream so np.random.seed() makes GPU runs repeatable.
    return int(np.random.randint(0, 2**62, dtype=np.int64))


def _build():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["seed"])
    def ddm_kernel(v_ptr, a_ptr, z_ptr, tau_ptr, sv_ptr, stau_ptr, out_ptr, dt64_ptr,
                   n, seed, dt, c, max_steps,
                   BLOCK: tl.constexpr, UNROLL: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offs < n
        v = tl.load(v_ptr + offs, mask=valid, other=0.0)
        a = tl.load(a_ptr + offs, mask=valid, other=1.0)
        z = tl.load(z_ptr + offs, mask=valid, other=0.0)
        tau = tl.load(tau_ptr + offs, mask=valid, other=0.0)          # float64
        s_v = tl.load(sv_ptr + offs, mask=valid, other=0.0)
        s_tau = tl.load(stau_ptr + offs, mask=valid, other=0.0)       # float64

        drift = (v + s_v * tl.randn(seed, offs)) * dt
        tau_i = tau + s_tau * tl.rand(seed + 2, offs).to(tl.float64)
        a0 = tl.maximum(a, 1e-6)
        x = z * a0

        done = ~valid
        hit = tl.full([BLOCK], 0, tl.int32)        # 1-based: numba advances t before the update
        choice = tl.full([BLOCK], -1.0, tl.float32)
        step = 0
        remaining = tl.sum(valid.to(tl.int32), axis=0)
        while (step < max_steps) & (remaining > 0):
            for j in tl.static_range(UNROLL // 4):
                e0, e1, e2, e3 = tl.randn4x(seed + 3 + step // 4 + j, offs)
                for q in tl.static_range(4):
                    if q == 0:
                        eps = e0
                    elif q == 1:
                        eps = e1
                    elif q == 2:
                        eps = e2
                    else:
                        eps = e3
                    x = tl.where(done, x, x + drift + c * eps)
                    up = (~done) & (x >= a0)
                    lo = (~done) & (x <= -a0)
                    hit = tl.where(up | lo, step + j * 4 + q + 1, hit)
                    choice = tl.where(up, 1.0, tl.where(lo, 0.0, choice))
                    done = done | up | lo
            step += UNROLL
            remaining = tl.sum((~done).to(tl.int32), axis=0)

        dt64 = tl.load(dt64_ptr)
        rt = tl.where(hit > 0, (tau_i + hit.to(tl.float64) * dt64).to(tl.float32), -1.0)
        tl.store(out_ptr + offs * 2, rt, mask=valid)
        tl.store(out_ptr + offs * 2 + 1, choice, mask=valid)

    @triton.jit(do_not_specialize=["seed"])
    def rdm_kernel(v_ptr, m_ptr, a_ptr, tau_ptr, sv_ptr, stau_ptr, out_ptr, dt64_ptr,
                   n, seed, dt, c, max_steps,
                   MAXM: tl.constexpr, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        acc = tl.arange(0, MAXM)
        valid = offs < n
        m = tl.load(m_ptr + offs, mask=valid, other=2)
        live = valid[:, None] & (acc[None, :] < m[:, None])            # (BLOCK, MAXM)
        cnt = offs[:, None] * MAXM + acc[None, :]
        v = tl.load(v_ptr + cnt, mask=live, other=0.0)
        a = tl.load(a_ptr + offs, mask=valid, other=1.0)
        tau = tl.load(tau_ptr + offs, mask=valid, other=0.0)          # float64
        s_v = tl.load(sv_ptr + offs, mask=valid, other=0.0)
        s_tau = tl.load(stau_ptr + offs, mask=valid, other=0.0)       # float64

        drift = (v + s_v[:, None] * tl.randn(seed, cnt)) * dt
        tau_i = tau + s_tau * tl.rand(seed + 2, offs).to(tl.float64)
        thr = a[:, None]
        X = tl.zeros([BLOCK, MAXM], tl.float32)

        done = ~valid
        hit = tl.full([BLOCK], -1, tl.int32)       # 0-based: rt = tau_i + i_iter * dt
        choice = tl.full([BLOCK], -1, tl.int32)
        step = 0
        remaining = tl.sum(valid.to(tl.int32), axis=0)
        while (step < max_steps) & (remaining > 0):
            for j in tl.static_range(UNROLL):
                eps = tl.randn(seed + 3 + step + j, cnt)
                act = live & (~done)[:, None]
                X = tl.where(act, X + drift + c * eps, X)
                crossed = act & (X >= thr)
                # Lowest crossing index wins -- numba updates accumulators in order and
                # stops at the first crossing, so later accumulators never get to cross.
                first = tl.min(tl.where(crossed, acc[None, :], MAXM), axis=1)
                now = first < MAXM
                hit = tl.where(now, step + j, hit)
                choice = tl.where(now, first, choice)
                done = done | now
            step += UNROLL
            remaining = tl.sum((~done).to(tl.int32), axis=0)

        dt64 = tl.load(dt64_ptr)
        ok = hit >= 0
        rt = tl.where(ok, (tau_i + hit.to(tl.float64) * dt64).to(tl.float32), -1.0)
        tl.store(out_ptr + offs * 2, rt, mask=valid)
        tl.store(out_ptr + offs * 2 + 1, tl.where(ok, choice.to(tl.float32), -1.0), mask=valid)

    @triton.jit
    def _rdm_acc(X, d, eps, c, thr, act, q, step_idx, hit, choice, done):
        # One accumulator's update + crossing check, in numba's order: an accumulator
        # is skipped once an earlier one has crossed this step (act excludes done).
        X = tl.where(act, X + d + c * eps, X)
        crossed = act & (X >= thr)
        hit = tl.where(crossed, step_idx, hit)
        choice = tl.where(crossed, q, choice)
        return X, hit, choice, done | crossed

    @triton.jit(do_not_specialize=["seed"])
    def rdm4_kernel(v_ptr, m_ptr, a_ptr, tau_ptr, sv_ptr, stau_ptr, out_ptr, dt64_ptr,
                    n, seed, dt, c, max_steps,
                    MAXM: tl.constexpr, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
        # Specialization for <= 4 accumulators (the training range): accumulators in
        # separate registers and one randn4x per step (per 2 steps when MAXM == 2),
        # instead of the generic kernel's (BLOCK, MAXM) tile + per-step reduction.
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offs < n
        m = tl.load(m_ptr + offs, mask=valid, other=2)
        a = tl.load(a_ptr + offs, mask=valid, other=1.0)
        tau = tl.load(tau_ptr + offs, mask=valid, other=0.0)          # float64
        s_v = tl.load(sv_ptr + offs, mask=valid, other=0.0)
        s_tau = tl.load(stau_ptr + offs, mask=valid, other=0.0)       # float64
        r0, r1, r2, r3 = tl.randn4x(seed, offs)
        d0 = (tl.load(v_ptr + offs * MAXM, mask=valid, other=0.0) + s_v * r0) * dt
        d1 = (tl.load(v_ptr + offs * MAXM + 1, mask=valid, other=0.0) + s_v * r1) * dt
        if MAXM == 4:
            d2 = (tl.load(v_ptr + offs * MAXM + 2, mask=valid & (m > 2), other=0.0) + s_v * r2) * dt
            d3 = (tl.load(v_ptr + offs * MAXM + 3, mask=valid & (m > 3), other=0.0) + s_v * r3) * dt
        tau_i = tau + s_tau * tl.rand(seed + 2, offs).to(tl.float64)
        X0 = tl.zeros([BLOCK], tl.float32)
        X1 = tl.zeros([BLOCK], tl.float32)
        X2 = tl.zeros([BLOCK], tl.float32)
        X3 = tl.zeros([BLOCK], tl.float32)

        done = ~valid
        hit = tl.full([BLOCK], -1, tl.int32)       # 0-based: rt = tau_i + i_iter * dt
        choice = tl.full([BLOCK], -1, tl.int32)
        step = 0
        remaining = tl.sum(valid.to(tl.int32), axis=0)
        while (step < max_steps) & (remaining > 0):
            if MAXM == 4:
                for j in tl.static_range(UNROLL):
                    e0, e1, e2, e3 = tl.randn4x(seed + 3 + step + j, offs)
                    k = step + j
                    X0, hit, choice, done = _rdm_acc(X0, d0, e0, c, a, ~done, 0, k, hit, choice, done)
                    X1, hit, choice, done = _rdm_acc(X1, d1, e1, c, a, ~done, 1, k, hit, choice, done)
                    X2, hit, choice, done = _rdm_acc(X2, d2, e2, c, a, ~done & (m > 2), 2, k, hit, choice, done)
                    X3, hit, choice, done = _rdm_acc(X3, d3, e3, c, a, ~done & (m > 3), 3, k, hit, choice, done)
            else:
                for j in tl.static_range(UNROLL // 2):
                    e0, e1, e2, e3 = tl.randn4x(seed + 3 + step // 2 + j, offs)
                    k = step + 2 * j
                    X0, hit, choice, done = _rdm_acc(X0, d0, e0, c, a, ~done, 0, k, hit, choice, done)
                    X1, hit, choice, done = _rdm_acc(X1, d1, e1, c, a, ~done, 1, k, hit, choice, done)
                    X0, hit, choice, done = _rdm_acc(X0, d0, e2, c, a, ~done, 0, k + 1, hit, choice, done)
                    X1, hit, choice, done = _rdm_acc(X1, d1, e3, c, a, ~done, 1, k + 1, hit, choice, done)
            step += UNROLL
            remaining = tl.sum((~done).to(tl.int32), axis=0)

        dt64 = tl.load(dt64_ptr)
        ok = hit >= 0
        rt = tl.where(ok, (tau_i + hit.to(tl.float64) * dt64).to(tl.float32), -1.0)
        tl.store(out_ptr + offs * 2, rt, mask=valid)
        tl.store(out_ptr + offs * 2 + 1, tl.where(ok, choice.to(tl.float32), -1.0), mask=valid)

    @triton.jit(do_not_specialize=["seed"])
    def cdm_kernel(v_ptr, a_ptr, tau_ptr, sv_ptr, stau_ptr, out_ptr, dt64_ptr,
                   n, seed, dt, c, max_steps,
                   BLOCK: tl.constexpr, UNROLL: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offs < n
        v0 = tl.load(v_ptr + offs * 2, mask=valid, other=0.0)
        v1 = tl.load(v_ptr + offs * 2 + 1, mask=valid, other=0.0)
        a = tl.load(a_ptr + offs, mask=valid, other=1.0)
        tau = tl.load(tau_ptr + offs, mask=valid, other=0.0)          # float64
        s_v = tl.load(sv_ptr + offs, mask=valid, other=0.0)
        s_tau = tl.load(stau_ptr + offs, mask=valid, other=0.0)       # float64

        r0, r1, _r2, _r3 = tl.randn4x(seed, offs)
        d0 = (v0 + s_v * r0) * dt
        d1 = (v1 + s_v * r1) * dt
        tau_i = tau + s_tau * tl.rand(seed + 2, offs).to(tl.float64)
        thr_sq = a * a
        x0 = tl.zeros([BLOCK], tl.float32)
        x1 = tl.zeros([BLOCK], tl.float32)
        hx0 = tl.zeros([BLOCK], tl.float32)
        hx1 = tl.zeros([BLOCK], tl.float32)

        done = ~valid
        hit = tl.full([BLOCK], -1, tl.int32)       # 0-based: rt = tau_i + i_iter * dt
        step = 0
        remaining = tl.sum(valid.to(tl.int32), axis=0)
        while (step < max_steps) & (remaining > 0):
            for j in tl.static_range(UNROLL // 2):
                e0, e1, e2, e3 = tl.randn4x(seed + 3 + step // 2 + j, offs)
                for q in tl.static_range(2):
                    if q == 0:
                        n0 = e0
                        n1 = e1
                    else:
                        n0 = e2
                        n1 = e3
                    x0 = tl.where(done, x0, x0 + d0 + c * n0)
                    x1 = tl.where(done, x1, x1 + d1 + c * n1)
                    now = (~done) & (x0 * x0 + x1 * x1 >= thr_sq)
                    hit = tl.where(now, step + j * 2 + q, hit)
                    hx0 = tl.where(now, x0, hx0)
                    hx1 = tl.where(now, x1, hx1)
                    done = done | now
            step += UNROLL
            remaining = tl.sum((~done).to(tl.int32), axis=0)

        dt64 = tl.load(dt64_ptr)
        ok = hit >= 0
        rt = tl.where(ok, (tau_i + hit.to(tl.float64) * dt64).to(tl.float32), -4.0)
        tl.store(out_ptr + offs * 3, rt, mask=valid)
        tl.store(out_ptr + offs * 3 + 1, hx0, mask=valid)
        tl.store(out_ptr + offs * 3 + 2, hx1, mask=valid)

    @triton.jit(do_not_specialize=["seed"])
    def lba_kernel(v_ptr, m_ptr, A_ptr, b_ptr, tau_ptr, sv_ptr, stau_ptr, out_ptr,
                   n, seed, max_time, max_iter,
                   MAXM: tl.constexpr, BLOCK: tl.constexpr):
        # Same as lba.sample_lba_trial: per attempt, every accumulator draws a start
        # point U(0, A) and a drift N(v, s_v); the finishing time of the positive-drift
        # accumulators is (A + b - start) / drift and the earliest wins (lowest index on
        # ties, as numba's strict <). An attempt with no positive drift is redrawn, up
        # to max_iter times; then tau_i = tau + U(0, s_tau), and rt >= max_time or an
        # exhausted attempt budget report [-1, -1].
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        acc = tl.arange(0, MAXM)
        valid = offs < n
        m = tl.load(m_ptr + offs, mask=valid, other=2)
        live = valid[:, None] & (acc[None, :] < m[:, None])
        cnt = offs[:, None] * MAXM + acc[None, :]
        v = tl.load(v_ptr + cnt, mask=live, other=0.0)
        A = tl.load(A_ptr + offs, mask=valid, other=0.0)
        thr = A + tl.load(b_ptr + offs, mask=valid, other=1.0)
        tau = tl.load(tau_ptr + offs, mask=valid, other=0.0)
        s_v = tl.load(sv_ptr + offs, mask=valid, other=0.0)
        s_tau = tl.load(stau_ptr + offs, mask=valid, other=0.0)

        done = ~valid
        best_t = tl.full([BLOCK], float("inf"), tl.float32)
        winner = tl.full([BLOCK], -1, tl.int32)
        it = 0
        remaining = tl.sum(valid.to(tl.int32), axis=0)
        while (it < max_iter) & (remaining > 0):
            start = tl.rand(seed + 3 + 2 * it, cnt) * A[:, None]
            drift = v + s_v[:, None] * tl.randn(seed + 4 + 2 * it, cnt)
            ok = live & (drift > 0)
            t = tl.where(ok, (thr[:, None] - start) / tl.where(ok, drift, 1.0), float("inf"))
            tmin = tl.min(t, axis=1)
            first = tl.min(tl.where(ok & (t == tmin[:, None]), acc[None, :], MAXM), axis=1)
            now = (~done) & (first < MAXM)
            best_t = tl.where(now, tmin, best_t)
            winner = tl.where(now, first, winner)
            done = done | now
            it += 1
            remaining = tl.sum((~done).to(tl.int32), axis=0)

        rt = best_t + tau + s_tau * tl.rand(seed + 2, offs)
        ok = (winner >= 0) & (rt < max_time)
        tl.store(out_ptr + offs * 2, tl.where(ok, rt, -1.0), mask=valid)
        tl.store(out_ptr + offs * 2 + 1, tl.where(ok, winner.to(tl.float32), -1.0), mask=valid)

    @triton.jit(do_not_specialize=["seed"])
    def addm_kernel(v0_ptr, v1_ptr, om_ptr, a_ptr, tau_ptr, sv_ptr, stau_ptr, mfd_ptr, mode_ptr,
                    out_ptr, dt64_ptr, n, seed, dt, c, max_steps,
                    BLOCK: tl.constexpr, UNROLL: tl.constexpr):
        # Same as addm.sample_addm_trial: one relative accumulator on [-a, a] whose drift
        # depends on the fixated item (v0 - omega*v1 on item 0, omega*v0 - v1 on item 1).
        # Fixations alternate with durations int(Exp(mean_fix_dur) / dt) >= 1 steps; the
        # switch is checked BEFORE each step's update. Upper bound = choice 0.
        # Each lane's k-th fixation duration uses Philox counter (trial << 20) + k, so
        # lanes switching at different steps still draw distinct numbers.
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offs < n
        v0 = tl.load(v0_ptr + offs, mask=valid, other=0.0)
        v1 = tl.load(v1_ptr + offs, mask=valid, other=0.0)
        om = tl.load(om_ptr + offs, mask=valid, other=1.0)
        a = tl.load(a_ptr + offs, mask=valid, other=1.0)
        tau = tl.load(tau_ptr + offs, mask=valid, other=0.0)          # float64
        s_v = tl.load(sv_ptr + offs, mask=valid, other=0.0)
        s_tau = tl.load(stau_ptr + offs, mask=valid, other=0.0)       # float64
        mfd = tl.load(mfd_ptr + offs, mask=valid, other=1.0)
        mode = tl.load(mode_ptr + offs, mask=valid, other=0)

        r0, r1, _r2, _r3 = tl.randn4x(seed, offs)
        v0i = v0 + s_v * r0
        v1i = v1 + s_v * r1
        tau_i = tau + s_tau * tl.rand(seed + 2, offs).to(tl.float64)
        u_first = tl.rand(seed + 1, offs)
        fix = tl.where(mode == 1, 0, tl.where(mode == 2, 1, tl.where(u_first < 0.5, 0, 1)))
        sw_cnt = offs.to(tl.int64) << 20
        nsw = tl.zeros([BLOCK], tl.int64)
        u = tl.rand(seed + 3, sw_cnt + nsw)
        left = tl.maximum((-mfd * tl.log(1.0 - u) / dt).to(tl.int32), 1)
        d_on0 = (v0i - om * v1i) * dt
        d_on1 = (om * v0i - v1i) * dt
        x = tl.zeros([BLOCK], tl.float32)

        done = ~valid
        hit = tl.full([BLOCK], 0, tl.int32)        # 1-based: rt = tau_i + (step + 1) * dt
        choice = tl.full([BLOCK], -1.0, tl.float32)
        step = 0
        remaining = tl.sum(valid.to(tl.int32), axis=0)
        while (step < max_steps) & (remaining > 0):
            for j in tl.static_range(UNROLL // 4):
                e0, e1, e2, e3 = tl.randn4x(seed + 4 + step // 4 + j, offs)
                for q in tl.static_range(4):
                    if q == 0:
                        eps = e0
                    elif q == 1:
                        eps = e1
                    elif q == 2:
                        eps = e2
                    else:
                        eps = e3
                    switch = (~done) & (left <= 0)
                    nsw = tl.where(switch, nsw + 1, nsw)
                    u = tl.rand(seed + 3, sw_cnt + nsw)
                    fix = tl.where(switch, 1 - fix, fix)
                    left = tl.where(switch, tl.maximum((-mfd * tl.log(1.0 - u) / dt).to(tl.int32), 1), left)
                    x = tl.where(done, x, x + tl.where(fix == 0, d_on0, d_on1) + c * eps)
                    left = left - 1
                    up = (~done) & (x >= a)
                    lo = (~done) & (x <= -a)
                    hit = tl.where(up | lo, step + j * 4 + q + 1, hit)
                    choice = tl.where(up, 0.0, tl.where(lo, 1.0, choice))
                    done = done | up | lo
            step += UNROLL
            remaining = tl.sum((~done).to(tl.int32), axis=0)

        dt64 = tl.load(dt64_ptr)
        rt = tl.where(hit > 0, (tau_i + hit.to(tl.float64) * dt64).to(tl.float32), -1.0)
        tl.store(out_ptr + offs * 2, rt, mask=valid)
        tl.store(out_ptr + offs * 2 + 1, choice, mask=valid)

    return {"ddm": ddm_kernel, "rdm": rdm_kernel, "rdm4": rdm4_kernel, "cdm": cdm_kernel,
            "lba": lba_kernel, "addm": addm_kernel}


def _get(name):
    global _kernels
    if _kernels is None:
        _kernels = _build()
    return _kernels[name]


def _prep(max_steps):
    if max_steps % _UNROLL:
        raise ValueError(f"max_steps must be a multiple of {_UNROLL} for the GPU kernels")


def _t(x, dtype, device):
    import torch
    return torch.from_numpy(np.ascontiguousarray(x, dtype=dtype)).to(device, non_blocking=True)


def _launch(kernel, n, args, dt, sigma, max_steps, device, out_cols, **meta):
    import torch
    import triton
    out = torch.empty((n, out_cols), dtype=torch.float32, device=device)
    if n == 0:
        return out.cpu().numpy()
    dt64 = torch.tensor([dt], dtype=torch.float64, device=device)
    kernel[(triton.cdiv(n, _BLOCK),)](
        *args, out, dt64, n, _seed(), float(dt), float(sigma * np.sqrt(dt)), int(max_steps),
        BLOCK=_BLOCK, UNROLL=_UNROLL, num_warps=1, **meta)
    return out.cpu().numpy()


def simulate_ddm_gpu(v, a, z, tau, s_v, s_tau, sigma=1.0, dt=5e-4, max_steps=20000,
                     device=None):
    """GPU twin of ddm.simulate_ddm (constant bounds). Returns (n, 2) float32 [rt, choice]."""
    _prep(max_steps)
    device = device or _device()
    f32, f64 = np.float32, np.float64
    args = (_t(v, f32, device), _t(a, f32, device), _t(z, f32, device), _t(tau, f64, device),
            _t(s_v, f32, device), _t(s_tau, f64, device))
    return _launch(_get("ddm"), len(v), args, dt, sigma, max_steps, device, 2)


def simulate_rdm_gpu(v, a, tau, s_v, s_tau, sigma=1.0, dt=5e-4, max_steps=20000,
                     num_alternatives=None, device=None):
    """GPU twin of rdm.simulate_rdm (decay == 0).

    v : (n, K) drifts. To batch datasets with different K in one launch, pad v to the
    largest K and pass the per-trial ``num_alternatives`` (accumulators beyond it are
    ignored). Returns (n, 2) float32 [rt, choice].
    """
    _prep(max_steps)
    device = device or _device()
    vp, m, maxm = _pad_accumulators(v, num_alternatives)
    n = vp.shape[0]
    f32, f64 = np.float32, np.float64
    args = (_t(vp, f32, device), _t(m, np.int32, device), _t(a, f32, device), _t(tau, f64, device),
            _t(s_v, f32, device), _t(s_tau, f64, device))
    kernel = _get("rdm4") if maxm <= 4 else _get("rdm")
    return _launch(kernel, n, args, dt, sigma, max_steps, device, 2, MAXM=maxm)


def simulate_cdm_gpu(v, a, tau, s_v, s_tau, sigma=1.0, dt=5e-4, max_steps=20000, device=None):
    """GPU twin of cdm.simulate_cdm (decay == 0). Returns (n, 2) float32 [rt, angle]."""
    _prep(max_steps)
    device = device or _device()
    f32, f64 = np.float32, np.float64
    args = (_t(np.asarray(v).reshape(-1, 2), f32, device), _t(a, f32, device), _t(tau, f64, device),
            _t(s_v, f32, device), _t(s_tau, f64, device))
    raw = _launch(_get("cdm"), len(a), args, dt, sigma, max_steps, device, 3)
    out = np.empty((raw.shape[0], 2), dtype=np.float32)
    out[:, 0] = raw[:, 0]
    finished = raw[:, 0] != -4.0
    out[:, 1] = np.where(finished,
                         np.arctan2(raw[:, 2].astype(np.float64), raw[:, 1].astype(np.float64)),
                         -4.0)
    return out


def _pad_accumulators(v, num_alternatives):
    v = np.asarray(v)
    n, k = v.shape
    if k > 8:
        raise ValueError("GPU kernels support at most 8 accumulators")
    maxm = 2 if k <= 2 else 4 if k <= 4 else 8
    vp = np.zeros((n, maxm), dtype=np.float32)
    vp[:, :k] = v
    m = np.full(n, k, dtype=np.int32) if num_alternatives is None else np.asarray(num_alternatives, dtype=np.int32)
    return vp, m, maxm


def simulate_lba_gpu(v, A, b, tau, s_v, s_tau, max_time=10, max_iter=1000,
                     num_alternatives=None, device=None):
    """GPU twin of lba.simulate_lba. v : (n, K), padded + per-trial ``num_alternatives``
    to batch datasets with different K. Returns (n, 2) float32 [rt, choice]."""
    import torch
    import triton
    device = device or _device()
    vp, m, maxm = _pad_accumulators(v, num_alternatives)
    n = vp.shape[0]
    out = torch.empty((n, 2), dtype=torch.float32, device=device)
    if n == 0:
        return out.cpu().numpy()
    f32 = np.float32
    _get("lba")[(triton.cdiv(n, _BLOCK),)](
        _t(vp, f32, device), _t(m, np.int32, device), _t(A, f32, device), _t(b, f32, device),
        _t(tau, f32, device), _t(s_v, f32, device), _t(s_tau, f32, device), out,
        n, _seed(), float(max_time), int(max_iter), MAXM=maxm, BLOCK=_BLOCK, num_warps=1)
    return out.cpu().numpy()


def simulate_addm_gpu(v_0, v_1, omega, a, tau, s_v, s_tau, mean_fix_dur, first_fix_mode,
                      sigma=1.0, dt=1e-3, max_steps=20000, device=None):
    """GPU twin of addm.simulate_addm. ``first_fix_mode`` may be a scalar or per-trial
    array (addm.FIRST_FIX_*). Returns (n, 2) float32 [rt, choice]."""
    _prep(max_steps)
    device = device or _device()
    n = len(v_0)
    mode = np.array(np.broadcast_to(np.asarray(first_fix_mode, dtype=np.int32), (n,)))
    f32, f64 = np.float32, np.float64
    args = (_t(v_0, f32, device), _t(v_1, f32, device), _t(omega, f32, device), _t(a, f32, device),
            _t(tau, f64, device), _t(s_v, f32, device), _t(s_tau, f64, device),
            _t(mean_fix_dur, f32, device), _t(mode, np.int32, device))
    return _launch(_get("addm"), n, args, dt, sigma, max_steps, device, 2)
