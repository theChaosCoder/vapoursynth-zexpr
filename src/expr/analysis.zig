//! Iterative flat-DAG analysis: clip/coord/prop/spatial usage, compile-time reach,
//! LUT eligibility, `opt=1` value typing (Y15), use counts, and `i16` range analysis (Y1).

const std = @import("std");
const builtin = @import("builtin");
const ast = @import("ast.zig");
const interp = @import("interp.zig");

pub const LutKind = enum(u8) {
    none,
    lut1d_u8,
    lut1d_u16,
    lut2d_u8,
};

pub const ValueType = enum(u8) {
    f32,
    i32,
};

pub const U8Plan = struct {
    eligible: bool = false,
};

pub const U16Plan = struct {
    eligible: bool = false,
};

pub const Int16Plan = struct {
    eligible: bool = false,
    /// Optional final right shift `k` when the root is `expr / 2^k` (`0` = no shift).
    final_div_shift: u5 = 0,
    /// Subtree root evaluated in `i16` before applying `final_div_shift` (equals `prog.root` when shift == 0).
    eval_root: ast.NodeIdx = ast.invalid_node,
};

pub const Analysis = struct {
    used_clips: std.ArrayList(ast.ClipId) = .empty,
    use_counts: std.ArrayList(u32) = .empty,
    value_types: std.ArrayList(ValueType) = .empty,
    uses_x: bool = false,
    uses_y: bool = false,
    uses_n: bool = false,
    uses_width: bool = false,
    uses_height: bool = false,
    uses_props: bool = false,
    uses_rel_access: bool = false,
    uses_abs_access: bool = false,
    uses_f16: bool = false,
    /// Compile-time clamped horizontal/vertical spatial reach (pixels).
    max_left: i32 = 0,
    max_right: i32 = 0,
    max_up: i32 = 0,
    max_down: i32 = 0,
    live_node_count: u32 = 0,
    expr_cost: u32 = 0,
    lut_kind: LutKind = .none,
    lut_clips: [2]ast.ClipId = .{ 0, 0 },
    /// Pure-point LUT fallback used if JIT fails with `ExecMemDenied` (M1), avoiding per-pixel scalar interpreter.
    fallback_lut_kind: LutKind = .none,
    fallback_lut_clips: [2]ast.ClipId = .{ 0, 0 },
    u8_plan: U8Plan = .{},
    u16_plan: U16Plan = .{},
    int16_plan: Int16Plan = .{},

    pub fn deinit(self: *Analysis, allocator: std.mem.Allocator) void {
        self.used_clips.deinit(allocator);
        self.use_counts.deinit(allocator);
        self.value_types.deinit(allocator);
        self.* = .{};
    }
};

fn addUsedClip(allocator: std.mem.Allocator, list: *std.ArrayList(ast.ClipId), clip: ast.ClipId) std.mem.Allocator.Error!void {
    for (list.items) |c| {
        if (c == clip) return;
    }
    try list.append(allocator, clip);
}

pub fn analyze(
    allocator: std.mem.Allocator,
    prog: *const ast.Program,
    src_fmts: []const ast.SampleFormat,
    dst_fmt: ast.SampleFormat,
    plane_width: i32,
    plane_height: i32,
    has_fast_u8_lut: bool,
) std.mem.Allocator.Error!Analysis {
    var res: Analysis = .{};
    errdefer res.deinit(allocator);

    const n_nodes = prog.nodes.items.len;
    try res.use_counts.resize(allocator, n_nodes);
    @memset(res.use_counts.items, 0);
    try res.value_types.resize(allocator, n_nodes);

    if (dst_fmt.is_float and dst_fmt.bits_per_sample == 16) {
        res.uses_f16 = true;
    }

    // Mark live nodes iteratively from root so dead nodes do not disqualify LUT/i16/F16.
    const live = try allocator.alloc(bool, n_nodes);
    defer allocator.free(live);
    @memset(live, false);

    if (prog.root < n_nodes) {
        live[prog.root] = true;
        var i: usize = n_nodes;
        while (i > 0) {
            i -= 1;
            if (!live[i]) continue;
            switch (prog.nodes.items[i]) {
                .load_abs => |la| {
                    live[la.x] = true;
                    live[la.y] = true;
                    res.use_counts.items[la.x] += 1;
                    res.use_counts.items[la.y] += 1;
                },
                .unary => |u| {
                    live[u.a] = true;
                    res.use_counts.items[u.a] += 1;
                },
                .binary => |b| {
                    live[b.a] = true;
                    live[b.b] = true;
                    res.use_counts.items[b.a] += 1;
                    res.use_counts.items[b.b] += 1;
                },
                .ternary => |t| {
                    live[t.a] = true;
                    live[t.b] = true;
                    live[t.c] = true;
                    res.use_counts.items[t.a] += 1;
                    res.use_counts.items[t.b] += 1;
                    res.use_counts.items[t.c] += 1;
                },
                .arg_extrema => |ae| {
                    for (prog.extra_args.items[ae.args_start..][0..ae.count]) |arg| {
                        live[arg] = true;
                        res.use_counts.items[arg] += 1;
                    }
                },
                .arg_sort_elem => |as| {
                    const spec = prog.arg_sorts.items[as.sort_id];
                    for (prog.extra_args.items[spec.args_start..][0..spec.count]) |arg| {
                        live[arg] = true;
                        res.use_counts.items[arg] += 1;
                    }
                },
                else => {},
            }
        }
    }

    // Forward pass over live nodes: collect usage, reach, and `opt=1` value types (Y15).
    for (prog.nodes.items, 0..) |node, idx| {
        if (!live[idx]) {
            res.value_types.items[idx] = .f32;
            continue;
        }
        switch (node) {
            .const_f32 => res.value_types.items[idx] = .f32,
            .const_i32 => res.value_types.items[idx] = if (prog.opt_int) .i32 else .f32,
            .load_clip => |clip| {
                try addUsedClip(allocator, &res.used_clips, clip);
                const fmt = src_fmts[clip];
                if (fmt.is_float and fmt.bits_per_sample == 16) res.uses_f16 = true;
                res.value_types.items[idx] = if (prog.opt_int and !fmt.is_float) .i32 else .f32;
            },
            .load_rel => |rel| {
                res.uses_rel_access = true;
                try addUsedClip(allocator, &res.used_clips, rel.clip);
                const fmt = src_fmts[rel.clip];
                if (fmt.is_float and fmt.bits_per_sample == 16) res.uses_f16 = true;
                res.value_types.items[idx] = if (prog.opt_int and !fmt.is_float) .i32 else .f32;

                // V1: clamp relative offsets to [-W, W] / [-H, H] in both clamp and mirror modes.
                const eff_dx = interp.clampRelOffset(rel.dx, plane_width);
                const eff_dy = interp.clampRelOffset(rel.dy, plane_height);
                if (eff_dx < 0) res.max_left = @max(res.max_left, -eff_dx);
                if (eff_dx > 0) res.max_right = @max(res.max_right, eff_dx);
                if (eff_dy < 0) res.max_up = @max(res.max_up, -eff_dy);
                if (eff_dy > 0) res.max_down = @max(res.max_down, eff_dy);
            },
            .load_abs => |la| {
                res.uses_abs_access = true;
                try addUsedClip(allocator, &res.used_clips, la.clip);
                const fmt = src_fmts[la.clip];
                if (fmt.is_float and fmt.bits_per_sample == 16) res.uses_f16 = true;
                res.value_types.items[idx] = if (prog.opt_int and !fmt.is_float) .i32 else .f32;
            },
            .load_prop => {
                res.uses_props = true;
                res.value_types.items[idx] = .f32;
            },
            .coord_x => {
                res.uses_x = true;
                res.value_types.items[idx] = if (prog.opt_int) .i32 else .f32;
            },
            .coord_y => {
                res.uses_y = true;
                res.value_types.items[idx] = if (prog.opt_int) .i32 else .f32;
            },
            .frame_n => {
                res.uses_n = true;
                res.value_types.items[idx] = if (prog.opt_int) .i32 else .f32;
            },
            .plane_width => {
                res.uses_width = true;
                res.value_types.items[idx] = if (prog.opt_int) .i32 else .f32;
            },
            .plane_height => {
                res.uses_height = true;
                res.value_types.items[idx] = if (prog.opt_int) .i32 else .f32;
            },
            .unary => |u| {
                const ta = res.value_types.items[u.a];
                // V7: under `opt=1`, `not` and `bitnot` always yield `i32` even for float operands.
                const always_i32 = prog.opt_int and switch (u.op) {
                    .not, .bitnot => true,
                    else => false,
                };
                const stays_i32 = prog.opt_int and ta == .i32 and switch (u.op) {
                    .neg, .abs, .sgn => true,
                    else => false,
                };
                res.value_types.items[idx] = if (always_i32 or stays_i32) .i32 else .f32;
            },
            .binary => |b| {
                const ta = res.value_types.items[b.a];
                const tb = res.value_types.items[b.b];
                // V7: under `opt=1`, comparisons, logic, and bitwise ops always yield `i32`.
                const always_i32 = prog.opt_int and switch (b.op) {
                    .lt, .le, .gt, .ge, .eq, .and_op, .or_op, .xor_op, .bitand, .bitor, .bitxor => true,
                    else => false,
                };
                const stays_i32 = prog.opt_int and ta == .i32 and tb == .i32 and switch (b.op) {
                    .add, .sub, .mul, .min, .max => true,
                    else => false,
                };
                res.value_types.items[idx] = if (always_i32 or stays_i32) .i32 else .f32;
            },
            .ternary => |t| {
                const tb = res.value_types.items[t.b];
                const tc = res.value_types.items[t.c];
                const stays_i32 = prog.opt_int and t.op == .select and tb == .i32 and tc == .i32;
                res.value_types.items[idx] = if (stays_i32) .i32 else .f32;
            },
            .arg_extrema, .arg_sort_elem => res.value_types.items[idx] = .f32,
        }
    }

    // Count live nodes and estimate SIMD compute cost for LUT vs SIMD cost-gating.
    var live_nodes: u32 = 0;
    var cost: u32 = 0;
    for (prog.nodes.items, 0..) |node, idx| {
        if (!live[idx]) continue;
        live_nodes += 1;
        cost += switch (node) {
            .unary => |u| switch (u.op) {
                .sin, .cos, .tan, .atan, .asin, .acos => @as(u32, 24),
                .exp, .log => 18,
                .exp2, .log2 => 16,
                .sqrt => 5,
                .rsqrt => 4,
                .to_f32 => 0,
                else => 1,
            },
            .binary => |b| switch (b.op) {
                .pow, .atan2 => @as(u32, 32),
                .div, .mod => 6,
                .copysign => 2,
                else => 1,
            },
            .ternary => 2,
            else => 0,
        };
    }
    res.live_node_count = live_nodes;
    res.expr_cost = cost;

    // Native u8 interior fast-path eligibility (16/32/64 px/iter, zero unpack/pack).
    res.u8_plan = try analyzeU8(allocator, prog, live, src_fmts, dst_fmt, &res);

    // Native u16 interior fast-path eligibility (8/16/32 px/iter, zero unpack/pack).
    res.u16_plan = try analyzeU16(allocator, prog, live, src_fmts, dst_fmt, &res);

    // i16 interior fast-path eligibility (Y1): all referenced clips + dst are u8,
    // interval arithmetic proves every intermediate fits in i16, and optional final `/ 2^k`
    // satisfies `fits_i16(hi + 2^(k-1))`.
    res.int16_plan = try analyzeInt16(allocator, prog, src_fmts, dst_fmt, &res);

    // Cost-gated LUT eligibility (B4): pure point expression on 1 integer clip (u8 or u16) or 2 u8 clips.
    // Avoid sending trivial expressions to L2-hitting 64KB/128KB LUTs or scalar non-u8 LUTs when SIMD is faster.
    const is_pure_point = !res.uses_x and !res.uses_y and !res.uses_n and
        !res.uses_width and !res.uses_height and !res.uses_props and
        !res.uses_rel_access and !res.uses_abs_access;

    if (is_pure_point) {
        if (res.used_clips.items.len == 1) {
            const c0 = res.used_clips.items[0];
            const fmt0 = src_fmts[c0];
            if (!fmt0.is_float) {
                if (fmt0.bits_per_sample <= 8) {
                    res.fallback_lut_clips[0] = c0;
                    res.fallback_lut_kind = .lut1d_u8;

                    // 256-byte 1D u8->u8 LUT uses AVX2 vpshufb / AVX-512 VBMI vpermb / NEON tbl when
                    // `has_fast_u8_lut` is true; otherwise (non-u8 dst, or CPU lacking SIMD u8 LUT such
                    // as AVX-512 without VBMI or SSE2-only) it uses a scalar or narrower LUT.
                    // Always evaluate u8/i16 fast-path skips in both branches (M1) so cheap u8/i16 plans
                    // never regress onto the scalar table!
                    const has_simd_u8_lut = has_fast_u8_lut and !dst_fmt.is_float and dst_fmt.bits_per_sample == 8;
                    const u8_limit: u32 = if (has_simd_u8_lut) 12 else 24;
                    const i16_limit: u32 = if (has_simd_u8_lut) 10 else 24;
                    const skip_for_u8 = res.u8_plan.eligible and cost <= u8_limit;
                    const skip_for_i16 = res.int16_plan.eligible and cost <= i16_limit;
                    // Scalar non-u8 LUTs (~1.2k-1.4k fps 1T) lose to AVX2 f32 SIMD for prefilter/sqrt/div/poly8;
                    // transcendentals (cost >= 24) still win strongly via LUT.
                    const skip_scalar_lut = !has_simd_u8_lut and cost < 24; // V13 (non-u8 dst), K3
                    if (!skip_for_u8 and !skip_for_i16 and !skip_scalar_lut) {
                        res.lut_clips[0] = c0;
                        res.lut_kind = .lut1d_u8;
                    }
                } else if (fmt0.bits_per_sample <= 16) {
                    res.fallback_lut_clips[0] = c0;
                    res.fallback_lut_kind = .lut1d_u16;
                    // 128 KB 1D u16 LUT hits L2 (~1k fps 1T at 1080p); AVX2 f32 SIMD beats it for
                    // prefilter masks, sqrt, div, and short polynomials (cost < 24), while
                    // transcendentals (sin/cos/exp/log/pow, cost >= 24) still win via LUT.
                    // NEON is 4-wide f32, so the scalar LUT already wins from cost ~16 on Apple M4
                    // (exp: 720 -> 1310 fps 1T at 1080p YUV420P16); AVX2 keeps the x86-tuned 24.
                    const u16_lut_min_cost: u32 = if (builtin.cpu.arch.isAARCH64()) 16 else 24;
                    if (cost >= u16_lut_min_cost) {
                        res.lut_clips[0] = c0;
                        res.lut_kind = .lut1d_u16;
                    }
                }
            }
        } else if (res.used_clips.items.len == 2) {
            const c0 = res.used_clips.items[0];
            const c1 = res.used_clips.items[1];
            if (!src_fmts[c0].is_float and src_fmts[c0].bits_per_sample <= 8 and
                !src_fmts[c1].is_float and src_fmts[c1].bits_per_sample <= 8)
            {
                res.fallback_lut_clips = .{ c0, c1 };
                res.fallback_lut_kind = .lut2d_u8;
                // 64 KB 2D u8x8 LUT hits L2 (~940 fps 1T / ~6.1k fps 16T); SIMD beats it for
                // 2x sqrt / 2x div (cost < 24), while transcendentals / fine_sharp (cost >= 24) win via LUT.
                if (!res.u8_plan.eligible and !res.int16_plan.eligible and cost >= 24) {
                    res.lut_clips = .{ c0, c1 };
                    res.lut_kind = .lut2d_u8;
                }
            }
        }
    }

    return res;
}

fn analyzeU8(
    allocator: std.mem.Allocator,
    prog: *const ast.Program,
    live: []const bool,
    src_fmts: []const ast.SampleFormat,
    dst_fmt: ast.SampleFormat,
    info: *const Analysis,
) std.mem.Allocator.Error!U8Plan {
    if (dst_fmt.is_float or dst_fmt.bits_per_sample != 8) return .{};
    if (info.uses_x or info.uses_y or info.uses_n or info.uses_width or info.uses_height or info.uses_props or info.uses_abs_access) {
        return .{};
    }
    for (info.used_clips.items) |clip| {
        const fmt = src_fmts[clip];
        if (fmt.is_float or fmt.bits_per_sample != 8) return .{};
    }
    if (prog.root >= prog.nodes.items.len) return .{};

    // Track `abs(a - b)` patterns and saturating `sub` (root or inside `max(sub, C>=0)`) /
    // saturating `add` (root or inside `min(add, C<=255)`) so `vpaddusb`/`vpsubusb` are bit-exact.
    const sub_used_only_by_abs = try allocator.alloc(bool, prog.nodes.items.len);
    defer allocator.free(sub_used_only_by_abs);
    @memset(sub_used_only_by_abs, false);

    const sat_ok = try allocator.alloc(bool, prog.nodes.items.len);
    defer allocator.free(sat_ok);
    @memset(sat_ok, false);
    if (prog.nodes.items[prog.root] == .binary and
        (prog.nodes.items[prog.root].binary.op == .add or prog.nodes.items[prog.root].binary.op == .sub))
    {
        sat_ok[prog.root] = true;
    }

    for (prog.nodes.items, 0..) |node, idx| {
        if (!live[idx]) continue;
        if (node == .unary and node.unary.op == .abs) {
            const child = prog.nodes.items[node.unary.a];
            if (child == .binary and child.binary.op == .sub and info.use_counts.items[node.unary.a] == 1) {
                sub_used_only_by_abs[node.unary.a] = true;
            }
        } else if (node == .binary) {
            const bin = node.binary;
            if (bin.op == .max) {
                // `max(a - b, C)` with `C in [0, 255]` -> `max(subs_u8(a, b), C)` is exact.
                for ([_][2]ast.NodeIdx{ .{ bin.a, bin.b }, .{ bin.b, bin.a } }) |pair| {
                    const sub_i = pair[0];
                    const c_i = pair[1];
                    if (prog.nodes.items[sub_i] == .binary and prog.nodes.items[sub_i].binary.op == .sub and
                        info.use_counts.items[sub_i] == 1)
                    {
                        const cv = switch (prog.nodes.items[c_i]) {
                            .const_f32 => |v| if (std.math.isFinite(v) and @trunc(v) == v and v >= 0.0 and v <= 255.0) v else null,
                            .const_i32 => |v| if (v >= 0 and v <= 255) @as(f32, @floatFromInt(v)) else null,
                            else => null,
                        };
                        if (cv != null) sat_ok[sub_i] = true;
                    }
                }
            } else if (bin.op == .min) {
                // `min(a + b, C)` with `C in [0, 255]` -> `min(adds_u8(a, b), C)` is exact.
                for ([_][2]ast.NodeIdx{ .{ bin.a, bin.b }, .{ bin.b, bin.a } }) |pair| {
                    const add_i = pair[0];
                    const c_i = pair[1];
                    if (prog.nodes.items[add_i] == .binary and prog.nodes.items[add_i].binary.op == .add and
                        info.use_counts.items[add_i] == 1)
                    {
                        const cv = switch (prog.nodes.items[c_i]) {
                            .const_f32 => |v| if (std.math.isFinite(v) and @trunc(v) == v and v >= 0.0 and v <= 255.0) v else null,
                            .const_i32 => |v| if (v >= 0 and v <= 255) @as(f32, @floatFromInt(v)) else null,
                            else => null,
                        };
                        if (cv != null) sat_ok[add_i] = true;
                    }
                }
            }
        }
    }

    const intervals = try allocator.alloc(Interval, prog.nodes.items.len);
    defer allocator.free(intervals);

    for (prog.nodes.items, 0..) |node, idx| {
        if (!live[idx]) continue;
        if (sub_used_only_by_abs[idx]) {
            // Checked via parent `abs(a - b)` using `max_u8(a, b) - min_u8(a, b)`.
            intervals[idx] = .{ .lo = 0, .hi = 255 };
            continue;
        }
        const iv: Interval = switch (node) {
            .const_f32 => |v| blk: {
                if (!std.math.isFinite(v) or @trunc(v) != v) return .{};
                if (v < 0.0 or v > 255.0) return .{};
                const iv_i64: i64 = @intFromFloat(v);
                break :blk .{ .lo = iv_i64, .hi = iv_i64 };
            },
            .const_i32 => |v| blk: {
                if (v < 0 or v > 255) return .{};
                break :blk .{ .lo = v, .hi = v };
            },
            .load_clip, .load_rel => .{ .lo = 0, .hi = 255 },
            .unary => |u| blk: {
                if (u.op == .abs and sub_used_only_by_abs[u.a]) {
                    const sb = prog.nodes.items[u.a].binary;
                    const a = intervals[sb.a];
                    const b = intervals[sb.b];
                    const diff_lo = a.lo - b.hi;
                    const diff_hi = a.hi - b.lo;
                    const abs_hi = @max(@abs(diff_lo), @abs(diff_hi));
                    break :blk .{ .lo = 0, .hi = @intCast(abs_hi) };
                }
                return .{};
            },
            .binary => |b| blk: {
                const a = intervals[b.a];
                const rhs = intervals[b.b];
                break :blk switch (b.op) {
                    .min => .{ .lo = @min(a.lo, rhs.lo), .hi = @min(a.hi, rhs.hi) },
                    .max => .{ .lo = @max(a.lo, rhs.lo), .hi = @max(a.hi, rhs.hi) },
                    .add => if (sat_ok[idx])
                        .{ .lo = @min(a.lo + rhs.lo, 255), .hi = @min(a.hi + rhs.hi, 255) }
                    else
                        .{ .lo = a.lo + rhs.lo, .hi = a.hi + rhs.hi },
                    .sub => if (sat_ok[idx])
                        .{ .lo = @max(a.lo - rhs.hi, 0), .hi = @max(a.hi - rhs.lo, 0) }
                    else
                        .{ .lo = a.lo - rhs.hi, .hi = a.hi - rhs.lo },
                    .bitand => .{ .lo = 0, .hi = @min(a.hi, rhs.hi) },
                    .bitor => .{ .lo = @max(a.lo, rhs.lo), .hi = 255 },
                    .bitxor => .{ .lo = 0, .hi = 255 },
                    else => return .{},
                };
            },
            else => return .{},
        };
        if (iv.lo < 0 or iv.hi > 255) return .{};
        intervals[idx] = iv;
    }

    return .{ .eligible = true };
}

fn analyzeU16(
    allocator: std.mem.Allocator,
    prog: *const ast.Program,
    live: []const bool,
    src_fmts: []const ast.SampleFormat,
    dst_fmt: ast.SampleFormat,
    info: *const Analysis,
) std.mem.Allocator.Error!U16Plan {
    if (dst_fmt.is_float or dst_fmt.bits_per_sample < 9 or dst_fmt.bits_per_sample > 16) return .{};
    if (info.uses_x or info.uses_y or info.uses_n or info.uses_width or info.uses_height or info.uses_props or info.uses_abs_access) {
        return .{};
    }
    for (info.used_clips.items) |clip| {
        const fmt = src_fmts[clip];
        if (fmt.is_float or fmt.bits_per_sample < 9 or fmt.bits_per_sample > 16) return .{};
    }
    if (prog.root >= prog.nodes.items.len) return .{};

    // Track `abs(a - b)` patterns and saturating `sub` (root or inside `max(sub, C>=0)`) /
    // saturating `add` (root or inside `min(add, C<=65535)`) so `vpaddusw`/`vpsubusw` are bit-exact.
    const sub_used_only_by_abs = try allocator.alloc(bool, prog.nodes.items.len);
    defer allocator.free(sub_used_only_by_abs);
    @memset(sub_used_only_by_abs, false);

    const sat_ok = try allocator.alloc(bool, prog.nodes.items.len);
    defer allocator.free(sat_ok);
    @memset(sat_ok, false);
    if (prog.nodes.items[prog.root] == .binary and
        (prog.nodes.items[prog.root].binary.op == .add or prog.nodes.items[prog.root].binary.op == .sub))
    {
        sat_ok[prog.root] = true;
    }

    for (prog.nodes.items, 0..) |node, idx| {
        if (!live[idx]) continue;
        if (node == .unary and node.unary.op == .abs) {
            const child = prog.nodes.items[node.unary.a];
            if (child == .binary and child.binary.op == .sub and info.use_counts.items[node.unary.a] == 1) {
                sub_used_only_by_abs[node.unary.a] = true;
            }
        } else if (node == .binary) {
            const bin = node.binary;
            if (bin.op == .max) {
                for ([_][2]ast.NodeIdx{ .{ bin.a, bin.b }, .{ bin.b, bin.a } }) |pair| {
                    const sub_i = pair[0];
                    const c_i = pair[1];
                    if (prog.nodes.items[sub_i] == .binary and prog.nodes.items[sub_i].binary.op == .sub and
                        info.use_counts.items[sub_i] == 1)
                    {
                        const cv = switch (prog.nodes.items[c_i]) {
                            .const_f32 => |v| if (std.math.isFinite(v) and @trunc(v) == v and v >= 0.0 and v <= 65535.0) v else null,
                            .const_i32 => |v| if (v >= 0 and v <= 65535) @as(f32, @floatFromInt(v)) else null,
                            else => null,
                        };
                        if (cv != null) sat_ok[sub_i] = true;
                    }
                }
            } else if (bin.op == .min) {
                for ([_][2]ast.NodeIdx{ .{ bin.a, bin.b }, .{ bin.b, bin.a } }) |pair| {
                    const add_i = pair[0];
                    const c_i = pair[1];
                    if (prog.nodes.items[add_i] == .binary and prog.nodes.items[add_i].binary.op == .add and
                        info.use_counts.items[add_i] == 1)
                    {
                        const cv = switch (prog.nodes.items[c_i]) {
                            .const_f32 => |v| if (std.math.isFinite(v) and @trunc(v) == v and v >= 0.0 and v <= 65535.0) v else null,
                            .const_i32 => |v| if (v >= 0 and v <= 65535) @as(f32, @floatFromInt(v)) else null,
                            else => null,
                        };
                        if (cv != null) sat_ok[add_i] = true;
                    }
                }
            }
        }
    }

    const intervals = try allocator.alloc(Interval, prog.nodes.items.len);
    defer allocator.free(intervals);

    for (prog.nodes.items, 0..) |node, idx| {
        if (!live[idx]) continue;
        if (sub_used_only_by_abs[idx]) {
            intervals[idx] = .{ .lo = 0, .hi = 65535 };
            continue;
        }
        const iv: Interval = switch (node) {
            .const_f32 => |v| blk: {
                if (!std.math.isFinite(v) or @trunc(v) != v) return .{};
                if (v < 0.0 or v > 65535.0) return .{};
                const iv_i64: i64 = @intFromFloat(v);
                break :blk .{ .lo = iv_i64, .hi = iv_i64 };
            },
            .const_i32 => |v| blk: {
                if (v < 0 or v > 65535) return .{};
                break :blk .{ .lo = v, .hi = v };
            },
            // Always use full raw u16 container domain [0, 65535] so unvalidated u10/u12 samples never wrap.
            .load_clip, .load_rel => .{ .lo = 0, .hi = 65535 },
            .unary => |u| blk: {
                if (u.op == .abs and sub_used_only_by_abs[u.a]) {
                    const sb = prog.nodes.items[u.a].binary;
                    const a = intervals[sb.a];
                    const b = intervals[sb.b];
                    const diff_lo = a.lo - b.hi;
                    const diff_hi = a.hi - b.lo;
                    const abs_hi = @max(@abs(diff_lo), @abs(diff_hi));
                    break :blk .{ .lo = 0, .hi = @intCast(abs_hi) };
                }
                return .{};
            },
            .binary => |b| blk: {
                const a = intervals[b.a];
                const rhs = intervals[b.b];
                break :blk switch (b.op) {
                    .min => .{ .lo = @min(a.lo, rhs.lo), .hi = @min(a.hi, rhs.hi) },
                    .max => .{ .lo = @max(a.lo, rhs.lo), .hi = @max(a.hi, rhs.hi) },
                    .add => if (sat_ok[idx])
                        .{ .lo = @min(a.lo + rhs.lo, 65535), .hi = @min(a.hi + rhs.hi, 65535) }
                    else
                        .{ .lo = a.lo + rhs.lo, .hi = a.hi + rhs.hi },
                    .sub => if (sat_ok[idx])
                        .{ .lo = @max(a.lo - rhs.hi, 0), .hi = @max(a.hi - rhs.lo, 0) }
                    else
                        .{ .lo = a.lo - rhs.hi, .hi = a.hi - rhs.lo },
                    .bitand => .{ .lo = 0, .hi = @min(a.hi, rhs.hi) },
                    .bitor => .{ .lo = @max(a.lo, rhs.lo), .hi = 65535 },
                    .bitxor => .{ .lo = 0, .hi = 65535 },
                    else => return .{},
                };
            },
            else => return .{},
        };
        if (iv.lo < 0 or iv.hi > 65535) return .{};
        intervals[idx] = iv;
    }

    return .{ .eligible = true };
}

const Interval = struct {
    lo: i64,
    hi: i64,

    fn fitsI16(self: Interval) bool {
        return self.lo >= std.math.minInt(i16) and self.hi <= std.math.maxInt(i16);
    }
};

fn powerOfTwoShift(f: f32) ?u5 {
    if (!std.math.isFinite(f) or f < 2.0 or f > 16384.0) return null;
    const u: u32 = @intFromFloat(f);
    if (@as(f32, @floatFromInt(u)) != f) return null;
    if (!std.math.isPowerOfTwo(u)) return null;
    return @intCast(@ctz(u));
}

fn analyzeInt16(
    allocator: std.mem.Allocator,
    prog: *const ast.Program,
    src_fmts: []const ast.SampleFormat,
    dst_fmt: ast.SampleFormat,
    info: *const Analysis,
) std.mem.Allocator.Error!Int16Plan {
    if (prog.opt_int) return .{};
    if (dst_fmt.is_float or dst_fmt.bits_per_sample != 8) return .{};
    if (info.uses_x or info.uses_y or info.uses_n or info.uses_width or info.uses_height or info.uses_props or info.uses_abs_access) {
        return .{};
    }
    for (info.used_clips.items) |clip| {
        const fmt = src_fmts[clip];
        if (fmt.is_float or fmt.bits_per_sample != 8) return .{};
    }
    if (prog.root >= prog.nodes.items.len) return .{};

    // Detect optional final `/ 2^k`, `expr * 2^-k`, or `2^-k * expr` at root.
    var check_root = prog.root;
    var final_shift: u5 = 0;
    const root_node = prog.nodes.items[prog.root];
    if (root_node == .binary) {
        const b = root_node.binary;
        if (b.op == .div and prog.nodes.items[b.b] == .const_f32) {
            if (powerOfTwoShift(prog.nodes.items[b.b].const_f32)) |k| {
                check_root = b.a;
                final_shift = k;
            }
        } else if (b.op == .mul and prog.nodes.items[b.b] == .const_f32) {
            const inv = 1.0 / prog.nodes.items[b.b].const_f32;
            if (powerOfTwoShift(inv)) |k| {
                check_root = b.a;
                final_shift = k;
            }
        } else if (b.op == .mul and prog.nodes.items[b.a] == .const_f32) {
            const inv = 1.0 / prog.nodes.items[b.a].const_f32;
            if (powerOfTwoShift(inv)) |k| {
                check_root = b.b;
                final_shift = k;
            }
        }
    }

    // Compute liveness anchored at `check_root` so a fractional reciprocal constant
    // (e.g. `0.25` from `/ 4 -> * 0.25`) used only by `prog.root` is not checked as i16.
    const sub_live = try allocator.alloc(bool, prog.nodes.items.len);
    defer allocator.free(sub_live);
    @memset(sub_live, false);
    sub_live[check_root] = true;
    var si: usize = check_root + 1;
    while (si > 0) {
        si -= 1;
        if (!sub_live[si]) continue;
        switch (prog.nodes.items[si]) {
            .unary => |u| sub_live[u.a] = true,
            .binary => |b| {
                sub_live[b.a] = true;
                sub_live[b.b] = true;
            },
            .ternary => |t| {
                sub_live[t.a] = true;
                sub_live[t.b] = true;
                sub_live[t.c] = true;
            },
            .load_abs => |la| {
                sub_live[la.x] = true;
                sub_live[la.y] = true;
            },
            else => {},
        }
    }

    const intervals = try allocator.alloc(Interval, prog.nodes.items.len);
    defer allocator.free(intervals);

    for (prog.nodes.items[0 .. check_root + 1], 0..) |node, idx| {
        if (!sub_live[idx]) continue;

        const iv: Interval = switch (node) {
            .const_f32 => |v| blk: {
                if (!std.math.isFinite(v) or @trunc(v) != v) return .{};
                if (v < @as(f32, std.math.minInt(i16)) or v > @as(f32, std.math.maxInt(i16))) return .{};
                const iv_i64: i64 = @intFromFloat(v);
                break :blk .{ .lo = iv_i64, .hi = iv_i64 };
            },
            .const_i32 => |v| .{ .lo = v, .hi = v },
            .load_clip, .load_rel => .{ .lo = 0, .hi = 255 },
            .unary => |u| blk: {
                const a = intervals[u.a];
                break :blk switch (u.op) {
                    .neg => .{ .lo = -a.hi, .hi = -a.lo },
                    .abs => blk2: {
                        // a.lo and a.hi are already verified within i16 bounds, so -v fits in i64.
                        const abs_lo: i64 = if (a.lo < 0) -a.lo else a.lo;
                        const abs_hi: i64 = if (a.hi < 0) -a.hi else a.hi;
                        break :blk2 .{
                            .lo = if (a.lo <= 0 and a.hi >= 0) 0 else @min(abs_lo, abs_hi),
                            .hi = @max(abs_lo, abs_hi),
                        };
                    },
                    .not => .{ .lo = 0, .hi = 1 },
                    else => return .{},
                };
            },
            .binary => |b| blk: {
                const a = intervals[b.a];
                const rhs = intervals[b.b];
                break :blk switch (b.op) {
                    .add => .{ .lo = a.lo + rhs.lo, .hi = a.hi + rhs.hi },
                    .sub => .{ .lo = a.lo - rhs.hi, .hi = a.hi - rhs.lo },
                    .mul => blk2: {
                        const p0 = a.lo * rhs.lo;
                        const p1 = a.lo * rhs.hi;
                        const p2 = a.hi * rhs.lo;
                        const p3 = a.hi * rhs.hi;
                        break :blk2 .{
                            .lo = @min(@min(p0, p1), @min(p2, p3)),
                            .hi = @max(@max(p0, p1), @max(p2, p3)),
                        };
                    },
                    .min => .{ .lo = @min(a.lo, rhs.lo), .hi = @min(a.hi, rhs.hi) },
                    .max => .{ .lo = @max(a.lo, rhs.lo), .hi = @max(a.hi, rhs.hi) },
                    .lt, .le, .gt, .ge, .eq, .and_op, .or_op, .xor_op => .{ .lo = 0, .hi = 1 },
                    else => return .{},
                };
            },
            .ternary => |t| blk: {
                const a = intervals[t.a];
                const b = intervals[t.b];
                const c_iv = intervals[t.c];
                if (t.op == .select) {
                    break :blk .{ .lo = @min(b.lo, c_iv.lo), .hi = @max(b.hi, c_iv.hi) };
                }
                const p0 = a.lo * b.lo;
                const p1 = a.lo * b.hi;
                const p2 = a.hi * b.lo;
                const p3 = a.hi * b.hi;
                const prod: Interval = .{
                    .lo = @min(@min(p0, p1), @min(p2, p3)),
                    .hi = @max(@max(p0, p1), @max(p2, p3)),
                };
                if (!prod.fitsI16()) return .{};
                break :blk switch (t.op) {
                    .fmadd => .{ .lo = prod.lo + c_iv.lo, .hi = prod.hi + c_iv.hi },
                    .fmsub => .{ .lo = prod.lo - c_iv.hi, .hi = prod.hi - c_iv.lo },
                    .fnmadd => .{ .lo = c_iv.lo - prod.hi, .hi = c_iv.hi - prod.lo },
                    .fnmsub => .{ .lo = -prod.hi - c_iv.hi, .hi = -prod.lo - c_iv.lo },
                    .select, .linstep => unreachable,
                };
            },
            else => return .{},
        };

        if (!iv.fitsI16()) return .{};
        intervals[idx] = iv;
    }

    // Y1 guard: rounding bias added before `>> k` is in `[2^(k-1)-1, 2^(k-1)] >= 0`,
    // so only `hi + 2^(k-1)` can overflow i16!
    if (final_shift != 0) {
        const root_iv = intervals[check_root];
        const half: i64 = @as(i64, 1) << (final_shift - 1);
        if (root_iv.hi + half > std.math.maxInt(i16)) return .{};
    }

    return .{
        .eligible = true,
        .final_div_shift = final_shift,
        .eval_root = check_root,
    };
}

test "analysis detects LUT, spatial reach, opt=1 types, and Y1 i16 rounding-bias overflow guard" {
    const parser = @import("parser.zig");
    const allocator = std.testing.allocator;
    const fmt8: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 8 };
    const fmts = [_]ast.SampleFormat{ fmt8, fmt8 };

    // Y1 regression: `x 128 * 127 + 2 /` has hi = 255*128 + 127 = 32767, so hi + 1 = 32768 overflows i16!
    {
        var p = try parser.parse(allocator, "x 128 * 127 + 2 /", .{ .num_clips = 1 });
        defer p.deinit(allocator);
        var a = try analyze(allocator, &p, fmts[0..1], fmt8, 64, 64, true);
        defer a.deinit(allocator);
        try std.testing.expectEqual(LutKind.lut1d_u8, a.lut_kind);
        try std.testing.expect(!a.u8_plan.eligible);
        try std.testing.expect(!a.int16_plan.eligible);
    }
    // Native u8 path detects `sort9` / `255 x -` / `x y - abs` and skips slow 2D LUT for `x y + 2 /`.
    {
        var p = try parser.parse(allocator, "x y - abs", .{ .num_clips = 2 });
        defer p.deinit(allocator);
        var a = try analyze(allocator, &p, &fmts, fmt8, 64, 64, true);
        defer a.deinit(allocator);
        try std.testing.expect(a.u8_plan.eligible);
        try std.testing.expectEqual(LutKind.none, a.lut_kind);
    }
    {
        var p = try parser.parse(allocator, "x y + 2 /", .{ .num_clips = 2 });
        defer p.deinit(allocator);
        var a = try analyze(allocator, &p, &fmts, fmt8, 64, 64, true);
        defer a.deinit(allocator);
        try std.testing.expect(a.int16_plan.eligible);
        try std.testing.expectEqual(LutKind.none, a.lut_kind);
    }

    // Safe i16 blur: `x[-1,0] x + x[1,0] + x[0,-1] + 4 /` (hi = 1020, 1020 + 2 <= 32767 -> eligible!).
    {
        var p = try parser.parse(allocator, "x[-1,0] x + x[1,0] + x[0,-1] + 4 /", .{ .num_clips = 1 });
        defer p.deinit(allocator);
        var a = try analyze(allocator, &p, fmts[0..1], fmt8, 64, 64, true);
        defer a.deinit(allocator);
        try std.testing.expectEqual(LutKind.none, a.lut_kind);
        try std.testing.expect(a.int16_plan.eligible);
        try std.testing.expectEqual(@as(u5, 2), a.int16_plan.final_div_shift);
        try std.testing.expectEqual(@as(i32, 1), a.max_left);
        try std.testing.expectEqual(@as(i32, 1), a.max_right);
        try std.testing.expectEqual(@as(i32, 1), a.max_up);
        try std.testing.expectEqual(@as(i32, 0), a.max_down);
    }
}
