//! Automatic 1D (256 / 65,536) and 2D (65,536) lookup table strategy (B4, Y14).
//!
//! Always allocates the full container domain (65,536 entries for any 9..16-bit `u16` clip)
//! so unvalidated out-of-range sample values `> (1 << bps) - 1` never read out of bounds.
//! Populated once in `Create` via a temporary domain-sized JIT kernel released immediately
//! after fill (falling back to the scalar oracle if JIT is unavailable), guaranteeing
//! bit-exact LUT-vs-JIT parity on the host.
//!
//! For 1D `u8 -> u8` LUTs on SSE4.1 / AVX2 / AVX-512 VBMI / AArch64 NEON, compiles a
//! specialized AsmJit SIMD in-register table-lookup kernel (`vpshufb+vpblendvb` tree,
//! `vpermb`, or NEON `tbl`), avoiding Zig 0.17's disabled loop vectorization and achieving
//! ~2.5x higher throughput than scalar lookups (~16.8 GB/s single-core on AVX2).

const std = @import("std");
const builtin = @import("builtin");
const ast = @import("../expr/ast.zig");
const analysis_mod = @import("../expr/analysis.zig");
const interp = @import("../expr/interp.zig");
const aj = @import("asmjit.zig");
const cpu_mod = @import("cpu.zig");
const codegen = @import("codegen.zig");

pub const Lut = struct {
    kind: analysis_mod.LutKind,
    clips: [2]ast.ClipId,
    dst_fmt: ast.SampleFormat,
    /// Flat byte table of `numEntries() * dst_fmt.bytesPerSample()`.
    table: []align(64) u8,
    /// Optional AsmJit-compiled runtime-dimension SIMD plane kernel for 1D `u8 -> u8` LUTs.
    simd_kernel: ?aj.LutKernel = null,

    pub fn deinit(self: *Lut, allocator: std.mem.Allocator, rt: ?*aj.Runtime) void {
        if (self.simd_kernel) |k| {
            if (rt) |runtime| runtime.releaseLut(k.func);
        }
        allocator.free(self.table);
        self.* = undefined;
    }

    pub fn entryCount(kind: analysis_mod.LutKind) usize {
        return switch (kind) {
            .lut1d_u8 => 256,
            .lut1d_u16 => 65536,
            .lut2d_u8 => 65536,
            .none => 0,
        };
    }

    /// Build and populate a dimension-free LUT over the full integer container domain (B4, Y14).
    pub fn build(
        allocator: std.mem.Allocator,
        rt: ?*aj.Runtime,
        cpu: cpu_mod.CpuInfo,
        profile: interp.TargetProfile,
        prog: *const ast.Program,
        info: *const analysis_mod.Analysis,
        src_fmts: []const ast.SampleFormat,
        dst_fmt: ast.SampleFormat,
        out_execmem_denied: ?*bool,
    ) std.mem.Allocator.Error!Lut {
        const kind = info.lut_kind;
        std.debug.assert(kind != .none);
        const entries = entryCount(kind);
        const dst_bps: usize = dst_fmt.bytesPerSample();
        const table = try allocator.alignedAlloc(u8, .@"64", entries * dst_bps);
        errdefer allocator.free(table);

        // Allocate full-container domain input planes (width x height = 256x1, 65536x1, or 256x256).
        const dom_w: usize = switch (kind) {
            .lut1d_u8 => 256,
            .lut1d_u16 => 65536,
            .lut2d_u8 => 256,
            .none => unreachable,
        };
        const dom_h: usize = if (kind == .lut2d_u8) 256 else 1;
        const dst_stride: isize = @intCast(dom_w * dst_bps);

        const n_clips = src_fmts.len;
        const dummy_bufs = try allocator.alloc([]align(64) u8, n_clips);
        defer {
            for (dummy_bufs) |b| if (b.len > 0) allocator.free(b);
            allocator.free(dummy_bufs);
        }
        @memset(dummy_bufs, &.{});

        const src_ptrs = try allocator.alloc([*]const u8, n_clips);
        defer allocator.free(src_ptrs);
        @memset(src_ptrs, table.ptr);
        const src_strides = try allocator.alloc(isize, n_clips);
        defer allocator.free(src_strides);
        @memset(src_strides, 0);

        switch (kind) {
            .lut1d_u8 => {
                const c0 = info.lut_clips[0];
                const buf = try allocator.alignedAlloc(u8, .@"64", 256);
                dummy_bufs[c0] = buf;
                for (0..256) |i| buf[i] = @intCast(i);
                src_ptrs[c0] = buf.ptr;
                src_strides[c0] = 256;
            },
            .lut1d_u16 => {
                const c0 = info.lut_clips[0];
                const buf = try allocator.alignedAlloc(u8, .@"64", 65536 * 2);
                dummy_bufs[c0] = buf;
                const u16_view: [*]u16 = @ptrCast(@alignCast(buf.ptr));
                for (0..65536) |i| u16_view[i] = @intCast(i);
                src_ptrs[c0] = buf.ptr;
                src_strides[c0] = 65536 * 2;
            },
            .lut2d_u8 => {
                const c0 = info.lut_clips[0];
                const c1 = info.lut_clips[1];
                const buf0 = try allocator.alignedAlloc(u8, .@"64", 65536);
                dummy_bufs[c0] = buf0;
                const buf1 = try allocator.alignedAlloc(u8, .@"64", 65536);
                dummy_bufs[c1] = buf1;
                for (0..256) |y| {
                    for (0..256) |x| {
                        buf0[y * 256 + x] = @intCast(x);
                        buf1[y * 256 + x] = @intCast(y);
                    }
                }
                src_ptrs[c0] = buf0.ptr;
                src_strides[c0] = 256;
                src_ptrs[c1] = buf1.ptr;
                src_strides[c1] = 256;
            },
            .none => unreachable,
        }

        // Fill via temporary domain-sized JIT kernel (Y14), falling back to oracle on error/scalar.
        var filled_by_jit = false;
        if (rt) |runtime| {
            if (codegen.compilePlane(allocator, runtime, cpu, prog, info, .{
                .width = dom_w,
                .height = dom_h,
                .src_fmts = src_fmts,
                .dst_fmt = dst_fmt,
            })) |compiled_val| {
                var compiled = compiled_val;
                defer compiled.deinit(allocator, runtime);
                compiled.kernel.func(table.ptr, dst_stride, src_ptrs.ptr, src_strides.ptr, 0, null);
                filled_by_jit = true;
            } else |err| {
                if (err == error.ExecMemDenied) {
                    if (out_execmem_denied) |flag| flag.* = true;
                }
            }
        }

        if (!filled_by_jit) {
            try interp.evalPlane(allocator, prog, profile, .{
                .dst = table.ptr,
                .dst_stride = dst_stride,
                .dst_fmt = dst_fmt,
                .srcs = src_ptrs,
                .src_strides = src_strides,
                .src_fmts = src_fmts,
                .width = dom_w,
                .height = dom_h,
                .frame_n = 0,
                .props = &.{},
            });
        }

        // Compile fast runtime-dimension SIMD in-register 1D u8 -> u8 LUT kernel when supported.
        // M1: on AVX-512 hosts without VBMI (Skylake-SP/X, Cascade Lake), fall back to the 32B AVX2
        // vpshufb+vpblendvb kernel (~1.68 insns/px vs 3.30 scalar) instead of the scalar table loop.
        var simd_kernel: ?aj.LutKernel = null;
        if (kind == .lut1d_u8 and !dst_fmt.is_float and dst_fmt.bits_per_sample == 8) {
            const lut_cpu: ?cpu_mod.CpuInfo = if (aj.Builder.cpuSupportsFastLutU8(cpu))
                cpu
            else if (cpu.vec_bytes == 64 and cpu.has_avx2)
                cpu_mod.detectWithOverride(3)
            else
                null;
            if (lut_cpu) |lcpu| {
                if (rt) |runtime| {
                    if (compileU8LutPlane(runtime, lcpu, info.lut_clips[0], table[0..256])) |k| {
                        simd_kernel = k;
                    } else |err| {
                        if (err == error.ExecMemDenied) {
                            if (out_execmem_denied) |flag| flag.* = true;
                        }
                    }
                }
            }
        }

        return .{
            .kind = kind,
            .clips = info.lut_clips,
            .dst_fmt = dst_fmt,
            .table = table,
            .simd_kernel = simd_kernel,
        };
    }

    /// Apply the precomputed LUT to a destination plane.
    pub fn apply(
        self: *const Lut,
        dst: [*]u8,
        dst_stride: isize,
        srcs: []const [*]const u8,
        src_strides: []const isize,
        width: usize,
        height: usize,
    ) void {
        if (width == 0 or height == 0) return;
        if (self.simd_kernel) |k| {
            k.func(dst, dst_stride, srcs.ptr, src_strides.ptr, width, height);
            return;
        }
        const bps = self.dst_fmt.bytesPerSample();
        switch (bps) {
            1 => self.applyTyped(u8, dst, dst_stride, srcs, src_strides, width, height),
            2 => self.applyTyped(u16, dst, dst_stride, srcs, src_strides, width, height),
            4 => self.applyTyped(u32, dst, dst_stride, srcs, src_strides, width, height),
            else => unreachable,
        }
    }

    fn applyTyped(
        self: *const Lut,
        comptime DstT: type,
        dst_base: [*]u8,
        dst_stride: isize,
        srcs: []const [*]const u8,
        src_strides: []const isize,
        width: usize,
        height: usize,
    ) void {
        const lut_vals: [*]const DstT = @ptrCast(@alignCast(self.table.ptr));
        const c0 = self.clips[0];
        const s0_base = srcs[c0];
        const s0_stride = src_strides[c0];

        switch (self.kind) {
            .lut1d_u8 => {
                if (comptime builtin.cpu.arch.endian() == .little) {
                    if (DstT == u16) {
                        applyLut1dU8ToU16Packed8x(lut_vals, dst_base, dst_stride, s0_base, s0_stride, width, height);
                        return;
                    } else if (DstT == u32) {
                        applyLut1dU8ToU32Packed8x(lut_vals, dst_base, dst_stride, s0_base, s0_stride, width, height);
                        return;
                    }
                }
                for (0..height) |y| {
                    const d_row: [*]DstT = @ptrCast(@alignCast(offsetRowMut(dst_base, dst_stride, y)));
                    const s_row = offsetRow(s0_base, s0_stride, y);
                    var x: usize = 0;
                    while (x + 4 <= width) : (x += 4) {
                        const v0 = lut_vals[s_row[x]];
                        const v1 = lut_vals[s_row[x + 1]];
                        const v2 = lut_vals[s_row[x + 2]];
                        const v3 = lut_vals[s_row[x + 3]];
                        d_row[x] = v0;
                        d_row[x + 1] = v1;
                        d_row[x + 2] = v2;
                        d_row[x + 3] = v3;
                    }
                    while (x < width) : (x += 1) {
                        d_row[x] = lut_vals[s_row[x]];
                    }
                }
            },
            .lut1d_u16 => {
                if (comptime builtin.cpu.arch.endian() == .little) {
                    if (DstT == u16) {
                        applyLut1dU16Packed8x(lut_vals, dst_base, dst_stride, s0_base, s0_stride, width, height);
                        return;
                    } else if (DstT == u8) {
                        applyLut1dU16ToU8Packed8x(lut_vals, dst_base, dst_stride, s0_base, s0_stride, width, height);
                        return;
                    } else if (DstT == u32) {
                        applyLut1dU16ToU32Packed8x(lut_vals, dst_base, dst_stride, s0_base, s0_stride, width, height);
                        return;
                    }
                }
                for (0..height) |y| {
                    const d_row: [*]DstT = @ptrCast(@alignCast(offsetRowMut(dst_base, dst_stride, y)));
                    const s_row: [*]const u16 = @ptrCast(@alignCast(offsetRow(s0_base, s0_stride, y)));
                    var x: usize = 0;
                    while (x + 4 <= width) : (x += 4) {
                        const v0 = lut_vals[s_row[x]];
                        const v1 = lut_vals[s_row[x + 1]];
                        const v2 = lut_vals[s_row[x + 2]];
                        const v3 = lut_vals[s_row[x + 3]];
                        d_row[x] = v0;
                        d_row[x + 1] = v1;
                        d_row[x + 2] = v2;
                        d_row[x + 3] = v3;
                    }
                    while (x < width) : (x += 1) {
                        d_row[x] = lut_vals[s_row[x]];
                    }
                }
            },
            .lut2d_u8 => {
                const c1 = self.clips[1];
                const s1_base = srcs[c1];
                const s1_stride = src_strides[c1];
                // Packed 8x `u64` wins +28%..+61% on AArch64 (wide GP regfile + 1-cycle `ubfx`/`orr`),
                // whereas 4x scalar wins on x86-64 (1,080 vs 877 fps on Zen 3 due to GP pressure).
                if (comptime builtin.cpu.arch.isAARCH64() and builtin.cpu.arch.endian() == .little) {
                    if (DstT == u8) {
                        applyLut2dU8ToU8Packed8x(lut_vals, dst_base, dst_stride, s0_base, s0_stride, s1_base, s1_stride, width, height);
                        return;
                    } else if (DstT == u16) {
                        applyLut2dU8ToU16Packed8x(lut_vals, dst_base, dst_stride, s0_base, s0_stride, s1_base, s1_stride, width, height);
                        return;
                    }
                }
                for (0..height) |y| {
                    const d_row: [*]DstT = @ptrCast(@alignCast(offsetRowMut(dst_base, dst_stride, y)));
                    const r0 = offsetRow(s0_base, s0_stride, y);
                    const r1 = offsetRow(s1_base, s1_stride, y);
                    var x: usize = 0;
                    while (x + 4 <= width) : (x += 4) {
                        const v0 = lut_vals[(@as(usize, r1[x]) << 8) | r0[x]];
                        const v1 = lut_vals[(@as(usize, r1[x + 1]) << 8) | r0[x + 1]];
                        const v2 = lut_vals[(@as(usize, r1[x + 2]) << 8) | r0[x + 2]];
                        const v3 = lut_vals[(@as(usize, r1[x + 3]) << 8) | r0[x + 3]];
                        d_row[x] = v0;
                        d_row[x + 1] = v1;
                        d_row[x + 2] = v2;
                        d_row[x + 3] = v3;
                    }
                    while (x < width) : (x += 1) {
                        const idx: usize = (@as(usize, r1[x]) << 8) | r0[x];
                        d_row[x] = lut_vals[idx];
                    }
                }
            },
            .none => unreachable,
        }
    }
};

/// Compile a runtime-dimension AsmJit plane kernel for 1D `u8 -> u8` LUTs using
/// AVX-512 VBMI `vpermb`, AVX2/SSE4.1 `vpshufb + vpblendvb` tree, or AArch64 NEON `tbl`.
fn compileU8LutPlane(
    rt: *aj.Runtime,
    cpu: cpu_mod.CpuInfo,
    clip_idx: ast.ClipId,
    table_256: *const [256]u8,
) aj.JitError!aj.LutKernel {
    if (cpu.isa == .neon) {
        return aj.Builder.compileLut1dU8Neon(rt, table_256, @intCast(clip_idx));
    }
    var b = try aj.Builder.init(rt, cpu, false);
    defer b.deinit();
    try b.beginLutFunc();

    const dst_ptr = b.argGp(0);
    const dst_stride = b.argGp(1);
    const srcs_ptr = b.argGp(2);
    const src_strides_ptr = b.argGp(3);
    const width_gp = b.argGp(4);
    const height_gp = b.argGp(5);

    const done_lbl = b.newLabel();
    b.jccImm(.eq, width_gp, 0, done_lbl);
    b.jccImm(.eq, height_gp, 0, done_lbl);

    const src_row = b.newGp64();
    const src_stride = b.newGp64();
    const clip_off: i32 = @intCast(@as(usize, clip_idx) * @sizeOf(usize));
    b.gpLoad(src_row, aj.Mem.ptr(srcs_ptr, clip_off), 8, false);
    b.gpLoad(src_stride, aj.Mem.ptr(src_strides_ptr, clip_off), 8, false);

    const step: i64 = @intCast(cpu.vec_bytes);
    const y_gp = b.newGp64();
    b.gpMovImm(y_gp, 0);
    const y_loop = b.newLabel();
    b.bind(y_loop);

    const x_gp = b.newGp64();
    b.gpMovImm(x_gp, 0);
    const x_loop = b.newLabel();
    b.bind(x_loop);

    const idx_v = b.newVec();
    const out_v = b.newVec();
    b.vecLoad(.u_vec, idx_v, aj.Mem.indexed(src_row, x_gp, 0, 0));
    b.vecLutU8_256(out_v, idx_v, table_256);
    b.vecStore(.u_vec, aj.Mem.indexed(dst_ptr, x_gp, 0, 0), out_v);

    b.gpOp3Imm(.add, x_gp, x_gp, step);
    b.jcc(.ult, x_gp, width_gp, x_loop);

    b.gpOp3(.add, dst_ptr, dst_ptr, dst_stride);
    b.gpOp3(.add, src_row, src_row, src_stride);
    b.gpOp3Imm(.add, y_gp, y_gp, 1);
    b.jcc(.ult, y_gp, height_gp, y_loop);

    b.bind(done_lbl);
    b.retVoid();
    b.endFunc();
    return try b.finalizeLut();
}

inline fn offsetRow(base: [*]const u8, stride: isize, y: usize) [*]const u8 {
    const byte_off = @as(isize, @intCast(y)) * stride;
    if (byte_off >= 0) return base + @as(usize, @intCast(byte_off));
    return base - @as(usize, @intCast(-byte_off));
}

inline fn offsetRowMut(base: [*]u8, stride: isize, y: usize) [*]u8 {
    const byte_off = @as(isize, @intCast(y)) * stride;
    if (byte_off >= 0) return base + @as(usize, @intCast(byte_off));
    return base - @as(usize, @intCast(-byte_off));
}

inline fn lookupPack4U16(lut_vals: [*]const u16, p: u64) u64 {
    const a0: u64 = lut_vals[@as(u16, @truncate(p))];
    const a1: u64 = lut_vals[@as(u16, @truncate(p >> 16))];
    const a2: u64 = lut_vals[@as(u16, @truncate(p >> 32))];
    const a3: u64 = lut_vals[@as(u16, @truncate(p >> 48))];
    return (a0 | (a1 << 16)) | ((a2 | (a3 << 16)) << 32);
}

/// 8-pixel/iter `u64`-packed 1D `u16 -> u16` (and `f16 -> f16`) LUT apply loop.
/// Compiles on AArch64 to `ldp` + 8 independent `ldrh` lookups + `orr` + `stp`,
/// cutting load/store uops by ~40% on wide-L1D cores (Apple M-series: +64% over 1x scalar).
fn applyLut1dU16Packed8x(
    noalias lut_vals: [*]const u16,
    dst_base: [*]u8,
    dst_stride: isize,
    s0_base: [*]const u8,
    s0_stride: isize,
    width: usize,
    height: usize,
) void {
    for (0..height) |y| {
        const d_row: [*]u16 = @ptrCast(@alignCast(offsetRowMut(dst_base, dst_stride, y)));
        const s_row: [*]const u16 = @ptrCast(@alignCast(offsetRow(s0_base, s0_stride, y)));
        lutRowU16ToU16(d_row, s_row, lut_vals, width);
    }
}

fn lutRowU16ToU16(
    noalias d_row: [*]u16,
    noalias s_row: [*]const u16,
    noalias lut_vals: [*]const u16,
    width: usize,
) void {
    if (builtin.cpu.arch == .aarch64) {
        lutRowU16ToU16A64(d_row, s_row, lut_vals, width);
        return;
    }
    var sp: [*]align(2) const u64 = @ptrCast(s_row);
    var dp: [*]align(2) u64 = @ptrCast(d_row);
    var chunks = width >> 3;
    while (chunks != 0) : (chunks -= 1) {
        const p0 = sp[0];
        const p1 = sp[1];
        sp += 2;
        dp[0] = lookupPack4U16(lut_vals, p0);
        dp[1] = lookupPack4U16(lut_vals, p1);
        dp += 2;
    }
    var x = width & ~@as(usize, 7);
    while (x < width) : (x += 1) {
        d_row[x] = lut_vals[s_row[x]];
    }
}

/// AArch64 version of `lutRowU16ToU16`: the same `ldp`/8x `ldrh`/`orr`/`stp` shape in fixed asm
/// (LLVM is free to split the u64 loads back into per-lane `ldrh`), but with a non-temporal `ldnp`
/// source load so the src stream does not evict the 128 KB table (= the whole M4 L1D) out of L1.
/// Apple M4, 1080p YUV420P16, 1 thread: ~1880 -> ~2180 fps; `stnp` for dst was slower.
fn lutRowU16ToU16A64(
    noalias d_row: [*]u16,
    noalias s_row: [*]const u16,
    noalias lut_vals: [*]const u16,
    width: usize,
) void {
    const blocks = width >> 3;
    if (blocks != 0) {
        var dp = d_row;
        var sp = s_row;
        var cnt = blocks;
        asm volatile (
            \\1:
            \\ldnp x8, x9, [%[sp]]
            \\add %[sp], %[sp], #16
            \\ubfx x10, x8, #16, #16
            \\ubfx x11, x8, #32, #16
            \\lsr x12, x8, #48
            \\and x8, x8, #0xffff
            \\ubfx x13, x9, #16, #16
            \\ubfx x14, x9, #32, #16
            \\lsr x15, x9, #48
            \\and x9, x9, #0xffff
            \\ldrh w8, [%[t], x8, lsl #1]
            \\ldrh w10, [%[t], x10, lsl #1]
            \\ldrh w11, [%[t], x11, lsl #1]
            \\ldrh w12, [%[t], x12, lsl #1]
            \\ldrh w9, [%[t], x9, lsl #1]
            \\ldrh w13, [%[t], x13, lsl #1]
            \\ldrh w14, [%[t], x14, lsl #1]
            \\ldrh w15, [%[t], x15, lsl #1]
            \\orr x8, x8, x10, lsl #16
            \\orr x11, x11, x12, lsl #16
            \\orr x8, x8, x11, lsl #32
            \\orr x9, x9, x13, lsl #16
            \\orr x14, x14, x15, lsl #16
            \\orr x9, x9, x14, lsl #32
            \\stp x8, x9, [%[dp]], #16
            \\subs %[cnt], %[cnt], #1
            \\b.ne 1b
            : [dp] "+r" (dp),
              [sp] "+r" (sp),
              [cnt] "+r" (cnt),
            : [t] "r" (lut_vals),
            : .{ .x8 = true, .x9 = true, .x10 = true, .x11 = true, .x12 = true, .x13 = true, .x14 = true, .x15 = true, .nzcv = true, .memory = true });
    }
    var x = width & ~@as(usize, 7);
    while (x < width) : (x += 1) {
        d_row[x] = lut_vals[s_row[x]];
    }
}

fn applyLut1dU16ToU8Packed8x(
    noalias lut_vals: [*]const u8,
    dst_base: [*]u8,
    dst_stride: isize,
    s0_base: [*]const u8,
    s0_stride: isize,
    width: usize,
    height: usize,
) void {
    for (0..height) |y| {
        const d_row = offsetRowMut(dst_base, dst_stride, y);
        const s_row: [*]const u16 = @ptrCast(@alignCast(offsetRow(s0_base, s0_stride, y)));
        lutRowU16ToU8(d_row, s_row, lut_vals, width);
    }
}

fn lutRowU16ToU8(
    noalias d_row: [*]u8,
    noalias s_row: [*]const u16,
    noalias lut_vals: [*]const u8,
    width: usize,
) void {
    var sp: [*]align(2) const u64 = @ptrCast(s_row);
    var dp: [*]align(1) u64 = @ptrCast(d_row);
    var chunks = width >> 3;
    while (chunks != 0) : (chunks -= 1) {
        const p0 = sp[0];
        const p1 = sp[1];
        sp += 2;
        const v0: u64 = lut_vals[@as(u16, @truncate(p0))];
        const v1: u64 = lut_vals[@as(u16, @truncate(p0 >> 16))];
        const v2: u64 = lut_vals[@as(u16, @truncate(p0 >> 32))];
        const v3: u64 = lut_vals[@as(u16, @truncate(p0 >> 48))];
        const v4: u64 = lut_vals[@as(u16, @truncate(p1))];
        const v5: u64 = lut_vals[@as(u16, @truncate(p1 >> 16))];
        const v6: u64 = lut_vals[@as(u16, @truncate(p1 >> 32))];
        const v7: u64 = lut_vals[@as(u16, @truncate(p1 >> 48))];
        const lo = (v0 | (v1 << 8)) | ((v2 | (v3 << 8)) << 16);
        const hi = (v4 | (v5 << 8)) | ((v6 | (v7 << 8)) << 16);
        dp[0] = lo | (hi << 32);
        dp += 1;
    }
    var x = width & ~@as(usize, 7);
    while (x < width) : (x += 1) {
        d_row[x] = lut_vals[s_row[x]];
    }
}

fn applyLut1dU16ToU32Packed8x(
    lut_vals: [*]const u32,
    dst_base: [*]u8,
    dst_stride: isize,
    s0_base: [*]const u8,
    s0_stride: isize,
    width: usize,
    height: usize,
) void {
    const chunks_per_row = width >> 3;
    const tail_start = width & ~@as(usize, 7);
    for (0..height) |y| {
        const d_row: [*]u32 = @ptrCast(@alignCast(offsetRowMut(dst_base, dst_stride, y)));
        const s_row: [*]const u16 = @ptrCast(@alignCast(offsetRow(s0_base, s0_stride, y)));
        var sp: [*]align(2) const u64 = @ptrCast(s_row);
        var dp: [*]align(4) u64 = @ptrCast(d_row);
        var chunks = chunks_per_row;
        while (chunks != 0) : (chunks -= 1) {
            const p0 = sp[0];
            const p1 = sp[1];
            sp += 2;
            const v0: u64 = lut_vals[@as(u16, @truncate(p0))];
            const v1: u64 = lut_vals[@as(u16, @truncate(p0 >> 16))];
            const v2: u64 = lut_vals[@as(u16, @truncate(p0 >> 32))];
            const v3: u64 = lut_vals[@as(u16, @truncate(p0 >> 48))];
            const v4: u64 = lut_vals[@as(u16, @truncate(p1))];
            const v5: u64 = lut_vals[@as(u16, @truncate(p1 >> 16))];
            const v6: u64 = lut_vals[@as(u16, @truncate(p1 >> 32))];
            const v7: u64 = lut_vals[@as(u16, @truncate(p1 >> 48))];
            dp[0] = v0 | (v1 << 32);
            dp[1] = v2 | (v3 << 32);
            dp[2] = v4 | (v5 << 32);
            dp[3] = v6 | (v7 << 32);
            dp += 4;
        }
        var x = tail_start;
        while (x < width) : (x += 1) {
            d_row[x] = lut_vals[s_row[x]];
        }
    }
}

fn applyLut1dU8ToU16Packed8x(
    lut_vals: [*]const u16,
    dst_base: [*]u8,
    dst_stride: isize,
    s0_base: [*]const u8,
    s0_stride: isize,
    width: usize,
    height: usize,
) void {
    const chunks_per_row = width >> 3;
    const tail_start = width & ~@as(usize, 7);
    for (0..height) |y| {
        const d_row: [*]u16 = @ptrCast(@alignCast(offsetRowMut(dst_base, dst_stride, y)));
        const s_row = offsetRow(s0_base, s0_stride, y);
        var sp: [*]align(1) const u64 = @ptrCast(s_row);
        var dp: [*]align(2) u64 = @ptrCast(d_row);
        var chunks = chunks_per_row;
        while (chunks != 0) : (chunks -= 1) {
            const p = sp[0];
            sp += 1;
            const v0: u64 = lut_vals[@as(u8, @truncate(p))];
            const v1: u64 = lut_vals[@as(u8, @truncate(p >> 8))];
            const v2: u64 = lut_vals[@as(u8, @truncate(p >> 16))];
            const v3: u64 = lut_vals[@as(u8, @truncate(p >> 24))];
            const v4: u64 = lut_vals[@as(u8, @truncate(p >> 32))];
            const v5: u64 = lut_vals[@as(u8, @truncate(p >> 40))];
            const v6: u64 = lut_vals[@as(u8, @truncate(p >> 48))];
            const v7: u64 = lut_vals[@as(u8, @truncate(p >> 56))];
            dp[0] = (v0 | (v1 << 16)) | ((v2 | (v3 << 16)) << 32);
            dp[1] = (v4 | (v5 << 16)) | ((v6 | (v7 << 16)) << 32);
            dp += 2;
        }
        var x = tail_start;
        while (x < width) : (x += 1) {
            d_row[x] = lut_vals[s_row[x]];
        }
    }
}

fn applyLut1dU8ToU32Packed8x(
    lut_vals: [*]const u32,
    dst_base: [*]u8,
    dst_stride: isize,
    s0_base: [*]const u8,
    s0_stride: isize,
    width: usize,
    height: usize,
) void {
    const chunks_per_row = width >> 3;
    const tail_start = width & ~@as(usize, 7);
    for (0..height) |y| {
        const d_row: [*]u32 = @ptrCast(@alignCast(offsetRowMut(dst_base, dst_stride, y)));
        const s_row = offsetRow(s0_base, s0_stride, y);
        var sp: [*]align(1) const u64 = @ptrCast(s_row);
        var dp: [*]align(4) u64 = @ptrCast(d_row);
        var chunks = chunks_per_row;
        while (chunks != 0) : (chunks -= 1) {
            const p = sp[0];
            sp += 1;
            const v0: u64 = lut_vals[@as(u8, @truncate(p))];
            const v1: u64 = lut_vals[@as(u8, @truncate(p >> 8))];
            const v2: u64 = lut_vals[@as(u8, @truncate(p >> 16))];
            const v3: u64 = lut_vals[@as(u8, @truncate(p >> 24))];
            const v4: u64 = lut_vals[@as(u8, @truncate(p >> 32))];
            const v5: u64 = lut_vals[@as(u8, @truncate(p >> 40))];
            const v6: u64 = lut_vals[@as(u8, @truncate(p >> 48))];
            const v7: u64 = lut_vals[@as(u8, @truncate(p >> 56))];
            dp[0] = v0 | (v1 << 32);
            dp[1] = v2 | (v3 << 32);
            dp[2] = v4 | (v5 << 32);
            dp[3] = v6 | (v7 << 32);
            dp += 4;
        }
        var x = tail_start;
        while (x < width) : (x += 1) {
            d_row[x] = lut_vals[s_row[x]];
        }
    }
}

fn applyLut2dU8ToU8Packed8x(
    noalias lut_vals: [*]const u8,
    dst_base: [*]u8,
    dst_stride: isize,
    s0_base: [*]const u8,
    s0_stride: isize,
    s1_base: [*]const u8,
    s1_stride: isize,
    width: usize,
    height: usize,
) void {
    for (0..height) |y| {
        const d_row = offsetRowMut(dst_base, dst_stride, y);
        const r0 = offsetRow(s0_base, s0_stride, y);
        const r1 = offsetRow(s1_base, s1_stride, y);
        lutRow2dU8ToU8(d_row, r0, r1, lut_vals, width);
    }
}

fn lutRow2dU8ToU8(
    noalias d_row: [*]u8,
    noalias r0: [*]const u8,
    noalias r1: [*]const u8,
    noalias lut_vals: [*]const u8,
    width: usize,
) void {
    var sp0: [*]align(1) const u64 = @ptrCast(r0);
    var sp1: [*]align(1) const u64 = @ptrCast(r1);
    var dp: [*]align(1) u64 = @ptrCast(d_row);
    var chunks = width >> 3;
    while (chunks != 0) : (chunks -= 1) {
        const a = sp0[0];
        const b = sp1[0];
        sp0 += 1;
        sp1 += 1;
        const v0: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b))) << 8) | @as(u8, @truncate(a))];
        const v1: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 8))) << 8) | @as(u8, @truncate(a >> 8))];
        const v2: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 16))) << 8) | @as(u8, @truncate(a >> 16))];
        const v3: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 24))) << 8) | @as(u8, @truncate(a >> 24))];
        const v4: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 32))) << 8) | @as(u8, @truncate(a >> 32))];
        const v5: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 40))) << 8) | @as(u8, @truncate(a >> 40))];
        const v6: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 48))) << 8) | @as(u8, @truncate(a >> 48))];
        const v7: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 56))) << 8) | @as(u8, @truncate(a >> 56))];
        const lo = (v0 | (v1 << 8)) | ((v2 | (v3 << 8)) << 16);
        const hi = (v4 | (v5 << 8)) | ((v6 | (v7 << 8)) << 16);
        dp[0] = lo | (hi << 32);
        dp += 1;
    }
    var x = width & ~@as(usize, 7);
    while (x < width) : (x += 1) {
        const idx: usize = (@as(usize, r1[x]) << 8) | r0[x];
        d_row[x] = lut_vals[idx];
    }
}

fn applyLut2dU8ToU16Packed8x(
    lut_vals: [*]const u16,
    dst_base: [*]u8,
    dst_stride: isize,
    s0_base: [*]const u8,
    s0_stride: isize,
    s1_base: [*]const u8,
    s1_stride: isize,
    width: usize,
    height: usize,
) void {
    const chunks_per_row = width >> 3;
    const tail_start = width & ~@as(usize, 7);
    for (0..height) |y| {
        const d_row: [*]u16 = @ptrCast(@alignCast(offsetRowMut(dst_base, dst_stride, y)));
        const r0 = offsetRow(s0_base, s0_stride, y);
        const r1 = offsetRow(s1_base, s1_stride, y);
        var sp0: [*]align(1) const u64 = @ptrCast(r0);
        var sp1: [*]align(1) const u64 = @ptrCast(r1);
        var dp: [*]align(2) u64 = @ptrCast(d_row);
        var chunks = chunks_per_row;
        while (chunks != 0) : (chunks -= 1) {
            const a = sp0[0];
            const b = sp1[0];
            sp0 += 1;
            sp1 += 1;
            const v0: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b))) << 8) | @as(u8, @truncate(a))];
            const v1: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 8))) << 8) | @as(u8, @truncate(a >> 8))];
            const v2: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 16))) << 8) | @as(u8, @truncate(a >> 16))];
            const v3: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 24))) << 8) | @as(u8, @truncate(a >> 24))];
            const v4: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 32))) << 8) | @as(u8, @truncate(a >> 32))];
            const v5: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 40))) << 8) | @as(u8, @truncate(a >> 40))];
            const v6: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 48))) << 8) | @as(u8, @truncate(a >> 48))];
            const v7: u64 = lut_vals[(@as(usize, @as(u8, @truncate(b >> 56))) << 8) | @as(u8, @truncate(a >> 56))];
            dp[0] = (v0 | (v1 << 16)) | ((v2 | (v3 << 16)) << 32);
            dp[1] = (v4 | (v5 << 16)) | ((v6 | (v7 << 16)) << 32);
            dp += 2;
        }
        var x = tail_start;
        while (x < width) : (x += 1) {
            const idx: usize = (@as(usize, r1[x]) << 8) | r0[x];
            d_row[x] = lut_vals[idx];
        }
    }
}

test "LUT matches JIT bit-for-bit including B4 out-of-range u16 samples and SIMD u8 LUT" {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");
    const allocator = std.testing.allocator;

    const cpu = cpu_mod.detectWithOverride(null);
    const profile = interp.TargetProfile.fromCpu(cpu);
    var rt = try aj.Runtime.init();
    defer rt.deinit();

    // 1. 10-bit clip carrying synthetic out-of-range u16 values (> 1023) to verify B4!
    {
        const fmt10 = ast.SampleFormat{ .is_float = false, .bits_per_sample = 10 };
        const fmts = [_]ast.SampleFormat{fmt10};
        var prog = try parser.parse(allocator, "x 1023 / 2.2 pow 1023 *", .{ .num_clips = 1 });
        defer prog.deinit(allocator);
        try lower_mod.lower(allocator, &prog, profile);
        try optimize_mod.optimize(allocator, &prog, profile, .{});

        var info = try analysis_mod.analyze(allocator, &prog, &fmts, fmt10, 16, 1, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
        defer info.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u16, info.lut_kind);

        var lut = try Lut.build(allocator, &rt, cpu, profile, &prog, &info, &fmts, fmt10, null);
        defer lut.deinit(allocator, &rt);

        var src align(64) = [_]u16{ 0, 1, 512, 1023, 1024, 5000, 32768, 65535, 100, 200, 300, 400, 500, 600, 700, 800 };
        var dst_lut: [16]u16 align(64) = @splat(0);
        var dst_ref: [16]u16 align(64) = @splat(0);
        const srcs = [_][*]const u8{@ptrCast(&src)};
        const strides = [_]isize{32};

        lut.apply(@ptrCast(&dst_lut), 32, &srcs, &strides, 16, 1);
        try interp.evalPlane(allocator, &prog, profile, .{
            .dst = @ptrCast(&dst_ref),
            .dst_stride = 32,
            .dst_fmt = fmt10,
            .srcs = &srcs,
            .src_strides = &strides,
            .src_fmts = &fmts,
            .width = 16,
            .height = 1,
            .frame_n = 0,
            .props = &.{},
        });
        try std.testing.expectEqualSlices(u16, &dst_ref, &dst_lut);

        // Verify 8x u64-packed 1D u16 LUT across odd and multi-chunk widths (1..33, height=3).
        var src_wide: [3 * 36]u16 align(64) = undefined;
        for (&src_wide, 0..) |*slot, i| {
            slot.* = @intCast((i * 19937 + 12345) & 0xffff);
        }
        const table_u16: [*]const u16 = @ptrCast(@alignCast(lut.table.ptr));
        const wide_widths = [_]usize{ 1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33 };
        const wide_srcs = [_][*]const u8{@ptrCast(&src_wide)};
        const wide_strides = [_]isize{72};

        for (wide_widths) |w| {
            var ref_buf: [3 * 36]u16 align(64) = @splat(0);
            for (0..3) |y| {
                for (0..w) |x| {
                    ref_buf[y * 36 + x] = table_u16[src_wide[y * 36 + x]];
                }
            }
            var out_buf: [3 * 36]u16 align(64) = @splat(0);
            lut.apply(@ptrCast(&out_buf), 72, &wide_srcs, &wide_strides, w, 3);
            try std.testing.expectEqualSlices(u16, &ref_buf, &out_buf);
        }
    }

    // 1b. Low-cost u16 expressions (cost < 16, e.g. prefilter range mask) stay on AVX2/NEON SIMD
    // rather than hitting L2 via the 128 KB 1D u16 LUT; exp/poly8 use LUT on 4-lane CPUs and SIMD on 8+ lanes.
    {
        const fmt16 = ast.SampleFormat{ .is_float = false, .bits_per_sample = 16 };
        const fmts = [_]ast.SampleFormat{fmt16};
        var prog = try parser.parse(allocator, "x 4112 < 65535 x 19275 > 0 65535 x 4112 - 65535 19275 4112 - / * - ? ?", .{ .num_clips = 1 });
        defer prog.deinit(allocator);
        try lower_mod.lower(allocator, &prog, profile);
        try optimize_mod.optimize(allocator, &prog, profile, .{});
        var info = try analysis_mod.analyze(allocator, &prog, &fmts, fmt16, 16, 1, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
        defer info.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.none, info.lut_kind);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u16, info.fallback_lut_kind);

        var p_exp = try parser.parse(allocator, "x 65535 / exp 24000 *", .{ .num_clips = 1 });
        defer p_exp.deinit(allocator);
        try lower_mod.lower(allocator, &p_exp, profile);
        try optimize_mod.optimize(allocator, &p_exp, profile, .{});
        var i_exp4 = try analysis_mod.analyze(allocator, &p_exp, &fmts, fmt16, 16, 1, .{ .has_fast_u8_lut = true, .f32_lanes = 4 });
        defer i_exp4.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u16, i_exp4.lut_kind);
        var i_exp8 = try analysis_mod.analyze(allocator, &p_exp, &fmts, fmt16, 16, 1, .{ .has_fast_u8_lut = true, .f32_lanes = 8 });
        defer i_exp8.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.none, i_exp8.lut_kind);

        var p_poly = try parser.parse(allocator, "x 65535 / T! T@ 0.1 * 0.2 + T@ * 0.3 + T@ * 0.4 + T@ * 0.5 + T@ * 0.6 + T@ * 0.7 + T@ * 0.8 + 65535 *", .{ .num_clips = 1 });
        defer p_poly.deinit(allocator);
        try lower_mod.lower(allocator, &p_poly, profile);
        try optimize_mod.optimize(allocator, &p_poly, profile, .{ .enable_fma = true });
        var i_poly4 = try analysis_mod.analyze(allocator, &p_poly, &fmts, fmt16, 16, 1, .{ .has_fast_u8_lut = true, .f32_lanes = 4 });
        defer i_poly4.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u16, i_poly4.lut_kind);
        var i_poly8 = try analysis_mod.analyze(allocator, &p_poly, &fmts, fmt16, 16, 1, .{ .has_fast_u8_lut = true, .f32_lanes = 8 });
        defer i_poly8.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.none, i_poly8.lut_kind);
    }

    // 1c. Scalar non-u8 LUTs (`u8 -> f32`) and 64 KB 2D `u8x8` LUTs bypass LUT for cost < 24
    // (prefilter / 2x sqrt) while transcendentals (`pow`) stay on LUT; `i16` cost <= 10 beats vpshufb LUT.
    {
        const fmt8 = ast.SampleFormat{ .is_float = false, .bits_per_sample = 8 };
        const fmtf32 = ast.SampleFormat{ .is_float = true, .bits_per_sample = 32 };
        const fmts1 = [_]ast.SampleFormat{fmt8};
        const fmts2 = [_]ast.SampleFormat{ fmt8, fmt8 };

        var p_pref = try parser.parse(allocator, "x 16 < 1 x 75 > 0 1 x 16 - 1 75 16 - / * - ? ?", .{ .num_clips = 1 });
        defer p_pref.deinit(allocator);
        try lower_mod.lower(allocator, &p_pref, profile);
        try optimize_mod.optimize(allocator, &p_pref, profile, .{});
        var i_pref = try analysis_mod.analyze(allocator, &p_pref, &fmts1, fmtf32, 16, 1, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
        defer i_pref.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.none, i_pref.lut_kind);

        var p_gam = try parser.parse(allocator, "x 255 / 2.2 pow", .{ .num_clips = 1 });
        defer p_gam.deinit(allocator);
        try lower_mod.lower(allocator, &p_gam, profile);
        try optimize_mod.optimize(allocator, &p_gam, profile, .{});
        var i_gam = try analysis_mod.analyze(allocator, &p_gam, &fmts1, fmtf32, 16, 1, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
        defer i_gam.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u8, i_gam.lut_kind);

        var p_2sqrt = try parser.parse(allocator, "x 1 + sqrt y 1 + sqrt + 8 *", .{ .num_clips = 2 });
        defer p_2sqrt.deinit(allocator);
        try lower_mod.lower(allocator, &p_2sqrt, profile);
        try optimize_mod.optimize(allocator, &p_2sqrt, profile, .{});
        var i_2sqrt = try analysis_mod.analyze(allocator, &p_2sqrt, &fmts2, fmt8, 16, 1, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
        defer i_2sqrt.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.none, i_2sqrt.lut_kind);

        var p_2pow = try parser.parse(allocator, "x 255 / 0.01 + y 255 / 2 * pow 255 *", .{ .num_clips = 2 });
        defer p_2pow.deinit(allocator);
        try lower_mod.lower(allocator, &p_2pow, profile);
        try optimize_mod.optimize(allocator, &p_2pow, profile, .{});
        var i_2pow = try analysis_mod.analyze(allocator, &p_2pow, &fmts2, fmt8, 16, 1, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
        defer i_2pow.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.lut2d_u8, i_2pow.lut_kind);

        var p_i16 = try parser.parse(allocator, "x 2 * 10 + 200 min 20 max 1 +", .{ .num_clips = 1 });
        defer p_i16.deinit(allocator);
        try lower_mod.lower(allocator, &p_i16, profile);
        try optimize_mod.optimize(allocator, &p_i16, profile, .{});
        var i_i16 = try analysis_mod.analyze(allocator, &p_i16, &fmts1, fmt8, 16, 1, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
        defer i_i16.deinit(allocator);
        try std.testing.expect(i_i16.int16_plan.eligible);
        try std.testing.expectEqual(analysis_mod.LutKind.none, i_i16.lut_kind);
    }

    // 2. 8-bit gamma 1D LUT verifying SIMD vpshufb/vpermb/NEON-tbl kernel across all 256 byte values!
    {
        const fmt8 = ast.SampleFormat{ .is_float = false, .bits_per_sample = 8 };
        const fmts = [_]ast.SampleFormat{fmt8};
        var prog = try parser.parse(allocator, "x 255 / 2.2 pow 255 *", .{ .num_clips = 1 });
        defer prog.deinit(allocator);
        try lower_mod.lower(allocator, &prog, profile);
        try optimize_mod.optimize(allocator, &prog, profile, .{});

        var info = try analysis_mod.analyze(allocator, &prog, &fmts, fmt8, 256, 1, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
        defer info.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u8, info.lut_kind);

        var lut = try Lut.build(allocator, &rt, cpu, profile, &prog, &info, &fmts, fmt8, null);
        defer lut.deinit(allocator, &rt);
        if (aj.Builder.cpuSupportsFastLutU8(cpu)) {
            try std.testing.expect(lut.simd_kernel != null);
        }

        var src: [256]u8 align(64) = undefined;
        for (0..256) |i| src[i] = @intCast(i);
        var dst_lut: [256]u8 align(64) = @splat(0);
        var dst_ref: [256]u8 align(64) = @splat(0);
        const srcs = [_][*]const u8{&src};
        const strides = [_]isize{256};

        lut.apply(&dst_lut, 256, &srcs, &strides, 256, 1);
        try interp.evalPlane(allocator, &prog, profile, .{
            .dst = &dst_ref,
            .dst_stride = 256,
            .dst_fmt = fmt8,
            .srcs = &srcs,
            .src_strides = &strides,
            .src_fmts = &fmts,
            .width = 256,
            .height = 1,
            .frame_n = 0,
            .props = &.{},
        });
        try std.testing.expectEqualSlices(u8, &dst_ref, &dst_lut);

        // Reuse ONE single compiled SIMD u8 LUT across widths on both sides of 16/32/64B boundaries,
        // multiple heights (>= 2), and varying strides without rebuilding it.
        var src_tail: [5 * 192]u8 align(64) = undefined;
        for (0..src_tail.len) |i| src_tail[i] = @intCast((i * 53 + 11) & 255);
        const heights = [_]usize{ 2, 3, 5 };
        for ([_]usize{ 1, 2, 3, 7, 8, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 49, 63, 64, 65, 97 }) |w| {
            const h = heights[w % heights.len];
            const min_stride: usize = std.mem.alignForward(usize, w, if (cpu.has_avx512) 64 else 32);
            // Vary stride (minimal aligned vs padded +64B) to exercise row-stride advancement.
            const row_stride: usize = if ((w & 1) == 0) min_stride + 64 else min_stride;
            var out_lut: [5 * 192]u8 align(64) = @splat(0xA5);
            var out_ref: [5 * 192]u8 align(64) = @splat(0xA5);
            const s_arr = [_][*]const u8{&src_tail};
            const st_arr = [_]isize{@intCast(row_stride)};
            lut.apply(&out_lut, @intCast(row_stride), &s_arr, &st_arr, w, h);
            try interp.evalPlane(allocator, &prog, profile, .{
                .dst = &out_ref,
                .dst_stride = @intCast(row_stride),
                .dst_fmt = fmt8,
                .srcs = &s_arr,
                .src_strides = &st_arr,
                .src_fmts = &fmts,
                .width = w,
                .height = h,
                .frame_n = 0,
                .props = &.{},
            });
            for (0..h) |y| {
                const row_off = y * row_stride;
                try std.testing.expectEqualSlices(u8, out_ref[row_off .. row_off + w], out_lut[row_off .. row_off + w]);
            }
        }
    }

    // 3. Asymmetric 2D u8x8 LUT (`lut2d_u8`) matches oracle bit-for-bit (V14).
    {
        const fmt8 = ast.SampleFormat{ .is_float = false, .bits_per_sample = 8 };
        const fmts = [_]ast.SampleFormat{ fmt8, fmt8 };
        var prog = try parser.parse(allocator, "x 255 / 2.2 pow y 255 / 1.8 pow + 127.5 *", .{ .num_clips = 2 });
        defer prog.deinit(allocator);
        try lower_mod.lower(allocator, &prog, profile);
        try optimize_mod.optimize(allocator, &prog, profile, .{});

        var info = try analysis_mod.analyze(allocator, &prog, &fmts, fmt8, 64, 4, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
        defer info.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.lut2d_u8, info.lut_kind);

        var lut = try Lut.build(allocator, &rt, cpu, profile, &prog, &info, &fmts, fmt8, null);
        defer lut.deinit(allocator, &rt);

        var s0: [256]u8 align(64) = undefined;
        var s1: [256]u8 align(64) = undefined;
        for (0..256) |i| {
            s0[i] = @intCast((i * 17 + 3) & 255);
            s1[i] = @intCast((i * 43 + 9) & 255);
        }
        var dst_lut: [256]u8 align(64) = @splat(0);
        var dst_ref: [256]u8 align(64) = @splat(0);
        const srcs = [_][*]const u8{ &s0, &s1 };
        const strides = [_]isize{ 64, 64 };

        lut.apply(&dst_lut, 64, &srcs, &strides, 64, 4);
        try interp.evalPlane(allocator, &prog, profile, .{
            .dst = &dst_ref,
            .dst_stride = 64,
            .dst_fmt = fmt8,
            .srcs = &srcs,
            .src_strides = &strides,
            .src_fmts = &fmts,
            .width = 64,
            .height = 4,
            .frame_n = 0,
            .props = &.{},
        });
        try std.testing.expectEqualSlices(u8, &dst_ref, &dst_lut);
    }

    // 4. V13 & K3 regression: scalar LUTs (non-u8 dst or u8->u8 without fast SIMD LUT) skip cheap ops and keep heavy ops.
    {
        const fmt8 = ast.SampleFormat{ .is_float = false, .bits_per_sample = 8 };
        const fmt_f32 = ast.SampleFormat{ .is_float = true, .bits_per_sample = 32 };
        const fmts = [_]ast.SampleFormat{fmt8};

        var p_cheap = try parser.parse(allocator, "x 255 /", .{ .num_clips = 1 });
        defer p_cheap.deinit(allocator);
        try lower_mod.lower(allocator, &p_cheap, profile);
        try optimize_mod.optimize(allocator, &p_cheap, profile, .{});
        var info_cheap = try analysis_mod.analyze(allocator, &p_cheap, &fmts, fmt_f32, 1920, 1080, .{ .has_fast_u8_lut = true });
        defer info_cheap.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.none, info_cheap.lut_kind);

        var p_heavy = try parser.parse(allocator, "x 255 / 2.2 pow", .{ .num_clips = 1 });
        defer p_heavy.deinit(allocator);
        try lower_mod.lower(allocator, &p_heavy, profile);
        try optimize_mod.optimize(allocator, &p_heavy, profile, .{});
        var info_heavy = try analysis_mod.analyze(allocator, &p_heavy, &fmts, fmt_f32, 1920, 1080, .{ .has_fast_u8_lut = true });
        defer info_heavy.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u8, info_heavy.lut_kind);

        // K3: u8 -> u8 `x 1.5 * 10 +` takes LUT only when CPU supports fast SIMD u8 LUT;
        // heavy `gamma` still takes LUT even when CPU lacks fast SIMD u8 LUT (e.g. SKX without VBMI / SSE2).
        var p_lin = try parser.parse(allocator, "x 1.5 * 10 +", .{ .num_clips = 1 });
        defer p_lin.deinit(allocator);
        try lower_mod.lower(allocator, &p_lin, profile);
        try optimize_mod.optimize(allocator, &p_lin, profile, .{});
        var info_lin_simd = try analysis_mod.analyze(allocator, &p_lin, &fmts, fmt8, 1920, 1080, .{ .has_fast_u8_lut = true });
        defer info_lin_simd.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u8, info_lin_simd.lut_kind);
        var info_lin_nosimd = try analysis_mod.analyze(allocator, &p_lin, &fmts, fmt8, 1920, 1080, .{ .has_fast_u8_lut = false });
        defer info_lin_nosimd.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.none, info_lin_nosimd.lut_kind);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u8, info_lin_nosimd.fallback_lut_kind);

        // M1: u8-plan cost-10 and i16-plan cost-9 expressions must stay on SIMD JIT (not scalar LUT)
        // when the CPU lacks fast SIMD u8 LUT, while still exposing `fallback_lut_kind` for ExecMemDenied.
        var p_u8_c10 = try parser.parse(allocator, "x 200 min 10 + 30 max 5 - 220 min 7 + 40 max 9 - 230 min 11 +", .{ .num_clips = 1 });
        defer p_u8_c10.deinit(allocator);
        try lower_mod.lower(allocator, &p_u8_c10, profile);
        try optimize_mod.optimize(allocator, &p_u8_c10, profile, .{});
        var info_u8_c10 = try analysis_mod.analyze(allocator, &p_u8_c10, &fmts, fmt8, 1920, 1080, .{ .has_fast_u8_lut = false });
        defer info_u8_c10.deinit(allocator);
        try std.testing.expect(info_u8_c10.u8_plan.eligible);
        try std.testing.expectEqual(analysis_mod.LutKind.none, info_u8_c10.lut_kind);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u8, info_u8_c10.fallback_lut_kind);

        var p_i16_c9 = try parser.parse(allocator, "x 128 - 3 * 128 + 0 max 255 min 2 * 100 - 0 max 255 min", .{ .num_clips = 1 });
        defer p_i16_c9.deinit(allocator);
        try lower_mod.lower(allocator, &p_i16_c9, profile);
        try optimize_mod.optimize(allocator, &p_i16_c9, profile, .{});
        var info_i16_c9 = try analysis_mod.analyze(allocator, &p_i16_c9, &fmts, fmt8, 1920, 1080, .{ .has_fast_u8_lut = false });
        defer info_i16_c9.deinit(allocator);
        try std.testing.expect(info_i16_c9.int16_plan.eligible);
        try std.testing.expectEqual(analysis_mod.LutKind.none, info_i16_c9.lut_kind);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u8, info_i16_c9.fallback_lut_kind);

        var p_gam = try parser.parse(allocator, "x 255 / 2.2 pow 255 *", .{ .num_clips = 1 });
        defer p_gam.deinit(allocator);
        try lower_mod.lower(allocator, &p_gam, profile);
        try optimize_mod.optimize(allocator, &p_gam, profile, .{});
        var info_gam_nosimd = try analysis_mod.analyze(allocator, &p_gam, &fmts, fmt8, 1920, 1080, .{ .has_fast_u8_lut = false });
        defer info_gam_nosimd.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u8, info_gam_nosimd.lut_kind);
    }

    // 5. 1D f16 -> f16 LUT (128 KB) matches oracle bit-for-bit across special f16 bit patterns,
    //    plus packed 8x mixed-depth (`u16->u8`, `u16->f32`, `u8->u16`, `u8->f32`) and `lut2d_u8` (`u8`, `u16`).
    {
        const fmt_f16 = ast.SampleFormat{ .is_float = true, .bits_per_sample = 16 };
        const fmts_f16 = [_]ast.SampleFormat{fmt_f16};
        var p_f16 = try parser.parse(allocator, "x abs 2.2 pow", .{ .num_clips = 1 });
        defer p_f16.deinit(allocator);
        try lower_mod.lower(allocator, &p_f16, profile);
        try optimize_mod.optimize(allocator, &p_f16, profile, .{ .enable_fma = profile.use_fma });
        var info_f16 = try analysis_mod.analyze(allocator, &p_f16, &fmts_f16, fmt_f16, 17, 2, .{ .has_fast_u8_lut = true, .f32_lanes = cpu.f32_lanes });
        defer info_f16.deinit(allocator);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u16, info_f16.lut_kind);

        var lut_f16 = try Lut.build(allocator, &rt, cpu, profile, &p_f16, &info_f16, &fmts_f16, fmt_f16, null);
        defer lut_f16.deinit(allocator, &rt);

        // Cover +0.0, -0.0, subnormals, normal floats, negative floats, +inf, -inf, qNaN.
        var src_f16 align(64) = [_]u16{
            0x0000, 0x8000, 0x0001, 0x03ff, 0x3c00, 0xbc00, 0x3800, 0x4000,
            0x4500, 0xc500, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x3555,
            0x3999,
        };
        var dst_lut: [17]u16 align(64) = @splat(0);
        var dst_ref: [17]u16 align(64) = @splat(0);
        const srcs = [_][*]const u8{@ptrCast(&src_f16)};
        const strides = [_]isize{34};
        lut_f16.apply(@ptrCast(&dst_lut), 34, &srcs, &strides, 17, 1);
        try interp.evalPlane(allocator, &p_f16, profile, .{
            .dst = @ptrCast(&dst_ref),
            .dst_stride = 34,
            .dst_fmt = fmt_f16,
            .srcs = &srcs,
            .src_strides = &strides,
            .src_fmts = &fmts_f16,
            .width = 17,
            .height = 1,
            .frame_n = 0,
            .props = &.{},
        });
        try std.testing.expectEqualSlices(u16, &dst_ref, &dst_lut);

        // Verify packed 8x helpers across odd & multi-chunk widths.
        var s8_a: [2 * 28]u8 align(64) = undefined;
        var s8_b: [2 * 28]u8 align(64) = undefined;
        var s16_a: [2 * 28]u16 align(64) = undefined;
        for (0..56) |i| {
            s8_a[i] = @intCast((i * 37 + 5) & 255);
            s8_b[i] = @intCast((i * 73 + 11) & 255);
            s16_a[i] = @intCast((i * 19937 + 12345) & 0xffff);
        }
        const tbl8 = try allocator.alloc(u8, 65536);
        defer allocator.free(tbl8);
        const tbl16 = try allocator.alloc(u16, 65536);
        defer allocator.free(tbl16);
        const tbl32 = try allocator.alloc(u32, 65536);
        defer allocator.free(tbl32);
        for (0..65536) |i| {
            tbl8[i] = @intCast((i * 29 + 7) & 255);
            tbl16[i] = @intCast((i * 257 + 13) & 0xffff);
            tbl32[i] = @intCast((i * 65537 + 19) & 0xffff_ffff);
        }
        for ([_]usize{ 1, 3, 7, 8, 9, 15, 16, 17, 25 }) |w| {
            var d8: [56]u8 align(64) = @splat(0);
            var r8: [56]u8 align(64) = @splat(0);
            for (0..2) |y| for (0..w) |x| {
                r8[y * 28 + x] = tbl8[(@as(usize, s8_b[y * 28 + x]) << 8) | s8_a[y * 28 + x]];
            };
            applyLut2dU8ToU8Packed8x(tbl8.ptr, &d8, 28, &s8_a, 28, &s8_b, 28, w, 2);
            try std.testing.expectEqualSlices(u8, &r8, &d8);

            var d16: [56]u16 align(64) = @splat(0);
            var r16: [56]u16 align(64) = @splat(0);
            applyLut2dU8ToU16Packed8x(tbl16.ptr, @ptrCast(&d16), 56, &s8_a, 28, &s8_b, 28, w, 2);
            for (0..2) |y| for (0..w) |x| {
                r16[y * 28 + x] = tbl16[(@as(usize, s8_b[y * 28 + x]) << 8) | s8_a[y * 28 + x]];
            };
            try std.testing.expectEqualSlices(u16, &r16, &d16);

            @memset(&d16, 0);
            @memset(&r16, 0);
            for (0..2) |y| for (0..w) |x| {
                r16[y * 28 + x] = tbl16[s16_a[y * 28 + x]];
            };
            applyLut1dU16Packed8x(tbl16.ptr, @ptrCast(&d16), 56, @ptrCast(&s16_a), 56, w, 2);
            try std.testing.expectEqualSlices(u16, &r16, &d16);

            @memset(&d8, 0);
            @memset(&r8, 0);
            for (0..2) |y| for (0..w) |x| {
                r8[y * 28 + x] = tbl8[s16_a[y * 28 + x]];
            };
            applyLut1dU16ToU8Packed8x(tbl8.ptr, &d8, 28, @ptrCast(&s16_a), 56, w, 2);
            try std.testing.expectEqualSlices(u8, &r8, &d8);

            @memset(&d16, 0);
            @memset(&r16, 0);
            applyLut1dU8ToU16Packed8x(tbl16.ptr, @ptrCast(&d16), 56, &s8_a, 28, w, 2);
            for (0..2) |y| for (0..w) |x| {
                r16[y * 28 + x] = tbl16[s8_a[y * 28 + x]];
            };
            try std.testing.expectEqualSlices(u16, &r16, &d16);

            var d32: [56]u32 align(64) = @splat(0);
            var r32: [56]u32 align(64) = @splat(0);
            applyLut1dU8ToU32Packed8x(tbl32.ptr, @ptrCast(&d32), 112, &s8_a, 28, w, 2);
            for (0..2) |y| for (0..w) |x| {
                r32[y * 28 + x] = tbl32[s8_a[y * 28 + x]];
            };
            try std.testing.expectEqualSlices(u32, &r32, &d32);

            @memset(&d32, 0);
            @memset(&r32, 0);
            applyLut1dU16ToU32Packed8x(tbl32.ptr, @ptrCast(&d32), 112, @ptrCast(&s16_a), 56, w, 2);
            for (0..2) |y| for (0..w) |x| {
                r32[y * 28 + x] = tbl32[s16_a[y * 28 + x]];
            };
            try std.testing.expectEqualSlices(u32, &r32, &d32);
        }
    }
}
