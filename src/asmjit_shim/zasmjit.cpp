// Thin C++20 C-ABI shim wrapping AsmJit (ujit::UniCompiler + JitRuntime).
// Compiled with -fno-exceptions -fno-rtti -fvisibility=hidden (NO -fno-threadsafe-statics).

#include "zasmjit.h"

#include <asmjit/core.h>
#include <asmjit/ujit/ujit_base.h>
#include <asmjit/ujit/uni_compiler.h>
#include <asmjit/ujit/uni_condition.h>
#include <asmjit/ujit/uni_op.h>

#if defined(ASMJIT_UJIT_X86)
#include <asmjit/x86.h>
#elif defined(ASMJIT_UJIT_AARCH64)
#include <asmjit/a64.h>
#else
#error "zasmjit.cpp requires x86-64 or AArch64 host architecture"
#endif

#include <cstring>
#include <new>
#include <vector>

namespace aj = asmjit;
namespace ujit = asmjit::ujit;

namespace {

struct ShimErrorHandler final : public aj::ErrorHandler {
    aj::Error first_error = aj::Error::kOk;
    char first_msg[256]{};
    void handle_error(aj::Error err, const char* msg, aj::BaseEmitter* /*origin*/) override {
        if (first_error == aj::Error::kOk) {
            first_error = err;
            if (msg != nullptr) {
                std::strncpy(first_msg, msg, sizeof(first_msg) - 1u);
                first_msg[sizeof(first_msg) - 1u] = '\0';
            }
        }
    }
};

struct ConstEntry {
    uint8_t bytes[64];
    uint32_t len;
    uint32_t reg_id;
    uint32_t reg_sig;
};

static inline zj_reg_t pack_reg(const aj::Reg& r) noexcept {
    return zj_reg_t{r.id(), r.signature().bits()};
}

static inline ujit::Gp unpack_gp(zj_reg_t r) noexcept {
    return ujit::Gp(aj::OperandSignature{r.sig}, r.id);
}

static inline ujit::Vec unpack_vec(zj_reg_t r) noexcept {
    return ujit::Vec(aj::OperandSignature{r.sig}, r.id);
}

static inline aj::Label unpack_label(zj_label_t l) noexcept {
    return aj::Label(l.id);
}

static inline ujit::CondCode to_cond_code(zj_cond_t c) noexcept {
    switch (c) {
        case ZJ_COND_EQ:  return ujit::CondCode::kEqual;
        case ZJ_COND_NE:  return ujit::CondCode::kNotEqual;
        case ZJ_COND_SLT: return ujit::CondCode::kSignedLT;
        case ZJ_COND_SLE: return ujit::CondCode::kSignedLE;
        case ZJ_COND_SGT: return ujit::CondCode::kSignedGT;
        case ZJ_COND_SGE: return ujit::CondCode::kSignedGE;
        case ZJ_COND_ULT: return ujit::CondCode::kUnsignedLT;
        case ZJ_COND_ULE: return ujit::CondCode::kUnsignedLE;
        case ZJ_COND_UGT: return ujit::CondCode::kUnsignedGT;
        case ZJ_COND_UGE: return ujit::CondCode::kUnsignedGE;
        default:          return ujit::CondCode::kEqual;
    }
}

static aj::CpuFeatures build_cpu_features(const zj_cpu_info_t& cpu) noexcept {
    aj::CpuFeatures f;
#if defined(ASMJIT_UJIT_X86)
    const aj::CpuFeatures& host = aj::CpuInfo::host().features();
    f.add(aj::CpuFeatures::X86::kCMOV, aj::CpuFeatures::X86::kSSE, aj::CpuFeatures::X86::kSSE2);
    if (cpu.isa >= ZJ_ISA_V2) {
        if (host.x86().has_sse3())   f.add(aj::CpuFeatures::X86::kSSE3);
        if (host.x86().has_ssse3())  f.add(aj::CpuFeatures::X86::kSSSE3);
        if (host.x86().has_sse4_1()) f.add(aj::CpuFeatures::X86::kSSE4_1);
        if (host.x86().has_sse4_2()) f.add(aj::CpuFeatures::X86::kSSE4_2);
        if (host.x86().has_popcnt()) f.add(aj::CpuFeatures::X86::kPOPCNT);
    }
    if (cpu.isa >= ZJ_ISA_V3) {
        if (host.x86().has_avx())   f.add(aj::CpuFeatures::X86::kAVX);
        if (host.x86().has_avx2())  f.add(aj::CpuFeatures::X86::kAVX2);
        if (host.x86().has_fma())   f.add(aj::CpuFeatures::X86::kFMA);
        if (host.x86().has_f16c())  f.add(aj::CpuFeatures::X86::kF16C);
        if (host.x86().has_bmi())   f.add(aj::CpuFeatures::X86::kBMI);
        if (host.x86().has_bmi2())  f.add(aj::CpuFeatures::X86::kBMI2);
        if (host.x86().has_lzcnt()) f.add(aj::CpuFeatures::X86::kLZCNT);
        if (host.x86().has_movbe()) f.add(aj::CpuFeatures::X86::kMOVBE);
    }
    if (cpu.isa >= ZJ_ISA_V4) {
        if (host.x86().has_avx512_f())    f.add(aj::CpuFeatures::X86::kAVX512_F);
        if (host.x86().has_avx512_cd())   f.add(aj::CpuFeatures::X86::kAVX512_CD);
        if (host.x86().has_avx512_bw())   f.add(aj::CpuFeatures::X86::kAVX512_BW);
        if (host.x86().has_avx512_dq())   f.add(aj::CpuFeatures::X86::kAVX512_DQ);
        if (host.x86().has_avx512_vl())   f.add(aj::CpuFeatures::X86::kAVX512_VL);
        if (host.x86().has_avx512_vbmi()) f.add(aj::CpuFeatures::X86::kAVX512_VBMI);
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    (void)cpu;
    const aj::CpuFeatures& host = aj::CpuInfo::host().features();
    f.add(aj::CpuFeatures::ARM::kARMv8a, aj::CpuFeatures::ARM::kFP, aj::CpuFeatures::ARM::kASIMD);
    if (host.arm().has_cssc()) f.add(aj::CpuFeatures::ARM::kCSSC);
#endif
    return f;
}

} // namespace

struct zj_runtime {
    aj::JitRuntime rt;
};

struct zj_ctx {
    zj_runtime* rt;
    zj_cpu_info_t cpu;
    aj::CodeHolder code;
    aj::StringLogger logger;
    ShimErrorHandler err_handler;
    ujit::BackendCompiler cc;
    ujit::UniCompiler uc;
    aj::FuncNode* func_node = nullptr;
    std::vector<ConstEntry> const_cache;
    zj_reg_t zero_vec = {0u, 0u};
    bool has_zero_vec = false;
    bool capture_disasm = false;

    zj_ctx(zj_runtime* runtime, const zj_cpu_info_t& cpu_info, bool disasm) noexcept
        : rt(runtime),
          cpu(cpu_info),
          cc(&code),
          uc(&cc, build_cpu_features(cpu_info), aj::CpuHints::kNone),
          capture_disasm(disasm) {
        const aj::CpuFeatures target_features = build_cpu_features(cpu_info);
        code.init(rt->rt.environment(), target_features);
        code.set_error_handler(&err_handler);
        if (capture_disasm) {
            logger.add_flags(aj::FormatFlags::kMachineCode | aj::FormatFlags::kRegCasts);
            code.set_logger(&logger);
        }
        code.attach(&cc);

        ujit::VecWidth vw = ujit::VecWidth::k128;
        if (cpu.vec_bytes == 64u) {
            vw = ujit::VecWidth::k512;
        } else if (cpu.vec_bytes == 32u) {
            vw = ujit::VecWidth::k256;
        }
        uc.init_vec_width(vw);
    }
};

namespace {

static inline bool ctx_ok(const zj_ctx* ctx) noexcept {
    return ctx != nullptr && ctx->err_handler.first_error == aj::Error::kOk;
}

static inline void set_shim_error(zj_ctx* ctx, aj::Error err, const char* msg) noexcept {
    if (ctx != nullptr) {
        ctx->err_handler.handle_error(err, msg, nullptr);
    }
}

static inline ujit::Mem legalise_mem(zj_ctx* ctx, zj_mem_t m, uint32_t access_bytes) noexcept {
#if defined(ASMJIT_UJIT_X86)
    (void)ctx;
    const uint32_t sz = m.size != 0u ? uint32_t(m.size) : access_bytes;
    if (m.is_label) {
        return aj::x86::ptr(aj::Label(m.label_id), m.disp, sz);
    }
    const aj::x86::Gp base = unpack_gp(m.base);
    if (m.has_index) {
        const aj::x86::Gp index = unpack_gp(m.index);
        return aj::x86::ptr(base, index, m.shift, m.disp, sz);
    }
    return aj::x86::ptr(base, m.disp, sz);
#elif defined(ASMJIT_UJIT_AARCH64)
    if (m.is_label) {
        return aj::a64::ptr(aj::Label(m.label_id), m.disp);
    }
    const aj::a64::Gp base = unpack_gp(m.base);
    if (m.has_index) {
        const aj::a64::Gp index = unpack_gp(m.index);
        aj::a64::Gp addr = ctx->uc.new_gp64();
        if (m.shift == 0u) {
            ctx->cc.add(addr, base, index);
        } else {
            ctx->cc.add(addr, base, index, aj::a64::lsl(m.shift));
        }
        if (m.disp != 0) {
            // Fold the displacement into the load/store immediate when it fits (ldur imm9 or
            // scaled uimm12) instead of spending an extra `add` per access.
            const uint32_t isz = m.size != 0u ? uint32_t(m.size) : access_bytes;
            const bool ifits_unscaled = (m.disp >= -256 && m.disp <= 255);
            const bool ifits_scaled = (isz > 0u) && (m.disp >= 0) && ((uint32_t(m.disp) % isz) == 0u) &&
                                      ((uint32_t(m.disp) / isz) <= 4095u);
            if (ifits_unscaled || ifits_scaled) {
                return aj::a64::ptr(addr, m.disp);
            }
            ctx->uc.emit_3i(ujit::UniOpRRR::kAdd, addr, addr, aj::Imm(m.disp));
        }
        return aj::a64::ptr(addr);
    }
    if (m.disp == 0) {
        return aj::a64::ptr(base);
    }
    const uint32_t sz = m.size != 0u ? uint32_t(m.size) : access_bytes;
    const bool fits_unscaled = (m.disp >= -256 && m.disp <= 255);
    const bool fits_scaled =
        (sz > 0u) && (m.disp >= 0) && ((uint32_t(m.disp) % sz) == 0u) && ((uint32_t(m.disp) / sz) <= 4095u);
    if (fits_unscaled || fits_scaled) {
        return aj::a64::ptr(base, m.disp);
    }
    aj::a64::Gp addr = ctx->uc.new_gp64();
    ctx->uc.emit_3i(ujit::UniOpRRR::kAdd, addr, base, aj::Imm(m.disp));
    return aj::a64::ptr(addr);
#endif
}

static zj_reg_t intern_const_vec(zj_ctx* ctx, const uint8_t* raw, uint32_t vec_bytes) noexcept {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};

    // Fast path for all-zero constant.
    bool all_zero = true;
    for (uint32_t i = 0; i < vec_bytes; ++i) {
        if (raw[i] != 0u) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return zj_vec_zero(ctx);
    }

    for (const ConstEntry& e : ctx->const_cache) {
        if (e.len == vec_bytes && std::memcmp(e.bytes, raw, vec_bytes) == 0) {
            return zj_reg_t{e.reg_id, e.reg_sig};
        }
    }

    ujit::Vec v = ctx->uc.new_vec();
    ujit::Mem mem = ctx->cc.new_const(aj::ConstPoolScope::kLocal, raw, vec_bytes);
    {
        ujit::ScopedInjector inject(&ctx->cc, &ctx->uc._func_init_hook);
        ctx->uc.v_loadavec(v, mem);
    }

    ConstEntry entry{};
    std::memcpy(entry.bytes, raw, vec_bytes);
    entry.len = vec_bytes;
    entry.reg_id = v.id();
    entry.reg_sig = v.signature().bits();
    ctx->const_cache.push_back(entry);
    return pack_reg(v);
}

} // namespace

extern "C" {

uint32_t zj_abi_version(void) {
    return ZJ_ABI_VERSION;
}

zj_runtime* zj_runtime_create(void) {
    return new (std::nothrow) zj_runtime();
}

void zj_runtime_destroy(zj_runtime* rt) {
    delete rt;
}

void zj_runtime_release(zj_runtime* rt, void* fn_ptr) {
    if (rt != nullptr && fn_ptr != nullptr) {
        rt->rt.release(fn_ptr);
    }
}

void zj_cpu_detect(zj_cpu_info_t* out, int32_t max_isa_override) {
    if (out == nullptr) return;
    std::memset(out, 0, sizeof(*out));

    if (max_isa_override == 0) {
        out->isa = ZJ_ISA_SCALAR;
        return;
    }

    const aj::CpuFeatures& f = aj::CpuInfo::host().features();

#if defined(ASMJIT_UJIT_X86)
    const bool has_v2 =
        f.x86().has_sse3() && f.x86().has_ssse3() && f.x86().has_sse4_1() &&
        f.x86().has_sse4_2() && f.x86().has_popcnt();
    const bool has_v3 =
        has_v2 && f.x86().has_avx() && f.x86().has_avx2() && f.x86().has_fma() &&
        f.x86().has_f16c() && f.x86().has_bmi() && f.x86().has_bmi2() &&
        f.x86().has_lzcnt() && f.x86().has_movbe();
    const bool has_v4 =
        has_v3 && f.x86().has_avx512_f() && f.x86().has_avx512_cd() &&
        f.x86().has_avx512_bw() && f.x86().has_avx512_dq() && f.x86().has_avx512_vl();

    // AVX-512 (v4) enabled by default on AVX-512 hosts after passing SDE -spr -mix/-ast & pytest gates.
    zj_isa_t target_isa = ZJ_ISA_SSE2;
    if (has_v4) {
        target_isa = ZJ_ISA_V4;
    } else if (has_v3) {
        target_isa = ZJ_ISA_V3;
    } else if (has_v2) {
        target_isa = ZJ_ISA_V2;
    }

    if (max_isa_override > 0) {
        const zj_isa_t req = static_cast<zj_isa_t>(max_isa_override);
        if (req >= ZJ_ISA_V4 && has_v4) {
            target_isa = ZJ_ISA_V4;
        } else if (req >= ZJ_ISA_V3 && has_v3) {
            target_isa = ZJ_ISA_V3;
        } else if (req >= ZJ_ISA_V2 && has_v2) {
            target_isa = ZJ_ISA_V2;
        } else {
            target_isa = ZJ_ISA_SSE2;
        }
    }

    out->isa = target_isa;
    if (target_isa == ZJ_ISA_V4) {
        out->vec_bytes = 64u;
        out->f32_lanes = 16u;
        out->has_ssse3 = 1u;
        out->has_sse4_1 = 1u;
        out->has_fma = 1u;
        out->has_f16c = 1u;
        out->has_avx2 = 1u;
        out->has_avx512 = 1u;
        out->prefer_hw_gather_f32 = 1u;
    } else if (target_isa == ZJ_ISA_V3) {
        out->vec_bytes = 32u;
        out->f32_lanes = 8u;
        out->has_ssse3 = 1u;
        out->has_sse4_1 = 1u;
        out->has_fma = 1u;
        out->has_f16c = 1u;
        out->has_avx2 = 1u;
        out->has_avx512 = 0u;
        // Default false on AVX2: register gather beats Zen 3 vpgatherdd by ~35% and avoids
        // Intel GDS microcode penalties (overrideable via ZEXPR_HW_GATHER=1, P1).
        out->prefer_hw_gather_f32 = 0u;
    } else if (target_isa == ZJ_ISA_V2) {
        out->vec_bytes = 16u;
        out->f32_lanes = 4u;
        out->has_ssse3 = 1u;
        out->has_sse4_1 = 1u;
        out->has_fma = 0u;
        out->has_f16c = 0u;
        out->has_avx2 = 0u;
        out->has_avx512 = 0u;
    } else {
        out->vec_bytes = 16u;
        out->f32_lanes = 4u;
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    (void)f;
    out->isa = ZJ_ISA_NEON;
    out->vec_bytes = 16u;
    out->f32_lanes = 4u;
    out->has_fma = 1u; // ARMv8.0 ASIMD has FMA (fmla/fmls) and fcvtl/fcvtn baseline
#endif
}

zj_ctx* zj_ctx_create(zj_runtime* rt, const zj_cpu_info_t* cpu, uint8_t capture_disasm) {
    if (rt == nullptr || cpu == nullptr || cpu->isa == ZJ_ISA_SCALAR) {
        return nullptr;
    }
    return new (std::nothrow) zj_ctx(rt, *cpu, capture_disasm != 0u);
}

void zj_ctx_destroy(zj_ctx* ctx) {
    delete ctx;
}

int32_t zj_ctx_begin_func(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return -1;
    aj::FuncSignature sig = aj::FuncSignature::build<
        void,
        uint8_t*,
        intptr_t,
        const uint8_t* const*,
        const intptr_t*,
        intptr_t,
        const float*
    >();
    ctx->func_node = ctx->cc.add_func(sig);
    if (ctx->func_node == nullptr) {
        return -1;
    }
    ctx->uc.hook_func();
    return 0;
}

zj_reg_t zj_ctx_arg_gp(zj_ctx* ctx, uint32_t index) {
    if (!ctx_ok(ctx) || ctx->func_node == nullptr || index >= 6u) {
        return zj_reg_t{0u, 0u};
    }
    ujit::Gp reg = ctx->uc.new_gp64();
    ctx->func_node->set_arg(index, reg);
    return pack_reg(reg);
}

void zj_ctx_ret_void(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return;
    ctx->cc.ret();
}

void zj_ctx_end_func(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return;
    ctx->uc.unhook_func();
    ctx->cc.end_func();
}

int32_t zj_ctx_finalize(zj_ctx* ctx, void** out_fn, size_t* out_size, uint8_t* out_execmem_denied) {
    if (out_fn != nullptr) *out_fn = nullptr;
    if (out_size != nullptr) *out_size = 0u;
    if (out_execmem_denied != nullptr) *out_execmem_denied = 0u;
    if (!ctx_ok(ctx)) return -1;

    aj::Error err = ctx->cc.finalize();
    if (err != aj::Error::kOk) {
        if (ctx->err_handler.first_error == aj::Error::kOk) {
            ctx->err_handler.first_error = err;
        }
        return -1;
    }

    const size_t code_size = ctx->code.code_size();
    void* fn = nullptr;
    err = ctx->rt->rt.add(&fn, &ctx->code);
    if (err != aj::Error::kOk) {
        if (ctx->err_handler.first_error == aj::Error::kOk) {
            ctx->err_handler.first_error = err;
        }
        if (out_execmem_denied != nullptr &&
            (err == aj::Error::kInvalidState || err == aj::Error::kOutOfMemory)) {
            *out_execmem_denied = 1u;
        }
        return -1;
    }

    if (out_fn != nullptr) *out_fn = fn;
    if (out_size != nullptr) *out_size = code_size;
    return 0;
}

void zj_ctx_get_disasm(const zj_ctx* ctx, const char** out_ptr, size_t* out_len) {
    if (out_ptr != nullptr) *out_ptr = "";
    if (out_len != nullptr) *out_len = 0u;
    if (ctx == nullptr || !ctx->capture_disasm) return;
    if (out_ptr != nullptr) *out_ptr = ctx->logger.data();
    if (out_len != nullptr) *out_len = ctx->logger.data_size();
}

uint32_t zj_ctx_last_error(const zj_ctx* ctx) {
    if (ctx == nullptr) return uint32_t(aj::Error::kInvalidArgument);
    return uint32_t(ctx->err_handler.first_error);
}

zj_reg_t zj_new_gp64(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};
    return pack_reg(ctx->uc.new_gp64());
}

zj_reg_t zj_new_gp32(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};
    return pack_reg(ctx->uc.new_gp32());
}

zj_reg_t zj_new_vec(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};
    return pack_reg(ctx->uc.new_vec());
}

zj_reg_t zj_new_vec128(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};
    return pack_reg(ctx->uc.new_vec128());
}

zj_label_t zj_new_label(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return zj_label_t{0u};
    return zj_label_t{ctx->uc.new_label().id()};
}

void zj_bind(zj_ctx* ctx, zj_label_t label) {
    if (!ctx_ok(ctx)) return;
    ctx->uc.bind(unpack_label(label));
}

zj_reg_t zj_vec_zero(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};
    if (ctx->has_zero_vec) return ctx->zero_vec;
    ujit::Vec v = ctx->uc.new_vec();
    {
        ujit::ScopedInjector inject(&ctx->cc, &ctx->uc._func_init_hook);
        ctx->uc.v_zero_i(v);
    }
    ctx->zero_vec = pack_reg(v);
    ctx->has_zero_vec = true;
    return ctx->zero_vec;
}

zj_reg_t zj_const_f32_splat(zj_ctx* ctx, float val) {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};
    uint8_t buf[64]{};
    const uint32_t lanes = ctx->cpu.f32_lanes;
    for (uint32_t i = 0; i < lanes; ++i) {
        std::memcpy(buf + i * 4u, &val, 4u);
    }
    return intern_const_vec(ctx, buf, ctx->cpu.vec_bytes);
}

zj_reg_t zj_const_i32_splat(zj_ctx* ctx, int32_t val) {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};
    uint8_t buf[64]{};
    const uint32_t lanes = ctx->cpu.f32_lanes;
    for (uint32_t i = 0; i < lanes; ++i) {
        std::memcpy(buf + i * 4u, &val, 4u);
    }
    return intern_const_vec(ctx, buf, ctx->cpu.vec_bytes);
}

zj_reg_t zj_const_bytes(zj_ctx* ctx, const uint8_t* data, uint32_t len) {
    if (!ctx_ok(ctx) || data == nullptr || len == 0u || len > 64u) return zj_reg_t{0u, 0u};
    uint8_t buf[64]{};
    const uint32_t vb = ctx->cpu.vec_bytes;
    if (len >= vb) {
        std::memcpy(buf, data, vb);
    } else {
        for (uint32_t off = 0; off < vb; off += len) {
            const uint32_t n = (vb - off) < len ? (vb - off) : len;
            std::memcpy(buf + off, data, n);
        }
    }
    return intern_const_vec(ctx, buf, vb);
}

void zj_gp_mov(zj_ctx* ctx, zj_reg_t dst, zj_reg_t src) {
    if (!ctx_ok(ctx)) return;
    ctx->uc.emit_mov(unpack_gp(dst), unpack_gp(src));
}

void zj_gp_mov_imm(zj_ctx* ctx, zj_reg_t dst, int64_t imm) {
    if (!ctx_ok(ctx)) return;
    ctx->uc.emit_mov(unpack_gp(dst), aj::Imm(imm));
}

void zj_gp_load(zj_ctx* ctx, zj_reg_t dst, zj_mem_t mem, uint32_t bytes, uint8_t sign_extend) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpRM op = ujit::UniOpRM::kLoadU64;
    switch (bytes) {
        case 1u: op = sign_extend ? ujit::UniOpRM::kLoadI8 : ujit::UniOpRM::kLoadU8; break;
        case 2u: op = sign_extend ? ujit::UniOpRM::kLoadI16 : ujit::UniOpRM::kLoadU16; break;
        case 4u: op = sign_extend ? ujit::UniOpRM::kLoadI32 : ujit::UniOpRM::kLoadU32; break;
        case 8u: op = ujit::UniOpRM::kLoadU64; break;
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid gp_load width"); return;
    }
    ctx->uc.emit_rm(op, unpack_gp(dst), legalise_mem(ctx, mem, bytes));
}

void zj_gp_store(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src, uint32_t bytes) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpMR op = ujit::UniOpMR::kStoreU64;
    switch (bytes) {
        case 1u: op = ujit::UniOpMR::kStoreU8; break;
        case 2u: op = ujit::UniOpMR::kStoreU16; break;
        case 4u: op = ujit::UniOpMR::kStoreU32; break;
        case 8u: op = ujit::UniOpMR::kStoreU64; break;
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid gp_store width"); return;
    }
    ctx->uc.emit_mr(op, legalise_mem(ctx, mem, bytes), unpack_gp(src));
}

void zj_gp_op2(zj_ctx* ctx, zj_gp_op2_t op, zj_reg_t dst, zj_reg_t src) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpRR uop = ujit::UniOpRR::kNeg;
    switch (op) {
        case ZJ_GP_NEG:   uop = ujit::UniOpRR::kNeg; break;
        case ZJ_GP_NOT:   uop = ujit::UniOpRR::kNot; break;
        case ZJ_GP_BSWAP: uop = ujit::UniOpRR::kBSwap; break;
        case ZJ_GP_CLZ:   uop = ujit::UniOpRR::kCLZ; break;
        case ZJ_GP_CTZ:   uop = ujit::UniOpRR::kCTZ; break;
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid gp_op2 opcode"); return;
    }
    ctx->uc.emit_2i(uop, unpack_gp(dst), unpack_gp(src));
}

static inline bool map_gp_op3(zj_gp_op3_t op, ujit::UniOpRRR* out) noexcept {
    switch (op) {
        case ZJ_GP_ADD:  *out = ujit::UniOpRRR::kAdd; return true;
        case ZJ_GP_SUB:  *out = ujit::UniOpRRR::kSub; return true;
        case ZJ_GP_MUL:  *out = ujit::UniOpRRR::kMul; return true;
        case ZJ_GP_AND:  *out = ujit::UniOpRRR::kAnd; return true;
        case ZJ_GP_OR:   *out = ujit::UniOpRRR::kOr; return true;
        case ZJ_GP_XOR:  *out = ujit::UniOpRRR::kXor; return true;
        case ZJ_GP_SHL:  *out = ujit::UniOpRRR::kSll; return true;
        case ZJ_GP_SHR:  *out = ujit::UniOpRRR::kSrl; return true;
        case ZJ_GP_SAR:  *out = ujit::UniOpRRR::kSra; return true;
        case ZJ_GP_SMIN: *out = ujit::UniOpRRR::kSMin; return true;
        case ZJ_GP_SMAX: *out = ujit::UniOpRRR::kSMax; return true;
        case ZJ_GP_UMIN: *out = ujit::UniOpRRR::kUMin; return true;
        case ZJ_GP_UMAX: *out = ujit::UniOpRRR::kUMax; return true;
        default: return false;
    }
}

void zj_gp_op3(zj_ctx* ctx, zj_gp_op3_t op, zj_reg_t dst, zj_reg_t a, zj_reg_t b) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpRRR uop;
    if (!map_gp_op3(op, &uop)) {
        set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid gp_op3 opcode");
        return;
    }
    ctx->uc.emit_3i(uop, unpack_gp(dst), unpack_gp(a), unpack_gp(b));
}

void zj_gp_op3_imm(zj_ctx* ctx, zj_gp_op3_t op, zj_reg_t dst, zj_reg_t a, int64_t imm) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpRRR uop;
    if (!map_gp_op3(op, &uop)) {
        set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid gp_op3_imm opcode");
        return;
    }
    ctx->uc.emit_3i(uop, unpack_gp(dst), unpack_gp(a), aj::Imm(imm));
}

void zj_gp_cmov(zj_ctx* ctx, zj_cond_t cond, zj_reg_t dst, zj_reg_t lhs, zj_reg_t rhs, zj_reg_t src) {
    if (!ctx_ok(ctx)) return;
    ujit::UniCondition c(ujit::UniOpCond::kCompare, to_cond_code(cond), unpack_gp(lhs), unpack_gp(rhs));
    ctx->uc.emit_cmov(unpack_gp(dst), unpack_gp(src), c);
}

void zj_gp_select(zj_ctx* ctx, zj_cond_t cond, zj_reg_t dst, zj_reg_t lhs, zj_reg_t rhs, zj_reg_t true_val, zj_reg_t false_val) {
    if (!ctx_ok(ctx)) return;
    ujit::UniCondition c(ujit::UniOpCond::kCompare, to_cond_code(cond), unpack_gp(lhs), unpack_gp(rhs));
    if (dst.id == lhs.id || dst.id == rhs.id) {
        ujit::Gp tmp = ctx->uc.new_gp64();
        ctx->uc.emit_select(tmp, unpack_gp(true_val), unpack_gp(false_val), c);
        ctx->uc.emit_mov(unpack_gp(dst), tmp);
    } else {
        ctx->uc.emit_select(unpack_gp(dst), unpack_gp(true_val), unpack_gp(false_val), c);
    }
}

void zj_jmp(zj_ctx* ctx, zj_label_t target) {
    if (!ctx_ok(ctx)) return;
    ctx->uc.emit_j(unpack_label(target));
}

void zj_jcc(zj_ctx* ctx, zj_cond_t cond, zj_reg_t lhs, zj_reg_t rhs, zj_label_t target) {
    if (!ctx_ok(ctx)) return;
    ujit::UniCondition c(ujit::UniOpCond::kCompare, to_cond_code(cond), unpack_gp(lhs), unpack_gp(rhs));
    ctx->uc.emit_j_if(unpack_label(target), c);
}

void zj_jcc_imm(zj_ctx* ctx, zj_cond_t cond, zj_reg_t lhs, int64_t rhs_imm, zj_label_t target) {
    if (!ctx_ok(ctx)) return;
    ujit::UniCondition c(ujit::UniOpCond::kCompare, to_cond_code(cond), unpack_gp(lhs), aj::Imm(rhs_imm));
    ctx->uc.emit_j_if(unpack_label(target), c);
}

void zj_vec_op2(zj_ctx* ctx, zj_vec_op2_t op, zj_reg_t dst, zj_reg_t src) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpVV uop;
    switch (op) {
        case ZJ_V2_MOV:                  uop = ujit::UniOpVV::kMov; break;
        case ZJ_V2_ABS_F32:              uop = ujit::UniOpVV::kAbsF32; break;
        case ZJ_V2_NEG_F32:              uop = ujit::UniOpVV::kNegF32; break;
        case ZJ_V2_SQRT_F32:             uop = ujit::UniOpVV::kSqrtF32; break;
        case ZJ_V2_TRUNC_F32:            uop = ujit::UniOpVV::kTruncF32; break;
        case ZJ_V2_FLOOR_F32:            uop = ujit::UniOpVV::kFloorF32; break;
        case ZJ_V2_CEIL_F32:             uop = ujit::UniOpVV::kCeilF32; break;
        case ZJ_V2_ROUND_EVEN_F32:       uop = ujit::UniOpVV::kRoundEvenF32; break;
        case ZJ_V2_CVT_I32_TO_F32:       uop = ujit::UniOpVV::kCvtI32ToF32; break;
        case ZJ_V2_CVT_ROUND_F32_TO_I32: uop = ujit::UniOpVV::kCvtRoundF32ToI32; break;
        case ZJ_V2_CVT_TRUNC_F32_TO_I32: uop = ujit::UniOpVV::kCvtTruncF32ToI32; break;
        case ZJ_V2_NOT_U32:              uop = ujit::UniOpVV::kNotU32; break;
        case ZJ_V2_ABS_I16:              uop = ujit::UniOpVV::kAbsI16; break;
        case ZJ_V2_ABS_I32:              uop = ujit::UniOpVV::kAbsI32; break;
        case ZJ_V2_NEG_I16: {
            ujit::Vec zero = unpack_vec(zj_vec_zero(ctx));
            ctx->uc.emit_3v(ujit::UniOpVVV::kSubU16, unpack_vec(dst), zero, unpack_vec(src));
            return;
        }
        case ZJ_V2_NEG_I32: {
            ujit::Vec zero = unpack_vec(zj_vec_zero(ctx));
            ctx->uc.emit_3v(ujit::UniOpVVV::kSubU32, unpack_vec(dst), zero, unpack_vec(src));
            return;
        }
        case ZJ_V2_CVT_U8_LO_TO_U16:     uop = ujit::UniOpVV::kCvtU8LoToU16; break;
        case ZJ_V2_CVT_U8_HI_TO_U16:     uop = ujit::UniOpVV::kCvtU8HiToU16; break;
        case ZJ_V2_CVT_U16_LO_TO_U32:    uop = ujit::UniOpVV::kCvtU16LoToU32; break;
        case ZJ_V2_CVT_U16_HI_TO_U32:    uop = ujit::UniOpVV::kCvtU16HiToU32; break;
        case ZJ_V2_CVT_U8_TO_U32:        uop = ujit::UniOpVV::kCvtU8ToU32; break;
        case ZJ_V2_CVT_I16_LO_TO_I32:    uop = ujit::UniOpVV::kCvtI16LoToI32; break;
        case ZJ_V2_CVT_I16_HI_TO_I32:    uop = ujit::UniOpVV::kCvtI16HiToI32; break;
        case ZJ_V2_BROADCAST_U8:         uop = ujit::UniOpVV::kBroadcastU8; break;
        case ZJ_V2_BROADCAST_U16:        uop = ujit::UniOpVV::kBroadcastU16; break;
        case ZJ_V2_BROADCAST_U32:        uop = ujit::UniOpVV::kBroadcastU32; break;
        case ZJ_V2_BROADCAST_U64:        uop = ujit::UniOpVV::kBroadcastU64; break;
        case ZJ_V2_RSQRT_F32: {
            // Refined hardware rsqrt(x) on clamped positive x (max_rel <= 2e-7 across all ISAs):
            // - x86 SSE2/v2: rsqrtps + 1 NR; AVX2: vrsqrtps + 1 NR; AVX-512: vrsqrt14ps + 1 NR.
            // - AArch64 NEON: frsqrte + 2x frsqrts (< 0.5 ULP).
            ujit::Vec s = unpack_vec(src);
            ujit::Vec d = unpack_vec(dst);
#if defined(ASMJIT_UJIT_X86)
            ujit::Vec y = ctx->uc.new_vec();
            if (ctx->cpu.vec_bytes == 64u) {
                ctx->cc.vrsqrt14ps(y.zmm(), s.zmm());
            } else if (ctx->cpu.vec_bytes == 32u) {
                ctx->cc.vrsqrtps(y.ymm(), s.ymm());
            } else {
                ctx->cc.rsqrtps(y.xmm(), s.xmm());
            }
            ujit::Vec xh = ctx->uc.new_vec();
            ujit::Vec yy = ctx->uc.new_vec();
            ujit::Vec c_half = unpack_vec(zj_const_f32_splat(ctx, 0.5f));
            ujit::Vec c_three_half = unpack_vec(zj_const_f32_splat(ctx, 1.5f));
            ctx->uc.emit_3v(ujit::UniOpVVV::kMulF32, xh, s, c_half);
            ctx->uc.emit_3v(ujit::UniOpVVV::kMulF32, yy, y, y);
            ctx->uc.emit_4v(ujit::UniOpVVVV::kNMAddF32, yy, xh, yy, c_three_half);
            ctx->uc.emit_3v(ujit::UniOpVVV::kMulF32, d, y, yy);
#elif defined(ASMJIT_UJIT_AARCH64)
            ujit::Vec y = ctx->uc.new_vec128();
            ujit::Vec t = ctx->uc.new_vec128();
            ctx->cc.frsqrte(y.s4(), s.s4());
            ctx->cc.fmul(t.s4(), y.s4(), y.s4());
            ctx->cc.frsqrts(t.s4(), s.s4(), t.s4());
            ctx->cc.fmul(y.s4(), y.s4(), t.s4());
            ctx->cc.fmul(t.s4(), y.s4(), y.s4());
            ctx->cc.frsqrts(t.s4(), s.s4(), t.s4());
            ctx->cc.fmul(d.s4(), y.s4(), t.s4());
#endif
            return;
        }
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid vec_op2 opcode"); return;
    }
    ctx->uc.emit_2v(uop, unpack_vec(dst), unpack_vec(src));
}

void zj_vec_op3(zj_ctx* ctx, zj_vec_op3_t op, zj_reg_t dst, zj_reg_t a, zj_reg_t b) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpVVV uop;
    switch (op) {
        case ZJ_V3_ADD_F32:       uop = ujit::UniOpVVV::kAddF32; break;
        case ZJ_V3_SUB_F32:       uop = ujit::UniOpVVV::kSubF32; break;
        case ZJ_V3_MUL_F32:       uop = ujit::UniOpVVV::kMulF32; break;
        case ZJ_V3_DIV_F32:       uop = ujit::UniOpVVV::kDivF32; break;
        case ZJ_V3_MIN_F32:       uop = ujit::UniOpVVV::kMinF32; break;
        case ZJ_V3_MAX_F32:       uop = ujit::UniOpVVV::kMaxF32; break;
        case ZJ_V3_CMP_EQ_F32:    uop = ujit::UniOpVVV::kCmpEqF32; break;
        case ZJ_V3_CMP_NE_F32:    uop = ujit::UniOpVVV::kCmpNeF32; break;
        case ZJ_V3_CMP_LT_F32:    uop = ujit::UniOpVVV::kCmpLtF32; break;
        case ZJ_V3_CMP_LE_F32:    uop = ujit::UniOpVVV::kCmpLeF32; break;
        case ZJ_V3_CMP_GT_F32:    uop = ujit::UniOpVVV::kCmpGtF32; break;
        case ZJ_V3_CMP_GE_F32:    uop = ujit::UniOpVVV::kCmpGeF32; break;
        case ZJ_V3_AND_U32:       uop = ujit::UniOpVVV::kAndU32; break;
        case ZJ_V3_OR_U32:        uop = ujit::UniOpVVV::kOrU32; break;
        case ZJ_V3_XOR_U32:       uop = ujit::UniOpVVV::kXorU32; break;
        case ZJ_V3_ANDN_U32:      uop = ujit::UniOpVVV::kAndnU32; break;
        case ZJ_V3_BIC_U32:       uop = ujit::UniOpVVV::kBicU32; break;
        case ZJ_V3_ADD_I32:       uop = ujit::UniOpVVV::kAddU32; break;
        case ZJ_V3_SUB_I32:       uop = ujit::UniOpVVV::kSubU32; break;
        case ZJ_V3_MUL_I32:       uop = ujit::UniOpVVV::kMulU32; break;
        case ZJ_V3_MIN_I32:       uop = ujit::UniOpVVV::kMinI32; break;
        case ZJ_V3_MAX_I32:       uop = ujit::UniOpVVV::kMaxI32; break;
        case ZJ_V3_CMP_EQ_I32:    uop = ujit::UniOpVVV::kCmpEqU32; break;
        case ZJ_V3_CMP_GT_I32:    uop = ujit::UniOpVVV::kCmpGtI32; break;
        case ZJ_V3_ADD_I16:       uop = ujit::UniOpVVV::kAddU16; break;
        case ZJ_V3_SUB_I16:       uop = ujit::UniOpVVV::kSubU16; break;
        case ZJ_V3_ADDS_I16:      uop = ujit::UniOpVVV::kAddsI16; break;
        case ZJ_V3_SUBS_I16:      uop = ujit::UniOpVVV::kSubsI16; break;
        case ZJ_V3_ADDS_U16:      uop = ujit::UniOpVVV::kAddsU16; break;
        case ZJ_V3_SUBS_U16:      uop = ujit::UniOpVVV::kSubsU16; break;
        case ZJ_V3_MUL_I16:       uop = ujit::UniOpVVV::kMulU16; break;
        case ZJ_V3_MIN_I16:       uop = ujit::UniOpVVV::kMinI16; break;
        case ZJ_V3_MAX_I16:       uop = ujit::UniOpVVV::kMaxI16; break;
        case ZJ_V3_MIN_U16:       uop = ujit::UniOpVVV::kMinU16; break;
        case ZJ_V3_MAX_U16:       uop = ujit::UniOpVVV::kMaxU16; break;
        case ZJ_V3_CMP_EQ_I16:    uop = ujit::UniOpVVV::kCmpEqU16; break;
        case ZJ_V3_CMP_GT_I16:    uop = ujit::UniOpVVV::kCmpGtI16; break;
        case ZJ_V3_PACKS_I32_I16: uop = ujit::UniOpVVV::kPacksI32_I16; break;
        case ZJ_V3_PACKS_I32_U16: uop = ujit::UniOpVVV::kPacksI32_U16; break;
        case ZJ_V3_PACKS_I16_U8:  uop = ujit::UniOpVVV::kPacksI16_U8; break;
        case ZJ_V3_PACKS_I16_I8:  uop = ujit::UniOpVVV::kPacksI16_I8; break;
        case ZJ_V3_UNPACK_LO_U8:  uop = ujit::UniOpVVV::kInterleaveLoU8; break;
        case ZJ_V3_UNPACK_HI_U8:  uop = ujit::UniOpVVV::kInterleaveHiU8; break;
        case ZJ_V3_UNPACK_LO_U16: uop = ujit::UniOpVVV::kInterleaveLoU16; break;
        case ZJ_V3_UNPACK_HI_U16: uop = ujit::UniOpVVV::kInterleaveHiU16; break;
        case ZJ_V3_UNPACK_LO_U32: uop = ujit::UniOpVVV::kInterleaveLoU32; break;
        case ZJ_V3_UNPACK_HI_U32: uop = ujit::UniOpVVV::kInterleaveHiU32; break;
#if defined(ASMJIT_UJIT_X86)
        case ZJ_V3_PERMUTE_U32:   uop = ujit::UniOpVVV::kPermuteU32; break;
#endif
        case ZJ_V3_ADD_U8:        uop = ujit::UniOpVVV::kAddU8; break;
        case ZJ_V3_SUB_U8:        uop = ujit::UniOpVVV::kSubU8; break;
        case ZJ_V3_ADDS_U8:       uop = ujit::UniOpVVV::kAddsU8; break;
        case ZJ_V3_SUBS_U8:       uop = ujit::UniOpVVV::kSubsU8; break;
        case ZJ_V3_MIN_U8:        uop = ujit::UniOpVVV::kMinU8; break;
        case ZJ_V3_MAX_U8:        uop = ujit::UniOpVVV::kMaxU8; break;
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid or unsupported vec_op3 opcode"); return;
    }
    ctx->uc.emit_3v(uop, unpack_vec(dst), unpack_vec(a), unpack_vec(b));
}

void zj_vec_op4(zj_ctx* ctx, zj_vec_op4_t op, zj_reg_t dst, zj_reg_t a, zj_reg_t b, zj_reg_t c) {
    if (!ctx_ok(ctx)) return;
    // On non-FMA x86 (SSE2/v2), UniCompiler lowers kNMAddF32 as `(-a)*b + c` via sse_fsign_flip,
    // flipping the NaN sign bit of `a` (and adding an xorps). Emit `c - a * b` directly to match
    // interp.fnmaF32 bit-for-bit on NaN payloads/signs (review.11.md sin/cos/tan L1-L2 parity).
    if (op == ZJ_V4_NMADD_F32 && !ctx->cpu.has_fma) {
        ujit::Vec tmp = ctx->uc.new_vec();
        ctx->uc.emit_3v(ujit::UniOpVVV::kMulF32, tmp, unpack_vec(a), unpack_vec(b));
        ctx->uc.emit_3v(ujit::UniOpVVV::kSubF32, unpack_vec(dst), unpack_vec(c), tmp);
        return;
    }
    ujit::UniOpVVVV uop;
    switch (op) {
        case ZJ_V4_MADD_F32:  uop = ujit::UniOpVVVV::kMAddF32; break;
        case ZJ_V4_MSUB_F32:  uop = ujit::UniOpVVVV::kMSubF32; break;
        case ZJ_V4_NMADD_F32: uop = ujit::UniOpVVVV::kNMAddF32; break;
        case ZJ_V4_NMSUB_F32: uop = ujit::UniOpVVVV::kNMSubF32; break;
        case ZJ_V4_BLENDV_U8: uop = ujit::UniOpVVVV::kBlendV_U8; break;
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid vec_op4 opcode"); return;
    }
    ctx->uc.emit_4v(uop, unpack_vec(dst), unpack_vec(a), unpack_vec(b), unpack_vec(c));
}

void zj_vec_op2i(zj_ctx* ctx, zj_vec_op2i_t op, zj_reg_t dst, zj_reg_t src, uint32_t imm) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpVVI uop;
    switch (op) {
        case ZJ_V2I_SLL_U16:       uop = ujit::UniOpVVI::kSllU16; break;
        case ZJ_V2I_SRL_U16:       uop = ujit::UniOpVVI::kSrlU16; break;
        case ZJ_V2I_SRA_I16:       uop = ujit::UniOpVVI::kSraI16; break;
        case ZJ_V2I_SLL_U32:       uop = ujit::UniOpVVI::kSllU32; break;
        case ZJ_V2I_SRL_U32:       uop = ujit::UniOpVVI::kSrlU32; break;
        case ZJ_V2I_SRA_I32:       uop = ujit::UniOpVVI::kSraI32; break;
        case ZJ_V2I_SWIZZLE_U32X4: uop = ujit::UniOpVVI::kSwizzleU32x4; break;
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid vec_op2i opcode"); return;
    }
    ctx->uc.emit_2vi(uop, unpack_vec(dst), unpack_vec(src), imm);
}

void zj_vec_load(zj_ctx* ctx, zj_vec_load_t op, zj_reg_t dst, zj_mem_t mem) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec d = unpack_vec(dst);
    switch (op) {
        case ZJ_VLOAD_A_VEC:
            ctx->uc.v_loadavec(d, legalise_mem(ctx, mem, ctx->cpu.vec_bytes));
            return;
        case ZJ_VLOAD_U_VEC:
            ctx->uc.v_loaduvec(d, legalise_mem(ctx, mem, ctx->cpu.vec_bytes));
            return;
        case ZJ_VLOAD_U_32:
            ctx->uc.emit_vm(ujit::UniOpVM::kLoad32_U32, d, legalise_mem(ctx, mem, 4u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_U_64:
            ctx->uc.emit_vm(ujit::UniOpVM::kLoad64_U32, d, legalise_mem(ctx, mem, 8u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_U_128:
            ctx->uc.emit_vm(ujit::UniOpVM::kLoad128_U32, d, legalise_mem(ctx, mem, 16u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_U_256:
            ctx->uc.emit_vm(ujit::UniOpVM::kLoad256_U32, d, legalise_mem(ctx, mem, 32u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_A_128:
            ctx->uc.emit_vm(ujit::UniOpVM::kLoad128_U32, d, legalise_mem(ctx, mem, 16u), ujit::Alignment(16));
            return;
        case ZJ_VLOAD_CVT_U8_TO_U16:
            ctx->uc.emit_vm(ujit::UniOpVM::kLoadCvtN_U8ToU16, d, legalise_mem(ctx, mem, ctx->cpu.vec_bytes / 2u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_CVT_U8_TO_U32:
#if defined(ASMJIT_UJIT_X86)
            if (!ctx->cpu.has_sse4_1) {
                // Work around upstream AsmJit 1.23.0 bug at uni_compiler_x86.cpp:5402
                // where kLoadCvtN_U8ToU32 emits VEX `vmovd` on pre-SSE4.1 CPUs.
                ujit::Mem m4 = legalise_mem(ctx, mem, 4u);
                m4.set_size(4);
                ctx->cc.movd(d.xmm(), m4);
                ctx->uc.emit_2v(ujit::UniOpVV::kCvtU8LoToU16, d, d);
                ctx->uc.emit_2v(ujit::UniOpVV::kCvtU16LoToU32, d, d);
                return;
            }
#elif defined(ASMJIT_UJIT_AARCH64)
            // V2: UniCompiler a64 kLoadCvtN_U8ToU32 loads dst.size()/2 = 8 bytes (OOB read and
            // unaligned scaled offset fault); load exactly 4 bytes and zero-extend to u32.
            ctx->uc.emit_vm(ujit::UniOpVM::kLoad32_U32, d, legalise_mem(ctx, mem, 4u), ujit::Alignment(1));
            ctx->uc.emit_2v(ujit::UniOpVV::kCvtU8ToU32, d, d);
            return;
#endif
            ctx->uc.emit_vm(ujit::UniOpVM::kLoadCvtN_U8ToU32, d, legalise_mem(ctx, mem, ctx->cpu.vec_bytes / 4u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_CVT_U16_TO_U32:
            ctx->uc.emit_vm(ujit::UniOpVM::kLoadCvtN_U16ToU32, d, legalise_mem(ctx, mem, ctx->cpu.vec_bytes / 2u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_BROADCAST_U32:
            ctx->uc.v_broadcast_u32(d, legalise_mem(ctx, mem, 4u));
            return;
        case ZJ_VLOAD_BROADCAST_U64:
            ctx->uc.v_broadcast_u64(d, legalise_mem(ctx, mem, 8u));
            return;
        default:
            set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid vec_load opcode");
            return;
    }
}

void zj_vec_store(zj_ctx* ctx, zj_vec_store_t op, zj_mem_t mem, zj_reg_t src) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s = unpack_vec(src);
    switch (op) {
        case ZJ_VSTORE_A_VEC:
            ctx->uc.v_storeavec(legalise_mem(ctx, mem, ctx->cpu.vec_bytes), s);
            return;
        case ZJ_VSTORE_U_VEC:
            ctx->uc.v_storeuvec(legalise_mem(ctx, mem, ctx->cpu.vec_bytes), s);
            return;
        case ZJ_VSTORE_U_32:
            ctx->uc.emit_mv(ujit::UniOpMV::kStore32_U32, legalise_mem(ctx, mem, 4u), s, ujit::Alignment(1));
            return;
        case ZJ_VSTORE_U_64:
            ctx->uc.emit_mv(ujit::UniOpMV::kStore64_U32, legalise_mem(ctx, mem, 8u), s, ujit::Alignment(1));
            return;
        case ZJ_VSTORE_U_128:
            ctx->uc.emit_mv(ujit::UniOpMV::kStore128_U32, legalise_mem(ctx, mem, 16u), s, ujit::Alignment(1));
            return;
        case ZJ_VSTORE_U_256:
            ctx->uc.emit_mv(ujit::UniOpMV::kStore256_U32, legalise_mem(ctx, mem, 32u), s, ujit::Alignment(1));
            return;
        case ZJ_VSTORE_A_128:
            ctx->uc.emit_mv(ujit::UniOpMV::kStore128_U32, legalise_mem(ctx, mem, 16u), s, ujit::Alignment(16));
            return;
        default:
            set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid vec_store opcode");
            return;
    }
}

void zj_vec_from_gp32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t gp32) {
    if (!ctx_ok(ctx)) return;
    ctx->uc.s_mov_u32(unpack_vec(dst), unpack_gp(gp32).r32());
}

void zj_vec_broadcast_gp32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t gp32) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec v = unpack_vec(dst);
    ctx->uc.s_mov_u32(v, unpack_gp(gp32).r32());
    ctx->uc.emit_2v(ujit::UniOpVV::kBroadcastU32, v, v);
}

void zj_gp32_from_vec(zj_ctx* ctx, zj_reg_t gp32, zj_reg_t vec) {
    if (!ctx_ok(ctx)) return;
    ctx->uc.s_mov_u32(unpack_gp(gp32).r32(), unpack_vec(vec));
}

// ============================================================================
// Arch-specific escape hatches
// ============================================================================

void zj_vec_cvt_f16_to_f32(zj_ctx* ctx, zj_reg_t dst, zj_mem_t mem) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec d = unpack_vec(dst);
    const uint32_t load_bytes = ctx->cpu.f32_lanes * 2u;
    ujit::Mem m = legalise_mem(ctx, mem, load_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (!ctx->cpu.has_f16c) {
        set_shim_error(ctx, aj::Error::kFeatureNotEnabled, "f16 load requires F16C");
        return;
    }
    if (ctx->cpu.vec_bytes == 64u) {
        ctx->cc.vcvtph2ps(d.zmm(), m);
    } else if (ctx->cpu.vec_bytes == 32u) {
        ctx->cc.vcvtph2ps(d.ymm(), m);
    } else {
        ctx->cc.vcvtph2ps(d.xmm(), m);
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec tmp = ctx->uc.new_vec128();
    ctx->cc.ldr(tmp.d(), m);
    ctx->cc.fcvtl(d.s4(), tmp.h4());
#endif
}

void zj_vec_cvt_f32_to_f16(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s = unpack_vec(src);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 2u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (!ctx->cpu.has_f16c) {
        set_shim_error(ctx, aj::Error::kFeatureNotEnabled, "f16 store requires F16C");
        return;
    }
    // Rounding mode 4 = MXCSR (round to nearest even).
    if (ctx->cpu.vec_bytes == 64u) {
        ctx->cc.vcvtps2ph(m, s.zmm(), 4);
    } else if (ctx->cpu.vec_bytes == 32u) {
        ctx->cc.vcvtps2ph(m, s.ymm(), 4);
    } else {
        ctx->cc.vcvtps2ph(m, s.xmm(), 4);
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec tmp = ctx->uc.new_vec128();
    ctx->cc.fcvtn(tmp.h4(), s.s4());
    ctx->cc.str(tmp.d(), m);
#endif
}

void zj_vec_store_u8_from_i32(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src_i32) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s = unpack_vec(src_i32);
    const uint32_t store_bytes = ctx->cpu.f32_lanes;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        // Caller clamped src_i32 to [0, peak] before converting to i32.
        ctx->cc.vpmovusdb(m, s.zmm());
    } else if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackssdw(t.ymm(), s.ymm(), s.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vpackuswb(t.ymm(), t.ymm(), t.ymm());
        ctx->cc.vmovq(m, t.xmm());
    } else {
        ujit::Vec t = ctx->uc.new_vec128();
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_I16, t, s, s);
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, t, t);
        ctx->uc.emit_mv(ujit::UniOpMV::kStore32_U32, m, t, ujit::Alignment(1));
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    // L3: narrow signed i32 -> signed i16 -> unsigned u8 so values >= 32768 saturate to 255.
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_I16, t, s, s);
    ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, t, t);
    ctx->uc.emit_mv(ujit::UniOpMV::kStore32_U32, m, t, ujit::Alignment(1));
#endif
}

void zj_vec_store_u8_from_i32x2(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src0_i32, zj_reg_t src1_i32) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s0 = unpack_vec(src0_i32);
    ujit::Vec s1 = unpack_vec(src1_i32);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 2u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        ctx->cc.vpmovusdb(m, s0.zmm());
        zj_mem_t m1 = mem;
        m1.disp += 16;
        ctx->cc.vpmovusdb(legalise_mem(ctx, m1, 16u), s1.zmm());
    } else if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackssdw(t.ymm(), s0.ymm(), s1.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vpackuswb(t.ymm(), t.ymm(), t.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vmovdqu(m, t.xmm());
    } else {
        ujit::Vec t = ctx->uc.new_vec128();
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_I16, t, s0, s1);
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, t, t);
        ctx->uc.emit_mv(ujit::UniOpMV::kStore64_U32, m, t, ujit::Alignment(1));
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    // L3: narrow signed i32 -> signed i16 -> unsigned u8 so values >= 32768 saturate to 255.
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_I16, t, s0, s1);
    ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, t, t);
    ctx->uc.emit_mv(ujit::UniOpMV::kStore64_U32, m, t, ujit::Alignment(1));
#endif
}

void zj_vec_store_u8_from_i16(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src_i16) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s = unpack_vec(src_i16);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 2u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->uc.emit_3v(ujit::UniOpVVV::kMaxI16, t, s, unpack_vec(zj_vec_zero(ctx)));
        ctx->cc.vpmovuswb(m, t.zmm());
    } else if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackuswb(t.ymm(), s.ymm(), s.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vmovdqu(m, t.xmm());
    } else {
        ujit::Vec t = ctx->uc.new_vec128();
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, s, s);
        ctx->uc.emit_mv(ujit::UniOpMV::kStore64_U32, m, t, ujit::Alignment(1));
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, s, s);
    ctx->uc.emit_mv(ujit::UniOpMV::kStore64_U32, m, t, ujit::Alignment(1));
#endif
}

void zj_vec_store_u8_from_i16x2(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src0_i16, zj_reg_t src1_i16) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s0 = unpack_vec(src0_i16);
    ujit::Vec s1 = unpack_vec(src1_i16);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 4u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        ujit::Vec zero = unpack_vec(zj_vec_zero(ctx));
        ujit::Vec t0 = ctx->uc.new_vec();
        ujit::Vec t1 = ctx->uc.new_vec();
        ctx->uc.emit_3v(ujit::UniOpVVV::kMaxI16, t0, s0, zero);
        ctx->uc.emit_3v(ujit::UniOpVVV::kMaxI16, t1, s1, zero);
        ctx->cc.vpmovuswb(m, t0.zmm());
        zj_mem_t m1 = mem;
        m1.disp += 32;
        ctx->cc.vpmovuswb(legalise_mem(ctx, m1, 32u), t1.zmm());
    } else if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackuswb(t.ymm(), s0.ymm(), s1.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vmovdqu(m, t.ymm());
    } else {
        ujit::Vec t = ctx->uc.new_vec128();
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, s0, s1);
        ctx->uc.v_storeuvec(m, t);
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, s0, s1);
    ctx->uc.v_storeuvec(m, t);
#endif
}

void zj_vec_store_u16_from_i32(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src_i32) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s = unpack_vec(src_i32);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 2u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        // Caller clamped src_i32 to [0, peak] before converting to i32.
        ctx->cc.vpmovusdw(m, s.zmm());
    } else if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackusdw(t.ymm(), s.ymm(), s.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vmovdqu(m, t.xmm());
    } else {
        ujit::Vec t = ctx->uc.new_vec128();
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_U16, t, s, s);
        ctx->uc.emit_mv(ujit::UniOpMV::kStore64_U32, m, t, ujit::Alignment(1));
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_U16, t, s, s);
    ctx->uc.emit_mv(ujit::UniOpMV::kStore64_U32, m, t, ujit::Alignment(1));
#endif
}

void zj_vec_store_u16_from_i32x2(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src0_i32, zj_reg_t src1_i32) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s0 = unpack_vec(src0_i32);
    ujit::Vec s1 = unpack_vec(src1_i32);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 4u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        ctx->cc.vpmovusdw(m, s0.zmm());
        zj_mem_t m1 = mem;
        m1.disp += 32;
        ctx->cc.vpmovusdw(legalise_mem(ctx, m1, 32u), s1.zmm());
    } else if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackusdw(t.ymm(), s0.ymm(), s1.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vmovdqu(m, t.ymm());
    } else {
        ujit::Vec t = ctx->uc.new_vec128();
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_U16, t, s0, s1);
        ctx->uc.v_storeuvec(m, t);
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_U16, t, s0, s1);
    ctx->uc.v_storeuvec(m, t);
#endif
}

uint8_t zj_cpu_supports_fast_lut_u8(const zj_cpu_info_t* cpu) {
    if (cpu == nullptr || cpu->isa == ZJ_ISA_SCALAR) return 0u;
#if defined(ASMJIT_UJIT_X86)
    if (cpu->vec_bytes == 64u) {
        return aj::CpuInfo::host().features().x86().has_avx512_vbmi() ? 1u : 0u;
    }
    return cpu->has_sse4_1 ? 1u : 0u;
#elif defined(ASMJIT_UJIT_AARCH64)
    return 1u;
#else
    return 0u;
#endif
}

void zj_vec_lut_u8_256(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t idx_reg, const uint8_t* table_256) {
    if (!ctx_ok(ctx) || table_256 == nullptr) return;
    ujit::Vec dst = unpack_vec(dst_reg);
    ujit::Vec idx = unpack_vec(idx_reg);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u && aj::CpuInfo::host().features().x86().has_avx512_vbmi()) {
        // AVX-512 VBMI: 4 x vpermb (64 bytes/lookup) + 3 masked moves = 64 pixels per vector!
        ujit::Vec t0 = unpack_vec(intern_const_vec(ctx, table_256 + 0, 64u));
        ujit::Vec t1 = unpack_vec(intern_const_vec(ctx, table_256 + 64, 64u));
        ujit::Vec t2 = unpack_vec(intern_const_vec(ctx, table_256 + 128, 64u));
        ujit::Vec t3 = unpack_vec(intern_const_vec(ctx, table_256 + 192, 64u));
        uint8_t b64[64], b128[64];
        std::memset(b64, 0x40, 64);
        std::memset(b128, 0x80, 64);
        ujit::Vec c64 = unpack_vec(intern_const_vec(ctx, b64, 64u));
        ujit::Vec c128 = unpack_vec(intern_const_vec(ctx, b128, 64u));

        aj::x86::KReg k1 = ctx->cc.new_kq();
        aj::x86::KReg k2 = ctx->cc.new_kq();
        ctx->cc.vptestmb(k1, idx.zmm(), c64.zmm());
        ctx->cc.vptestmb(k2, idx.zmm(), c128.zmm());

        ujit::Vec r0 = ctx->uc.new_vec();
        ujit::Vec r1 = ctx->uc.new_vec();
        ujit::Vec r2 = ctx->uc.new_vec();
        ujit::Vec r3 = ctx->uc.new_vec();
        ctx->cc.vpermb(r0.zmm(), idx.zmm(), t0.zmm());
        ctx->cc.vpermb(r1.zmm(), idx.zmm(), t1.zmm());
        ctx->cc.vpermb(r2.zmm(), idx.zmm(), t2.zmm());
        ctx->cc.vpermb(r3.zmm(), idx.zmm(), t3.zmm());
        ctx->cc.k(k1).vmovdqu8(r0.zmm(), r1.zmm());
        ctx->cc.k(k1).vmovdqu8(r2.zmm(), r3.zmm());
        ctx->cc.k(k2).vmovdqu8(r0.zmm(), r2.zmm());
        ctx->uc.emit_2v(ujit::UniOpVV::kMov, dst, r0);
        return;
    }

    // AVX2 (32B) or SSE4.1 (16B): depth-first 4-level vpshufb + vpblendvb binary tree (0 spills!).
    // Note: AsmJit's ConstPool max single-entry size is 64 bytes, so intern each 16B/32B chunk individually.
    const uint32_t vb = ctx->cpu.vec_bytes;
    if (vb > 32u || !ctx->cpu.has_sse4_1) {
        set_shim_error(ctx, aj::Error::kFeatureNotEnabled, "vec_lut_u8_256 requires AVX-512 VBMI, AVX2, or SSE4.1");
        return;
    }
    ujit::Mem chunk_mems[16];
    for (uint32_t c = 0; c < 16u; ++c) {
        uint8_t chunk_buf[32]{};
        std::memcpy(chunk_buf, table_256 + c * 16u, 16u);
        if (vb == 32u) {
            std::memcpy(chunk_buf + 16u, table_256 + c * 16u, 16u);
        }
        chunk_mems[c] = ctx->cc.new_const(aj::ConstPoolScope::kLocal, chunk_buf, vb);
    }

    uint8_t m0f_raw[32];
    std::memset(m0f_raw, 0x0F, 32);
    ujit::Vec m0f = unpack_vec(intern_const_vec(ctx, m0f_raw, vb));

    ujit::Vec lo = ctx->uc.new_vec();
    ctx->uc.emit_3v(ujit::UniOpVVV::kAndU32, lo, idx, m0f);
    ujit::Vec b6 = ctx->uc.new_vec();
    ujit::Vec b5 = ctx->uc.new_vec();
    ujit::Vec b4 = ctx->uc.new_vec();
    ctx->uc.emit_3v(ujit::UniOpVVV::kAddU8, b6, idx, idx);
    ctx->uc.emit_3v(ujit::UniOpVVV::kAddU8, b5, b6, b6);
    ctx->uc.emit_3v(ujit::UniOpVVV::kAddU8, b4, b5, b5);

    auto load_shuf = [&](uint32_t c) -> ujit::Vec {
        ujit::Vec v = ctx->uc.new_vec();
        ctx->uc.v_loadavec(v, chunk_mems[c]);
        ctx->uc.emit_3v(ujit::UniOpVVV::kSwizzlev_U8, v, v, lo);
        return v;
    };
    auto eval_pair = [&](uint32_t c0) -> ujit::Vec {
        ujit::Vec a = load_shuf(c0);
        ujit::Vec b = load_shuf(c0 + 1u);
        ctx->uc.emit_4v(ujit::UniOpVVVV::kBlendV_U8, a, a, b, b4);
        return a;
    };
    auto eval_quad = [&](uint32_t c0) -> ujit::Vec {
        ujit::Vec a = eval_pair(c0);
        ujit::Vec b = eval_pair(c0 + 2u);
        ctx->uc.emit_4v(ujit::UniOpVVVV::kBlendV_U8, a, a, b, b5);
        return a;
    };
    auto eval_oct = [&](uint32_t c0) -> ujit::Vec {
        ujit::Vec a = eval_quad(c0);
        ujit::Vec b = eval_quad(c0 + 4u);
        ctx->uc.emit_4v(ujit::UniOpVVVV::kBlendV_U8, a, a, b, b6);
        return a;
    };

    ujit::Vec lo_half = eval_oct(0u);
    ujit::Vec hi_half = eval_oct(8u);
    ctx->uc.emit_4v(ujit::UniOpVVVV::kBlendV_U8, dst, lo_half, hi_half, idx);
#elif defined(ASMJIT_UJIT_AARCH64)
    // AArch64 NEON: `tbl` (kSwizzlev_U8) returns 0 for any lane >= 16 unsigned, so
    // `OR_{c=0..15} tbl(chunk[c], idx - 16*c)` is exact with zero blends!
    // Intern each 16B chunk individually (ConstPool max entry size is 64B).
    ujit::Mem chunk_mems[16];
    for (uint32_t c = 0; c < 16u; ++c) {
        chunk_mems[c] = ctx->cc.new_const(aj::ConstPoolScope::kLocal, table_256 + c * 16u, 16u);
    }
    uint8_t step_raw[16];
    std::memset(step_raw, 16u, 16u);
    ujit::Vec v_step = unpack_vec(intern_const_vec(ctx, step_raw, 16u));

    ujit::Vec cur_idx = ctx->uc.new_vec();
    ctx->uc.emit_2v(ujit::UniOpVV::kMov, cur_idx, idx);

    ujit::Vec acc = ctx->uc.new_vec();
    ujit::Vec tmp = ctx->uc.new_vec();
    for (uint32_t c = 0; c < 16u; ++c) {
        ctx->uc.v_loadavec(tmp, chunk_mems[c]);
        ctx->uc.emit_3v(ujit::UniOpVVV::kSwizzlev_U8, tmp, tmp, cur_idx);
        if (c == 0u) {
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, acc, tmp);
        } else {
            ctx->uc.emit_3v(ujit::UniOpVVV::kOrU32, acc, acc, tmp);
        }
        if (c + 1u < 16u) {
            ctx->uc.emit_3v(ujit::UniOpVVV::kSubU8, cur_idx, cur_idx, v_step);
        }
    }
    ctx->uc.emit_2v(ujit::UniOpVV::kMov, dst, acc);
#endif
}

static ujit::Mem make_f32_const_mem(zj_ctx* ctx, float val) noexcept {
    uint8_t raw[64]{};
    const uint32_t lanes = ctx->cpu.f32_lanes;
    for (uint32_t i = 0; i < lanes; ++i) {
        std::memcpy(raw + i * 4u, &val, 4u);
    }
    return ctx->cc.new_const(aj::ConstPoolScope::kLocal, raw, ctx->cpu.vec_bytes);
}

void zj_vec_mul_c_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u) {
        ujit::Vec d = unpack_vec(dst_reg);
        ujit::Vec a = unpack_vec(a_reg);
        ujit::Mem m = make_f32_const_mem(ctx, b_const);
        if (ctx->cpu.vec_bytes == 64u) {
            ctx->cc.vmulps(d.zmm(), a.zmm(), m);
        } else {
            ctx->cc.vmulps(d.ymm(), a.ymm(), m);
        }
        return;
    }
#endif
    ctx->uc.emit_3v(ujit::UniOpVVV::kMulF32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)));
}

void zj_vec_add_c_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u) {
        ujit::Vec d = unpack_vec(dst_reg);
        ujit::Vec a = unpack_vec(a_reg);
        ujit::Mem m = make_f32_const_mem(ctx, b_const);
        if (ctx->cpu.vec_bytes == 64u) {
            ctx->cc.vaddps(d.zmm(), a.zmm(), m);
        } else {
            ctx->cc.vaddps(d.ymm(), a.ymm(), m);
        }
        return;
    }
#endif
    ctx->uc.emit_3v(ujit::UniOpVVV::kAddF32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)));
}

void zj_vec_sub_c_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u) {
        ujit::Vec d = unpack_vec(dst_reg);
        ujit::Vec a = unpack_vec(a_reg);
        ujit::Mem m = make_f32_const_mem(ctx, b_const);
        if (ctx->cpu.vec_bytes == 64u) {
            ctx->cc.vsubps(d.zmm(), a.zmm(), m);
        } else {
            ctx->cc.vsubps(d.ymm(), a.ymm(), m);
        }
        return;
    }
#endif
    ctx->uc.emit_3v(ujit::UniOpVVV::kSubF32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)));
}

void zj_vec_min_c_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u) {
        ujit::Vec d = unpack_vec(dst_reg);
        ujit::Vec a = unpack_vec(a_reg);
        ujit::Mem m = make_f32_const_mem(ctx, b_const);
        if (ctx->cpu.vec_bytes == 64u) {
            ctx->cc.vminps(d.zmm(), a.zmm(), m);
        } else {
            ctx->cc.vminps(d.ymm(), a.ymm(), m);
        }
        return;
    }
#endif
    ctx->uc.emit_3v(ujit::UniOpVVV::kMinF32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)));
}

void zj_vec_max_c_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u) {
        ujit::Vec d = unpack_vec(dst_reg);
        ujit::Vec a = unpack_vec(a_reg);
        ujit::Mem m = make_f32_const_mem(ctx, b_const);
        if (ctx->cpu.vec_bytes == 64u) {
            ctx->cc.vmaxps(d.zmm(), a.zmm(), m);
        } else {
            ctx->cc.vmaxps(d.ymm(), a.ymm(), m);
        }
        return;
    }
#endif
    ctx->uc.emit_3v(ujit::UniOpVVV::kMaxF32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)));
}

void zj_vec_cmp_c_f32(zj_ctx* ctx, zj_vec_op3_t cmp_op, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 32u) {
        uint8_t imm = 0u;
        bool valid = true;
        switch (cmp_op) {
            case ZJ_V3_CMP_EQ_F32: imm = 0u; break;   // _CMP_EQ_OQ
            case ZJ_V3_CMP_LT_F32: imm = 17u; break;  // _CMP_LT_OQ
            case ZJ_V3_CMP_LE_F32: imm = 18u; break;  // _CMP_LE_OQ
            case ZJ_V3_CMP_NE_F32: imm = 4u; break;   // _CMP_NEQ_UQ
            case ZJ_V3_CMP_GE_F32: imm = 29u; break;  // _CMP_GE_OQ
            case ZJ_V3_CMP_GT_F32: imm = 30u; break;  // _CMP_GT_OQ
            default: valid = false; break;
        }
        if (valid) {
            ujit::Vec d = unpack_vec(dst_reg);
            ujit::Vec a = unpack_vec(a_reg);
            ujit::Mem m = make_f32_const_mem(ctx, b_const);
            ctx->cc.vcmpps(d.ymm(), a.ymm(), m, imm);
            return;
        }
    }
#endif
    zj_vec_op3(ctx, cmp_op, dst_reg, a_reg, zj_const_f32_splat(ctx, b_const));
}

void zj_vec_fma_ac_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const, zj_reg_t c_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.has_fma && ctx->cpu.vec_bytes >= 32u) {
        ujit::Vec d = unpack_vec(dst_reg);
        ujit::Vec a = unpack_vec(a_reg);
        ujit::Vec c = unpack_vec(c_reg);
        ujit::Mem mb = make_f32_const_mem(ctx, b_const);
        // dst = a * [mb] + c. Guard dst == a && dst != c before copying c into acc.
        ujit::Vec acc = (d.id() == a.id() && d.id() != c.id()) ? ctx->uc.new_vec() : d;
        if (acc.id() != c.id()) {
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, acc, c);
        }
        if (ctx->cpu.vec_bytes == 64u) {
            ctx->cc.vfmadd231ps(acc.zmm(), a.zmm(), mb);
        } else {
            ctx->cc.vfmadd231ps(acc.ymm(), a.ymm(), mb);
        }
        if (acc.id() != d.id()) {
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, d, acc);
        }
        return;
    }
#endif
    ctx->uc.emit_4v(ujit::UniOpVVVV::kMAddF32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)), unpack_vec(c_reg));
}

void zj_vec_blendv_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t false_val, zj_reg_t true_val, zj_reg_t mask_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 32u) {
        ctx->cc.vblendvps(unpack_vec(dst_reg).ymm(), unpack_vec(false_val).ymm(), unpack_vec(true_val).ymm(), unpack_vec(mask_reg).ymm());
        return;
    }
#endif
    ctx->uc.emit_4v(ujit::UniOpVVVV::kBlendV_U8, unpack_vec(dst_reg), unpack_vec(false_val), unpack_vec(true_val), unpack_vec(mask_reg));
}

void zj_vec_and_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, zj_reg_t b_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 32u) {
        ctx->cc.vandps(unpack_vec(dst_reg).ymm(), unpack_vec(a_reg).ymm(), unpack_vec(b_reg).ymm());
        return;
    }
#endif
    ctx->uc.emit_3v(ujit::UniOpVVV::kAndU32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(b_reg));
}

void zj_vec_andn_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t mask_not, zj_reg_t b_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 32u) {
        ctx->cc.vandnps(unpack_vec(dst_reg).ymm(), unpack_vec(mask_not).ymm(), unpack_vec(b_reg).ymm());
        return;
    }
#endif
    ctx->uc.emit_3v(ujit::UniOpVVV::kAndnU32, unpack_vec(dst_reg), unpack_vec(mask_not), unpack_vec(b_reg));
}

void zj_vec_fma_bc_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, zj_reg_t b_reg, float c_const) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.has_fma && ctx->cpu.vec_bytes >= 32u) {
        ujit::Vec d = unpack_vec(dst_reg);
        ujit::Vec a = unpack_vec(a_reg);
        ujit::Vec b = unpack_vec(b_reg);
        ujit::Mem mc = make_f32_const_mem(ctx, c_const);
        // L1: dst = a * b + [mc]. Guard dst == b (clobbering b before vfmadd213ps).
        ujit::Vec acc = d;
        ujit::Vec rhs = b;
        if (d.id() == b.id() && d.id() != a.id()) {
            // Commutative: a * b == b * a, so keep d (= b) as acc and use a as rhs!
            rhs = a;
        } else if (d.id() != a.id()) {
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, acc, a);
        }
        if (ctx->cpu.vec_bytes == 64u) {
            ctx->cc.vfmadd213ps(acc.zmm(), rhs.zmm(), mc);
        } else {
            ctx->cc.vfmadd213ps(acc.ymm(), rhs.ymm(), mc);
        }
        return;
    }
#endif
    ctx->uc.emit_4v(ujit::UniOpVVVV::kMAddF32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(b_reg), unpack_vec(zj_const_f32_splat(ctx, c_const)));
}

void zj_vec_fma_cc_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const, float c_const) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.has_fma && ctx->cpu.vec_bytes >= 32u) {
        ujit::Vec d = unpack_vec(dst_reg);
        ujit::Vec a = unpack_vec(a_reg);
        ujit::Mem mb = make_f32_const_mem(ctx, b_const);
        ujit::Mem mc = make_f32_const_mem(ctx, c_const);
        // L1: guard dst == a before loading [mb] into acc.
        ujit::Vec acc = (d.id() == a.id()) ? ctx->uc.new_vec() : d;
        ctx->uc.v_loadavec(acc, mb);
        if (ctx->cpu.vec_bytes == 64u) {
            ctx->cc.vfmadd213ps(acc.zmm(), a.zmm(), mc);
        } else {
            ctx->cc.vfmadd213ps(acc.ymm(), a.ymm(), mc);
        }
        if (acc.id() != d.id()) {
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, d, acc);
        }
        return;
    }
#endif
    ctx->uc.emit_4v(
        ujit::UniOpVVVV::kMAddF32,
        unpack_vec(dst_reg),
        unpack_vec(a_reg),
        unpack_vec(zj_const_f32_splat(ctx, b_const)),
        unpack_vec(zj_const_f32_splat(ctx, c_const))
    );
}

void zj_vec_fnma_bc_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const, zj_reg_t c_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.has_fma && ctx->cpu.vec_bytes >= 32u) {
        ujit::Vec d = unpack_vec(dst_reg);
        ujit::Vec a = unpack_vec(a_reg);
        ujit::Vec c = unpack_vec(c_reg);
        ujit::Mem mb = make_f32_const_mem(ctx, b_const);
        // L1: dst = c - a * [mb]. Guard dst == a && dst != c before copying c into acc.
        ujit::Vec acc = (d.id() == a.id() && d.id() != c.id()) ? ctx->uc.new_vec() : d;
        if (acc.id() != c.id()) {
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, acc, c);
        }
        if (ctx->cpu.vec_bytes == 64u) {
            ctx->cc.vfnmadd231ps(acc.zmm(), a.zmm(), mb);
        } else {
            ctx->cc.vfnmadd231ps(acc.ymm(), a.ymm(), mb);
        }
        if (acc.id() != d.id()) {
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, d, acc);
        }
        return;
    }
#endif
    if (!ctx->cpu.has_fma) {
        ujit::Vec tmp = ctx->uc.new_vec();
        ctx->uc.emit_3v(ujit::UniOpVVV::kMulF32, tmp, unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)));
        ctx->uc.emit_3v(ujit::UniOpVVV::kSubF32, unpack_vec(dst_reg), unpack_vec(c_reg), tmp);
        return;
    }
    ctx->uc.emit_4v(ujit::UniOpVVVV::kNMAddF32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)), unpack_vec(c_reg));
}

void zj_vec_permute_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t src, const uint32_t* lane_indices, uint32_t lane_count) {
    if (!ctx_ok(ctx)) return;
    if (lane_indices == nullptr || lane_count != ctx->cpu.f32_lanes) {
        set_shim_error(ctx, aj::Error::kInvalidArgument, "vec_permute_f32 lane_count mismatch");
        return;
    }
    ujit::Vec d = unpack_vec(dst);
    ujit::Vec s = unpack_vec(src);
    if (lane_count == 4u) {
        const uint32_t imm = ujit::swizzle(
            uint8_t(lane_indices[3] & 3u),
            uint8_t(lane_indices[2] & 3u),
            uint8_t(lane_indices[1] & 3u),
            uint8_t(lane_indices[0] & 3u)
        ).value;
        ctx->uc.emit_2vi(ujit::UniOpVVI::kSwizzleU32x4, d, s, imm);
        return;
    }
#if defined(ASMJIT_UJIT_X86)
    uint8_t idx_bytes[64]{};
    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t v = lane_indices[i];
        std::memcpy(idx_bytes + i * 4u, &v, 4u);
    }
    ujit::Vec idx = unpack_vec(intern_const_vec(ctx, idx_bytes, ctx->cpu.vec_bytes));
    if (lane_count == 8u) {
        ctx->cc.vpermps(d.ymm(), idx.ymm(), s.ymm());
    } else if (lane_count == 16u) {
        ctx->cc.vpermps(d.zmm(), idx.zmm(), s.zmm());
    }
#endif
}

static void emit_scalar_gather_impl(zj_ctx* ctx, zj_reg_t dst, zj_reg_t base_reg, zj_reg_t offsets_reg, uint32_t elem_bytes) noexcept {
    const uint32_t lanes = ctx->cpu.f32_lanes;
    const uint32_t vec_bytes = ctx->cpu.vec_bytes;
    ujit::Mem off_slot = ctx->cc.new_stack(vec_bytes, vec_bytes);
    ujit::Mem res_slot = ctx->cc.new_stack(vec_bytes, vec_bytes);
    ctx->uc.v_storeavec(off_slot, unpack_vec(offsets_reg));

    ujit::Gp base = unpack_gp(base_reg);
    ujit::Gp off = ctx->uc.new_gp64();
    ujit::Gp val = ctx->uc.new_gp32();

    for (uint32_t i = 0; i < lanes; ++i) {
        const int32_t slot_disp = int32_t(i * 4u);
        ujit::Mem off_m = off_slot;
        off_m.add_offset(slot_disp);
        // Zero-extended 32-bit offset into 64-bit register (callers clamp offsets >= 0).
        ctx->uc.emit_rm(ujit::UniOpRM::kLoadU32, off.r32(), off_m);

        zj_mem_t src_m{};
        src_m.base = pack_reg(base);
        src_m.index = pack_reg(off);
        src_m.has_index = 1u;
        src_m.size = uint8_t(elem_bytes);
        ujit::UniOpRM load_op = ujit::UniOpRM::kLoadU32;
        if (elem_bytes == 1u) load_op = ujit::UniOpRM::kLoadU8;
        else if (elem_bytes == 2u) load_op = ujit::UniOpRM::kLoadU16;
        ctx->uc.emit_rm(load_op, val, legalise_mem(ctx, src_m, elem_bytes));

        ujit::Mem res_m = res_slot;
        res_m.add_offset(slot_disp);
        ctx->uc.emit_mr(ujit::UniOpMR::kStoreU32, res_m, val);
    }

    ctx->uc.v_loadavec(unpack_vec(dst), res_slot);
}

#if defined(ASMJIT_UJIT_X86)
static void emit_avx2_reg_gather_impl(zj_ctx* ctx, zj_reg_t dst, zj_reg_t base_reg, zj_reg_t offsets_reg, uint32_t elem_bytes, bool is_f16) noexcept {
    // Register-only 8-lane gather on AVX2: 4x 64-bit extracts (vmovq/vpextrq), 0 stack spills, 0 STL stalls.
    ujit::Gp base = unpack_gp(base_reg);
    ujit::Vec off = unpack_vec(offsets_reg);
    ujit::Vec d = unpack_vec(dst);

    ujit::Vec hi_off = ctx->uc.new_vec();
    ctx->cc.vextracti128(hi_off.xmm(), off.ymm(), 1);

    ujit::Gp p01 = ctx->uc.new_gp64();
    ujit::Gp p23 = ctx->uc.new_gp64();
    ujit::Gp i0 = ctx->uc.new_gp64();
    ujit::Gp i2 = ctx->uc.new_gp64();
    ujit::Gp tmp = ctx->uc.new_gp32();

    ujit::Vec lo_v = ctx->uc.new_vec();
    ujit::Vec hi_v = ctx->uc.new_vec();

    auto emit_half = [&](const ujit::Vec& off_x, const ujit::Vec& out_x) {
        ctx->cc.vmovq(p01, off_x);
        ctx->cc.vpextrq(p23, off_x, 1);
        ctx->cc.mov(i0.r32(), p01.r32());
        ctx->cc.shr(p01, 32);
        ctx->cc.mov(i2.r32(), p23.r32());
        ctx->cc.shr(p23, 32);

        if (elem_bytes == 4u) {
            ctx->cc.vmovss(out_x, aj::x86::dword_ptr(base, i0));
            ctx->cc.vinsertps(out_x, out_x, aj::x86::dword_ptr(base, p01), 0x10);
            ctx->cc.vinsertps(out_x, out_x, aj::x86::dword_ptr(base, i2), 0x20);
            ctx->cc.vinsertps(out_x, out_x, aj::x86::dword_ptr(base, p23), 0x30);
        } else if (elem_bytes == 2u) {
            ctx->cc.movzx(tmp, aj::x86::word_ptr(base, i0));
            ctx->cc.vmovd(out_x, tmp);
            ctx->cc.vpinsrw(out_x, out_x, aj::x86::word_ptr(base, p01), 1);
            ctx->cc.vpinsrw(out_x, out_x, aj::x86::word_ptr(base, i2), 2);
            ctx->cc.vpinsrw(out_x, out_x, aj::x86::word_ptr(base, p23), 3);
        } else {
            ctx->cc.movzx(tmp, aj::x86::byte_ptr(base, i0));
            ctx->cc.vmovd(out_x, tmp);
            ctx->cc.vpinsrb(out_x, out_x, aj::x86::byte_ptr(base, p01), 1);
            ctx->cc.vpinsrb(out_x, out_x, aj::x86::byte_ptr(base, i2), 2);
            ctx->cc.vpinsrb(out_x, out_x, aj::x86::byte_ptr(base, p23), 3);
        }
    };

    emit_half(off.xmm(), lo_v.xmm());
    emit_half(hi_off.xmm(), hi_v.xmm());

    if (elem_bytes == 4u) {
        ctx->cc.vinsertf128(d.ymm(), lo_v.ymm(), hi_v.xmm(), 1);
    } else if (elem_bytes == 2u) {
        ctx->cc.vpunpcklqdq(lo_v.xmm(), lo_v.xmm(), hi_v.xmm());
        if (is_f16) {
            ctx->cc.vcvtph2ps(d.ymm(), lo_v.xmm());
        } else {
            ctx->cc.vpmovzxwd(d.ymm(), lo_v.xmm());
        }
    } else {
        ctx->cc.vpunpckldq(lo_v.xmm(), lo_v.xmm(), hi_v.xmm());
        ctx->cc.vpmovzxbd(d.ymm(), lo_v.xmm());
    }
}
#endif

void zj_vec_gather_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t base, zj_reg_t byte_offsets_i32, uint8_t can_use_hw_gather) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    const bool use_hw = (can_use_hw_gather != 0u) && (ctx->cpu.prefer_hw_gather_f32 != 0u);
    if (use_hw && ctx->cpu.vec_bytes == 64u) {
        aj::x86::KReg k = ctx->cc.new_kw();
        ctx->cc.kxnorw(k, k, k);
        ujit::Vec d = unpack_vec(dst);
        ujit::Vec off = unpack_vec(byte_offsets_i32);
        // L2: EVEX vpgatherdd forbids dst aliasing index register; use a fresh tmp if needed.
        if (d.id() == off.id()) {
            ujit::Vec tmp = ctx->uc.new_vec();
            ctx->cc.k(k).vpgatherdd(tmp.zmm(), aj::x86::ptr(unpack_gp(base), off.zmm(), 0));
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, d, tmp);
        } else {
            ctx->cc.k(k).vpgatherdd(d.zmm(), aj::x86::ptr(unpack_gp(base), off.zmm(), 0));
        }
        return;
    }
    if (use_hw && ctx->cpu.vec_bytes == 32u) {
        ujit::Vec mask = ctx->uc.new_vec();
        ctx->cc.vpcmpeqd(mask.ymm(), mask.ymm(), mask.ymm());
        ujit::Vec d = unpack_vec(dst);
        ujit::Vec off = unpack_vec(byte_offsets_i32);
        // vpgatherdd forbids dst aliasing index or mask; use a fresh tmp if needed.
        if (d.id() == off.id()) {
            ujit::Vec tmp = ctx->uc.new_vec();
            ctx->cc.vpgatherdd(tmp.ymm(), aj::x86::ptr(unpack_gp(base), off.ymm(), 0), mask.ymm());
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, d, tmp);
        } else {
            ctx->cc.vpgatherdd(d.ymm(), aj::x86::ptr(unpack_gp(base), off.ymm(), 0), mask.ymm());
        }
        return;
    }
    if (ctx->cpu.vec_bytes == 32u) {
        emit_avx2_reg_gather_impl(ctx, dst, base, byte_offsets_i32, 4u, false);
        return;
    }
#else
    (void)can_use_hw_gather;
#endif
    emit_scalar_gather_impl(ctx, dst, base, byte_offsets_i32, 4u);
}

void zj_vec_gather_f16(zj_ctx* ctx, zj_reg_t dst_f32, zj_reg_t base_reg, zj_reg_t byte_offsets_i32) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 32u && ctx->cpu.has_f16c) {
        emit_avx2_reg_gather_impl(ctx, dst_f32, base_reg, byte_offsets_i32, 2u, true);
        return;
    }
#endif
    const uint32_t lanes = ctx->cpu.f32_lanes;
    const uint32_t vec_bytes = ctx->cpu.vec_bytes;
    ujit::Mem off_slot = ctx->cc.new_stack(vec_bytes, vec_bytes);
    ujit::Mem f16_slot = ctx->cc.new_stack(lanes * 2u, 16u);
    ctx->uc.v_storeavec(off_slot, unpack_vec(byte_offsets_i32));

    ujit::Gp base = unpack_gp(base_reg);
    ujit::Gp off = ctx->uc.new_gp64();
    ujit::Gp val = ctx->uc.new_gp32();

    for (uint32_t i = 0; i < lanes; ++i) {
        ujit::Mem off_m = off_slot;
        off_m.add_offset(int32_t(i * 4u));
        ctx->uc.emit_rm(ujit::UniOpRM::kLoadU32, off.r32(), off_m);

        zj_mem_t src_m{};
        src_m.base = pack_reg(base);
        src_m.index = pack_reg(off);
        src_m.has_index = 1u;
        src_m.size = 2u;
        ctx->uc.emit_rm(ujit::UniOpRM::kLoadU16, val, legalise_mem(ctx, src_m, 2u));

        ujit::Mem dst_m = f16_slot;
        dst_m.add_offset(int32_t(i * 2u));
        ctx->uc.emit_mr(ujit::UniOpMR::kStoreU16, dst_m, val);
    }

    ujit::Vec d = unpack_vec(dst_f32);
#if defined(ASMJIT_UJIT_X86)
    if (!ctx->cpu.has_f16c) {
        set_shim_error(ctx, aj::Error::kFeatureNotEnabled, "f16 gather requires F16C");
        return;
    }
    if (ctx->cpu.vec_bytes == 64u) {
        ctx->cc.vcvtph2ps(d.zmm(), f16_slot);
    } else if (ctx->cpu.vec_bytes == 32u) {
        ctx->cc.vcvtph2ps(d.ymm(), f16_slot);
    } else {
        ctx->cc.vcvtph2ps(d.xmm(), f16_slot);
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec tmp = ctx->uc.new_vec128();
    ctx->cc.ldr(tmp.d(), f16_slot);
    ctx->cc.fcvtl(d.s4(), tmp.h4());
#endif
}

void zj_vec_gather_u16(zj_ctx* ctx, zj_reg_t dst_i32, zj_reg_t base, zj_reg_t byte_offsets_i32) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 32u) {
        emit_avx2_reg_gather_impl(ctx, dst_i32, base, byte_offsets_i32, 2u, false);
        return;
    }
#endif
    emit_scalar_gather_impl(ctx, dst_i32, base, byte_offsets_i32, 2u);
}

void zj_vec_gather_u8(zj_ctx* ctx, zj_reg_t dst_i32, zj_reg_t base, zj_reg_t byte_offsets_i32) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 32u) {
        emit_avx2_reg_gather_impl(ctx, dst_i32, base, byte_offsets_i32, 1u, false);
        return;
    }
#endif
    emit_scalar_gather_impl(ctx, dst_i32, base, byte_offsets_i32, 1u);
}

} // extern "C"
