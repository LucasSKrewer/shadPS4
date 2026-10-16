// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/profile.h"

namespace Shader::Gcn {

using Interpolation = Info::Interpolation;

static Interpolation GetInterpolation(IR::Attribute attribute) {
    switch (attribute) {
    case IR::Attribute::BaryCoordNoPersp:
        return {Qualifier::NoPerspective, Qualifier::None};
    case IR::Attribute::BaryCoordNoPerspCentroid:
        return {Qualifier::NoPerspective, Qualifier::Centroid};
    case IR::Attribute::BaryCoordNoPerspSample:
        return {Qualifier::NoPerspective, Qualifier::Sample};
    case IR::Attribute::BaryCoordSmooth:
        return {Qualifier::Smooth, Qualifier::None};
    case IR::Attribute::BaryCoordSmoothCentroid:
        return {Qualifier::Smooth, Qualifier::Centroid};
    case IR::Attribute::BaryCoordSmoothSample:
        return {Qualifier::Smooth, Qualifier::Sample};
    default:
        UNREACHABLE_MSG("Unhandled barycentric attribute {}", NameOf(attribute));
    }
}

static constexpr std::array DefaultValTable = {
    std::array{0.0f, 0.0f, 0.0f, 0.0f},
    std::array{0.0f, 0.0f, 0.0f, 1.0f},
    std::array{1.0f, 1.0f, 1.0f, 0.0f},
    std::array{1.0f, 1.0f, 1.0f, 1.0f},
};

void Translator::EmitVectorInterpolation(const GcnInst& inst) {
    switch (inst.opcode) {
        // VINTRP
    case Opcode::V_INTERP_P1_F32:
        return V_INTERP_P1_F32(inst);
    case Opcode::V_INTERP_P2_F32:
        return V_INTERP_P2_F32(inst);
    case Opcode::V_INTERP_MOV_F32:
        return V_INTERP_MOV_F32(inst);
    default:
        LogMissingOpcode(inst);
    }
}

// VINTRP

void Translator::V_INTERP_P1_F32(const GcnInst& inst) {
    if (!profile.needs_manual_interpolation) {
        return;
    }
    const u32 attr_index = inst.control.vintrp.attr;
    const auto& attr = runtime_info.fs_info.inputs[attr_index];
    if (attr.IsDefault()) {
        return;
    }
    // VDST = P10 * VSRC + P0
    const IR::Attribute attrib = IR::Attribute::Param0 + attr_index;
    const IR::F32 p0 = ir.GetAttribute(attrib, inst.control.vintrp.chan, 0);
    const IR::F32 p1 = ir.GetAttribute(attrib, inst.control.vintrp.chan, 1);
    const IR::F32 i = GetSrc<IR::F32>(inst.src[0]);
    const IR::F32 result = ir.FPFma(ir.FPSub(p1, p0), i, p0);
    SetDst(inst.dst[0], result);
}

void Translator::V_INTERP_P2_F32(const GcnInst& inst) {
    const u32 attr_index = inst.control.vintrp.attr;
    const IR::Attribute attrib = IR::Attribute::Param0 + attr_index;
    const auto& attr = runtime_info.fs_info.inputs[attr_index];
    auto& interp = info.fs_interpolation[attr_index];
    if (attr.IsDefault()) {
        SetDst(inst.dst[0],
               ir.Imm32(DefaultValTable[attr.default_value][inst.control.vintrp.chan]));
        return;
    }
    ASSERT(!attr.is_flat);
    if (!profile.needs_manual_interpolation) {
        interp = GetInterpolation(vgpr_to_interp[inst.src[0].code]);
        SetDst(inst.dst[0], ir.GetAttribute(attrib, inst.control.vintrp.chan));
        return;
    }
    // VDST = P20 * VSRC + VDST
    const IR::F32 p0 = ir.GetAttribute(attrib, inst.control.vintrp.chan, 0);
    const IR::F32 p2 = ir.GetAttribute(attrib, inst.control.vintrp.chan, 2);
    const IR::F32 j = GetSrc<IR::F32>(inst.src[0]);
    const IR::F32 result = ir.FPFma(ir.FPSub(p2, p0), j, GetSrc<IR::F32>(inst.dst[0]));
    interp.primary = Qualifier::PerVertex;
    SetDst(inst.dst[0], result);
}

void Translator::V_INTERP_MOV_F32(const GcnInst& inst) {
    const u32 attr_index = inst.control.vintrp.attr;
    const IR::Attribute attrib = IR::Attribute::Param0 + attr_index;
    const auto& attr = runtime_info.fs_info.inputs[attr_index];
    auto& interp = info.fs_interpolation[attr_index];
    // MLB The Show 24 (CUSA43942) uses V_INTERP_MOV_F32 with non-flat attributes and
    // src codes other than P0 (2) in its in-game/stadium fragment shaders. The generic
    // paths below already handle any src code, so the original strict assert was too
    // conservative and is relaxed to a warning.
    if (!(attr.is_flat || inst.src[0].code == 2)) {
        LOG_WARNING(Render_Recompiler,
                    "V_INTERP_MOV_F32 non-flat attr with src code {} (attr {})",
                    inst.src[0].code, attr_index);
    }
    if (profile.supports_amd_shader_explicit_vertex_parameter ||
        profile.supports_fragment_shader_barycentric) {
        // VSRC selects a stored primitive parameter: 0=P10 (v1-v0), 1=P20 (v2-v0), 2=P0 (v0).
        // GetAttribute(attrib, chan, N) returns vertex N's raw value, so P10/P20 must be computed
        // as deltas. The old (code+1)%3 mapping returned the raw vertex value instead, which
        // corrupts shaders that read P10/P20 for manual interpolation (e.g. MLB The Show 24
        // stadium dirt, which came out blocky).
        interp.primary = Qualifier::PerVertex;
        const u32 chan = inst.control.vintrp.chan;
        const IR::F32 v0 = ir.GetAttribute(attrib, chan, 0);
        IR::F32 result;
        switch (inst.src[0].code) {
        case 0: // P10 = v1 - v0
            result = ir.FPSub(ir.GetAttribute(attrib, chan, 1), v0);
            break;
        case 1: // P20 = v2 - v0
            result = ir.FPSub(ir.GetAttribute(attrib, chan, 2), v0);
            break;
        default: // 2 = P0 = v0
            result = v0;
            break;
        }
        SetDst(inst.dst[0], result);
    } else {
        interp.primary = Qualifier::Flat;
        SetDst(inst.dst[0], ir.GetAttribute(attrib, inst.control.vintrp.chan));
    }
}

} // namespace Shader::Gcn
