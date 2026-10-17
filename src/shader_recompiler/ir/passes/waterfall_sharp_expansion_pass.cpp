// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <vector>
#include "common/assert.h"
#include "common/logging/log.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"

// Waterfall loops that pick a T# per lane from a descriptor array (v_readlane of a per-lane
// index, then s_buffer_load of the T# at index * 32 bytes) cannot be flattened by the CPU SRT
// walker, which has no lane values. MLB The Show 24 draws its crowd this way: every instance
// picks its albedo/normal textures from an array, so with a single flattened element the whole
// crowd shared one texture. This pass expands such a sample into one sample per array element
// 0..N-1 (each with a constant offset, which the walker handles) and selects the result that
// matches the lane's index. Indices >= N fall back to element 0.

namespace Shader::Optimization {

namespace {

constexpr u32 DefaultElements = 8;
constexpr u32 MaxElements = 32;

u32 NumElements() {
    static const u32 num = [] {
        if (const char* env = std::getenv("SHADPS4_WATERFALL_ELEMENTS")) {
            const u32 value = static_cast<u32>(std::strtoul(env, nullptr, 10));
            return std::clamp<u32>(value, 1, MaxElements);
        }
        return DefaultElements;
    }();
    return num;
}

bool IsOffsetArithmetic(IR::Opcode op) {
    switch (op) {
    case IR::Opcode::IAdd32:
    case IR::Opcode::ISub32:
    case IR::Opcode::IMul32:
    case IR::Opcode::ShiftLeftLogical32:
    case IR::Opcode::ShiftRightLogical32:
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::BitwiseOr32:
        return true;
    default:
        return false;
    }
}

// Finds the single ReadLane an offset expression depends on. Leaves that are not arithmetic are
// treated as uniform and kept as they are.
bool FindReadLane(const IR::Value& value, IR::Inst*& read_lane) {
    if (value.IsImmediate()) {
        return true;
    }
    IR::Inst* const inst = value.Inst();
    if (inst->GetOpcode() == IR::Opcode::ReadLane) {
        if (read_lane && read_lane != inst) {
            return false;
        }
        read_lane = inst;
        return true;
    }
    if (!IsOffsetArithmetic(inst->GetOpcode())) {
        return true;
    }
    for (size_t i = 0; i < inst->NumArgs(); ++i) {
        if (!FindReadLane(inst->Arg(i), read_lane)) {
            return false;
        }
    }
    return true;
}

using CloneMap = std::unordered_map<IR::Inst*, IR::Value>;

// The SRT walker expects offset arithmetic to have at least one non-immediate operand, so fold
// the expressions that become constant once the lane index is replaced.
u32 Fold(IR::Opcode op, u32 a, u32 b) {
    switch (op) {
    case IR::Opcode::IAdd32:
        return a + b;
    case IR::Opcode::ISub32:
        return a - b;
    case IR::Opcode::IMul32:
        return a * b;
    case IR::Opcode::ShiftLeftLogical32:
        return a << (b & 31);
    case IR::Opcode::ShiftRightLogical32:
        return a >> (b & 31);
    case IR::Opcode::BitwiseAnd32:
        return a & b;
    case IR::Opcode::BitwiseOr32:
        return a | b;
    default:
        UNREACHABLE();
    }
}

IR::Value CloneWithElement(IR::Block* block, IR::Block::iterator it, const IR::Value& value,
                           IR::Inst* read_lane, u32 element, CloneMap& clones) {
    if (value.IsImmediate()) {
        return value;
    }
    IR::Inst* const inst = value.Inst();
    if (inst == read_lane) {
        return IR::Value{element};
    }
    if (!IsOffsetArithmetic(inst->GetOpcode())) {
        return value;
    }
    if (const auto found = clones.find(inst); found != clones.end()) {
        return found->second;
    }
    const IR::Value a = CloneWithElement(block, it, inst->Arg(0), read_lane, element, clones);
    const IR::Value b = CloneWithElement(block, it, inst->Arg(1), read_lane, element, clones);
    if (a.IsImmediate() && b.IsImmediate()) {
        return clones[inst] = IR::Value{Fold(inst->GetOpcode(), a.U32(), b.U32())};
    }
    IR::Inst* const clone = &*block->PrependNewInst(it, *inst);
    clone->SetArg(0, a);
    clone->SetArg(1, b);
    return clones[inst] = IR::Value{clone};
}

struct WaterfallSample {
    IR::Inst* sample;
    IR::Inst* read_lane;
    std::array<IR::Inst*, 8> dwords;
};

std::optional<WaterfallSample> MatchWaterfallSample(IR::Inst& inst) {
    if (inst.GetOpcode() != IR::Opcode::ImageSampleRaw) {
        return std::nullopt;
    }
    IR::Inst* const handle = inst.Arg(0).Inst();
    if (handle->GetOpcode() != IR::Opcode::ImageHandle) {
        return std::nullopt;
    }
    WaterfallSample match{.sample = &inst, .read_lane = nullptr, .dwords = {}};
    for (u32 half = 0; half < 2; ++half) {
        IR::Inst* const composite = handle->Arg(half).Inst();
        if (composite->GetOpcode() != IR::Opcode::CompositeConstructU32x4) {
            return std::nullopt;
        }
        for (u32 i = 0; i < 4; ++i) {
            const IR::Value dword = composite->Arg(i);
            if (dword.IsImmediate()) {
                return std::nullopt;
            }
            IR::Inst* const load = dword.Inst();
            if (load->GetOpcode() != IR::Opcode::ReadConstBuffer ||
                !FindReadLane(load->Arg(1), match.read_lane)) {
                return std::nullopt;
            }
            match.dwords[half * 4 + i] = load;
        }
    }
    if (!match.read_lane) {
        return std::nullopt;
    }
    return match;
}

void Expand(const WaterfallSample& match) {
    IR::Inst* const sample = match.sample;
    IR::Block* const block = sample->GetParent();
    const auto it = IR::Block::InstructionList::s_iterator_to(*sample);
    const u32 num_elements = NumElements();

    IR::Value result{};
    for (u32 element = 0; element < num_elements; ++element) {
        CloneMap clones;
        std::array<IR::Value, 8> dwords;
        for (u32 i = 0; i < 8; ++i) {
            IR::Inst* const load = &*block->PrependNewInst(it, *match.dwords[i]);
            load->SetArg(1, CloneWithElement(block, it, match.dwords[i]->Arg(1), match.read_lane,
                                             element, clones));
            dwords[i] = IR::Value{load};
        }
        const IR::Value lo{&*block->PrependNewInst(
            it, IR::Opcode::CompositeConstructU32x4, {dwords[0], dwords[1], dwords[2], dwords[3]})};
        const IR::Value hi{&*block->PrependNewInst(
            it, IR::Opcode::CompositeConstructU32x4, {dwords[4], dwords[5], dwords[6], dwords[7]})};
        const IR::Value handle{&*block->PrependNewInst(it, IR::Opcode::ImageHandle, {lo, hi})};
        IR::Inst* const element_sample = &*block->PrependNewInst(it, *sample);
        element_sample->SetArg(0, handle);
        const IR::Value element_value{element_sample};

        if (element == 0) {
            result = element_value;
            continue;
        }
        const IR::Value is_element{&*block->PrependNewInst(
            it, IR::Opcode::IEqual32, {IR::Value{match.read_lane}, IR::Value{element}})};
        std::array<IR::Value, 4> components;
        for (u32 c = 0; c < 4; ++c) {
            const IR::Value from_element{&*block->PrependNewInst(
                it, IR::Opcode::CompositeExtractF32x4, {element_value, IR::Value{c}})};
            const IR::Value from_previous{&*block->PrependNewInst(
                it, IR::Opcode::CompositeExtractF32x4, {result, IR::Value{c}})};
            components[c] = IR::Value{&*block->PrependNewInst(
                it, IR::Opcode::SelectF32, {is_element, from_element, from_previous})};
        }
        result = IR::Value{&*block->PrependNewInst(
            it, IR::Opcode::CompositeConstructF32x4,
            {components[0], components[1], components[2], components[3]})};
    }
    sample->ReplaceUsesWith(result);
}

} // Anonymous namespace

void WaterfallSharpExpansionPass(IR::Program& program) {
    std::vector<WaterfallSample> matches;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (auto match = MatchWaterfallSample(inst)) {
                matches.push_back(*match);
            }
        }
    }
    for (const auto& match : matches) {
        Expand(match);
    }
    if (!matches.empty()) {
        LOG_INFO(Render_Recompiler, "Expanded {} waterfall descriptor sample(s) into {} elements",
                 matches.size(), NumElements());
    }
}

} // namespace Shader::Optimization
