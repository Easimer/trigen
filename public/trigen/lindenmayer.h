// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Lindenmayer-system
//

#pragma once
#include <vector>
#include <string>
#include <stack>
#include <unordered_map>
#include <trigen/linear_math.h>
#include <trigen/tree_meshifier.h>

namespace Lindenmayer {
    enum class Op {
        Noop,
        Forward,
        Push, Pop,
        Yaw_Pos, Yaw_Neg,
        Pitch_Pos, Pitch_Neg,
        Roll_Pos, Roll_Neg,
    };

    class System {
    public:
        using Alphabet = std::unordered_map<char, std::vector<Op>>;
        using Rule_Set = std::unordered_map<char, std::string>;

        System(std::string const& axiom, Alphabet const& alphabet, Rule_Set const& rules)
            : axiom(axiom), alphabet(alphabet), rules(rules) {
            // Check that every character is present in the alphabet
            for (auto& ch : axiom) {
                assert(alphabet.count(ch));
            }
            for (auto& rule : rules) {
                assert(alphabet.count(rule.first) > 0);
                for (auto& ch : rule.second) {
                    assert(alphabet.count(ch) > 0);
                }
            }
        }

        std::vector<Op> Iterate(size_t unIterations) const {
            std::vector<Op> ret;
            std::string current = axiom;
            std::string buf;

            for (size_t uiCurrentIteration = 0; uiCurrentIteration < unIterations; uiCurrentIteration++) {
                for (size_t iOff = 0; iOff < current.size(); iOff++) {
                    auto cur = current[iOff];
                    assert(alphabet.count(cur) == 1);
                    if (rules.count(cur)) {
                        buf += rules.at(cur);
                    } else {
                        buf += cur;
                    }
                }
                current = std::move(buf);
            }

            ret.reserve(current.size());
            for (auto ch : current) {
                auto const& Y = alphabet.at(ch);
                for (auto op : Y) {
                    ret.push_back(op);
                }
            }

            return ret;
        }
    private:
        std::string const axiom;
        Alphabet const alphabet;
        Rule_Set const rules;
    };

    struct Execution_State {
        uint32_t uiNodeCurrent;
        float flYaw, flPitch, flRoll;
    };

    inline lm::Vector4 GetDirectionVector(Execution_State const& s) {
        auto u = cosf(s.flYaw) * cosf(s.flPitch);
        auto v = cosf(s.flPitch) * sinf(s.flYaw);
        auto w = sinf(s.flPitch);
        return lm::Vector4(-v, w, -u);
    }

    struct Parameters {
        constexpr Parameters()
            : Parameters(64.0f) {}
        constexpr Parameters(float flStep)
            : Parameters(flStep, 0.785398163f, 0.785398163f, 0.785398163f) {}
        constexpr Parameters(float flStep, float flYaw, float flPitch, float flRoll)
            : flStep(flStep), flYaw(flYaw), flPitch(flPitch), flRoll(flRoll) {}

        float flStep, flYaw, flPitch, flRoll;
    };

    inline Tree_Node_Pool Execute(std::vector<Op> const& ops, Parameters const& params) {
        Tree_Node_Pool pool;
        std::stack<Execution_State> stack;
        Execution_State state = {};
        uint32_t uiRoot;

        state.flPitch = M_PI / 2.0f;
        pool.Allocate(uiRoot);
        state.uiNodeCurrent = uiRoot;

        for (uint32_t pc = 0; pc < ops.size(); pc++) {
            auto dir = GetDirectionVector(state);
            switch (ops[pc]) {
            case Op::Forward:
            {
                uint32_t nextNodeIdx;
                auto& nextNode = pool.Allocate(nextNodeIdx);
                auto& curNode = pool.GetNode(state.uiNodeCurrent);
                nextNode.vPosition = curNode.vPosition + params.flStep * dir;
                nextNode.unUser = 0;
                printf("Node: (%f, %f, %f)\n", nextNode.vPosition[0], nextNode.vPosition[1], nextNode.vPosition[2]);
                curNode.AddChild(nextNodeIdx);
                state.uiNodeCurrent = nextNodeIdx;
                break;
            }
            case Op::Push:
            {
                stack.push(state);
                break;
            }
            case Op::Pop:
            {
                state = stack.top();
                stack.pop();
                break;
            }
            case Op::Yaw_Pos:
            {
                state.flYaw += params.flYaw;
                break;
            }
            case Op::Yaw_Neg:
            {
                state.flYaw -= params.flYaw;
                break;
            }
            case Op::Roll_Pos:
            {
                state.flRoll += params.flRoll;
                break;
            }
            case Op::Roll_Neg:
            {
                state.flRoll -= params.flRoll;
                break;
            }
            case Op::Pitch_Pos:
            {
                state.flPitch += params.flPitch;
                break;
            }
            case Op::Pitch_Neg:
            {
                state.flPitch -= params.flPitch;
                break;
            }
            default:
                break;
            }
        }

        return pool;
    }
}