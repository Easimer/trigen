#include <trigen.h>

namespace Net::Easimer::Trigen {
public
ref struct Position {
    Position(tg_f32 x, tg_f32 y, tg_f32 z)
        : x(x)
        , y(y)
        , z(z) {
        }
    tg_f32 x, y, z;
};
public
ref class Parameters {
public:
    tg_u32 flags = Trigen_F_None;

    Position ^seed_position = gcnew Position(0, 0, 0);

    tg_f32 density;
    tg_f32 attachment_strength;
    tg_f32 surface_adaption_strength;
    tg_f32 stiffness;
    tg_f32 aging_rate;
    tg_f32 phototropism_response_strength;
    tg_f32 branching_probability;
    tg_f32 branch_angle_variance;

    tg_u32 particle_count_limit;
};
public
ref class Session {
public:
    static Session ^ Create(Parameters ^ params) { return gcnew Session(params); }

protected :
    Session(Parameters ^ params) {
    }
};
}