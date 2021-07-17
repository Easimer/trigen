#include <array>
#include <thread>
#include <vector>

#include <stb_image.h>

#include <topo.h>
#include <topo_sdl.h>

#include <imgui.h>
#include <arcball_camera.h>

static glm::vec3 positionOffsets[] = { {
                                           1.0f,
                                           1.0f,
                                           0.0f,
                                       },
                                       {
                                           1.0f,
                                           -1.0f,
                                           0.0f,
                                       },
                                       {
                                           -1.0f,
                                           -1.0f,
                                           0.0f,
                                       },
                                       {
                                           -1.0f,
                                           1.0f,
                                           0.0f,
                                       } };

static glm::vec3 positions[4];

static float normals[] = {
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
};

static float texcoords[] = {
    1, 1, 1, 0, 0, 0, 0, 1,
};

static unsigned elements[] = {
    0, 1, 3, 1, 2, 3,
};

static uint8_t diffuse[] = {
    255, 128, 128,
};

static void
make_floor(topo::UPtr<topo::ISDL_Window> &instance, topo::Renderable_ID *floorWhite, topo::Renderable_ID *floorGreen) {
    static glm::vec3 floor_vtx[] = {
        { -100, 0, -100 }, { -100, 0, 100 }, { 100, 0, 100 }, { 100, 0, -100 }
    };
    static glm::vec3 floor_normals[]
        = { { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 } };
    static unsigned floor_elements[] = {
        0, 1, 3, 1, 2, 3,
    };

    topo::Model_Descriptor model;
    model.elements = floor_elements;
    model.element_count = 6;
    model.vertices = floor_vtx;
    model.uv = texcoords;
    model.normals = floor_normals;
    model.vertex_count = 4;

    topo::Model_ID handle;
    instance->CreateModel(&handle, &model);

    topo::Material_ID matSolidWhite;
    topo::Material_ID matSolidGreen;
    instance->CreateSolidColorMaterial(&matSolidWhite, { 1.0, 1.0, 1.0 });
    instance->CreateSolidColorMaterial(&matSolidGreen, { 0.2, 0.8, 0.2 });

    instance->CreateRenderable(floorWhite, handle, matSolidWhite);
    instance->CreateRenderable(floorGreen, handle, matSolidGreen);
}

int
main(int argc, char **argv) {
    topo::Surface_Config surf;
    surf.width = 1280;
    surf.height = 720;
    surf.title = "Renderer demo";
    
    auto camera = create_arcball_camera();

    auto window = topo::MakeWindow(surf);

    if (!window) {
        return 1;
    }

    ImGui::SetCurrentContext(window->ImguiContext());

    std::vector<topo::Model_ID> models;
    std::vector<topo::Renderable_ID> renderables;

    topo::Texture_ID texDiffuse;
    topo::Material_ID material;

    topo::Model_Descriptor model;
    model.elements = elements;
    model.element_count = 6;
    model.vertices = positions;
    model.uv = texcoords;
    model.normals = normals;
    model.vertex_count = 4;

    window->BeginModelManagement();

    for (int x = -10; x < 10; x++) {
        for (int y = -10; y < 10; y++) {
            for (int z = 0; z > -10; z--) {
                positions[0] = glm::vec3(x, y, z) + positionOffsets[0];
                positions[1] = glm::vec3(x, y, z) + positionOffsets[1];
                positions[2] = glm::vec3(x, y, z) + positionOffsets[2];
                positions[3] = glm::vec3(x, y, z) + positionOffsets[3];

                topo::Model_ID handle;
                window->CreateModel(&handle, &model);
                models.emplace_back(handle);
            }
        }
    }

    topo::Renderable_ID floorWhite, floorGreen;
    make_floor(window, &floorWhite, &floorGreen);

    window->FinishModelManagement();

    window->CreateTexture(
        &texDiffuse, 1, 1, topo::Texture_Format::RGB888, diffuse);
    window->CreateUnlitMaterial(&material, texDiffuse);

    for (auto &model : models) {
        topo::Renderable_ID renderable;
        window->CreateRenderable(&renderable, model, material);
        renderables.emplace_back(renderable);
    }

    std::vector<glm::vec3> lines;
    topo::Renderable_ID linesRenderable;

    lines.emplace_back(-1, 0, 0);
    lines.emplace_back(1, 0, 0);
    lines.emplace_back(1, 0, 0);
    lines.emplace_back(1, 1, 0);
    lines.emplace_back(1, 1, 0);
    lines.emplace_back(2, 1, 0);

    window->CreateRenderableLinesStreaming(
        &linesRenderable, lines.data(), lines.size() / 2, { 1.0, 0, 0 },
        { 1.0, 0, 0 });

    camera->set_screen_size(surf.width, surf.height);

    bool quit = false;
    while (!quit) {
        SDL_Event ev;

        window->NewFrame();

        while (window->PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_QUIT: {
                quit = true;
                break;
            }
            case SDL_WINDOWEVENT: {
                switch (ev.window.event) {
                case SDL_WINDOWEVENT_RESIZED: {
                    camera->set_screen_size(ev.window.data1, ev.window.data2);
                    break;
                }
                }
            }
            case SDL_MOUSEMOTION: {
                camera->mouse_move(ev.motion.x, ev.motion.y);
                break;
            }
            case SDL_MOUSEBUTTONDOWN: {
                camera->mouse_down(ev.button.x, ev.button.y);
                break;
            }
            case SDL_MOUSEBUTTONUP: {
                camera->mouse_up(ev.button.x, ev.button.y);
                break;
            }
            case SDL_MOUSEWHEEL: {
                camera->mouse_wheel(ev.wheel.y);
                break;
            }
            }
        }

        window->SetEyeViewMatrix(camera->get_view_matrix());
        auto rq = window->BeginRendering();

        topo::Transform transform;
        transform.position = { 0, 0, 0 };
        transform.rotation = { 1, 0, 0, 0 };
        transform.scale = { 1, 1, 1 };

        rq->Submit(floorWhite, transform);
        for (auto &renderable : renderables) {
            rq->Submit(renderable, transform);
        }
        rq->Submit(linesRenderable, transform);

        rq->AddLight(
            { 1, 0, 0, 1 }, { { -100, 100, -100 }, { 1, 0, 0, 0 }, { 1, 1, 1 } });
        rq->AddLight(
            { 0, 1, 0, 1 }, { { +100, 100, -100 }, { 1, 0, 0, 0 }, { 1, 1, 1 } });
        rq->AddLight(
            { 0, 0, 1, 1 }, { { +100, 100, +100 }, { 1, 0, 0, 0 }, { 1, 1, 1 } });
        rq->AddLight(
            { 1, 0, 1, 1 }, { { -100, 100, +100 }, { 1, 0, 0, 0 }, { 1, 1, 1 } });

        window->FinishRendering();

        if (ImGui::Begin("Test")) {

        }
        ImGui::End();

        window->Present();
    }

    return 0;
}