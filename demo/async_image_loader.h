#pragma once

#include <functional>
#include <list>
#include <string>

#include <uv.h>

#include <stb_image.h>

struct Image_Load_Result {
    void *data;
    void *image;
    int width;
    int height;
    int channels;
};

struct Image_Load_Request {
    void *data;
    std::string path;
    int channels;
    std::function<void(void *data, Image_Load_Result *result)> callback;
};

class IAsync_Image_Loader;

struct Pending_Image_Load {
    Image_Load_Request request;
    uv_work_t work;
    IAsync_Image_Loader *loader;
    Image_Load_Result result;
};

class IAsync_Image_Loader {
public:
    virtual ~IAsync_Image_Loader() { }

    virtual void
    BeginAsyncImageLoad(Image_Load_Request &&request)
        = 0;

    virtual void
    PendingLoadFinished(Pending_Image_Load *load)
        = 0;
};

class Async_Image_Loader : public IAsync_Image_Loader {
public:
    ~Async_Image_Loader() override = default;

    Async_Image_Loader(uv_loop_t *loop)
        : _loop(loop) { }

    void
    BeginAsyncImageLoad(Image_Load_Request &&request) override {
        _pendingImageLoads.push_front({ std::move(request), {} });
        auto &pendingLoad = _pendingImageLoads.front();
        pendingLoad.work.data = &pendingLoad;
        pendingLoad.loader = this;
        printf(
            "[AIL] Queueing image load '%s'\n",
            pendingLoad.request.path.c_str());
        uv_queue_work(
            _loop, &pendingLoad.work, &Async_Image_Loader::Work,
            &Async_Image_Loader::AfterWork);
    }

    void
    PendingLoadFinished(Pending_Image_Load *finishedLoad) override {
        printf(
            "[AIL] Finished loading '%s'\n",
            finishedLoad->request.path.c_str());
        auto it = std::find_if(
            _pendingImageLoads.begin(), _pendingImageLoads.end(),
            [&](Pending_Image_Load const &load) {
                return &load == finishedLoad;
            });

        if (it == _pendingImageLoads.end()) {
            fprintf(stderr, "Finished image load is not in list of pending image loads\n");
            return;
        }

        printf(
            "[AIL] Calling callback for '%s'\n",
            finishedLoad->request.path.c_str());
        it->request.callback(it->request.data, &it->result);

        printf(
            "[AIL] Freeing image for '%s'\n",
            finishedLoad->request.path.c_str());
        stbi_image_free(it->result.data);
        _pendingImageLoads.erase(it);
    }

protected:
    static void
    Work(uv_work_t* work) {
        auto *load = (Pending_Image_Load *)work->data;
        auto &req = load->request;
        int width, height, channels;
        printf("[AIL] Loading '%s'\n", load->request.path.c_str());
        auto image = stbi_load(
            req.path.c_str(), &width, &height, &channels,
            load->request.channels);
        if (image == nullptr) {
            load->result.image = nullptr;
            printf("[AIL] Failed to load '%s'\n", load->request.path.c_str());
            return;
        }
        load->result.image = image;
        load->result.width = width;
        load->result.height = height;
        load->result.channels = channels;
        printf("[AIL] Loaded '%s'\n", load->request.path.c_str());
    }

    static void
    AfterWork(uv_work_t* work, int status) {
        auto *load = (Pending_Image_Load *)work->data;
        load->loader->PendingLoadFinished(load);
    }

private:
    uv_loop_t *_loop;
    std::list<Pending_Image_Load> _pendingImageLoads;
};