// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: worker classes related to async libtrigen operations
//

#include "stdafx.h"
#include "trigen_worker.h"

void Trigen_Worker::execute(Stage_Tag tag, std::function<Trigen_Status(Trigen_Session)> const &func, Trigen_Session session) {
    auto rc = func(session);
    emit onResult(tag, rc, session);
}

Trigen_Controller::Trigen_Controller()
    : session(nullptr)
    , _worker(new Trigen_Worker()) {
    _worker->moveToThread(&_thread);

    // Delete the worker when the thread is destroyed
    connect(&_thread, &QThread::finished, _worker, &QObject::deleteLater);
    // Attach the session handle to the request and forward it
    connect(this, &Trigen_Controller::execute, [this](Stage_Tag tag, std::function<Trigen_Status(Trigen_Session)> const &func) {
        emit execute2(tag, func, session);
    });
    // Forward the execution request to the worker
    connect(this, &Trigen_Controller::execute2, _worker, &Trigen_Worker::execute);
    // Forward the results to the client code
    connect(_worker, &Trigen_Worker::onResult, [&](Stage_Tag tag, Trigen_Status rc, Trigen_Session session) {
        emit onResult(tag, rc, session);
    });

    // Start the thread
    _thread.start();
}

Trigen_Controller::~Trigen_Controller() {
    _thread.quit();
    _thread.wait();
}