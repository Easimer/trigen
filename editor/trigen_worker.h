// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: worker classes related to async libtrigen operations
//

#pragma once

#include <QThread>
#include <trigen.h>

enum class Stage_Tag {
    Metaballs,
    Mesh,
    Foliage,
    Painting,
};

Q_DECLARE_METATYPE(Stage_Tag);
Q_DECLARE_METATYPE(Trigen_Status);
Q_DECLARE_OPAQUE_POINTER(Trigen_Session);
Q_DECLARE_METATYPE(Trigen_Session);
Q_DECLARE_METATYPE(std::function<Trigen_Status(Trigen_Session)>);

class Trigen_Worker : public QObject {
    Q_OBJECT;
public:

public slots:
    void execute(Stage_Tag tag, std::function<Trigen_Status(Trigen_Session)> const &func, Trigen_Session session);

signals:
    void onResult(Stage_Tag tag, Trigen_Status rc, Trigen_Session session);
};

class Trigen_Controller : public QObject {
    Q_OBJECT;
public:
    Trigen_Controller();
    ~Trigen_Controller();

    Trigen_Session session = nullptr;

signals:
    /** Client code calls this when they want to run a libtrigen call on the
     * worker thread.
     * The tag is used to identify what kind of call finished.
     */
    void execute(Stage_Tag tag, std::function<Trigen_Status(Trigen_Session)> const &func);
    /** Client code connects to this signal to get notified when a task
     * finishes
     */
    void onResult(Stage_Tag tag, Trigen_Status rc, Trigen_Session session);

    /** Do not use.
     * Used internally to forward the session handle to the worker object
     */
    void execute2(Stage_Tag tag, std::function<Trigen_Status(Trigen_Session)> const &func, Trigen_Session session);

private:
    QThread _thread;
    Trigen_Worker *_worker;
};
