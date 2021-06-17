set(SRC
    src/trigen.cs
    src/Properties/AssemblyInfo.cs
)

if(${TRIGEN_BUILD_CSHARP_BINDING})
    add_library(trigen-net SHARED ${SRC})
    set_target_properties(trigen-net PROPERTIES FOLDER "Libs")
    install(TARGETS trigen-net)
endif()

