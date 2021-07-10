if(${TRIGEN_BUILD_CSHARP_BINDING})
    add_executable(nettest)
    target_sources(nettest
        PRIVATE
            Program.cs Properties/AssemblyInfo.cs
    )
    target_link_libraries(nettest PRIVATE trigen-net)
    install(TARGETS nettest)
endif()

