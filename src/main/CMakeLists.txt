set(MAIN_SOURCES config.cpp gui.cpp Main.cpp settings.cpp qtconfiggui.cpp halide_experiments.cpp)

#Compile
add_library(main ${MAIN_SOURCES})

include_directories(../imacq ../tld ../config)

target_link_libraries(main imacq tld config ${PROJECT_SOURCE_DIR}/dependencies/Halide/bin/libHalide.so)
