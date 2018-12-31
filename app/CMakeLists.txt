
cmake_minimum_required(VERSION 3.4.1)
#扫描opencv头文件
include_directories(src/main/cpp/include)
#扫描opencv库文件
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${CMAKE_SOURCE_DIR}/src/main/jinLibs/${ANDROID_ABI}" )

#扫描所有源文件
file(GLOB source_file src/main/cpp/*.cpp)
file(GLOB base src/main/cpp/base/*.cpp)
add_library( native-lib SHARED ${source_file})
add_library(base SHARED ${base})

target_link_libraries(
        base
        native-lib
        opencv_java3
        log
         android)