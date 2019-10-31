cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(modelAcceleration)

find_package(Torch REQUIRED)
find_package(Cuda REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main "${CUDA_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 11)