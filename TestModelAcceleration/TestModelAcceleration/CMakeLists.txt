# CMakeList.txt: TestModelAcceleration 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)



# TODO: 如有需要，请添加测试并安装目标。
set(CUDNN_INCLUDE_PATH  "D:\\cuda\\include")
set(CUDNN_LIBRARY_PATH  "D:\\cuda\\lib\\x64")

set(TORCH_INCLUDE_PATH  "D:\\libtorch\\include")
set(TORCH_LIBRARY_PATH  "D:\\libtorch\\lib")

#set(CUDA_INCLUDE_PATH  "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\include")
#set(CUDA_LIBRARY_PATH  "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\lib\\x64")

#find_package(Torch REQUIRED)

find_package(Cuda REQUIRED)

include_directories(${CUDNN_INCLUDE_PATH})
link_directories(${CUDNN_LIBRARY_PATH})

include_directories(${TORCH_INCLUDE_PATH})
include_directories(${TORCH_INCLUDE_PATH}\\torch\\csrc\\api\\include)
link_directories(${TORCH_LIBRARY_PATH})
link_libraries(c10)
link_libraries(torch)

#link_libraries(torch)

#link_directories(${CUDA_LIBRARY_PATH})
#link_libraries(cuda)


#if(NOT Torch_FOUND)
#    message(FATAL_ERROR "Pytorch Not Found!")
#endif(NOT Torch_FOUND)
# 将源代码添加到此项目的可执行文件。
add_executable (TestModelAcceleration "TestModelAcceleration.cpp" "TestModelAcceleration.h")

#target_link_libraries(TestModelAcceleration cudnn torch cuda c10 c10_cuda)
target_link_libraries(TestModelAcceleration "cudnn" "torch" "c10_cuda" "c10" "caffe2_detectron_ops_gpu" "caffe2_nvrtc" "clog" "cpuinfo" "libprotobuf" "libprotoc" "${CUDA_LIBRARIES}")
#target_link_libraries(modelAcceleration cudnn.lib torch.lib cuda.lib c10.lib c10_cuda.lib)
#target_link_libraries(modelAcceleration torch)
#target_link_libraries(modelAcceleration cuda)
#target_link_libraries(modelAcceleration c10)
#target_link_libraries(modelAcceleration c10_cuda)

set_property(TARGET TestModelAcceleration PROPERTY CXX_STANDARD 11)





