project(gemms)

cmake_minimum_required(VERSION 3.5)
find_package(OpenMP)
# add_executable(gemms12x4u   main.cpp gemmu8.cpp clock.cpp)
# target_compile_options(gemms12x4u PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-fopenmp;-march=z15;-mtune=z15")
# target_include_directories(gemms12x4u PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(gemms12x4u PUBLIC OpenMP::OpenMP_CXX)
# target_compile_definitions(gemms12x4u PUBLIC MR=12 NR=4 DNNL_CPU_THREADING_RUNTIME=2)

# add_executable(gemms12x8u   main.cpp gemmu8.cpp clock.cpp)
# target_compile_options(gemms12x8u PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-fopenmp;-march=z15;-mtune=z15")
# target_include_directories(gemms12x8u PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(gemms12x8u PUBLIC OpenMP::OpenMP_CXX)
# target_compile_definitions(gemms12x8u PUBLIC MR=12 NR=8 DNNL_CPU_THREADING_RUNTIME=2)

# add_executable(gemms8x4u   main.cpp gemmu8.cpp clock.cpp)
# target_compile_options(gemms8x4u PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-fopenmp;-march=z15;-mtune=z15")
# target_include_directories(gemms8x4u PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(gemms8x4u   PUBLIC OpenMP::OpenMP_CXX)
# target_compile_definitions(gemms8x4u PUBLIC MR=8 NR=4 DNNL_CPU_THREADING_RUNTIME=2)

# add_executable(gemms4x4u   main.cpp gemmu8.cpp clock.cpp)
# target_compile_options(gemms4x4u PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-fopenmp;-march=z15;-mtune=z15")
# target_include_directories(gemms4x4u PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(gemms4x4u   PUBLIC OpenMP::OpenMP_CXX)
# target_compile_definitions(gemms4x4u PUBLIC MR=4 NR=4 DNNL_CPU_THREADING_RUNTIME=2)

# add_executable(gemms16x4u   main.cpp gemmu8.cpp clock.cpp)
# target_compile_options(gemms16x4u PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-fopenmp;-march=z15;-mtune=z15")
# target_include_directories(gemms16x4u PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(gemms16x4u   PUBLIC OpenMP::OpenMP_CXX)
# target_compile_definitions(gemms16x4u PUBLIC MR=16 NR=4 DNNL_CPU_THREADING_RUNTIME=2)


# add_executable(gemms16x4u_low   main.cpp gemmu8.cpp clock.cpp)
# target_compile_options(gemms16x4u_low PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-fopenmp;-march=z15;-mtune=z15")
# target_include_directories(gemms16x4u_low PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(gemms16x4u_low PUBLIC OpenMP::OpenMP_CXX)
# target_compile_definitions(gemms16x4u_low PUBLIC MR=16 NR=4 LOW_0_127 DNNL_CPU_THREADING_RUNTIME=2)

# add_executable(gemms12x4u_low   main.cpp gemmu8.cpp clock.cpp)
# target_compile_options(gemms12x4u_low PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-fopenmp;-march=z15;-mtune=z15")
# target_include_directories(gemms12x4u_low PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(gemms12x4u_low PUBLIC OpenMP::OpenMP_CXX)
# target_compile_definitions(gemms12x4u_low PUBLIC MR=12 NR=4 LOW_0_127 DNNL_CPU_THREADING_RUNTIME=2)

# add_executable(gemms8x4u_low   main.cpp gemmu8.cpp clock.cpp)
# target_compile_options(gemms8x4u_low PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-fopenmp;-march=z15;-mtune=z15")
# target_include_directories(gemms8x4u_low PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(gemms8x4u_low  PUBLIC OpenMP::OpenMP_CXX)
# target_compile_definitions(gemms8x4u_low PUBLIC MR=8 NR=4 LOW_0_127 DNNL_CPU_THREADING_RUNTIME=2)


add_executable(gemmsu16   main.cpp gemmu16.cpp clock.cpp)
target_compile_options(gemmsu16 PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-fopenmp;-march=z15;-mtune=z15")
target_include_directories(gemmsu16 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gemmsu16   PUBLIC OpenMP::OpenMP_CXX)
target_compile_definitions(gemmsu16 PUBLIC MR=16 NR=4 DNNL_CPU_THREADING_RUNTIME=2)



# add_executable(gemmsu16_debug   main.cpp gemmu16.cpp clock.cpp)
# target_compile_options(gemmsu16_debug PUBLIC "-std=gnu++11;-g;-mvx;-mzvector;-march=z15;-fsanitize=address")
# target_include_directories(gemmsu16_debug PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(gemmsu16_debug PUBLIC asan ) 
# target_compile_definitions(gemmsu16_debug PUBLIC MR=16 NR=4 DNNL_CPU_THREADING_RUNTIME=1)