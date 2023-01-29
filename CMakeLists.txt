project(gemms)

cmake_minimum_required(VERSION 3.5)

# add_executable(gemms12x4u   main.cpp gemmu8.cpp clock.cpp)
# target_compile_options(gemms12x4u PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-march=z15;-mtune=z15")
# target_include_directories(gemms12x4u PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_compile_definitions(gemms12x4u PUBLIC MR=12 NR=4)

# add_executable(gemms12x8u   main.cpp gemmu8.cpp clock.cpp)
# target_compile_options(gemms12x8u PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-march=z15;-mtune=z15")
# target_include_directories(gemms12x8u PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_compile_definitions(gemms12x8u PUBLIC MR=12 NR=8)

# add_executable(gemms8x4u   main.cpp gemmu8.cpp clock.cpp)
# target_compile_options(gemms8x4u PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-march=z15;-mtune=z15")
# target_include_directories(gemms8x4u PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_compile_definitions(gemms8x4u PUBLIC MR=8 NR=4)



# add_executable(gemms16x4u_low   main.cpp gemmu8_low.cpp clock.cpp)
# target_compile_options(gemms16x4u_low PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-march=z15;-mtune=z15")
# target_include_directories(gemms16x4u_low PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_compile_definitions(gemms16x4u_low PUBLIC MR=16 NR=4 LOW_0_127)

# add_executable(gemms12x4u_low   main.cpp gemmu8_low.cpp clock.cpp)
# target_compile_options(gemms12x4u_low PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-march=z15;-mtune=z15")
# target_include_directories(gemms12x4u_low PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_compile_definitions(gemms12x4u_low PUBLIC MR=12 NR=4 LOW_0_127)


add_executable(gemmsu16   main.cpp gemmu16.cpp clock.cpp)
target_compile_options(gemmsu16 PUBLIC "-mvx;-std=gnu++11;-O3;-mvx;-mzvector;-march=z15;-mtune=z15")
target_include_directories(gemmsu16 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(gemmsu16 PUBLIC MR=16 NR=4)