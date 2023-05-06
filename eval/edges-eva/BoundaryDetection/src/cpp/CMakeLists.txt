cmake_minimum_required ( VERSION 2.8 )

project ( sed )
find_package ( OpenCV REQUIRED )

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall" )

find_package ( OpenMP )
if ( OPENMP_FOUND )
    set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
    set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif()

add_executable ( sed sed.cpp gaussian.cpp utils.cpp )
target_link_libraries ( sed ${OpenCV_LIBS} )
