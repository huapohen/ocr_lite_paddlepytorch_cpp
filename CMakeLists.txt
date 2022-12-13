cmake_minimum_required(VERSION 3.1)

project(
  OCRLib
  VERSION 1.0
  DESCRIPTION "OCRLib"
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -fuse-ld=gold")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(3rdparty_ROOT ${CMAKE_SOURCE_DIR}/3rdparty)
# online version: opencv 4
include(${3rdparty_ROOT}/opencv-4.3.0/opencv_config.cmake)
# include(${3rdparty_ROOT}/opencv-3.4/opencv_config.cmake)
include(${3rdparty_ROOT}/ncnn/ncnn_config.cmake)

if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
  add_definitions(-DIS_LINUX)
endif()

file(
  GLOB_RECURSE
  OCRLib_SRCS
  "ocr/*.cc"
  "ocr/*/*.cc"
  "3rdparty/jsoncpp/*.cc"
)

list(
  APPEND
  OCRLib_HEADERS
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/ocr
  )

add_library(ocrlib SHARED ${OCRLib_SRCS})
target_include_directories(
  ocrlib PUBLIC ${OCRLib_HEADERS} ${OpenCV_INCLUDE_DIRS}
                ${NCNN_INCLUDE_DIRS}
                ${3rdparty_ROOT})
target_link_libraries(ocrlib PUBLIC 
                      ${OpenCV_LIBS} ${NCNN_LIBS})



set(ocr_DIR ${CMAKE_SOURCE_DIR})
add_executable(ocr ${ocr_DIR}/demo.cpp)
target_include_directories(
  ocr PUBLIC  ${OpenCV_INCLUDE_DIRS} ${OCRLib_HEADERS} 
              ${ocr_DIR} ${3rdparty_ROOT})
target_link_libraries(ocr ocrlib)

