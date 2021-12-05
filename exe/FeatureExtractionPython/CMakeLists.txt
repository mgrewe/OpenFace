# setup pybind11
set(PYBIND11_PATH "${CMAKE_SOURCE_DIR}/lib/3rdParty/pybind11")
# If this fails, the repo has probably not been cloned with submodules. Run: git submodule update --init
add_subdirectory("${PYBIND11_PATH}" "${CMAKE_CURRENT_BINARY_DIR}/pybind11") # add and initialise pybind11
install(DIRECTORY "${PYBIND11_PATH}/include/" DESTINATION "${PYBIND11_PATH}/include") # pybind11 headers
install(FILES "${PYBIND11_PATH}/LICENSE" DESTINATION "${PYBIND11_PATH}") # pybind11 licence

# Local libraries
include_directories("${LandmarkDetector_SOURCE_DIR}/include")

pybind11_add_module(FeatureExtractionPython FeatureExtractionPython.cpp)
target_link_libraries(FeatureExtractionPython PRIVATE LandmarkDetector)
target_link_libraries(FeatureExtractionPython PRIVATE FaceAnalyser)
target_link_libraries(FeatureExtractionPython PRIVATE GazeAnalyser)
target_link_libraries(FeatureExtractionPython PRIVATE Utilities)
set_target_properties(FeatureExtractionPython PROPERTIES OUTPUT_NAME openface LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

install(TARGETS FeatureExtractionPython DESTINATION bin)
