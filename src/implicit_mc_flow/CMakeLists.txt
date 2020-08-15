#INCLUDE(CheckLibraryExists)

SET( MCFLOW_SRCS apply_mc_flow.cpp)

SET( MCFLOW_CHOLESKY_SRCS apply_mc_flow_cholesky.cpp)

INCLUDE_DIRECTORIES(
  #${BOOST_INCLUDEDIR}
  #/usr/local/include
  ${PROJECT_ROOT}/include/
  ${CMAKE_SOURCE_DIR}
  ./
  ${PROJECT_ROOT}/gmm-5.4/bin/include
)



LINK_DIRECTORIES(
  ${CMAKE_SOURCE_DIR}/include
  #${BOOST_LIBRARYDIR}
  #/usr/local/lib/OpenMesh
  #/usr/lib
  #/usr/local/lib
  ${PROJECT_ROOT}/lib
)

LINK_LIBRARIES( #m pthread ${Boost_LIBRARIES}
  OpenMeshCore OpenMeshTools 
  #taucs
  #gfortran
  #cblas
  #lapack
  #metis
  )
            
ADD_EXECUTABLE( apply_mc_flow ${MCFLOW_SRCS} )

if (WIN32)
    target_compile_definitions( apply_mc_flow
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS
	  PRIVATE _SCL_SECURE_NO_DEPRECATE
  )
endif ()

#ADD_EXECUTABLE( apply_mc_flow_cholesky ${MCFLOW_CHOLESKY_SRCS} )