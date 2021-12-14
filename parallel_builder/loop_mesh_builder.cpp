/**
 * @file    loop_mesh_builder.cpp
 *
 * @author  Alexander Polok <xpolok03@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP loops
 *
 * @date    27.11.2021
 **/

#include <cmath>
#include <limits>
#include <omp.h>
#include "loop_mesh_builder.h"

LoopMeshBuilder::LoopMeshBuilder(unsigned gridEdgeSize)
        : BaseMeshBuilder(gridEdgeSize, "OpenMP Loop") {
    int threads = omp_get_max_threads();
    std::vector<std::vector<Triangle_t>> triangles(threads);
    mThreadTriangles = triangles;
    pointsCount = 0;
}

unsigned LoopMeshBuilder::marchCubes(const ParametricScalarField &field) {
    // 0. Retype vec of structs to struct of vectors
    pointsCount = unsigned(field.getPoints().size());
    const Vec3_t<float> *pPoints = field.getPoints().data();

    for (size_t i = 0; i < pointsCount; i++) {
        pPointsX.push_back(pPoints[i].x);
        pPointsY.push_back(pPoints[i].y);
        pPointsZ.push_back(pPoints[i].z);
    }

    // 1. Compute total number of cubes in the grid.
    size_t totalCubesCount = mGridSize * mGridSize * mGridSize;

    unsigned totalTriangles = 0;

    // 2. Loop over each coordinate in the 3D grid.
    #pragma omp parallel for default(none) shared(totalCubesCount, field) reduction(+: totalTriangles) schedule(static)
    for (size_t i = 0; i < totalCubesCount; ++i) {
        // 3. Compute 3D position in the grid.
        Vec3_t<float> cubeOffset(static_cast<float>(i % mGridSize),
                                 static_cast<float>((i / mGridSize) % mGridSize),
                                 static_cast<float>(i / (mGridSize * mGridSize))); // NOLINT

        // 4. Evaluate "Marching Cube" at given position in the grid and
        //    store the number of triangles generated.
        totalTriangles += buildCube(cubeOffset, field);
    }

    // 5. Flatten Triangles vector
    for(auto tVec: mThreadTriangles) {
        mTriangles.insert(std::end(mTriangles), std::begin(tVec), std::end(tVec));
    }

    // 6. Return total number of triangles generated.
    return totalTriangles;
}

float LoopMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field) {
    // NOTE: This method is called from "buildCube(...)"!

    float value = std::numeric_limits<float>::max();

    // 1. Find minimum square distance from points "pos" to any point in the field.
    #pragma omp simd reduction(min: value)
    for (unsigned i = 0; i < pointsCount; ++i) {
        float x = pPointsX[i];
        float y = pPointsY[i];
        float z = pPointsZ[i];
        float distanceSquared = (pos.x - x) * (pos.x - x);
        distanceSquared += (pos.y - y) * (pos.y - y);
        distanceSquared += (pos.z - z) * (pos.z - z);
        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 2. Finally, take square root of the minimal square distance to get the real distance
    return std::sqrt(value);
}



void LoopMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle) {
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.
    mThreadTriangles[omp_get_thread_num()].push_back(triangle);
}