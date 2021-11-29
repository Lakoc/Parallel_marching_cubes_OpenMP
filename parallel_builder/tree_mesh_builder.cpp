/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Alexander Polok <xpolok03@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    27.11.2021
 **/

#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
        : BaseMeshBuilder(gridEdgeSize, "Octree") {
    int threads = omp_get_max_threads();
    std::vector<std::vector<Triangle_t>> triangles(threads);
    mThreadTriangles = triangles;
    std::vector<unsigned> t_count(threads, 0);
    triangles_count = t_count;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field) {
    unsigned totalTriangles = 0;
    Vec3_t<unsigned> startPoint(0, 0, 0);

    // 1. Start recursion with single thread
    #pragma omp parallel default(none) shared(totalTriangles, mGridSize, startPoint, field)
    {
        #pragma omp single
        {
            divideCube(field, startPoint, mGridSize, 1);
        }

    }

    // 2. Flatten Triangles vector
    for (auto tVec: mThreadTriangles) {
        mTriangles.insert(std::end(mTriangles), std::begin(tVec), std::end(tVec));
    }

    // 3. Sum up triangles
    for (auto thread_c: triangles_count) {
        totalTriangles += thread_c;
    }

    return totalTriangles;
}


void
TreeMeshBuilder::divideCube(const ParametricScalarField &field, Vec3_t<unsigned> &pos, unsigned edgeSize,
                            unsigned gridDivision) {
    unsigned halfOfEdgeLen = edgeSize / 2;
    unsigned nextGridDivision= gridDivision * 2;

    // 1. If sequential threshold is met start seq
    if (edgeSize == 1 || gridDivision >=8) {
        size_t totalCubesCount = edgeSize * edgeSize * edgeSize;
        unsigned count = 0;

        // 2a. Loop over each coordinate in the 3D grid.
        for (size_t i = 0; i < totalCubesCount; ++i) {
            // 3a. Compute 3D position in the grid.
            Vec3_t<float> cubeOffset(static_cast<float>(pos.x + (i % edgeSize)),
                                     static_cast<float>(pos.y + ((i / edgeSize) % edgeSize)),
                                     static_cast<float>(pos.z + (i / (edgeSize * edgeSize)))); // NOLINT
            // 4a. Evaluate "Marching Cube" at given position in the grid and
            //    store the number of triangles generated.
            count += buildCube(cubeOffset, field);
        }
        // 5a. Add number to current thread count
        triangles_count[omp_get_thread_num()] += count;
        return;
    }

    // 2b. Calculate object range
    auto objectRange = static_cast<float>( mIsoLevel + ((sqrt(3.0) / 2.0) * halfOfEdgeLen));

    // 3b. Iterate over offsets and calculate startPosition and MidPoint
    for (auto offset: offsets) {
        Vec3_t<unsigned> startPos(pos.x + (offset.x / nextGridDivision),
                                  pos.y + (offset.y / nextGridDivision),
                                  pos.z + (offset.z / nextGridDivision));

        Vec3_t<float> cubeMidPoint(static_cast<float>((startPos.x + halfOfEdgeLen)) * mGridResolution,
                                   static_cast<float>((startPos.y + halfOfEdgeLen)) * mGridResolution,
                                   static_cast<float>((startPos.z + +halfOfEdgeLen)) * mGridResolution);


        // 4b. Calculate cube val and check if object is in current sub-block
        float cubeVal = evaluateFieldAt(cubeMidPoint, field);
        if (cubeVal <= objectRange) {
            #pragma omp task default(none) shared(field) firstprivate(startPos, halfOfEdgeLen, nextGridDivision)
            {
                // 5b. If condition is met create task of that sub-space
                divideCube(field, startPos, halfOfEdgeLen, nextGridDivision);
            }
        }
    }
    #pragma omp taskwait
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field) {
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const auto count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    #pragma omp simd reduction(min: value)
    for (unsigned i = 0; i < count; ++i) {
        float distanceSquared = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally, take square root of the minimal square distance to get the real distance
    return std::sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle) {
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.
    mThreadTriangles[omp_get_thread_num()].push_back(triangle);
}


