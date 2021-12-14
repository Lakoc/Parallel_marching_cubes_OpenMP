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
#include <limits>
#include <omp.h>
#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
        : BaseMeshBuilder(gridEdgeSize, "Octree") {
    int threads = omp_get_max_threads();
    std::vector<std::vector<Triangle_t>> triangles(threads);
    mThreadTriangles = triangles;
    std::vector<unsigned> t_count(threads, 0);
    trianglesCount = t_count;
    pointsCount = 0;

    // Init offsets
    for (size_t i = 1; i <= maxDepth; i++) {
        std::vector<Vec3_t<unsigned >> offset_prev = offsets[i - 1];
        std::vector<Vec3_t<unsigned >> offset_new;
        for (auto tVec: offset_prev) {
            auto newX = tVec.x / 2;
            auto newY = tVec.y / 2;
            auto newZ = tVec.z / 2;
            offset_new.emplace_back(newX, newY, newZ);
        }
        offsets.push_back(offset_new);
    }
}


unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field) {
    unsigned totalTriangles = 0;
    Vec3_t<unsigned> startPoint(0, 0, 0);

    // 0. Retype vec of structs to struct of vectors
    pointsCount = unsigned(field.getPoints().size());
    const Vec3_t<float> *pPoints = field.getPoints().data();

    for (size_t i=0; i < pointsCount; i++) {
        pPointsX.push_back(pPoints[i].x);
        pPointsY.push_back(pPoints[i].y);
        pPointsZ.push_back(pPoints[i].z);
    }

    // 1. Start recursion with single thread
    #pragma omp parallel default(none) shared(totalTriangles, mGridSize, startPoint, field)
    {
        #pragma omp single
        {
            divideCube(field, startPoint, mGridSize, 0);
        }

    }

    // 2. Flatten Triangles vector
    for (auto tVec: mThreadTriangles) {
        mTriangles.insert(std::end(mTriangles), std::begin(tVec), std::end(tVec));
    }

    // 3. Sum up triangles
    for (auto thread_c: trianglesCount) {
        totalTriangles += thread_c;
    }

    return totalTriangles;
}


void
TreeMeshBuilder::divideCube(const ParametricScalarField &field, Vec3_t<unsigned> &pos, unsigned edgeSize,
                            unsigned depth) {
    unsigned halfOfEdgeLen = edgeSize / 2;
    unsigned nextDepth = depth + 1;

    // 1. If sequential threshold is met start seq
    if (edgeSize == 1 || depth >= maxDepth) {
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
        trianglesCount[omp_get_thread_num()] += count;
        return;
    }

    // 2b. Calculate object range
    auto objectRange = static_cast<float>( mIsoLevel + ((sqrt(3.0) / 2.0) * halfOfEdgeLen));

    // 3b. Iterate over offsets and calculate startPosition and MidPoint
    for (auto offset: offsets[nextDepth]) {
        Vec3_t<unsigned> startPos(pos.x + (offset.x),
                                  pos.y + (offset.y),
                                  pos.z + (offset.z));

        Vec3_t<float> cubeMidPoint(static_cast<float>((startPos.x + halfOfEdgeLen)) * mGridResolution,
                                   static_cast<float>((startPos.y + halfOfEdgeLen)) * mGridResolution,
                                   static_cast<float>((startPos.z + +halfOfEdgeLen)) * mGridResolution);


        // 4b. Calculate cube val and check if object is in current sub-block
        float cubeVal = evaluateFieldAt(cubeMidPoint, field);
        if (cubeVal <= objectRange) {
            #pragma omp task default(none) firstprivate(field, startPos, halfOfEdgeLen, nextDepth)
            {
                // 5b. If condition is met create task of that subspace
                divideCube(field, startPos, halfOfEdgeLen, nextDepth);
            }
        }
    }
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field) {
    // NOTE: This method is called from "buildCube(...)"!
    float value = std::numeric_limits<float>::max();

    // 1. Find minimum square distance from points "pos" to any point in the
    //    field.
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

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle) {
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.
    mThreadTriangles[omp_get_thread_num()].push_back(triangle);
}


