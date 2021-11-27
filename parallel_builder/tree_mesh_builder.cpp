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
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field) {
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    // 1. Compute total number of cubes in the grid.

    unsigned totalTriangles;
    Vec3_t<unsigned> startPoint(0, 0, 0);

    // 2. Loop over each coordinate in the 3D grid.
#pragma omp parallel default(none) shared(totalTriangles, mGridSize, startPoint, field)
    {
#pragma omp master
        {
            totalTriangles = divideCube(field, startPoint, mGridSize, 1);
        }

    }
    // Flatten Triangles vector
    for (auto tVec: mThreadTriangles) {
        mTriangles.insert(std::end(mTriangles), std::begin(tVec), std::end(tVec));
    }
    return totalTriangles;
}


unsigned
TreeMeshBuilder::divideCube(const ParametricScalarField &field, Vec3_t<unsigned> &pos, unsigned edgeSize,
                            unsigned depth) {
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    unsigned halfOfEdgeLen = edgeSize / 2;
    unsigned nextDepth = depth * 2;
    if (edgeSize <= 4) {

        size_t totalCubesCount = edgeSize * edgeSize * edgeSize;

        unsigned totalTriangles = 0;

        // 2. Loop over each coordinate in the 3D grid.
        for (size_t i = 0; i < totalCubesCount; ++i) {
            // 3. Compute 3D position in the grid.
            Vec3_t<float> cubeOffset(static_cast<float>(pos.x + (i % edgeSize)),
                                     static_cast<float>(pos.y + ((i / edgeSize) % edgeSize)),
                                     static_cast<float>(pos.z + (i / (edgeSize * edgeSize)))); // NOLINT
            // 4. Evaluate "Marching Cube" at given position in the grid and
            //    store the number of triangles generated.
            totalTriangles += buildCube(cubeOffset, field);
        }
        return totalTriangles;
    }


    unsigned triangles = 0;
    auto objectRange = static_cast<float>( mIsoLevel + sqrt(3.0) / 2.0 * mGridSize);
    for (auto offset: offsets) {
        Vec3_t<unsigned> startPos(pos.x + (offset.x / nextDepth),
                                  pos.y + (offset.y / nextDepth),
                                  pos.z + (offset.z / nextDepth));

        Vec3_t<float> cubeMidPoint(static_cast<float>((startPos.x + halfOfEdgeLen)) * mGridResolution,
                                   static_cast<float>((startPos.y + halfOfEdgeLen)) * mGridResolution,
                                   static_cast<float>((startPos.z + +halfOfEdgeLen)) * mGridResolution);


        float cube_val = evaluateFieldAt(cubeMidPoint, field);
        if (cube_val <= objectRange) {
#pragma omp task default(none) shared(field, triangles) firstprivate( startPos, halfOfEdgeLen, nextDepth)
            {
#pragma omp atomic
                triangles += divideCube(field, startPos, halfOfEdgeLen, nextDepth);

            }
        }

    }


#pragma omp taskwait

    return triangles;


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
