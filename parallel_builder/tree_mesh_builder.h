/**
 * @file    tree_mesh_builder.h
 *
 * @author  Alexander Polok <xpolok03@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    27.11.2021
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"
#include <cmath>

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    explicit TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field) override;
    void divideCube(const ParametricScalarField &field, Vec3_t<unsigned> &pos, unsigned edgeSize, unsigned depth);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field) override;
    void emitTriangle(const Triangle_t &triangle) override;
    const Triangle_t *getTrianglesArray() const override { return mTriangles.data(); }

    std::vector<Triangle_t> mTriangles;
    std::vector<std::vector<Triangle_t>> mThreadTriangles;

    std::vector<std::vector<Vec3_t<unsigned >>> offsets{
        {
            {0,                    0,                    0},
            {0 + mGridSize, 0,                    0},
            {0,                    0 + mGridSize, 0},
            {0,                    0,                    0 + mGridSize},
            {0 + mGridSize, 0 + mGridSize, 0},
            {0 + mGridSize, 0,                    0 + mGridSize},
            {0,                    0 + mGridSize, 0 + mGridSize},
            {0 + mGridSize, 0 + mGridSize, 0 + mGridSize},
        },
    };
    std::vector<unsigned> trianglesCount;
    std::vector<float> pPointsX;
    std::vector<float> pPointsY;
    std::vector<float> pPointsZ;
    unsigned pointsCount;
    size_t maxDepth = 3;
};

#endif // TREE_MESH_BUILDER_H
