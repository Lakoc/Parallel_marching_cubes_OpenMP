/**
 * @file    loop_mesh_builder.h
 *
 * @author  Alexander Polok <xpolok03@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP loops
 *
 * @date    27.11.2021
 **/

#ifndef LOOP_MESH_BUILDER_H
#define LOOP_MESH_BUILDER_H

#include <vector>
#include "base_mesh_builder.h"

class LoopMeshBuilder : public BaseMeshBuilder {
public:
    explicit LoopMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field) override;

    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field) override;

    void emitTriangle(const Triangle_t &triangle) override;

    const Triangle_t *getTrianglesArray() const override { return mTriangles.data(); }

    std::vector<Triangle_t> mTriangles;
    std::vector<std::vector<Triangle_t>> mThreadTriangles;
    std::vector<float> pPointsX;
    std::vector<float> pPointsY;
    std::vector<float> pPointsZ;
    unsigned pointsCount;
};

#endif // LOOP_MESH_BUILDER_H
