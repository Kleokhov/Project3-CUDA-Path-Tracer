#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
            newMaterial.color = glm::vec3(0.0f);

            // Read roughness if provided, default to 0.0f
            if (p.contains("ROUGHNESS")) {
                newMaterial.roughness = p["ROUGHNESS"];
            } else {
                newMaterial.roughness = 0.0f;
            }
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;

            // check if IOR exists, otherwise set a default
            if (p.contains("IOR")) {
                newMaterial.indexOfRefraction = p["IOR"];
            } else {
                newMaterial.indexOfRefraction = 1.0f;
            }

            // specular reflection for refractive materials
            if (p.contains("SPECULAR_COLOR")) {
                const auto& specCol = p["SPECULAR_COLOR"];
                newMaterial.specular.color = glm::vec3(specCol[0], specCol[1], specCol[2]);
                newMaterial.hasReflective = 1.0f;
            } else {
                newMaterial.specular.color = glm::vec3(1.0f);
            }

            // Read roughness
            if (p.contains("ROUGHNESS")) {
                newMaterial.roughness = p["ROUGHNESS"];
            } else {
                newMaterial.roughness = 0.0f;
            }
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
//    for (const auto& pair : MatNameToID) {
//        std::cout << "Material Name: " << pair.first << ", ID: " << pair.second << std::endl;
//    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        Geom newGeom;
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation,
                newGeom.rotation,
                newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        const auto& type = p["TYPE"];
        if (type == "gltf") {

        } else if (type == "obj") {
            newGeom.type = MESH;
            const std::string filename = p["FILE"].get<std::string>();

            std::string sceneDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
            std::string fullPath = sceneDir + filename;

            // Load the mesh
            int ret = loadObj(fullPath, newGeom);
            if (ret != 0) {
                std::cerr << "Failed to load obj mesh: " << filename << std::endl;
                exit(-1);
            }
        } else if (type == "cube") {
            newGeom.type = CUBE;
        } else if (type == "sphere") {
            newGeom.type = SPHERE;
        } else {
            std::cerr << "Unknown object type: " << type << std::endl;
            exit(-1);
        }
        geoms.push_back(newGeom);
    }

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    if (cameraData.contains("FOCALDISTANCE")) {
        camera.focalDistance = cameraData["FOCALDISTANCE"];
    } else {
        camera.focalDistance = 1.0f;
    }

    if (cameraData.contains("APERTURE")) {
        camera.aperture = cameraData["APERTURE"];
    } else {
        camera.aperture = 0.0f;
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

int Scene::loadObj(const string &fullPath, Geom &geom) {
    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(fullPath, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error() << std::endl;
        }
        return -1;
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning() << std::endl;
    }

    std::cout << "Loaded obj file: " << fullPath << std::endl;

    auto& objAttrib = reader.GetAttrib();
    auto& objShapes = reader.GetShapes();
    auto& objMaterials = reader.GetMaterials();

    // Scale and center obj mesh
    std::vector<glm::vec3> tempVertices(objAttrib.vertices.size() / 3);
    glm::vec3 minBounds, maxBounds;

    for (size_t i = 0; i < objAttrib.vertices.size() / 3; ++i) {
        tempVertices[i] = glm::vec3(
                objAttrib.vertices[3 * i + 0],
                objAttrib.vertices[3 * i + 1],
                objAttrib.vertices[3 * i + 2]
        );

        if (i == 0) {
            minBounds = tempVertices[i];
            maxBounds = tempVertices[i];
        } else {
            minBounds = glm::min(minBounds, tempVertices[i]);
            maxBounds = glm::max(maxBounds, tempVertices[i]);
        }
    }

    // Compute center and scale factor
    glm::vec3 center = (minBounds + maxBounds) * 0.5f;
    glm::vec3 extents = maxBounds - minBounds;
    float maxExtent = glm::max(extents.x, glm::max(extents.y, extents.z));
    float scaleFactor = 1.0f / maxExtent;

    glm::mat4 translation = glm::mat4(1.0f);
    translation[3] = glm::vec4(-center, 1.0f);

    glm::mat4 scale = glm::mat4(1.0f);
    scale[0][0] = scaleFactor;
    scale[1][1] = scaleFactor;
    scale[2][2] = scaleFactor;

    glm::mat4 normalizationTransform = scale * translation;
    glm::mat4 totalTransform = geom.transform * normalizationTransform;

    // Process vertices
    size_t baseVertexIndex = vertices.size();
    glm::vec3 transformedMinBounds(FLT_MAX);
    glm::vec3 transformedMaxBounds(-FLT_MAX);

    for (const auto& vertex : tempVertices) {
        glm::vec3 transformedVertex = glm::vec3(totalTransform * glm::vec4(vertex, 1.0f));
        vertices.push_back(transformedVertex);

        transformedMinBounds = glm::min(transformedMinBounds, transformedVertex);
        transformedMaxBounds = glm::max(transformedMaxBounds, transformedVertex);
    }

    geom.minBounds = transformedMinBounds;
    geom.maxBounds = transformedMaxBounds;

    // Process normals
    size_t baseNormalIndex = normals.size();
    glm::mat3 normalTransform = glm::transpose(glm::inverse(glm::mat3(totalTransform)));

    for (size_t i = 0; i < objAttrib.normals.size() / 3; ++i) {
        glm::vec3 normal(
                objAttrib.normals[3 * i + 0],
                objAttrib.normals[3 * i + 1],
                objAttrib.normals[3 * i + 2]
        );
        normals.push_back(glm::normalize(normalTransform * normal));
    }

    // Process UVs
    size_t baseUVIndex = uvs.size();

    for (size_t i = 0; i < objAttrib.texcoords.size() / 2; ++i) {
        uvs.emplace_back(
                objAttrib.texcoords[2 * i + 0],
                objAttrib.texcoords[2 * i + 1]
        );
    }

    // Process triangles
    geom.meshStart = triangles.size();
    for (const auto& shape : objShapes) {
        size_t indexOffset = 0;

        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            auto fv = size_t(shape.mesh.num_face_vertices[f]);

            // expect only triangles
            if (fv != 3) {
                std::cerr << "Error: Only triangular faces are supported." << std::endl;
                continue;
            }

            Triangle triangle{};
            triangle.materialId = geom.materialid;

            for (size_t v = 0; v < 3; ++v) {
                const tinyobj::index_t& idx = shape.mesh.indices[indexOffset + v];

                triangle.v[v] = baseVertexIndex + idx.vertex_index;
                triangle.n[v] = (idx.normal_index >= 0) ? baseNormalIndex + idx.normal_index : -1;
                triangle.uv[v] = (idx.texcoord_index >= 0) ? baseUVIndex + idx.texcoord_index : -1;
            }

            triangles.emplace_back(triangle);
            indexOffset += fv;
        }
    }
    geom.meshCount = triangles.size() - geom.meshStart;

    // for debugging, print the sizes
    std::cout << "Vertices: " << vertices.size() << std::endl;
    std::cout << "Normals: " << normals.size() << std::endl;
    std::cout << "UVs: " << uvs.size() << std::endl;

    return 0;
}