#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool intersectRayAABB(
    const Ray& ray,
    const glm::vec3& minBounds,
    const glm::vec3& maxBounds)
{
    glm::vec3 invDir = 1.0f / ray.direction;

    float t1 = (minBounds.x - ray.origin.x) * invDir.x;
    float t2 = (maxBounds.x - ray.origin.x) * invDir.x;
    float tmin = glm::min(t1, t2);
    float tmax = glm::max(t1, t2);

    t1 = (minBounds.y - ray.origin.y) * invDir.y;
    t2 = (maxBounds.y - ray.origin.y) * invDir.y;
    tmin = glm::max(tmin, glm::min(glm::min(t1, t2), tmax));
    tmax = glm::min(tmax, glm::max(glm::max(t1, t2), tmin));

    t1 = (minBounds.z - ray.origin.z) * invDir.z;
    t2 = (maxBounds.z - ray.origin.z) * invDir.z;
    tmin = glm::max(tmin, glm::min(glm::min(t1, t2), tmax));
    tmax = glm::min(tmax, glm::max(glm::max(t1, t2), tmin));

    return tmax >= tmin && tmax >= 0.0f;
}

__host__ __device__ bool intersectRayTriangle(
        const glm::vec3& orig,
        const glm::vec3& dir,
        const glm::vec3& v0,
        const glm::vec3& v1,
        const glm::vec3& v2,
        float& t,
        float& u,
        float& v)
{
    glm::vec3 v0v1 = v1 - v0;
    glm::vec3 v0v2 = v2 - v0;
    glm::vec3 pvec = glm::cross(dir, v0v2);
    float det = glm::dot(v0v1, pvec);

    // backface culling
    if (det < EPSILON) return false;

    // Ray and triangle are parallel if det is close to 0
    if (fabs(det) < EPSILON) return false;

    float invDet = 1.0f / det;

    glm::vec3 tvec = orig - v0;
    u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    glm::vec3 qvec = glm::cross(tvec, v0v1);
    v = glm::dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    t = glm::dot(v0v2, qvec) * invDet;

    return t > EPSILON;
}

__host__ __device__ float meshIntersectionTest(
        const Geom& geom,
        const Ray& ray,
        glm::vec3& intersectionPoint,
        glm::vec3& normal,
        bool& outside,
        const glm::vec3* vertices,
        const glm::vec3* normals,
        const glm::vec2* uvs,
        const Triangle* triangles) {
    float t_min = FLT_MAX;
    bool hit = false;
    bool normalsExist = normals != nullptr && normals[0] != glm::vec3(0.0f);

    // Iterate over all triangles in the mesh
    for (int i = geom.meshStart; i < geom.meshStart + geom.meshCount; i++) {
        const Triangle &triangle = triangles[i];

        glm::vec3 v0 = vertices[triangle.v[0]];
        glm::vec3 v1 = vertices[triangle.v[1]];
        glm::vec3 v2 = vertices[triangle.v[2]];

        float t, u, v;
        bool triangleHit = intersectRayTriangle(
                ray.origin, ray.direction,
                v0, v1, v2,
                t, u, v
        );

        if (triangleHit && t > EPSILON)
        {
            if (t < t_min)
            {
                t_min = t;
                intersectionPoint = ray.origin + t * ray.direction;

                if (normalsExist) {
                    // Interpolate the vertex normals using barycentric coordinates if normals exist
                    glm::vec3 n0 = normals[triangle.n[0]];
                    glm::vec3 n1 = normals[triangle.n[1]];
                    glm::vec3 n2 = normals[triangle.n[2]];

                    // Barycentric interpolation of normals
                    glm::vec3 interpolatedNormal = glm::normalize((1.0f - u - v) * n0 + u * n1 + v * n2);

                    normal = interpolatedNormal;
                } else {
                    // Compute the flat normal (if normals are missing)
                    normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                }

                // Determine if the intersection is from outside
                outside = glm::dot(ray.direction, normal) < 0.0f;
                if (!outside)
                {
                    normal = -normal;
                }

                hit = true;
            }
        }
    }

    if (hit) {
        return t_min;
    } else {
        return -1.0f;
    }
}

__host__ __device__ bool IntersectAABB(
        const Ray& ray,
        const glm::vec3& minBounds,
        const glm::vec3& maxBounds,
        float t) {
    glm::vec3 invDir = glm::vec3(
            ray.direction.x != 0 ? 1.0f / ray.direction.x : FLT_MAX,
            ray.direction.y != 0 ? 1.0f / ray.direction.y : FLT_MAX,
            ray.direction.z != 0 ? 1.0f / ray.direction.z : FLT_MAX
    );

    float tx1 = (minBounds.x - ray.origin.x) * invDir.x;
    float tx2 = (maxBounds.x - ray.origin.x) * invDir.x;
    float tmin = glm::min(tx1, tx2);
    float tmax = glm::max(tx1, tx2);

    float ty1 = (minBounds.y - ray.origin.y) * invDir.y;
    float ty2 = (maxBounds.y - ray.origin.y) * invDir.y;
    tmin = glm::max(tmin, glm::min(ty1, ty2));
    tmax = glm::min(tmax, glm::max(ty1, ty2));

    float tz1 = (minBounds.z - ray.origin.z) * invDir.z;
    float tz2 = (maxBounds.z - ray.origin.z) * invDir.z;
    tmin = glm::max(tmin, glm::min(tz1, tz2));
    tmax = glm::min(tmax, glm::max(tz1, tz2));

    // Ensure that the intersection happens in front of the ray's origin
    return tmax >= tmin && tmax > 0.0f;
}

__host__ __device__ float meshIntersectionTestWithBVH(
        const Geom& geom,
        const Ray& ray,
        glm::vec3& intersectionPoint,
        glm::vec3& normal,
        bool& outside,
        const glm::vec3* vertices,
        const glm::vec3* normals,
        const glm::vec2* uvs,
        const Triangle* triangles,
        const BVHNode* bvhNodes)
{
    float t_min = FLT_MAX;
    bool hit = false;

    // Recursive BVH traversal
    int nodeStack[64]; // A stack to hold the nodes to visit (maximum depth)
    int stackPtr = 0;
    nodeStack[stackPtr++] = 0;  // Start with the root node

    while (stackPtr > 0) {
        int nodeIdx = nodeStack[--stackPtr];
        const BVHNode& node = bvhNodes[nodeIdx];

        if (!IntersectAABB(ray, node.aabb.minBounds, node.aabb.maxBounds, t_min)) {
            continue;
        }

        // If this is a leaf node, test the ray against the triangles in the node
        if (node.primCount > 0) {
            for (int i = 0; i < node.primCount; i++) {
                int triIdx = node.firstPrim + i;
                const Triangle& triangle = triangles[triIdx];

                glm::vec3 v0 = vertices[triangle.v[0]];
                glm::vec3 v1 = vertices[triangle.v[1]];
                glm::vec3 v2 = vertices[triangle.v[2]];

                float t, u, v;
                if (intersectRayTriangle(ray.origin, ray.direction, v0, v1, v2, t, u, v) && t > EPSILON) {
                    if (t < t_min) {
                        t_min = t;
                        intersectionPoint = ray.origin + t * ray.direction;

                        // Compute flat normal if no normals are provided
                        normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

                        // Determine if the intersection is from outside
                        outside = glm::dot(ray.direction, normal) < 0.0f;
                        if (!outside) {
                            normal = -normal;
                        }

                        hit = true;
                    }
                }
            }
        } else {
            if (node.leftChild >= 0) {
                nodeStack[stackPtr++] = node.leftChild;
            }
            if (node.rightChild >= 0) {
                nodeStack[stackPtr++] = node.rightChild;
            }
        }
    }

    // Return the t-value of the intersection if a hit is found, otherwise return -1.0
    return hit ? t_min : -1.0f;
}




