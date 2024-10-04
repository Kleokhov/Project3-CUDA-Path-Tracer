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

    float tminX = (minBounds.x - ray.origin.x) * invDir.x;
    float tmaxX = (maxBounds.x - ray.origin.x) * invDir.x;
    if (invDir.x < 0.0f) std::swap(tminX, tmaxX);

    float tminY = (minBounds.y - ray.origin.y) * invDir.y;
    float tmaxY = (maxBounds.y - ray.origin.y) * invDir.y;
    if (invDir.y < 0.0f) std::swap(tminY, tmaxY);

    float tminZ = (minBounds.z - ray.origin.z) * invDir.z;
    float tmaxZ = (maxBounds.z - ray.origin.z) * invDir.z;
    if (invDir.z < 0.0f) std::swap(tminZ, tmaxZ);

    float tmin = glm::max(glm::max(tminX, tminY), tminZ);
    float tmax = glm::min(glm::min(tmaxX, tmaxY), tmaxZ);

    // If tmax is less than tmin, no intersection occurs
    if (tmax < tmin || tmax < 0) return false;

    return true;
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
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(dir, edge2);
    float a = glm::dot(edge1, h);
    if (fabs(a) < EPSILON) return false;

    float f = 1.0f / a;
    glm::vec3 s = orig - v0;
    u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    glm::vec3 q = glm::cross(s, edge1);
    v = f * glm::dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * glm::dot(edge2, q);
    if (t > EPSILON) return true;

    return false;
}

__host__ __device__ float meshIntersectionTest(
        const Geom& meshGeom,
        const Ray& ray,
        const glm::vec3* vertices,
        const Triangle* triangles,
        glm::vec3& intersectionPoint,
        glm::vec3& normal,
        bool& outside,
        int& materialId) {
    float t_min = FLT_MAX;
    bool hit = false;

    // Iterate over all triangles in the mesh
    for (int i = meshGeom.meshStart; i < meshGeom.meshStart + meshGeom.meshCount; i++) {
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

        if (triangleHit && t > 0.0f)
        {
            if (t < t_min)
            {
                t_min = t;
                intersectionPoint = ray.origin + t * ray.direction;

                normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

                // Determine if the intersection is from outside
                outside = glm::dot(ray.direction, normal) < 0.0f;
                if (!outside)
                {
                    normal = -normal;
                }

                // Set the material ID
                materialId = triangle.materialId;
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

