#include "pathtrace.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

// implement SampleUniformDiskConcentric from PBRT
__host__ __device__ glm::vec2 sampleUniformDiskConcentric(glm::vec2 u) {
    // Map u to [-1,1]^2
    glm::vec2 uOffset = 2.0f * u - glm::vec2(1.0f);
    if (uOffset.x == 0 && uOffset.y == 0)
        return glm::vec2(0.0f);

    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = (PI / 4.0f) * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = (PI / 2.0f) - (PI / 4.0f) * (uOffset.x / uOffset.y);
    }

    return r * glm::vec2(cos(theta), sin(theta));
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

static glm::vec3* dev_vertices = NULL;
static glm::vec3* dev_normals = NULL;
static glm::vec2* dev_uvs = NULL;
static Triangle* dev_triangles = NULL;
static LinearBVHNode* dev_linearBVHNodes = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_normals, scene->normals.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_normals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_uvs, scene->uvs.size() * sizeof(glm::vec2));
    cudaMemcpy(dev_uvs, scene->uvs.data(), scene->uvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

#if USE_BVH
    cudaMalloc(&dev_linearBVHNodes, scene->linearBVH.size() * sizeof(LinearBVHNode));
    cudaMemcpy(dev_linearBVHNodes, scene->linearBVH.data(), scene->linearBVH.size() * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    cudaFree(dev_vertices);
    cudaFree(dev_normals);
    cudaFree(dev_uvs);
    cudaFree(dev_triangles);

#if USE_BVH
    cudaFree(dev_linearBVHNodes);
#endif

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.insideObject = false;

        // Stochastic Antialiasing
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jitterX = u01(rng) - 0.5f;
        float jitterY = u01(rng) - 0.5f;

        glm::vec3 rayDirection = glm::normalize(cam.view
           - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
           - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        );

        // Apply depth of field
        if (cam.aperture > 0.0f) {
            // Sample point on the lens aperture using concentric mapping
            glm::vec2 pLens = cam.aperture * sampleUniformDiskConcentric(glm::vec2(u01(rng), u01(rng)));
            glm::vec3 lensOffset = pLens.x * cam.right + pLens.y * cam.up;

            // Compute focal point
            float t = cam.focalDistance / glm::dot(rayDirection, glm::normalize(cam.view));
            glm::vec3 focalPoint = cam.position + rayDirection * t;

            // Adjust ray origin and direction
            segment.ray.origin = cam.position + lensOffset;
            segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
        } else {
            segment.ray.direction = rayDirection;
        }
    }
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    glm::vec3* vertices,
    glm::vec3* normals,
    glm::vec2* uvs,
    Triangle* triangles,
    LinearBVHNode* linearBVHNodes)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;
        bool tmp_outside = true;
        int tmp_materialid = -1;

        // naive parse through global geoms
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);

            } else if (geom.type == MESH)
            {
#if USE_BVH
                t = meshIntersectionTestWithLinearBVH(geom,
                                                pathSegment.ray,
                                                tmp_intersect,
                                                tmp_normal,
                                                tmp_uv,
                                                tmp_outside,
                                                vertices,
                                                normals,
                                                uvs,
                                                triangles,
                                                linearBVHNodes,
                                                tmp_materialid);
#else
                t = meshIntersectionTest(geom,
                                         pathSegment.ray,
                                         tmp_intersect,
                                         tmp_normal,
                                         tmp_uv,
                                         tmp_outside,
                                         vertices,
                                         normals,
                                         uvs,
                                         triangles,
                                         tmp_materialid);
#endif
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
                outside = tmp_outside;

                if (geom.type == MESH) {
                    intersections[path_index].materialId = tmp_materialid;
                } else {
                    intersections[path_index].materialId = geoms[hit_geom_index].materialid;
                }
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;
            intersections[path_index].outside = outside;
            intersections[path_index].intersectionPoint = intersect_point;
        }
    }
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    // Preload frequently accessed variables into registers
    PathSegment& segment = pathSegments[idx];
    int remainingBounces = segment.remainingBounces;
    if (remainingBounces <= 0) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];

    if (intersection.t > 0.0f) { // If the intersection exists...
        // Set up RNG
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

        Material material = materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

        if (material.emittance > 0.0f) {
            segment.color *= (materialColor * material.emittance);
            segment.remainingBounces = 0;  // Terminate ray if it hits a light source
        } else {
            glm::vec3 intersectionPoint = intersection.intersectionPoint;
            segment.insideObject = !intersection.outside;

            scatterRay(segment, intersectionPoint, intersection.surfaceNormal, material, rng);

            segment.remainingBounces--;

           if (segment.remainingBounces == 0) {
               segment.color = glm::vec3(0.0f);
           }

            // Apply Russian roulette. Reference to PBRT
#if RUSSIAN_ROULETTE
            // start after a minimum number of bounces
            if (depth > MIN_BOUNCES && segment.remainingBounces > 0) {
                float y = glm::max(glm::max(segment.color.r, segment.color.g), segment.color.b);
                float q = glm::max(MIN_SURVIVAL_PROB, 1.0f - y);

                thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
                float randVal = u01(rng);

                if (randVal < q) {
                    // Terminate the path
                    segment.color = glm::vec3(0.0f);
                    segment.remainingBounces = 0;
                } else {
                    // Survive
                    segment.color /= (1.0f - q);
                }
            }
#endif
        }
    } else {
        // If no intersection, black out the ray
        segment.color = glm::vec3(0.0f);
        segment.remainingBounces = 0;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];

        // Clamp the color
        glm::vec3 color = iterationPath.color;
        float maxColorValue = 10.0f;
        color = glm::min(color, glm::vec3(maxColorValue));

        image[iterationPath.pixelIndex] += color;
    }
}

struct IsActive {
    __host__ __device__
    bool operator()(const PathSegment& path) {
        return path.remainingBounces > 0;
    }
};

struct CompareByMaterial {
    __host__ __device__
    bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};

void sortByMaterial(int num_paths) {
    thrust::sort_by_key(
            thrust::device,
            dev_intersections,
            dev_intersections + num_paths,
            dev_paths,
            CompareByMaterial()
    );
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 256;

    ///////////////////////////////////////////////////////////////////////////

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_vertices,
            dev_normals,
            dev_uvs,
            dev_triangles,
            dev_linearBVHNodes
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        //  path segments that have been reshuffled to be contiguous in memory.

        if (SORTMATERIAL) {
            sortByMaterial(num_paths);
        }

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials);

        // Stream compaction
        dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, IsActive());
        num_paths = dev_path_end - dev_paths;

        iterationComplete = depth > traceDepth || num_paths == 0;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
