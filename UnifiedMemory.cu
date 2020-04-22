#include <iostream>
#include <random>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include <vector>

struct RGBPoint {
    float x;
    float y;
    float z;
    float r;
    float g;
    float b;
    float a;
    RGBPoint() {}
    RGBPoint(float x, float y, float z, float r, float g, float b, float a) : x(x), y(y), z(z), r(r), g(g), b(b), a(a) {}
};

__global__ void TestKernelRGB(RGBPoint *img_RGB, const int row, const int col) {
    int h = threadIdx.x + blockIdx.x * blockDim.x;
    int w = threadIdx.y + blockIdx.y * blockDim.y;
    if ((h >= row) || (w >= col)) {
        return;
    }

    int index = h * col + w;

    if (index % 120 == 0) {
        img_RGB[index].a = 3.0f;
        img_RGB[index].r = 4.0f;
        img_RGB[index].g = 5.0f;
        img_RGB[index].b = 6.0f;
        img_RGB[index].x = 7.0f;
        img_RGB[index].y = 8.0f;
        img_RGB[index].z = 9.0f;
    }
}

__global__ void TestKernelFloat(float *img_float, float *img_float_a, const int row, const int col) {
    int h = threadIdx.x + blockIdx.x * blockDim.x;
    int w = threadIdx.y + blockIdx.y * blockDim.y;
    if ((h >= row) || (w >= col)) {
        return;
    }

    int index  = h * col + w;
    int index6 = index * 6;

    if (index % 120 == 0) {
        img_float_a[index] = 3.0f;
        img_float[index6] = 4.0f;
        img_float[index6 + 1] = 5.0f;
        img_float[index6 + 2] = 6.0f;
        img_float[index6 + 3] = 7.0f;
        img_float[index6 + 4] = 8.0f;
        img_float[index6 + 5] = 9.0f;
    }
}

int main() {
    struct timeval start, end;
    float t1, t2, t3, t4, t5;

    const int row = 640;
    const int col = 480;
    
    const size_t size_RGB   = row * col * sizeof(RGBPoint);
    const size_t size_float = row * col * sizeof(float);

    RGBPoint *img_RGB;
    cudaMallocManaged(&img_RGB, size_RGB);
    float *img_float, *img_float_a;
    cudaMallocManaged(&img_float, size_float * 6);
    cudaMallocManaged(&img_float_a, size_float);

    dim3 block_size(4, 32);
    dim3 grid_size((row - 1) / block_size.x + 1, (col - 1) / block_size.y + 1);
    gettimeofday(&start, nullptr);
    TestKernelRGB<<<grid_size, block_size>>>(img_RGB, row, col);
    cudaDeviceSynchronize();
    gettimeofday(&end, nullptr);
    t1 = ((end.tv_sec - start.tv_sec) * 1000000.0f + (end.tv_usec - start.tv_usec)) / 1000.0f;
    
    gettimeofday(&start, nullptr);
    TestKernelFloat<<<grid_size, block_size>>>(img_float, img_float_a, row, col);
    cudaDeviceSynchronize();
    gettimeofday(&end, nullptr);
    t2 = ((end.tv_sec - start.tv_sec) * 1000000.0f + (end.tv_usec - start.tv_usec)) / 1000.0f;

    std::vector<RGBPoint> result_RGB;
    gettimeofday(&start, nullptr);
    for (int h = 0; h < row; h++) {
        for (int w = 0; w < col; w++) {
            int index = h * col + w;
            if (img_RGB[index].a > 0) {
                result_RGB.push_back(img_RGB[index]);
            }
        }
    }
    gettimeofday(&end, nullptr);
    t3 = ((end.tv_sec - start.tv_sec) * 1000000.0f + (end.tv_usec - start.tv_usec)) / 1000.0f;
    
    std::vector<RGBPoint> result_float1;
    gettimeofday(&start, nullptr);
    for (int h = 0; h < row; h++) {
        for (int w = 0; w < col; w++) {
            int index = h * col + w;
            if (img_float_a[index] > 0) {
                //RGBPoint img;
                result_float1.push_back(img_RGB[index]);
            }
        }
    }
    gettimeofday(&end, nullptr);
    t4 = ((end.tv_sec - start.tv_sec) * 1000000.0f + (end.tv_usec - start.tv_usec)) / 1000.0f;

    std::vector<RGBPoint> result_float2;
    gettimeofday(&start, nullptr);
    for (int h = 0; h < row; h++) {
        for (int w = 0; w < col; w++) {
            int index = h * col + w;
            if (img_float_a[index] > 0) {
                RGBPoint img;
                int index6 = index * 6;

                img.a = img_float_a[index];
                img.r = img_float[index6];
                img.g = img_float[index6 + 1];
                img.b = img_float[index6 + 2];
                img.x = img_float[index6 + 3];
                img.y = img_float[index6 + 4];
                img.z = img_float[index6 + 5];
                result_float2.push_back(img);
            }
        }
    }
    gettimeofday(&end, nullptr);
    t5 = ((end.tv_sec - start.tv_sec) * 1000000.0f + (end.tv_usec - start.tv_usec)) / 1000.0f;

    cudaFree(img_RGB);
    cudaFree(img_float);
    cudaFree(img_float_a);

    std::cout << "RGB kernel time: " << t1 << " ms float kernel time: " << t2 << " ms" << std::endl;
    std::cout << "RGB postprocess time: " << t3 << " ms float postprocess time: " << t5 << " ms float postprocess time without data copy: " << t4 << " ms" << std::endl;

    return 0;
}
