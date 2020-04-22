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

__global__ void TestKernel(RGBPoint *d_img_RGB, const int row, const int col) {
    int h = threadIdx.x + blockIdx.x * blockDim.x;
    int w = threadIdx.y + blockIdx.y * blockDim.y;
    if ((h >= row) || (w >= col)) {
        return;
    }

    int index = h * col + w;

    if (index % 120 == 0) {
        d_img_RGB[index].a = 3.0f;
        d_img_RGB[index].r = 4.0f;
        d_img_RGB[index].g = 5.0f;
        d_img_RGB[index].b = 6.0f;
        d_img_RGB[index].x = 7.0f;
        d_img_RGB[index].y = 8.0f;
        d_img_RGB[index].z = 9.0f;
    }
}

int main() {
    struct timeval start, end;
    float t1, t2;

    const int row = 640;
    const int col = 480;
    
    const size_t size_RGB   = row * col * sizeof(RGBPoint);

    RGBPoint *img_RGB = (RGBPoint*)malloc(size_RGB);
    RGBPoint *d_img_RGB;
    cudaMalloc(&d_img_RGB, size_RGB);

    dim3 block_size(4, 32);
    dim3 grid_size((row - 1) / block_size.x + 1, (col - 1) / block_size.y + 1);
    gettimeofday(&start, nullptr);
    TestKernel<<<grid_size, block_size>>>(d_img_RGB, row, col);
    cudaMemcpy(img_RGB, d_img_RGB, size_RGB, cudaMemcpyDeviceToHost);
    gettimeofday(&end, nullptr);
    t1 = ((end.tv_sec - start.tv_sec) * 1000000.0f + (end.tv_usec - start.tv_usec)) / 1000.0f;
    
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
    t2 = ((end.tv_sec - start.tv_sec) * 1000000.0f + (end.tv_usec - start.tv_usec)) / 1000.0f;
    
    free(img_RGB);
    cudaFree(d_img_RGB);

    std::cout << "kernel and data transfer time: " << t1 << " ms" << std::endl;
    std::cout << "postprocess time: " << t2 << " ms" << std::endl;

    return 0;
}
