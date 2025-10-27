# 🧩 Image Filter CUDA

## Overview
This project demonstrates a **2D convolution filter implemented with CUDA**, focusing on **shared memory tiling and halo (apron) handling** for efficient image processing on the GPU.

It applies a **5×5 normalized blur filter** to a grayscale image using:
- **Constant memory** to store the filter kernel (fast broadcast to all threads)
- **Shared memory tiles** with border clamping to minimize global memory reads
- **Parameterized kernel templates** for different filter sizes and tile dimensions
- **CUDA error checking** and simple verification (impulse response ≈ 1/25)

This project illustrates key GPU concepts—**memory hierarchy**, **thread cooperation**, **tiling**, and **edge handling**—that apply to both image processing and general-purpose convolution operations in AI and vision workloads.

---

## 🚀 Key Learning Goals
- Implement a 2D convolution using CUDA kernels
- Optimize data reuse through shared memory
- Manage image borders using clamping
- Use constant memory for small filters
- Understand how threads are organized in 2D grids and blocks

---

## 🧠 Technical Highlights
| Feature | Description |
|----------|--------------|
| **Kernel Type** | 5×5 normalized box filter |
| **Optimization** | Shared memory tiling with halo |
| **Memory Usage** | Constant memory for kernel coefficients |
| **Launch Configuration** | 2D blocks & grids with tile size = 16 |
| **Verification** | Impulse test → 5×5 patch with value ≈ 0.04 |

---

## 📂 Project Structure
```
cpp-cuda-image-filter/
 ├── src/
 │    └── conv2d_shared.cu         # Main CUDA implementation
 ├── CMakeLists.txt                # Build configuration
 ├── README.md                     # Documentation
 ├── .gitignore                    # Ignore build artifacts
 └── build/                        # CMake build directory
```

---

## ⚙️ Build and Run

### 1. Configure and Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 2. Run
```bash
./build/conv2d_shared
```

Expected output:
```
Center value after blur: 0.040000 (expected ~0.04)
Success!
```

---

## 🧩 Understanding the CUDA Kernel
Each CUDA block loads a shared tile (with halo) into shared memory, then cooperatively computes a 16×16 patch of the output image. Constant memory stores the small convolution kernel for fast broadcast access.

```cpp
__shared__ float tile[TILE + 2*R][TILE + 2*R];
tile[threadIdx.y][threadIdx.x] = in[y * W + x];
__syncthreads();
```

Only inner threads perform convolution to produce valid output pixels.

---

## 🏷️ GitHub Topics
```
cuda
gpu-computing
parallel-programming
image-processing
computer-vision
shared-memory
2d-convolution
blur-filter
box-filter
cpp
cmake
nvidia
rtx-4070
```

---

## 🧪 Next Steps
- Replace the box filter with a **Gaussian kernel**
- Add support for **RGB (float3/float4)** images
- Implement **separable convolution** (horizontal + vertical passes)
- Benchmark different **TILE sizes** on RTX 4070 SUPER

---

## 🖋️ Author
**Samuel Huang (Chengliang Huang)**  
AI/ML Engineer | CUDA & LLM Systems Developer  
📍 Toronto, ON  
🔗 [GitHub: FlosMume](https://github.com/FlosMume)

