# 🧩 Image Filter CUDA (Windows + WSL2)

A compact CUDA C++ project implementing a **5×5 image blur** with **shared memory tiling**, **constant memory**, and **explicit halo handling**, optimized for Windows + WSL2 workflows.

---

## 🚀 Key Features
- CUDA **5×5** convolution (normalized box filter)
- **Shared memory tile + halo** for efficient stencil operations
- **Constant memory** for fast kernel coefficient broadcast
- CMake build, VS Code debug, CTest integration
- Status-check script for environment + runtime validation
- Impulse-response verification (~0.04 center value)

---

# 📐 GPU Memory Model & Kernel Workflow

## 1. Shared Memory Tile Layout (with Halo)

```
          Shared Memory Tile (TILE + 2R = 20 x 20)

     0   1   2   3  ...            18  19   <-- threadIdx.x
   +---+---+---+---+---+---+---+---+---+---+
 0 | H | H | H | H | H | H | H | H | H | H |
   +---+---+---+---+---+---+---+---+---+---+
 1 | H | H | H | H | H | H | H | H | H | H |
   +---+---+---+---+---+---+---+---+---+---+
 2 | H | H | T | T | T | T | T | … | H | H |
   +---+---+---+---+---+---+---+---+---+---+
 3 | H | H | T | T | T | T | T | … | H | H |
   +---+---+---+---+---+---+---+---+---+---+
 .                 (16×16 compute region)
 .
17| H | H | T | T | T | T | T | … | H | H |
   +---+---+---+---+---+---+---+---+---+---+
18| H | H | H | H | H | H | H | H | H | H |
   +---+---+---+---+---+---+---+---+---+---+
19| H | H | H | H | H | H | H | H | H | H |
   +---+---+---+---+---+---+---+---+---+---+

Legend:
H = Halo pixel (loaded but not producing output)
T = Tile pixel (used for output)
```

---

## 2. Convolution Dataflow Diagram
```
Global Memory                    Shared Memory                    Compute
(Full Image)                     (Tile + Halo)                    (Output Tile)
+--------------+        +--------------------------+        +-------------------+
|              |        | H H H H H H H H H H ... |        | O O O O O O O O  |
|   Image      | -----> | H T T T T T T T T H ... | -----> | O O O O O O O O  |
|              |  load  | H T T T T T T T T H ... |  K*K   | O O O O O O O O  |
+--------------+        | ...       (20x20)        |        |   (16x16 tile)   |
                        +--------------------------+        +-------------------+

       ↑                          ↑                             ↑
       |                          |                             |
       |                     cooperative load              write results
       |                  (blockDim.x * blockDim.y)         to global memory
```

---

# 🧩 Understanding the CUDA Implementation

## 1. Constant Memory Usage
```cpp
__constant__ float d_kernel[25];
```
Small, read-only, warp-broadcasted kernel coefficients.

---

## 2. Shared Memory Usage
```cpp
__shared__ float tile[TILE + 2*R][TILE + 2*R];
```
Shared tile + halo enabling efficient reuse of image data.

---

## 3. Why `#pragma unroll` Is Used
- Removes loop overhead
- Allows instruction-level optimization
- Ideal for small fixed-size kernels such as 5×5

---

## 4. Index & Boundary Handling
### Global coords with halo
```cpp
int ox = blockIdx.x * TILE + threadIdx.x - R;
```

### Clamped reads
```cpp
int x = max(0, min(W - 1, ox));
```

### Compute only in inner region
```cpp
if (threadIdx.x >= R && threadIdx.x < TILE + R)
```

### Output coords
```cpp
int outx = blockIdx.x * TILE + (threadIdx.x - R);
```

---

# 📂 Project Structure
```
cpp-cuda-image-filter/
├── src/
│   └── image_filter.cu
├── build/
├── .vscode/
│   └── launch.json
├── check_cuda_image_filter_status.sh
├── CMakeLists.txt
├── .gitignore
└── README.md
```

---

# ⚡ Status Check Script

Run:
```bash
./check_cuda_image_filter_status.sh
```

It will:
- Print CUDA Toolkit + GPU info  
- Validate presence of the compiled binary  
- Run the blur kernel  
- Parse center value and verify ≈ 0.04  
- Report timing information  

---

# 🧠 Environment

| Component | Recommended Version |
|----------|---------------------|
| CUDA Toolkit | 12.8+ |
| CMake | ≥ 3.24 |
| Compiler | NVCC + GCC |
| GPU | RTX 4070 SUPER (SM 8.9) |
| OS | Windows 11 + WSL2 (Ubuntu 22.04) |

---

---

# ⚙️ Build & Run

### Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Run
```bash
./build/conv2d_shared
```

Expected output:
```
Center value after blur: 0.040000 (expected ~0.04)
Success!
```

---

# 🧪 CTest
```bash
cd build
ctest
```

---

# ⚡ Status Check
```bash
./check_cuda_image_filter_status.sh
```

# 🛠 VS Code Integration
```json
{
  "name": "Run image_filter",
  "type": "cppdbg",
  "request": "launch",
  "program": "${workspaceFolder}/build/image_filter",
  "cwd": "${workspaceFolder}",
  "MIMode": "gdb"
}
```

---

# 📚 References
- *CUDA by Example* — Sanders & Kandrot  
- NVIDIA CUDA Toolkit Documentation  
- CMake CUDA Language Guide  

---

# © Author
**Samuel Huang**  
Toronto, ON  
GitHub: https://github.com/FlosMume
