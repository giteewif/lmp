```bash
apt install -y software-properties-common
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt update
apt install -y gcc-11 g++-11

apt install -y pkg-config libhwloc-dev gcc-11 g++-11
```

Then rebuild:

```bash
cd /mnt/zhengcf3/ktransformers/kt-kernel

PATH=/root/miniconda3/envs/kt-kernel/bin:$PATH \
CC=/usr/bin/gcc-11 \
CXX=/usr/bin/g++-11 \
CUDAHOSTCXX=/usr/bin/g++-11 \
CPUINFER_FORCE_REBUILD=1 \
CPUINFER_PARALLEL=8 \
CMAKE_ARGS='-DNUMA_LIBRARY=/usr/lib/x86_64-linux-gnu/libnuma.so -DNUMA_INCLUDE_DIR=/usr/include' \
/root/miniconda3/envs/kt-kernel/bin/python -m pip install . -v --no-build-isolation
```


```
./install.sh deps
./install.sh build
```

build in nvcc

apt-get install -y libhwloc-dev
install binutils 2.45.1


# 设置环境变量覆盖自动检测
export CPUINFER_CPU_INSTRUCT=NATIVE
export CPUINFER_ENABLE_AMX=OFF
export CPUINFER_ENABLE_AVX512_VNNI=ON
export CPUINFER_ENABLE_AVX512_BF16=ON
export CPUINFER_ENABLE_AVX512_VBMI=OFF
export CPUINFER_USE_CUDA=0

# 执行安装脚本（自动检测模式，但会使用环境变量的值）
./install.sh build

install binutils

cd /tmp
wget https://ftp.gnu.org/gnu/binutils/binutils-2.45.1.tar.xz
tar -xf binutils-2.45.1.tar.xz
cd binutils-2.45.1


export PATH=/opt/binutils-2.45.1/bin:$PATH

export CC="gcc-13 -B/opt/binutils-2.45.1/bin"
export CXX="g++-13 -B/opt/binutils-2.45.1/bin"
export CUDAHOSTCXX="g++-13 -B/opt/binutils-2.45.1/bin"


mkdir build && cd build
../configure --prefix=/opt/binutils-2.45.1 --disable-werror
make -j"$(nproc)"
sudo make install


# 1) 进入环境
source /mnt/zhengcf3/lmp_env/fslmp/bin/activate
cd /mnt/zhengcf3/ktransformers/kt-kernel

# 2) 使用新 binutils（你如果已装到 /opt/binutils-2.45.1）
export PATH=/opt/binutils-2.45.1/bin:$PATH
hash -r

# 3) 强制 gcc/g++ 调用新 as（关键）
export CC="gcc-13 -B/opt/binutils-2.45.1/bin"
export CXX="g++-13 -B/opt/binutils-2.45.1/bin"
export CUDAHOSTCXX="g++-13 -B/opt/binutils-2.45.1/bin"

# 4) 清理环境污染
unset CMAKE_ARGS PYTHONHOME PYTHONPATH
export CPUINFER_USE_CUDA=0

# 5) 开启 VNNI/BF16（manual）
export CPUINFER_CPU_INSTRUCT=AVX512
export CPUINFER_ENABLE_AMX=OFF
export CPUINFER_ENABLE_AVX512_VNNI=ON
export CPUINFER_ENABLE_AVX512_BF16=ON
export CPUINFER_ENABLE_AVX512_VBMI=OFF

rm -rf build dist *.egg-info
./install.sh build --manual


which as
as --version
gcc-13 -print-prog-name=as