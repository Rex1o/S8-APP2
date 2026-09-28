# Compression d'images

Projet de compression et de traitement d'images développé en **C++20** avec **OpenCV**.

Le projet implémente plusieurs méthodes de quantification, de compression et de redimensionnement d'images.

# Dépendances

Le projet nécessite :

- **CMake 3.16 ou plus récent**
- Un compilateur compatible avec **C++20**
- **OpenCV**

Le projet utilise CMake pour localiser OpenCV :

```cmake id="2tkp5x"
find_package(OpenCV REQUIRED)
```

OpenCV doit donc être installé sur la machine avant de configurer le projet.

---

# Linux

## Installation des dépendances

### Ubuntu / Debian

Installer CMake, un compilateur C++ et OpenCV :

```bash id="u62rkm"
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

Vérifier l'installation de CMake :

```bash id="dfg07x"
cmake --version
```

Il est également possible de vérifier la version d'OpenCV détectée par `pkg-config` :

```bash id="w1zgrm"
pkg-config --modversion opencv4
```

## Compilation

À partir de la racine du projet :

```bash id="w12bxg"
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
```

L'exécutable devrait ensuite se trouver dans :

```text id="gwgq95"
build/release/image_compression
```

Pour lancer le programme :

```bash id="1dfd6a"
./build/release/image_compression
```

## Compilation Debug

Pour compiler en mode Debug :

```bash id="3xszb1"
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j
```

Puis :

```bash id="80l86p"
./build/debug/image_compression
```

---

# Windows

Sous Windows, il faut installer :

- CMake
- Un compilateur C++ compatible avec C++20
- OpenCV

L'utilisation de **Visual Studio 2022** avec le module de développement C++ est recommandée.

Lors de l'installation de Visual Studio, sélectionner la charge de travail :

```text id="vqln7x"
Desktop development with C++
```

## Option 1 — Installation avec vcpkg

L'utilisation de `vcpkg` est une façon relativement simple d'installer OpenCV et de permettre à CMake de le trouver automatiquement.

### 1. Installer vcpkg

Dans PowerShell :

```powershell id="t67ujg"
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

### 2. Installer OpenCV

```powershell id="sv6yrh"
.\vcpkg install opencv4:x64-windows
```

La compilation et l'installation d'OpenCV peuvent prendre plusieurs minutes.

### 3. Configurer le projet

À partir du répertoire du projet :

```powershell id="wffifz"
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:/chemin/vers/vcpkg/scripts/buildsystems/vcpkg.cmake
```

Remplacer :

```text id="85d0c1"
C:/chemin/vers/vcpkg
```

par le chemin réel vers l'installation de `vcpkg`.

Par exemple :

```powershell id="a8fl8h"
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### 4. Compiler

```powershell id="4n8abp"
cmake --build build --config Release
```

L'exécutable devrait se trouver dans :

```text id="dxj43m"
build/Release/image_compression.exe
```

Pour lancer le programme :

```powershell id="db7ht4"
.\build\Release\image_compression.exe
```
