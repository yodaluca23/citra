#!/bin/sh -ex

brew update
brew unlink python@2 || true
rm '/usr/local/bin/2to3' || true
brew install qt5 p7zip ccache ninja wget || true
pip3 install macpack

export SDL_VER=2.0.16
export FFMPEG_VER=4.4
export VULKAN_SDK_VER=1.3.236.0

mkdir tmp
cd tmp/

# install SDL
wget https://github.com/SachinVin/ext-macos-bin/raw/main/sdl2/sdl-${SDL_VER}.7z
7z x sdl-${SDL_VER}.7z
cp -rv $(pwd)/sdl-${SDL_VER}/* /

# install FFMPEG
wget https://github.com/SachinVin/ext-macos-bin/raw/main/ffmpeg/ffmpeg-${FFMPEG_VER}.7z
7z x ffmpeg-${FFMPEG_VER}.7z
cp -rv $(pwd)/ffmpeg-${FFMPEG_VER}/* /

# install Vulkan SDK
wget https://sdk.lunarg.com/sdk/download/1.3.236.0/mac/vulkansdk-macos-${VULKAN_SDK_VER}.dmg
hdiutil attach vulkansdk-macos-${VULKAN_SDK_VER}.dmg
sudo /Volumes/vulkansdk-macos-${VULKAN_SDK_VER}/InstallVulkan.app/Contents/MacOS/InstallVulkan install --accept-licenses --confirm-command --default-answer com.lunarg.vulkan.core com.lunarg.vulkan.usr
hdiutil detach /Volumes/vulkansdk-macos-${VULKAN_SDK_VER}
