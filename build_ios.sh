#!/bin/bash

# you need to set some environment variables
# DEVELOPMENT_TEAM_ID: your development team id, you can check in https://developer.apple.com/account/ or somewhere in Xcode
# BUNDLE_IDENTIFIER: bundle identifier of Citra.app, you can set like `com.citra-emu.citra`
# MOLTENVK_IOS_ARTIFACT: MoltenVK's GitHub Actions Artifact. How to download (needs signed in with GitHub account):
# 1. https://github.com/KhronosGroup/MoltenVK/releases latest release
# 2. click commit hash 
# 3. click checkmark 
# 4. select "MoltenVK (Xcode xx.x - ios)"
# 5. select Summary
# 6. scroll down and find "Artifacts", then click "ios" package
# 7. unzip it
# 8. set path of "/path/to/Package/Release/MoltenVK" (that directory should includes "dylib", "MoltenVK.xcframework", "include")

set -xe
# if [ -d build/ios ]; then
#     rm -r build/ios
# fi
mkdir -p build/ios
cd build/ios
cmake ../.. \
    -DENABLE_SDL2=OFF -DENABLE_QT=OFF -DDISABLE_LIBUSB=ON \
    -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
    -DCMAKE_OSX_SYSROOT="$(xcrun --sdk iphoneos --show-sdk-path)" \
    -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM_ID" -DCMAKE_XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_IDENTIFIER" \
    -DMOLTENVK_IOS_ARTIFACT="$MOLTENVK_IOS_ARTIFACT" \
    -DMACOSX_BUNDLE_BUNDLE_NAME="citra_ios$CITRA_IOS_NAME_SUFFIX" -DCITRA_IOS_VERSION_STRING="$CITRA_IOS_SHORT_VERSION" \
    -DZSTD_BUILD_PROGRAMS=NO \
    -DENABLE_CUBEB=NO -DENABLE_WEB_SERVICE=NO \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="14.0" -DENABLE_LTO="$ENABLE_LTO" \
    -G Xcode
