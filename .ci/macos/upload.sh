#!/bin/bash -ex

. .ci/common/pre-upload.sh

REV_NAME="citra-osx-${GITDATE}-${GITREV}"
ARCHIVE_NAME="${REV_NAME}.tar.gz"
COMPRESSION_FLAGS="-czvf"

mkdir "$REV_NAME"

cp build/bin/Release/citra "$REV_NAME"
cp -r build/bin/Release/citra-qt.app "$REV_NAME"
cp build/bin/Release/citra-room "$REV_NAME"

BUNDLE_PATH="$REV_NAME/citra-qt.app"
BUNDLE_CONTENTS_PATH="$BUNDLE_PATH/Contents"
BUNDLE_EXECUTABLE_PATH="$BUNDLE_CONTENTS_PATH/MacOS/citra-qt"
BUNDLE_FRAMEWORKS_PATH="$BUNDLE_CONTENTS_PATH/Frameworks"
BUNDLE_RESOURCES_PATH="$BUNDLE_CONTENTS_PATH/Resources"

CITRA_STANDALONE_PATH="$REV_NAME/citra"

# move libs into folder for deployment
macpack $BUNDLE_EXECUTABLE_PATH -d "../Frameworks"
# move qt frameworks into app bundle for deployment
macdeployqt $BUNDLE_PATH -executable=$BUNDLE_EXECUTABLE_PATH

# move libs into folder for deployment
macpack $CITRA_STANDALONE_PATH -d "libs"

# bundle MoltenVK
VULKAN_SDK_PATH=~/VulkanSDK/*/macOS
cp $VULKAN_SDK_PATH/lib/libvulkan.dylib $BUNDLE_FRAMEWORKS_PATH
cp $VULKAN_SDK_PATH/lib/libMoltenVK.dylib $BUNDLE_FRAMEWORKS_PATH
cp $VULKAN_SDK_PATH/lib/libVkLayer_*.dylib $BUNDLE_FRAMEWORKS_PATH
cp -r $VULKAN_SDK_PATH/share/vulkan $BUNDLE_RESOURCES_PATH
find $BUNDLE_RESOURCES_PATH/vulkan -type f -name "*.json" -exec sed -i'' -e 's/..\/..\/..\/lib/..\/..\/..\/Frameworks/g' {} \;
install_name_tool -add_rpath "@loader_path/../Frameworks/" $BUNDLE_EXECUTABLE_PATH

# workaround for libc++
install_name_tool -change @loader_path/../Frameworks/libc++.1.0.dylib /usr/lib/libc++.1.dylib $BUNDLE_EXECUTABLE_PATH
install_name_tool -change @loader_path/libs/libc++.1.0.dylib /usr/lib/libc++.1.dylib $CITRA_STANDALONE_PATH

# Make the launching script executable
chmod +x $BUNDLE_EXECUTABLE_PATH

# Verify loader instructions
find "$REV_NAME" -type f -exec otool -L {} \;

. .ci/common/post-upload.sh
