add_executable(citra_ios
    AppDelegate.swift
    Bridge.h
    FileSelectorTableViewController.swift
    EmulatorViewController.swift
    Emulator.h
    Emulator.mm
    EmulatorInput.swift
    InputBridge.h
    InputBridge.mm
    TopViewController.swift
)

create_target_directory_groups(citra_ios)

target_link_libraries(citra_ios PRIVATE common core network)
target_link_libraries(citra_ios PRIVATE inih glad)
target_link_libraries(citra_ios PRIVATE ${PLATFORM_LIBRARIES} Threads::Threads)

set_target_properties(citra_ios PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST    "${CMAKE_CURRENT_SOURCE_DIR}/resources/Info.plist.in"
    MACOSX_BUNDLE_BUNDLE_VERSION    1
    MACOSX_BUNDLE_SHORT_VERSION_STRING  "0.1${CITRA_IOS_VERSION_STRING}"
    XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER  "${CMAKE_CURRENT_SOURCE_DIR}/Bridge.h"
    XCODE_EMBED_FRAMEWORKS  "${MOLTENVK_IOS_ARTIFACT}/dylib/iOS/libMoltenVK.dylib"
    XCODE_EMBED_FRAMEWORKS_CODE_SIGN_ON_COPY    YES
    MACOSX_RPATH    TRUE
    XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@executable_path/Frameworks"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY  "1,2"
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC   YES
)
