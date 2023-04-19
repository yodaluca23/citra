import Foundation

@objc class EmulatorInput: NSObject {
    @objc static let buttonA = ButtonInputBridge()
    @objc static let buttonB = ButtonInputBridge()
    @objc static let buttonX = ButtonInputBridge()
    @objc static let buttonY = ButtonInputBridge()
    @objc static let buttonL = ButtonInputBridge()
    @objc static let buttonR = ButtonInputBridge()
    @objc static let buttonZL = ButtonInputBridge()
    @objc static let buttonZR = ButtonInputBridge()
    @objc static let buttonStart = ButtonInputBridge()
    @objc static let buttonSelect = ButtonInputBridge()
    @objc static let dpadUp = ButtonInputBridge()
    @objc static let dpadDown = ButtonInputBridge()
    @objc static let dpadLeft = ButtonInputBridge()
    @objc static let dpadRight = ButtonInputBridge()
    @objc static let circlePad = StickInputBridge()
    @objc static let circlePadPro = StickInputBridge()
    @objc static let _buttonDummy = ButtonInputBridge()
}
