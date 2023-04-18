import UIKit
import MetalKit
import GameController

class EmulatorViewController: UIViewController {
    let mtkView: MTKView
    let metalLayer: CAMetalLayer
    let emulator: Emulator
    var virtualController: GCVirtualController?

    init() {
        mtkView = .init()
        metalLayer = mtkView.layer as! CAMetalLayer
        emulator = .init(metalLayer: metalLayer)
        super.init(nibName: nil, bundle: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(activeGameControllerWasChanged(notification:)), name: .GCControllerDidBecomeCurrent, object: nil)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func loadView() {
        view = UIView()
        view.backgroundColor = .black
        view.addSubview(mtkView)
        mtkView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            mtkView.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor),
            mtkView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),
            mtkView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            mtkView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
        ])
    }

    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            let virtualControllerConfiguration = GCVirtualController.Configuration()
            virtualControllerConfiguration.elements = [
                GCInputButtonA,
                GCInputButtonB,
                GCInputButtonX,
                GCInputButtonY,
                GCInputDirectionPad,
                GCInputLeftShoulder,
                GCInputRightShoulder,
            ]
            self.virtualController = GCVirtualController(configuration: virtualControllerConfiguration)
            self.virtualController?.connect { error in
                print("Virtual Controller", error)
            }
        }
        emulator.start()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        emulator.layerWasResized()
    }

    @objc func activeGameControllerWasChanged(notification: Notification) {
        print("active game controller was changed")
        if let gamepad = GCController.current?.extendedGamepad {
            gamepad.buttonA.valueChangedHandler = EmulatorInput.buttonB.valueChangedHandler
            gamepad.buttonB.valueChangedHandler = EmulatorInput.buttonA.valueChangedHandler
            gamepad.buttonX.valueChangedHandler = EmulatorInput.buttonY.valueChangedHandler
            gamepad.buttonY.valueChangedHandler = EmulatorInput.buttonX.valueChangedHandler
            gamepad.leftShoulder.valueChangedHandler = EmulatorInput.buttonL.valueChangedHandler
            gamepad.rightShoulder.valueChangedHandler = EmulatorInput.buttonR.valueChangedHandler
            gamepad.dpad.up.valueChangedHandler = EmulatorInput.dpadUp.valueChangedHandler
            gamepad.dpad.down.valueChangedHandler = EmulatorInput.dpadDown.valueChangedHandler
            gamepad.dpad.left.valueChangedHandler = EmulatorInput.dpadLeft.valueChangedHandler
            gamepad.dpad.right.valueChangedHandler = EmulatorInput.dpadRight.valueChangedHandler
        }
    }
}
