import UIKit
import MetalKit
import GameController

class EmulatorViewController: UIViewController {
    let mtkView: MTKView
    let metalLayer: CAMetalLayer
    lazy var emulator = Emulator(metalLayer: metalLayer, viewController: self)
    var virtualController: Any?

    init() {
        mtkView = .init()
        metalLayer = mtkView.layer as! CAMetalLayer
        super.init(nibName: nil, bundle: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(activeGameControllerWasChanged(notification:)), name: .GCControllerDidBecomeCurrent, object: nil)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func start(parent: UIViewController) {
        emulator.useJIT = Emulator.checkJITIsAvailable()
        if !emulator.useJIT {
            let alertController = UIAlertController(title: "Debugger is not attached", message: "You probably want to use JIT for stable fps and energy efficiency, but debugger is not attached.\n\nCurrently any integration about device-only JIT solution (like AltKit) is not introduced, so you need to manually launch citra_ios with AltStore's \"Enable JIT\" option or attach from Xcode or something else BEFORE launch game.\n\nBy the way, you can play games without JIT, but its may not run at full speed (especially OLD devices).", preferredStyle: .alert)
            alertController.addAction(.init(title: "Launch without JIT (SLOW), anyway", style: .destructive) { _ in
                parent.view.window?.rootViewController = self
            })
            alertController.addAction(.init(title: "Cancel", style: .cancel) { _ in

            })
            parent.present(alertController, animated: true)
            return
        }
        parent.view.window?.rootViewController = self
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
        if #available(iOS 15.0, *) {
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
                let virtualController = GCVirtualController(configuration: virtualControllerConfiguration)
                virtualController.connect { error in
                    print("Virtual Controller", error)
                }
                self.virtualController = virtualController
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
            gamepad.buttonMenu.valueChangedHandler = EmulatorInput.buttonStart.valueChangedHandler
            gamepad.leftThumbstick.valueChangedHandler = EmulatorInput.circlePad.valueChangedHandler
            gamepad.rightThumbstick.valueChangedHandler = EmulatorInput.circlePadPro.valueChangedHandler
        }
    }
}
