import UIKit
import MetalKit

class EmulatorViewController: UIViewController {
    let mtkView: MTKView
    let metalLayer: CAMetalLayer
    let emulator: Emulator

    init() {
        mtkView = .init()
        metalLayer = mtkView.layer as! CAMetalLayer
        emulator = .init(metalLayer: metalLayer)
        super.init(nibName: nil, bundle: nil)
        view = mtkView
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        emulator.start()
    }
}
