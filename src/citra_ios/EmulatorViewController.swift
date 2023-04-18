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
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func loadView() {
        view = UIView()
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
        emulator.start()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        emulator.layerWasResized()
    }
}
