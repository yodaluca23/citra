import UIKit

@main
class AppDelegate: NSObject, UIApplicationDelegate {
    var window: UIWindow?
    var alerts: [(title: String, description: String)] = []

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil) -> Bool {
        let window = UIWindow()
        self.window = window
        let documentDir = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        try? FileManager.default.createDirectory(at: documentDir.appendingPathComponent("Citra"), withIntermediateDirectories: true)

        let vc = UITabBarController()
        let fileSelectorVC = UINavigationController(rootViewController: FileSelectorTableViewController(at: documentDir))
        fileSelectorVC.tabBarItem.image = UIImage(systemName: "folder")
        fileSelectorVC.tabBarItem.title = "Files"
        let topVC = UINavigationController(rootViewController: TopViewController())
        topVC.tabBarItem.image = UIImage(systemName: "square.grid.3x3.fill")
        topVC.tabBarItem.title = "Installed Titles"
        vc.setViewControllers([
            fileSelectorVC,
            topVC,
        ], animated: false)

        window.rootViewController = vc

        window.makeKeyAndVisible()

        if #unavailable(iOS 15.0) {
            alerts.append((title: "Warning to old iOS users", description: "because GCVirtualController requires iOS 15.0+, currently theres no virtual controller about iOS 14.x users. which means you need to use physical controller (which supported by your version of iOS) to actual play."))
        }
        if let device = MTLCreateSystemDefaultDevice(), !device.supportsFamily(.apple5) {
            alerts.append((title: "Citra doesn't work with A11 or older device", description: "Citra requires Metal's Apple5 Feature Set (equals to A12, iPhone XS/XR) or later to use layered rendering.\n\nbut your device's GPU (\(device.name)) is not supporting Apple5 Feature Set.\n\nRun Citra anyway causes to crash when starting game.\n\nPLEASE DON'T REPORT ISSUE ABOUT THIS unless you are using A12 or later chip but still getting this alert."))
        }
        showAlertIfNeeded()

        return false // NO if the app cannot handle the URL resource or continue a user activity
    }

    func showAlertIfNeeded() {
        if let alert = alerts.popLast() {
            let alertController = UIAlertController(title: alert.title, message: alert.description, preferredStyle: .alert)
            alertController.addAction(.init(title: "OK", style: .default) { _ in
                self.showAlertIfNeeded()
            })
            window?.rootViewController?.present(alertController, animated: true)
        }
    }
}
