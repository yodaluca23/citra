import UIKit

@main
class AppDelegate: NSObject, UIApplicationDelegate {
    var window: UIWindow?

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
            let alertController = UIAlertController(title: "Warning to old iOS users", message: "because GCVirtualController requires iOS 15.0+, currently theres no virtual controller about iOS 14.x users. which means you need to use physical controller (which supported by your version of iOS) to actual play.", preferredStyle: .alert)
            alertController.addAction(.init(title: "OK", style: .default))
            window.rootViewController?.present(alertController, animated: true)
        }

        return false // NO if the app cannot handle the URL resource or continue a user activity
    }
}
