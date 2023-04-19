import UIKit

@main
class AppDelegate: NSObject, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil) -> Bool {
        let window = UIWindow()
        self.window = window
        let documentDir = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        try? FileManager.default.createDirectory(at: documentDir.appendingPathComponent("Citra"), withIntermediateDirectories: true)
        window.rootViewController = UINavigationController(rootViewController: FileSelectorTableViewController(
            at: documentDir
        ))

        window.makeKeyAndVisible()

        return false // NO if the app cannot handle the URL resource or continue a user activity
    }
}
