import UIKit

class TopViewController: UITableViewController {
    lazy var dataSource = UITableViewDiffableDataSource<Section, URL>(tableView: tableView) { tableView, indexPath, itemIdentifier in
        let cell = UITableViewCell(style: .subtitle, reuseIdentifier: nil)
        if let smdh = Emulator.getSMDH(itemIdentifier) {
            let regionPrefix = (1 * (0x180 + 0x80)) + 8
            cell.textLabel?.text = String(bytes: smdh[smdh.startIndex+regionPrefix+0x80..<smdh.startIndex+regionPrefix+0x80+0x100], encoding: .utf16LittleEndian)
            cell.detailTextLabel?.text = String(bytes: smdh[smdh.startIndex+regionPrefix+0x180..<smdh.startIndex+regionPrefix+0x180+0x80], encoding: .utf16LittleEndian)
            cell.imageView?.image = Emulator.getIcon(smdh, large: true)
        }
        return cell
    }

    enum Section {
        case installedTitles
    }

    override func viewDidLoad() {
        reload()
    }

    func reload() {
        do {
            let documentDir = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            var ds = NSDiffableDataSourceSnapshot<Section, URL>()
            ds.appendSections([.installedTitles])
            let titleDir = documentDir.appendingPathComponent("Citra/sdmc/Nintendo 3DS/00000000000000000000000000000000/00000000000000000000000000000000/title/00040000")
            if let enumerator = FileManager.default.enumerator(at: titleDir, includingPropertiesForKeys: [.isRegularFileKey]) {
                for case let url as URL in enumerator where try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile! {
                    let filename = url.lastPathComponent.lowercased()
                    if filename.hasSuffix(".app"), let smdh = Emulator.getSMDH(url) {
                        ds.appendItems([url])
                    }
                }
            }
            dataSource.apply(ds, animatingDifferences: true)
        } catch {
            print(error)
        }
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        guard let url = dataSource.itemIdentifier(for: indexPath) else {
            return
        }
        let emulatorVC = EmulatorViewController()
        emulatorVC.emulator.executableURL = url
        emulatorVC.start(parent: self)
    }
}
