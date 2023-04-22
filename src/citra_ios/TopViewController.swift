import UIKit

class TopViewController: UITableViewController {
    lazy var dataSource = UITableViewDiffableDataSource<Section, URL>(tableView: tableView) { tableView, indexPath, itemIdentifier in
        let cell = UITableViewCell(style: .subtitle, reuseIdentifier: nil)
        if let smdh = Emulator.getSMDH(itemIdentifier) {
            let regionPrefix = (1 * (0x180 + 0x80)) + 8
            cell.textLabel?.text = String(bytes: smdh[smdh.startIndex+regionPrefix+0x80..<smdh.startIndex+regionPrefix+0x80+0x100], encoding: .utf16LittleEndian)
            let vendorString = (String(bytes: smdh[smdh.startIndex+regionPrefix+0x180..<smdh.startIndex+regionPrefix+0x180+0x80], encoding: .utf16LittleEndian) ?? "")
            if #available(iOS 16.0, *), let documentsURL = try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
                cell.detailTextLabel?.text = vendorString.replacingOccurrences(of: "\0", with: "") + " | " +  (itemIdentifier.path.replacing(/0{9,}/, with: "000...000").replacingOccurrences(of: "/private/", with: "/").replacingOccurrences(of: documentsURL.path, with: ""))
            } else {
                cell.detailTextLabel?.text = vendorString
            }
            print(cell.detailTextLabel?.text)
            cell.imageView?.image = Emulator.getIcon(smdh, large: true)
        }
        return cell
    }

    enum Section {
        case installedTitles
        case installedNANDTitles
    }

    override func viewDidLoad() {
        reload()
    }

    func searchDir(titleDir: URL, ds: inout NSDiffableDataSourceSnapshot<Section, URL>) throws {
        if let enumerator = FileManager.default.enumerator(at: titleDir, includingPropertiesForKeys: [.isRegularFileKey]) {
            var lastUpdatedDate = Date()
            for case let url as URL in enumerator where try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile! {
                let filename = url.lastPathComponent.lowercased()
                if filename.hasSuffix(".app"), let smdh = Emulator.getSMDH(url) {
                    ds.appendItems([url])
                    if lastUpdatedDate.timeIntervalSinceNow < -0.1 {
                        dataSource.apply(ds, animatingDifferences: true)
                        lastUpdatedDate = .init()
                    }
                }
            }
            dataSource.apply(ds, animatingDifferences: true)
        }
    }

    func reload() {
        tableView.dataSource = dataSource
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let documentDir = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
                var ds = NSDiffableDataSourceSnapshot<Section, URL>()
                ds.appendSections([.installedTitles])
                try self?.searchDir(titleDir: documentDir.appendingPathComponent("Citra/sdmc/Nintendo 3DS/00000000000000000000000000000000/00000000000000000000000000000000/title/00040000"), ds: &ds)
                ds.appendSections([.installedNANDTitles])
                try self?.searchDir(titleDir: documentDir.appendingPathComponent("Citra/nand/00000000000000000000000000000000/title"), ds: &ds)
            } catch {
                print(error)
            }
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
