#www.virustotal.com/vtapi/v2/
import time

import requests
import json
import pickle
import csv

def getFileScanId(url,apikey,a,b):
    # /file/scan
    # /文件/扫描
    # 上传并扫描文件
    # 限制为32MB
    # params = {'apikey': apikey}
    # files = {'file': (a, open(b, 'rb'))}
    # response = requests.post(url, files=files, params=params)
    # my_scan_id = str(response.json()['scan_id'])
    # return my_scan_id
    try:
        # 尝试打开文件
        files = {'file': (a, open(b, 'rb'))}
        response = requests.post(url, files=files, params={'apikey': apikey})
        my_scan_id = str(response.json()['scan_id'])
        return my_scan_id
    except Exception as e:
        # 如果发生异常，打印错误信息并返回None
        print(f"Failed to upload {a}: {e}")
        return None

def getFieReportResult(url,apikey,my_scan_id):
    try:
        #/file/report
        # /文件/报告
        # 检索文件扫描报告
        #该resource参数可以是要获取最新的病毒报告文件的MD5，SHA-1或SHA-256。
        #还可以指定/ file / scan端点scan_id返回的值。
        #如果allinfo参数设置为true除了返回防病毒结果之外的其他信息。
        get_params = {'apikey': apikey, 'resource': my_scan_id,'allinfo': '1'}
        response2 = requests.get(url, params=get_params)
        jsondata = json.loads(response2.text)
        with open("jsonResult.json","w") as f:
            json.dump(jsondata, f, indent=4)
        return jsondata
    except Exception as e:
        print(e)
        return None

def getResult(json):
    result = {}
    for k,v in json["scans"].items():
        result[k] = v['detected']
    # print(result)
    print("一共有{0}条杀毒数据".format(len(result)))
    # with open("result.txt","w") as g:
    #     g.write(str(result))
    return result

def detect_one_file(file_name, file_src):
    a = str(file_name)

    b = str(file_src)

    url1 = 'https://www.virustotal.com/vtapi/v2/file/scan'
    url2 = "https://www.virustotal.com/vtapi/v2/file/report"
    #需要提供密钥，否者会出现403错误
    apikey = "1bfe1bb5ee6959affea8853f6c9efe0d72c7cffe13fd8e00307e30294d52bf07"

    # #获得文件scan_id
    # scan_id = getFileScanId(url1,apikey,a,b)
    # #获得返回的json结果并写入result文件
    # #getFieReportResult(url2, apikey, scan_id)
    # json = getFieReportResult(url2,apikey,scan_id)
    # if json == None:
    #     return None
    # res = getResult(json)
    # return res
    scan_id = getFileScanId(url1, apikey, a, b)
    json = None
    attempts = 0
    while attempts < 30 and json is None:
        json = getFieReportResult(url2, apikey, scan_id)
        if 'scans' not in json:
            print("Waiting for scan results to be ready...")
            time.sleep(30)  # Wait for a minute before retrying
            attempts += 1
        else:
            break

    if json is None or 'scans' not in json:
        print("Failed to retrieve scan results after several attempts.")
        return None

    return getResult(json)

def save_results(results, file_path='intermediate_results.pkl'):
    """将当前的结果字典保存到Pickle文件"""
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)


def process_files(file_paths, apikey, max_attempts=30):
    url_scan = 'https://www.virustotal.com/vtapi/v2/file/scan'
    url_report = "https://www.virustotal.com/vtapi/v2/file/report"

    scan_ids = {}
    results = {}
    completed = set()
    attempt_counts = {}

    # 批量上传所有文件并收集scan_id
    for file_path in file_paths:
        filename = file_path.split('/')[-1]
        scan_id = getFileScanId(url_scan, apikey, filename, file_path)
        print(f"Uploaded {filename} with scan_id {scan_id}")
        if scan_id:
            print(f"Uploaded {filename} with scan_id {scan_id}")
            scan_ids[filename] = scan_id
            results[filename] = None
            attempt_counts[filename] = 0
        else:
            print(f"Skipping {filename} due to upload failure.")
        time.sleep(30)  # 每上传一个文件休息30秒

    # 轮询获取报告结果
    while len(completed) < len(scan_ids):
        for filename, scan_id in scan_ids.items():
            if filename not in completed and attempt_counts[filename] < max_attempts:
                result = getFieReportResult(url_report, apikey, scan_id)
                if result and 'scans' in result:
                    results[filename] = result['scans']
                    completed.add(filename)
                attempt_counts[filename] += 1
                save_results(results)  # 每次结果更新时保存当前状态
                time.sleep(30)  # 每次查询间隔30秒
            elif attempt_counts[filename] >= max_attempts:
                print(f"Max attempts reached for {filename}, stopping attempts to retrieve report.")
                completed.add(filename)  # 停止尝试这个文件

    return results

def write_results_to_csv(results, filename="virustotal_results.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Detected Count'])
        for file_name, data in results.items():
            if data:
                detected_count = sum(1 for _, scan in data.items() if scan['detected'])
                writer.writerow([file_name, detected_count])
            else:
                writer.writerow([file_name, 'Failed to get report'])



if __name__ == '__main__':
    all_file_path = ['/data/c/shiwensong/dataset/virusshare_2018/VirusShare_810bd45dc23b9694b1deb98b3b620ec4.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_55d005fd4286947a74e258dfc8bd25e3.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_58bde35ee25f91c57ed7d3509603d824.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_37ca1c7a8f75b0cc52b93fbeb0e63026.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_4a5a18fde72e37661ec77217a47b25cd.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_b7bec341529aba4f3d0f49388e971926.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_daee372ad1a6dbb14545a3e95c6f5863.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_868287378a201d133a3f63e3cd4f4422.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_848446728cd75e1c5aa78f3208e9f843.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_dde01580010e8a65049b461f24ae48e9.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_dabec1761e85d22fe574040fb36ccd1e.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_dad40b7991d04253aab542cecf1e6d23.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_57f49d9b916d158f7ce61ac338736583.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_731c74f860bae10b06f5f0ca0b790af5.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_18352a1ee12ef5ff3aa7c940ec78e432.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_e1f0623ed3ef35bfe85e7df0c70b3542.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_5b1b326ab001186e38230594bcedcfa0.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_667beae610f1a7295d0ec2f2c69dd8c3.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_b2858a351457b5908ebcb454e651e420.apk',
                     '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_b4a54e733d52ed1893b81ae698596d29.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_508f0afea269e35f490d552a66ec2e14.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_8056be6fc9c0b8375602f36e16ec868b.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_840e10e732a68dbbb31c758eafe508e5.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_6116b6c2fdc534ad81955c54790489db.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_611d1913a9ea18e0ed9c473e5fa779d1.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_5039a1ef2b1babd84b4ba8d71912013b.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_603e00a1fb5cb50dae82248c9f810907.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_6989a9b5928e3a042ec0d1ed555d00c6.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_130bca6ceae31f0c0d7b381721076bac.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_758b10e4c290948227bae10d8e38f680.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_834d8bbe3d7c48dd0ed7fe652ea4741f.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_5364a823624186fa1d25b63c17c0dd82.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_6372ad9ef6a5230bcec2a110d27736bd.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_581ac0fad87ee029389dd4f53d111e12.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_0414c23228421cad41f84fb24ea0e83d.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_4643a2c734518f66ec4df4647e5454b3.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_3189d56bb74e6b59df90eb6cec493c98.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_756f766ff6946658aa204d648a9e1a63.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_721d553d6ad0741d1a9c6e29a1b27241.apk',
                     '/data/b/shiwensong/dataset/virusshare2019/VirusShare_93ecd966dd2a73d92f5d8e7b57d9aed2.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/24967a280568b39f66b188170c3d96209894dbc8c7dd9b893b80ceec37150e6c.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/e6e898c9fed6fdb16e5126743ee82f4d01963b02e80c996aa4c8d06dc9b93d5c.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/916b57831fad621c4e9c45f46abfd54667a8f3d66dc368dbb50c33eca76eb6f1.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/20c4060bc387473af6bb35578698988d14e8f16f8b02fb5fa0214b02a62d1d9c.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/01411844d63ecc8ef3a03069d314ee6f3d4064e7a40cfa9afa90ea74e3e96182.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/00a11192e6c2286c61bbaf9e8206de75d02a100f6ed7e410c1654fe19c07789b.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/cd237dfaff49d19fa6bfebb1b7a0842af73cc634b080afc4fc76277e1228e399.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/27e3dbaf27bcd2e313b9239897ef1af24a8472701fcbcc0a65f0ae473db9deed.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/186fd20f396a06a881da2e2b8c7b397c7a6bc1a6df232c4d5249c0d575a862a3.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/47844433ee4c2fbd4fd0dc0880bfbb23cfee7b5baa21146b172d53a63cbe4be0.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/504e7d85f6128d40beabc873eedb957eac9ed2ad0d5c67780125e97e4c959af4.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/31d352de8f8ee6e856f12e6b31a6929a1e5b0e782783d6bffe410cfd15389134.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/0e9b08aae475925e3f2c3aab950606f6dce986234bf5bd029a8144b157e2ab0d.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/1fd9e593276301ebb72cfb9c1bb509bb6791ea2299a4f8ea2536d8a7db7645b7.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/083e13abb82b795e521b02c825bde2c4e5fc5c078a04da04ad32826ba99e9361.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/2e68d4c9993030ab181631c851a21d565b8819ea76c2995b2954072ba3693b90.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/1c328f4e8823573fd040ccaa3670c416f991ffa88df68075625e7f89182d7374.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/3353399805c5805e2afdabf2f45ecfe33de4bf832f1965244a8917a4976ca965.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/4c7f60fac63a1d4e8d6569f3c933fcdbc69aff1d2db3afb116871f5eebd262d6.apk',
                     '/data/b/shiwensong/dataset/virusshare2020/0202507ce80ef8460650b42148b4c63c64dcf11df5de50874c6108bf495e930d.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/b4c7e0be7b01a784bf4f7208ed15e097a33e47f650775185488be534acf1549a.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/326589ee73391408f9b535b1a657554a504b7c48526267beedeea4a1bf106643.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/458d2174eb686a3fc3a349c09d72e3f9908624dde60a1296d862e2ea9d2f7d8f.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/0084834231010d1e184efe61fca89a3633f6e4493f7057a3427773bfd032cd40.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/37edb065edc41f5595265a6c3b9b2be69a7cbb0a5c4ba21ca7683b976a7cc070.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/2077aed3e3ed1b0fb9a7639537e844f00f17a4c06909a9d813f98889db2a103a.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/26b5a91b99173069aeabc03ec79bd7f8556103b1db75d9ca28900ba88223920c.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/1a7fd5886ca2725436e7fd8cf3aa495d1aafe1357449cb889c8d6b0640537dc5.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/a4a38f521588325125b254cb629fc05537a43bca74985a26cf407403ae428331.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/8e7f37c423430234fc83ce3d38a53c2444ed1cf06c2e7ff87a7c8e424165f509.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/4e90d3d2592c1b5ac447b23869e348105e491d90153a3c8fe249e98913777281.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/01c9424509a1a13d8079a8075511994da70f7ce6b74745f3f28b7400e48f34f4.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/eaa87b55496e16e16f9c0c6a4d4a8f9143a01b2d8a9fde0e8ca5070e43cfd242.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/89a6666f7836129e27add5d49eccb5deb224ac1cab5bc5a25a32bcb6411097dd.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/330356af2c27d773214aae12aa8c3df99a8d12564d04617406a4e6ea6e881e9e.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/33bfdaabce74b0418f0bc0a7ec03d2086c2ad83243a2f425a202110bdc7db251.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/46c2619d70e2de0884329cbbce88cc6bdc6255cfd7aaabd0cb5486d20229fe02.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/256535ffc9735292e56b44e9dc1a9fbed4ae8e8107dd761546f4015765ea838d.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/ad5a23554bca06436291fb727b0cd3b0751e1b82ee0feb04a7572c4e63e93f87.apk',
                     '/data/c/shiwensong/dataset/virusshare_2021/00891d288701241f12b43b21003bbe58414af0b18a03dc290d5df0e2c8aff55d.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/864d6667ce8ed56ef81d96e4908c87ed5d9dc0acf728f36d6ef6e98a30995e88.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/703ac7e5672a1f53c02bab2614361ffee92e67845ba813bc93da788c6fc141fa.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/b57ef999b106fcda3e9685bd9c963bd91b4fe93c0541d319a61d5ea9dfed9e03.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/9a701b5c58d3aec46863be8e4bf411e1a4a143dd9ffd6062c608e2caf3a7a269.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/a4705775a6bd946c278e58d5026529371fc8e2d29dffc74c29b086adc73bdaa4.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/843697bed377cf7c566de73a982be13fe53917a4661719b25b63103b7b9a0ca5.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/a89f2fb842fa4dddf1d581c057c9aede5fbd1f84e32ff853f3d12128bb48fb50.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/a09a278ce764911a79f7c548b182796c1b09a211b528fbef021101f7e2b37ba2.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/7e4f9eb10a21709f7866f4d94da45eeb3e0b9e4ecbc22087a19397d3f65d6070.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/b72b1d870ceb2aee805fb6ee066b6dc210a11b79debdc26404f84b0b5ed2acde.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/937870b7a6b931d5d3a2348f915c5545f5b425b5c6ce2340ef9bee69f0740247.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/9ee016506b865e4eef975815ddfb381f404642399ef4b9e6f8d0334df849e8b2.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/b8e24021bec54836921e26a7a8cf14941e26a9dae59d0cfab75fe46dcd557521.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/9ffe4e83bcf69577023061e6e122bfb604b45c66c43de4c9903a0a1c4d652029.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/a3b4f1a9799427cb715764ceccc0f429e148a3dce3f05d50bca0592bfcf233d7.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/b79b053fabbb99fd749da89672b64c4ab645ade3f6fdd22385f74b4ad9b92fda.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/85807b11a480cff3e2a597bfc25f997ee23098cdca2823723d76f48302d26ed9.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/b561a776288e97301af234848947a8365c794063999d8aae55e2ce79986cb8cd.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/a2f560a36711d1ad153dcd436a6dd3323902e798033d6d3b07cf52f36dd5e850.apk',
                     '/data/c/shiwensong/dataset/virusshare_2022_2/9ed5a9c986e25f2663185941bcd7c47e085838cb97f12cc08300abb4a241a39b.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/c4801b19c4adcd8c2e41e8677b1f1064535f52e37dd16874b635d06c82b07e60.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/c535ac41a2d3a086b6889ce1338af7e704417d5f309cea2b77deaf29fe3f0318.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/df72141b7c2fe5935fbb0c0784ea27198c108af4d2f52e835b1fe484e16170cc.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/d7e38806cf027102d5601b3c2b4c63e3fe150dd0181fcc954682c42c39aed1be.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/e16ff85f135fce41d562f2dc76e9f2101a509f14744a2b20e6f42ac83e26d417.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/e4ba2cb7f5b9ab40cf7ed44a424249572ced764b4b4640b91a8de8a95cb71c70.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/e1b77076c312c17885d53835d36426fa9d677ea631189b1ef80b9baeeca5f7d0.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/c17fcf308a3ccb12aa51da86890af8df1573a4cfde5d4e6266d0bbfda2922b96.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/c2862e57db8214670d36d8f1ab2088bbb946c0e5b6228290a864321bcab044a6.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/c3026e1a5d22230243d50031bccfcfba26a70268b9354bd85b9321cd938c3e79.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/edf37c541c83463413c8ffb45cd00aefeba0b8a2bcabce2198f7b3dbe1a273ce.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/f082f308acb55e76637bf08709a44d9542b8820876f0c2e4fb09b3c8cc7892cc.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/d5f252f426a25052aa82d613436037f282a7d43e1b614e19a847141cea4414a1.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/b9ca729413b15330cb05321067376bbfa3b2e8decf6a3fe5698356d86a596f03.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/f3a5d605d3fefc29e6b9a83aa4482d6af23d6dda1171ca01457547a282f4f1e6.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/d84ffbab7faa8aa63c8f98c52b66dffbd052efc72553b65b8073d30df1f3f39b.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/f6a4e8d37edae1e302ca640702f6254b0816d70e5eabf58e69da82ff75366b56.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/cd4a598ca4abfdd5fa9aca35135ed11e7eddf7048307cf6746834e4efb1ec8b9.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/ed555516e79d3e0a5f4898e5fb08ec43c92287b906349ef46c319b0eea1a4ef2.apk',
                     '/data/c/shiwensong/dataset/virusshare_2023_3/c0893dbacd2af9946aafe35a011f1ba9fbc7c86cc5a34d5ed3605d55ae80642e.apk']
    apikey = "1bfe1bb5ee6959affea8853f6c9efe0d72c7cffe13fd8e00307e30294d52bf07"
    results = process_files(all_file_path, apikey)
    write_results_to_csv(results)

    # with open('virustotal_results.csv', 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     # Write headers
    #     writer.writerow(['Filename', 'Detected Count'])
    #     # all_file_path = ['/data/c/shiwensong/dataset/virusshare_2018/VirusShare_810bd45dc23b9694b1deb98b3b620ec4.apk', '/data/c/shiwensong/dataset/virusshare_2018/VirusShare_55d005fd4286947a74e258dfc8bd25e3.apk']
    #     all_numbers = []
    #     all_pkl = {}
    #     for file_src in all_file_path:
    #         file_name = file_src.split('/')[-1]
    #         res = detect_one_file(file_name, file_src)
    #         # print(res)
    #         if res is None:
    #             all_numbers.append([file_name, None])
    #         else:
    #             all_numbers.append([file_name, len(res)])
    #         all_pkl[file_name] = res
    #         time.sleep(40)
    #
    #         # Debug print to check what's being written
    #     # print(f"Data to be written to CSV: {all_numbers}")
    #
    #     writer.writerows(all_numbers)
    #
    # # csv_file.close()
    # with open('virustotal_results.pkl', 'wb') as f:
    #     pickle.dump(all_pkl, f)
    
    
    # file_name = input("请输入文件名:")
    # file_src  = input("请输入文件路径:")
    # res = detect_one_file(file_name, file_src)
    # print(res)