import glob
import sys
import os
sys.path.append(os.path.abspath('../type'))
sys.path.append(os.path.abspath('task'))
from FCG import FCG
import pickle
import networkx as nx
import copy
import numpy as np
import time
from MutateFCG import load_MLP_model
import pandas as pd

def find_shap_value(shap_values, sample_name):
    name = [
        'VirusShare_97bc5adf5df9106efb885b78855c4838.gexf',  # 30.765587
        'VirusShare_e7ca640611fa2f8c630961199e13f6b5.gexf',  # 2.9279337
        'VirusShare_8e2b629a10625956f7609f2d939bcac4.gexf',  # 280.80453
        'VirusShare_63acec04855ac0c5641247f5ba3d48b9.gexf',  # 3.4467793
        'VirusShare_6ceede843c5dc4ca02509b35a3f40b28.gexf',  # 13.807023
        'VirusShare_ae165056c14a2cba5466cd69a28fc431.gexf',  # 6.2594376
        'VirusShare_86539705c6eb59c6acaf55e580be653a.gexf',  # 54.0898
        'VirusShare_b157472d61af978bee9d2c3b26df1e83.gexf',  # 15.3836565
        'VirusShare_5381c76ce28d84d3245efc4a19238d58.gexf',  # 6.5762854
        'VirusShare_8f7aff5ec7c3bb14331dfa3e981a0b73.gexf',  # 6.858757
        'VirusShare_6a3d9dfe6587141ace52c54d02e67e39.gexf',  # 6.5762854
        'VirusShare_6963b7ca41268cfa7470a3e8ad8e9766.gexf',  # 6.62138
        'VirusShare_41cec261cacf2f4bac3168740594361b.gexf',  # 0.7636745
        'VirusShare_d1df2a91bbe0594c1061ead71649ae09.gexf',  # 168.00703
        'VirusShare_74b1162820ca4095d8e911207a8a729a.gexf',  # 5.7805347
        'VirusShare_527c66dec303042a556a2349f29999dc.gexf',  # 6.621313
        'VirusShare_45e67a88d73488396eae77a7309d90e7.gexf',  # 6.621313
        'VirusShare_28a2725940ecb8b9686476cfa0dda209.gexf',  # 157.92831
        'VirusShare_c508d918a080b807fd23b51c350f86a5.gexf',  # 239.70909
        'VirusShare_9b0d37e5cb949ab86354788ce5488375.gexf',  # 69.2645
        'VirusShare_31e8119d2d0b14556eca26f06b679244.gexf',  # 6.5762854
        'VirusShare_32e2a5033829d8940bc50d5fb42f9785.gexf',  # 15.447115
        'VirusShare_d9263748ae4c6f2ec244351e054be5f4.gexf',  # 5.34118
        'VirusShare_250520d860f63afdf8c94affd0921253.gexf',  # 38.117565
        'VirusShare_0b2d190d17d50dfd4a589aca1a9caa49.gexf',  # 434.88388
        'VirusShare_fb8640f452beb0403462f3a036182ed7.gexf',  # 6.890682
        'VirusShare_f4db6fc9770ad3e0310f3e2a13b5cb8b.gexf',  # 6.5762854
        'VirusShare_6a9b2fac202d0ab867f89cd79c17f5c5.gexf',  # 6.5093713
        'VirusShare_17113d01df79f103c07f68b4ebfd51d3.gexf',  # 3.9143429
        'VirusShare_e3ae70d5e1f209bf4dc30ba013a165f2.gexf',  # 22.537872
        'VirusShare_0e78970a2de95e2d546c74cf9a134e75.gexf',  # 6.9511476
        'VirusShare_a0417e92830ad60ca9f58120a9af2f14.gexf',  # 5.000574
        'VirusShare_d985eeb52e4407c39b53e7426a8bc2e3.gexf',  # 142.4429
        'VirusShare_8a3d12947d7d98dac2e89b51f623bbf0.gexf',  # 434.88153
        'VirusShare_3b951e9452c02817a602753463958d67.gexf',  # 6.409724
        'VirusShare_15957ed1ff6bf19f2fa4c709409ebd70.gexf',  # 85.60583
        'VirusShare_5fe0c336f26b8eb2eb8c556299857cc1.gexf',  # 6.621313
        'VirusShare_962f12f1ef496da24b52eb87a8739521.gexf',  # 7.228757
        'VirusShare_0f54179597d9f1a5f075e7f64c722419.gexf',  # 15.447115
        'VirusShare_57657cc2a67c756100ef2b5c55dd47ba.gexf',  # 434.88153
        'VirusShare_387636a84ffcf5f318599e9723375ad4.gexf',  # 6.5762854
        'VirusShare_038d756ffdb66cf297fb4bbc6cd994ba.gexf',  # 6.5762854
        'VirusShare_5e1d0b712f856e0594ccceee2c2135e7.gexf',  # 434.88153
        'VirusShare_120712981dfad4dcf8bf085a74cc0baa.gexf',  # 6.858757
        'VirusShare_dd01bf58c4d54a5b5339b0b49e9799fb.gexf',  # 6.5762854
        'VirusShare_780a0e95e27fd60c516503fd9100e5bf.gexf',  # 6.858757
        'VirusShare_62f35131cf856d486a3433c9b94f8200.gexf',  # 193.92279
        'VirusShare_153bfed1fe7e7c813d9edb925764005f.gexf',  # 6.858757
        'VirusShare_bdc481637e36fccc0814df61cf8eb3d4.gexf',  # 6.5762854
        'VirusShare_7a35ec2f61888da33a78374ae73070e5.gexf',  # 4.1895137
        'VirusShare_2a944cca05ce869d504d2f1a15f66140.gexf',  # 382.2225
        'VirusShare_dbe9cb897c0a4b7edabb64a68c5a242f.gexf',  # 6.9511476
        'VirusShare_b93ea89c26caff768100b3b3734a9d74.gexf',  # 6.5762854
        'VirusShare_dbf2975d5765e960f6522a4b0128b81a.gexf',  # 10.421358
        'VirusShare_1cdebdc7075a2271ddc45c8fc19bda19.gexf',  # 1.8705304
        'VirusShare_4a4e5889fedccbe245be4489ce07dff1.gexf',  # 434.88153
        'VirusShare_ab0e4909e5316fa9886da9ec5f6bfa7c.gexf',  # 6.858757
        'VirusShare_0ecd899c27d8fb6e1c2ed5d7abeed74e.gexf',  # 63.353546
        'VirusShare_549130d50c2ce5a57debf51689c0a975.gexf',  # 6.858757
        'VirusShare_d33e77a93f902a7d5edf3e210539feeb.gexf',  # 6.5093713
        'VirusShare_9e68ea99c5b5bb5b17916f7c0c8191b7.gexf',  # 5.265791
        'VirusShare_4ae0387cd9f86182fe72e74afedbc4c1.gexf',  # 6.5762854
        'VirusShare_581fd33cc8acacb12cbbc715a766d9d9.gexf',  # 82.18351
        'VirusShare_8df785cbd4c179d4508d0090a90ef491.gexf',  # 6.2594376
        'VirusShare_2a16333a5cbe5f649977421336aded91.gexf',  # 15.457101
        'VirusShare_f06c6b9c3ccf73921f8570b10f9bb7fe.gexf',  # 169.74081
        'VirusShare_892e87a4d9955fe0d2e6e2b1edab635b.gexf',  # 6.1565742
        'VirusShare_2e814a5f5c114489ce38a117fabf3d18.gexf',  # 6.9511476
        'VirusShare_dada91fc9e8db0f7bd8a31d6f1ed75dc.gexf',  # 38.99535
        'VirusShare_6828f8c42b5a94471ca9faaddf9c1215.gexf',  # 3.103993
        'VirusShare_9f96f92dbe6fdc8db6bc2e134e1b9d77.gexf',  # 434.88153
        'VirusShare_4fca330cfe3b6529e3d272305639533a.gexf',  # 6.198564
        'VirusShare_f2f9f752b08753c06fab05473a793c0a.gexf',  # 4.5687366
        'VirusShare_4adc49e9d85518866b4f95ab645faf6d.gexf',  # 6.2594376
        'VirusShare_69c2490f5478801eb72355368757d48a.gexf',  # 6.5762854
        'VirusShare_64191b00e77481347d55397bf033d597.gexf',  # 147.4903
        'VirusShare_4836ea8ec7adc537d17d1357caedf305.gexf',  # 86.3658
        'VirusShare_ff3e003183a3830d0eea2abd1220e602.gexf',  # 6.675957
        'VirusShare_d944797ebd43393a123a3c28f330fc5d.gexf',  # 3.103993
        'VirusShare_d59b8493b1ad4b600051c5c28d4b52b5.gexf',  # 353.60672
        'VirusShare_3539bdfad58ae71df12ddfe382bd6a80.gexf',  # 15.447115
        'VirusShare_1fb5801da767a22ea898bb90a0b94a5c.gexf',  # 15.460242
        'VirusShare_85ba0aa794e2bab79947f2dc2a9fbed2.gexf',  # 346.54233
        'VirusShare_26e37cae425e318cca08e9e46d84e113.gexf',  # 3.141282
        'VirusShare_3f3eb3a8b29bc9000c209701271e3184.gexf',  # 3.103993
        'VirusShare_91778619e3e81e0814ba2e69181dd6d4.gexf',  # 68.53075
        'VirusShare_160b455cc19e0c1ba1bef1b3a9ca97ce.gexf',  # 6.621313
        'VirusShare_3f5bc078b918e1a16050d8c545b8abbf.gexf',  # 3.103993
        'VirusShare_381f1339d5a0778606afd201759bc481.gexf',  # 6.5762854
        'VirusShare_f8c512a64f06173ba7b6f948b3cc3b81.gexf',  # 3.1992087
        'VirusShare_574e59fd51e7e894b296f684eaa37356.gexf',  # 334.88132
        'VirusShare_e0049eb5345cfe181d61a503ffbbc561.gexf',  # 15.3836565
        'VirusShare_0d88318b0186b5d65b87899dbaa7a5b3.gexf',  # 16.324898
        'VirusShare_f3ddd6fc9631a8679c1adefc3fa778b9.gexf',  # 434.88388
        'VirusShare_f07d5e4136ba20e54ba6ba422de971e4.gexf',  # 6.5762854
        'VirusShare_b4ca4cd07d5e87821012bdaa1faa2096.gexf',  # 329.44675
        'VirusShare_ef8c08d3f9c25c0c09a0e323d15df259.gexf',  # 3.103993
        'VirusShare_641b0063130668d6a25f5cd6bd8a7eb5.gexf',  # 6.62138
        'VirusShare_b5199fabcf3eb1baeb57d26ea2b733b2.gexf',  # 6.2594376
        'VirusShare_0a72229bb504d270e508b15ace1b38a4.gexf',  # 6.5762854
        'VirusShare_c2a1fb355e422a7fc515546af0886b66.gexf',  # 5.34118
        'VirusShare_2a33933c4cbfeb35f65ccbb9cb661866.gexf',  # 4.5069394
        'VirusShare_c2c60639213052cdda6872f345ea8e8f.gexf',  # 6.5762854
        'VirusShare_124347ab9424ccedbf7b841e007efc07.gexf',  # 6.5762854
        'VirusShare_1a25ab2e6585605665b282dea61a499d.gexf',  # 6.198564
        'VirusShare_97fec2ce6b34ac37a6a5df0739bd3860.gexf',  # 6.598372
        'VirusShare_00d0118a7152d850741d4143e968ba56.gexf',  # 6.858757
        'VirusShare_6f237d25472d9d09fc44ece7dc9ced92.gexf',  # 8.673773
        'VirusShare_cb2fa6dc53f32acad90a3cf4bc5d51f9.gexf',  # 7.057972
        'VirusShare_242e3e0fd9d9fecbe7f741a03c07c1de.gexf',  # 6.198564
        'VirusShare_836a62bec037576e17d16bb1bd036ffb.gexf',  # 176.4146
        'VirusShare_fbe403540869b62e2d3cc3acc639c074.gexf',  # 6.5762854
        'VirusShare_e95a8b7be2ce47237e8d1b808c93e8d3.gexf',  # 6.5762854
        'VirusShare_e5b3273d5f61c99dcd85328f9f3f34fd.gexf',  # 6.5762854
        'VirusShare_35565177740efd453fb60e63042d22eb.gexf',  # 4.766976
        'VirusShare_7f1d201c88fa16e39ea198fbc5b99553.gexf',  # 340.8121
        'VirusShare_bdad9ff85f4f1e00829d06db530f9eb1.gexf',  # 6.198564
        'VirusShare_e8c8a765a1ed3a746c3ac5c728e1202a.gexf',  # 6.5762854
        'VirusShare_8589ec219ffe9f94d16c07243bcb0631.gexf',  # 6.198564
        'VirusShare_4da7692874c056831c380935f8c04cc4.gexf',  # 15.990224
        'VirusShare_a7aec2cc8b5357d6dff9d21e94d623f2.gexf',  # 6.62138
        'VirusShare_9e6ccb1f074a1a68fb1bcc0436a76beb.gexf',  # 25.109325
        'VirusShare_0237bf35b128a6665f59d500b458ac0a.gexf',  # 6.198564
        'VirusShare_0e5512cffc5e9e51dd47450aa79434f5.gexf',  # 1.9641466
        'VirusShare_490934ef49d8ac537c69c2c537f9d17f.gexf',  # 2.9555335
    ]
    for i in range(len(name)):
        if sample_name in name[i]:
            return shap_values[i]

    return None

def find_i(file_name):
    name = [
        'VirusShare_97bc5adf5df9106efb885b78855c4838.gexf',  # 30.765587
        'VirusShare_e7ca640611fa2f8c630961199e13f6b5.gexf',  # 2.9279337
        'VirusShare_8e2b629a10625956f7609f2d939bcac4.gexf',  # 280.80453
        'VirusShare_63acec04855ac0c5641247f5ba3d48b9.gexf',  # 3.4467793
        'VirusShare_6ceede843c5dc4ca02509b35a3f40b28.gexf',  # 13.807023
        'VirusShare_ae165056c14a2cba5466cd69a28fc431.gexf',  # 6.2594376
        'VirusShare_86539705c6eb59c6acaf55e580be653a.gexf',  # 54.0898
        'VirusShare_b157472d61af978bee9d2c3b26df1e83.gexf',  # 15.3836565
        'VirusShare_5381c76ce28d84d3245efc4a19238d58.gexf',  # 6.5762854
        'VirusShare_8f7aff5ec7c3bb14331dfa3e981a0b73.gexf',  # 6.858757
        'VirusShare_6a3d9dfe6587141ace52c54d02e67e39.gexf',  # 6.5762854
        'VirusShare_6963b7ca41268cfa7470a3e8ad8e9766.gexf',  # 6.62138
        'VirusShare_41cec261cacf2f4bac3168740594361b.gexf',  # 0.7636745
        'VirusShare_d1df2a91bbe0594c1061ead71649ae09.gexf',  # 168.00703
        'VirusShare_74b1162820ca4095d8e911207a8a729a.gexf',  # 5.7805347
        'VirusShare_527c66dec303042a556a2349f29999dc.gexf',  # 6.621313
        'VirusShare_45e67a88d73488396eae77a7309d90e7.gexf',  # 6.621313
        'VirusShare_28a2725940ecb8b9686476cfa0dda209.gexf',  # 157.92831
        'VirusShare_c508d918a080b807fd23b51c350f86a5.gexf',  # 239.70909
        'VirusShare_9b0d37e5cb949ab86354788ce5488375.gexf',  # 69.2645
        'VirusShare_31e8119d2d0b14556eca26f06b679244.gexf',  # 6.5762854
        'VirusShare_32e2a5033829d8940bc50d5fb42f9785.gexf',  # 15.447115
        'VirusShare_d9263748ae4c6f2ec244351e054be5f4.gexf',  # 5.34118
        'VirusShare_250520d860f63afdf8c94affd0921253.gexf',  # 38.117565
        'VirusShare_0b2d190d17d50dfd4a589aca1a9caa49.gexf',  # 434.88388
        'VirusShare_fb8640f452beb0403462f3a036182ed7.gexf',  # 6.890682
        'VirusShare_f4db6fc9770ad3e0310f3e2a13b5cb8b.gexf',  # 6.5762854
        'VirusShare_6a9b2fac202d0ab867f89cd79c17f5c5.gexf',  # 6.5093713
        'VirusShare_17113d01df79f103c07f68b4ebfd51d3.gexf',  # 3.9143429
        'VirusShare_e3ae70d5e1f209bf4dc30ba013a165f2.gexf',  # 22.537872
        'VirusShare_0e78970a2de95e2d546c74cf9a134e75.gexf',  # 6.9511476
        'VirusShare_a0417e92830ad60ca9f58120a9af2f14.gexf',  # 5.000574
        'VirusShare_d985eeb52e4407c39b53e7426a8bc2e3.gexf',  # 142.4429
        'VirusShare_8a3d12947d7d98dac2e89b51f623bbf0.gexf',  # 434.88153
        'VirusShare_3b951e9452c02817a602753463958d67.gexf',  # 6.409724
        'VirusShare_15957ed1ff6bf19f2fa4c709409ebd70.gexf',  # 85.60583
        'VirusShare_5fe0c336f26b8eb2eb8c556299857cc1.gexf',  # 6.621313
        'VirusShare_962f12f1ef496da24b52eb87a8739521.gexf',  # 7.228757
        'VirusShare_0f54179597d9f1a5f075e7f64c722419.gexf',  # 15.447115
        'VirusShare_57657cc2a67c756100ef2b5c55dd47ba.gexf',  # 434.88153
        'VirusShare_387636a84ffcf5f318599e9723375ad4.gexf',  # 6.5762854
        'VirusShare_038d756ffdb66cf297fb4bbc6cd994ba.gexf',  # 6.5762854
        'VirusShare_5e1d0b712f856e0594ccceee2c2135e7.gexf',  # 434.88153
        'VirusShare_120712981dfad4dcf8bf085a74cc0baa.gexf',  # 6.858757
        'VirusShare_dd01bf58c4d54a5b5339b0b49e9799fb.gexf',  # 6.5762854
        'VirusShare_780a0e95e27fd60c516503fd9100e5bf.gexf',  # 6.858757
        'VirusShare_62f35131cf856d486a3433c9b94f8200.gexf',  # 193.92279
        'VirusShare_153bfed1fe7e7c813d9edb925764005f.gexf',  # 6.858757
        'VirusShare_bdc481637e36fccc0814df61cf8eb3d4.gexf',  # 6.5762854
        'VirusShare_7a35ec2f61888da33a78374ae73070e5.gexf',  # 4.1895137
        'VirusShare_2a944cca05ce869d504d2f1a15f66140.gexf',  # 382.2225
        'VirusShare_dbe9cb897c0a4b7edabb64a68c5a242f.gexf',  # 6.9511476
        'VirusShare_b93ea89c26caff768100b3b3734a9d74.gexf',  # 6.5762854
        'VirusShare_dbf2975d5765e960f6522a4b0128b81a.gexf',  # 10.421358
        'VirusShare_1cdebdc7075a2271ddc45c8fc19bda19.gexf',  # 1.8705304
        'VirusShare_4a4e5889fedccbe245be4489ce07dff1.gexf',  # 434.88153
        'VirusShare_ab0e4909e5316fa9886da9ec5f6bfa7c.gexf',  # 6.858757
        'VirusShare_0ecd899c27d8fb6e1c2ed5d7abeed74e.gexf',  # 63.353546
        'VirusShare_549130d50c2ce5a57debf51689c0a975.gexf',  # 6.858757
        'VirusShare_d33e77a93f902a7d5edf3e210539feeb.gexf',  # 6.5093713
        'VirusShare_9e68ea99c5b5bb5b17916f7c0c8191b7.gexf',  # 5.265791
        'VirusShare_4ae0387cd9f86182fe72e74afedbc4c1.gexf',  # 6.5762854
        'VirusShare_581fd33cc8acacb12cbbc715a766d9d9.gexf',  # 82.18351
        'VirusShare_8df785cbd4c179d4508d0090a90ef491.gexf',  # 6.2594376
        'VirusShare_2a16333a5cbe5f649977421336aded91.gexf',  # 15.457101
        'VirusShare_f06c6b9c3ccf73921f8570b10f9bb7fe.gexf',  # 169.74081
        'VirusShare_892e87a4d9955fe0d2e6e2b1edab635b.gexf',  # 6.1565742
        'VirusShare_2e814a5f5c114489ce38a117fabf3d18.gexf',  # 6.9511476
        'VirusShare_dada91fc9e8db0f7bd8a31d6f1ed75dc.gexf',  # 38.99535
        'VirusShare_6828f8c42b5a94471ca9faaddf9c1215.gexf',  # 3.103993
        'VirusShare_9f96f92dbe6fdc8db6bc2e134e1b9d77.gexf',  # 434.88153
        'VirusShare_4fca330cfe3b6529e3d272305639533a.gexf',  # 6.198564
        'VirusShare_f2f9f752b08753c06fab05473a793c0a.gexf',  # 4.5687366
        'VirusShare_4adc49e9d85518866b4f95ab645faf6d.gexf',  # 6.2594376
        'VirusShare_69c2490f5478801eb72355368757d48a.gexf',  # 6.5762854
        'VirusShare_64191b00e77481347d55397bf033d597.gexf',  # 147.4903
        'VirusShare_4836ea8ec7adc537d17d1357caedf305.gexf',  # 86.3658
        'VirusShare_ff3e003183a3830d0eea2abd1220e602.gexf',  # 6.675957
        'VirusShare_d944797ebd43393a123a3c28f330fc5d.gexf',  # 3.103993
        'VirusShare_d59b8493b1ad4b600051c5c28d4b52b5.gexf',  # 353.60672
        'VirusShare_3539bdfad58ae71df12ddfe382bd6a80.gexf',  # 15.447115
        'VirusShare_1fb5801da767a22ea898bb90a0b94a5c.gexf',  # 15.460242
        'VirusShare_85ba0aa794e2bab79947f2dc2a9fbed2.gexf',  # 346.54233
        'VirusShare_26e37cae425e318cca08e9e46d84e113.gexf',  # 3.141282
        'VirusShare_3f3eb3a8b29bc9000c209701271e3184.gexf',  # 3.103993
        'VirusShare_91778619e3e81e0814ba2e69181dd6d4.gexf',  # 68.53075
        'VirusShare_160b455cc19e0c1ba1bef1b3a9ca97ce.gexf',  # 6.621313
        'VirusShare_3f5bc078b918e1a16050d8c545b8abbf.gexf',  # 3.103993
        'VirusShare_381f1339d5a0778606afd201759bc481.gexf',  # 6.5762854
        'VirusShare_f8c512a64f06173ba7b6f948b3cc3b81.gexf',  # 3.1992087
        'VirusShare_574e59fd51e7e894b296f684eaa37356.gexf',  # 334.88132
        'VirusShare_e0049eb5345cfe181d61a503ffbbc561.gexf',  # 15.3836565
        'VirusShare_0d88318b0186b5d65b87899dbaa7a5b3.gexf',  # 16.324898
        'VirusShare_f3ddd6fc9631a8679c1adefc3fa778b9.gexf',  # 434.88388
        'VirusShare_f07d5e4136ba20e54ba6ba422de971e4.gexf',  # 6.5762854
        'VirusShare_b4ca4cd07d5e87821012bdaa1faa2096.gexf',  # 329.44675
        'VirusShare_ef8c08d3f9c25c0c09a0e323d15df259.gexf',  # 3.103993
        'VirusShare_641b0063130668d6a25f5cd6bd8a7eb5.gexf',  # 6.62138
        'VirusShare_b5199fabcf3eb1baeb57d26ea2b733b2.gexf',  # 6.2594376
        'VirusShare_0a72229bb504d270e508b15ace1b38a4.gexf',  # 6.5762854
        'VirusShare_c2a1fb355e422a7fc515546af0886b66.gexf',  # 5.34118
        'VirusShare_2a33933c4cbfeb35f65ccbb9cb661866.gexf',  # 4.5069394
        'VirusShare_c2c60639213052cdda6872f345ea8e8f.gexf',  # 6.5762854
        'VirusShare_124347ab9424ccedbf7b841e007efc07.gexf',  # 6.5762854
        'VirusShare_1a25ab2e6585605665b282dea61a499d.gexf',  # 6.198564
        'VirusShare_97fec2ce6b34ac37a6a5df0739bd3860.gexf',  # 6.598372
        'VirusShare_00d0118a7152d850741d4143e968ba56.gexf',  # 6.858757
        'VirusShare_6f237d25472d9d09fc44ece7dc9ced92.gexf',  # 8.673773
        'VirusShare_cb2fa6dc53f32acad90a3cf4bc5d51f9.gexf',  # 7.057972
        'VirusShare_242e3e0fd9d9fecbe7f741a03c07c1de.gexf',  # 6.198564
        'VirusShare_836a62bec037576e17d16bb1bd036ffb.gexf',  # 176.4146
        'VirusShare_fbe403540869b62e2d3cc3acc639c074.gexf',  # 6.5762854
        'VirusShare_e95a8b7be2ce47237e8d1b808c93e8d3.gexf',  # 6.5762854
        'VirusShare_e5b3273d5f61c99dcd85328f9f3f34fd.gexf',  # 6.5762854
        'VirusShare_35565177740efd453fb60e63042d22eb.gexf',  # 4.766976
        'VirusShare_7f1d201c88fa16e39ea198fbc5b99553.gexf',  # 340.8121
        'VirusShare_bdad9ff85f4f1e00829d06db530f9eb1.gexf',  # 6.198564
        'VirusShare_e8c8a765a1ed3a746c3ac5c728e1202a.gexf',  # 6.5762854
        'VirusShare_8589ec219ffe9f94d16c07243bcb0631.gexf',  # 6.198564
        'VirusShare_4da7692874c056831c380935f8c04cc4.gexf',  # 15.990224
        'VirusShare_a7aec2cc8b5357d6dff9d21e94d623f2.gexf',  # 6.62138
        'VirusShare_9e6ccb1f074a1a68fb1bcc0436a76beb.gexf',  # 25.109325
        'VirusShare_0237bf35b128a6665f59d500b458ac0a.gexf',  # 6.198564
        'VirusShare_0e5512cffc5e9e51dd47450aa79434f5.gexf',  # 1.9641466
        'VirusShare_490934ef49d8ac537c69c2c537f9d17f.gexf',  # 2.9555335
    ]

    for i in range(len(name)):
        if file_name in name[i]:
            return i

    return -1

def load_graph(file_path):
    # 假设 G_loaded 是从 GEXF 文件中加载的图
    G_loaded = nx.read_gexf(file_path)

    # 将节点标识符转换回整数类型
    G_int = nx.relabel_nodes(G_loaded, lambda x: int(x))

    # 创建一个新图，用于存储转换后的节点和边
    G_int_edges = nx.DiGraph()

    # 遍历原图的所有节点和边，并在新图中添加它们
    for node in G_int.nodes():
        G_int_edges.add_node(node)

    for edge in G_int.edges():
        u, v = map(int, edge)  # 转换边的节点标识符为整数
        G_int_edges.add_edge(u, v)

    return G_int_edges

def test_MLP_model(vector):
    #model is a global variable
    Y_pred_probs = model.predict(vector)  # 预测概率
    return Y_pred_probs

def recover_graph(sample_name, shap_value, failed_graph_dir = 'log_init_200pop_100step_shap/feb7_MLP_test_ga_dominate_shap_score_init/ga_failed/'):
    original_graph_dir = '/data/b/shiwensong/dataset/virusshare2018_gexf/'

    # 1. load failed graphs
    failed_graphs = glob.glob(failed_graph_dir + sample_name + '*.gexf')
    print(len(failed_graphs))

    # 2. load original graph
    original_fcg_file = original_graph_dir + sample_name + '.gexf'
    original_fcg = FCG(original_fcg_file, 1, shap_value)

    # 3. cal the scores, and find the best failed graph
    min_score = 100000
    min_failed_graph = None
    for failed_path in failed_graphs:
        failed_fcg = load_graph(failed_path)

        current_fcg = copy.deepcopy(original_fcg)
        current_fcg.nodes = failed_fcg.nodes
        current_fcg.edges = failed_fcg.edges

        current_fcg.cal_centralities()
        degree = current_fcg.degree_feature
        katz = current_fcg.katz_feature
        closeness = current_fcg.closeness_feature
        harmonic = current_fcg.harmonic_feature
        combined_feature = np.hstack((degree, katz, closeness, harmonic))

        combined_feature = combined_feature.reshape(1, -1)

        Y_probs = test_MLP_model(combined_feature)

        if Y_probs[0][0] < min_score:
            min_score = Y_probs[0][1]
            min_failed_graph = current_fcg

    print(min_score)
    return min_failed_graph




if __name__ == '__main__':
    original_graph_dir = '/data/b/shiwensong/dataset/virusshare2018_gexf/'
    failed_graph_dir = 'log_init_200pop_100step_shap/feb7_MLP_test_ga_dominate_shap_score_init/ga_failed/'

    #load shap value
    with open('shap_values_120_samples_jan30.pkl', 'rb') as file:
        shap_values = pickle.load(file)
    #
    # #只输出了类别为1的shap value
    shap_values = shap_values[0]

    sample_names = [
        'VirusShare_5e1d0b712f856e0594ccceee2c2135e7',
        "VirusShare_836a62bec037576e17d16bb1bd036ffb",
        "VirusShare_45e67a88d73488396eae77a7309d90e7",
        "VirusShare_57657cc2a67c756100ef2b5c55dd47ba",
        "VirusShare_c508d918a080b807fd23b51c350f86a5",
        "VirusShare_9b0d37e5cb949ab86354788ce5488375",
        "VirusShare_574e59fd51e7e894b296f684eaa37356",
        "VirusShare_4836ea8ec7adc537d17d1357caedf305",
        "VirusShare_160b455cc19e0c1ba1bef1b3a9ca97ce",
        "VirusShare_0b2d190d17d50dfd4a589aca1a9caa49",
        "VirusShare_4a4e5889fedccbe245be4489ce07dff1",
        "VirusShare_0d88318b0186b5d65b87899dbaa7a5b3",
        "VirusShare_2a944cca05ce869d504d2f1a15f66140",
        "VirusShare_f3ddd6fc9631a8679c1adefc3fa778b9",
        "VirusShare_28a2725940ecb8b9686476cfa0dda209",
        "VirusShare_b4ca4cd07d5e87821012bdaa1faa2096",
        "VirusShare_d59b8493b1ad4b600051c5c28d4b52b5",
        "VirusShare_62f35131cf856d486a3433c9b94f8200",
        "VirusShare_7f1d201c88fa16e39ea198fbc5b99553",
        "VirusShare_6ceede843c5dc4ca02509b35a3f40b28",
        "VirusShare_85ba0aa794e2bab79947f2dc2a9fbed2",
        "VirusShare_15957ed1ff6bf19f2fa4c709409ebd70",
        "VirusShare_8e2b629a10625956f7609f2d939bcac4",
        "VirusShare_64191b00e77481347d55397bf033d597",
        "VirusShare_dada91fc9e8db0f7bd8a31d6f1ed75dc",
        "VirusShare_f06c6b9c3ccf73921f8570b10f9bb7fe",
        "VirusShare_d985eeb52e4407c39b53e7426a8bc2e3",
        "VirusShare_86539705c6eb59c6acaf55e580be653a"
    ]
    print(len(sample_names))

    model = load_MLP_model('./430features_3yearsdataset_all/MLP.h5')

    res = []

    for sample_name in sample_names:
        # 1. load failed graphs
        failed_graphs = glob.glob(failed_graph_dir + sample_name + '*.gexf')
        print(len(failed_graphs))

        #2. load original graph
        original_fcg_file = original_graph_dir + sample_name + '.gexf'
        shap_value = find_shap_value(shap_values, sample_name)
        original_fcg = FCG(original_fcg_file, 1, shap_value)
        print("original nodes", len(original_fcg.nodes))
        print("original edges", len(original_fcg.edges))

        #3. cal the scores
        scores = []
        for failed_path in failed_graphs:
            failed_fcg = load_graph(failed_path)

            current_fcg = copy.deepcopy(original_fcg)
            current_fcg.nodes = failed_fcg.nodes
            current_fcg.edges = failed_fcg.edges

            current_fcg.cal_centralities()
            degree = current_fcg.degree_feature
            katz = current_fcg.katz_feature
            closeness = current_fcg.closeness_feature
            harmonic = current_fcg.harmonic_feature
            combined_feature = np.hstack((degree, katz, closeness, harmonic))

            combined_feature = combined_feature.reshape(1, -1)

            Y_probs = test_MLP_model(combined_feature)

            scores.append(Y_probs[0][0])

        # print("max", max(scores))
        # print("min", min(scores))
        data = [
            sample_name,
            len(failed_graphs),
            max(scores),
            min(scores)
        ]
        res.append(data)

    df = pd.DataFrame(res, columns=['sample_name', 'num_failed_graphs', 'max_score', 'min_score'])
    df.to_csv('feb7_MLP_test_ga_dominate_shap_score_init_recover_graph.csv', index=False)


