# all string ids grouped by blog vs newswire
global all_ids
global test_fnames
all_ids = [
    '010aaf594ae6ef20eb28e3ee26038375',
    '018fb4e59ac5474167ffc5940d7e55e7',
    '01f69c4c2206e7c3fa3706ccd5b8b350',
    '026bd1c7eae9f14da9480a4b88ba2fb6',
    '02905b7ce3a6b8b0961c6c2310392ef9',
    '04134f2be20afbb868d7a8292f49e277',
    '0648a08469a3be9eb972f0d213562805',
    '0659c87d9fd3d5efd258ee6de3ba1003',
    '086e26ec92d1cc02f3900e9ac46d6962',
    '087f58983ef5e94e54024bc9f0f009ae',
    '08b0dfe15192c063055ed7db8d24c625',
    '08ebdc5f0ec8588af38ab1684318d99c',
    '09098ae4e956a51b038876197814735e',
    '0a421343005f3241376fa01e1cb3c6fb',
    '0ba982819aaf9f5b94a7cebd48ac6018',
    '0c49bb860962aa0d5b8e3fc277592da0',
    '0cfdfe102b7a4cb34e1a181c1d36d23d',
    '0eb03fc279066b84ed49d44b2405469a',
    '0f03cc5a508d630c6c8c8c61396e31a9',
    '0f565d3822dca80336582ffac4adaf78',
    '0fab386f8b6527439481f526c92341c7',
    '101d0fc4a78dc1b84953ebd399b2fad5',
    '11329f1cdb44019afc8f48b6fdc5376d',
    '11a29a0d63a79b0f5d19ccae1838b125',
    '11c906f2f798abb05f143b206edf77a5',
    '120fe19a9bc68fd85fc4963c166e9345',
    '130a86739522ab7c56232e798d04cbf9',
    '14294db341956a71811c9dd015b04ed7',
    '1473ea2ded50c05b29b4f55f1b83ada3',
    '15ba31cca04cc5300361f46319247c40',
    '1656bbad43fee4569b5c5f14110c1342',
    '178e7de35eccad0df800f0c7539cf614',
    '17a2dc40635ec239e9e16d10b6dd45e8',
    '17f98f0c6cda0227e732e6761f396d1f',
    '186ef6837e001cd9b97a132c86705545',
    '18a89cdd00dadc593a88c924111575f1',
    '18e8a277f2659f79291efa0e12e80cb3',
    '19569b08f07d751d6ac4a07633653c50',
    '1ae45904ad12b1540dc390e162b61235',
    '1b0f90c029f75d326ea39c0371901ef4',
    '1b268b27094ba9c5feb11192dad940ab',
    '1d16a571f14fb1032bc19e9314a46deb',
    '1f60eb9697e240af089b134b69c2042d',
    '21dbe23f56aaef87fd0980234895b321',
    '2251a78817e67a2adaf0722fd05c7ac0',
    '22696c601df1a7359e9b629c689700ad',
    '22ca1a5aa492b429d274169c54554a7c',
    '23987125927d321ec6f0c30c8f453cb3',
    '24d93564f48ae17904aa82f937db8c21',
    '26175bdbe49b712d7412c273c111e813',
    '26542fb5b83cdb4b98a3fe31e0226b39',
    '2701285c791f423cd2f8fd827df9c2c9',
    '2a10c5cc27e7504dc9df92396b9e28b8',
    '2a54459212636289034af844f8634e37',
    '2aaa319d1e1a0600837d013cb84290ea',
    '2ac34d012c8d909d4a29aa3f6be1f23d',
    '2ac3b55a10d5395ded9e8e54c345553b',
    '2b96d1172d37f60aea5ce64a0b410248',
    '2ba8bbf004fe30c0a01f6fcd25f01dcc',
    '2bdb9d86091c6f412ffa767bdc749be9',
    '2bebb50073ceefd0c9ccfdf3e07b3258',
    '2c2e8b3286bd34e30a4cb57cb7e26ce5',
    '2c8bcca93da4097da338a8754e4f03b0',
    '2ca0238925d38f345acbf826854ea448',
    '2d2a4ddb1c8f4a669541704f9fb78472',
    '2d7d6761aad911a63a235a571fa7862f',
    '2d8d3572658fdb8754fdc84d2b15f302',
    '2f5ee4e363c30678dc3b55caf43bc63d',
    '3059538a2542c71687871b3444f8d921',
    '324274e50f2d07757e2d88ff58a0c33b',
    '3322caacf140c92366a639ee004560ce',
    '334de29f692ef2c5460b78fcad5c6c9e',
    '33ed1c9fdee1000e2340ac7f92c77752',
    '3446f8cbcf53eaca5692913ced012b11',
    '34d49f3357eaf14c849e9cdfeb893273',
    '34f729e5ac124e9898b2744a6598d50e',
    '35587c6d8aa67724ba23231dd16f7b44',
    '361e1c2ca3a1e21c618e0e8fab959e30',
    '36b12cef6f7a805e3e74a4f430129028',
    '36d45aff571e3fbe036f309c18d31668',
    '376c304800b734b2a5a2c87b19eddc2a',
    '37b56b6dd846ad0dd6e8cd00ba2efaf4',
    '37d781089c669131c5118415cf470422',
    '389c70a4859f7528cc6e8b84c10766d7',
    '39280a4d31d81837e17469e18a854116',
    '3a0d64b5cb2bc7319e803e344dc695b5',
    '3b34a76a3589417f5db02883b47280a6',
    '3b4d58c0a53671c6ce03f0529bb6089d',
    '3b9c27eda65c635e109a547930942486',
    '3c9fb643a48360935c1044efca570514',
    '3dc7812b2b39ed067cc7c8ab1218e128',
    '3ddbad6f438c88eec387131477ffe1b9',
    '3dff15d768dbfe27e4d6b81fb63aee95',
    '3e6c7121211de578d7fd831eae801438',
    '3e9bbf75058a3f16585889bb9c64a903',
    '3f0e2f2fb9b773bc178522a6535a9651',
    '4042cd8643253f65df3a4e8de320a1c9',
    '408dff173c599256711f23238e280c15',
    '40f1f697a457e39c30ad94b7cc712c96',
    '41404718f9c1e94cf58aad1fc90c70a7',
    '416cfc6a5717682cd35d381c5be07734',
    '4175e3da216dcc8710a26359e4ecaaad',
    '428e1e095b4e6e830b47e72f133faf87',
    '43341a312ffd84a4ad3c3ab0df8bcd7c',
    '44087d95184e9d94f3948f47e9b602af',
    '44169f6a3f5b04e8dbab2a26e572a136',
    '44a65adb7f74e6c99d05eb2721fd0baf',
    '44b011cd504c9ed71beb851324db886a',
    '44fd27d40ae65547c3b584c2ff360cd7',
    '4572d22caf3e1924f894002b724f958b',
    '459f9a2b3eddd436f0232395f129dfd0',
    '45b9b8f7d17ce5f352c16a339e96705f',
    '4622b60202cf3944119daf2be53aa74f',
    '4683e6affe801713ed4cc9d596b57fac',
    '4743a10c1d5f1ad35c31646049acb9db',
    '4764f1400fa336d1fb972719b10b939a',
    '4798bc0e166fe93893bdf2d922f06258',
    '47c26ba3563092e41c5a42252931baf1',
    '4829d3d91263ed9d8801e6d94c3569a5',
    '48c498c9762046efbece8d183ed996ca',
    '4a3d067b19686b281e0beb437573a28c',
    '4deb48e2b0ab194ce37c1bd31c73586a',
    '4df3dfff1ee1683ac6e1c2ea24ce2589',
    '4eb58398a5c2ef35b16d885c5573b3d4',
    '4edd239ce7d1f7274154cd05081f8995',
    '4f7eedf44076ea050d7db3715f9333fa',
    '4fbb1eec7dfd5c2fefb94a2d873ddfa5',
    '4fca88a5c29716cbb7c0f9aa9b84007a',
    '502c46cc149d30f9ad0c25194636dcb6',
    '51d64c51a2363954454ee9e921b590ce',
    '52355a4167e6ac3a80d19c94ad6259a7',
    '5254f96ac3a601e99b6357c4f7627991',
    '52a77871923a7f86bb1a52812bc7f2e1',
    '52e569e00b6428b94205d3dd5c457c54',
    '563b1e8fcb1de7a4c0e01da9100d6e09',
    '565fa81d640f451b20955887a43b3a23',
    '5685a6069312d52a897fe69973269338',
    '56af144a4d1d2e662531bdfd00d3c725',
    '57026b7bcb8f855de3e26d572db35285',
    '57b2773ab54bbc5c119a46fd9be2c4f0',
    '584b6272bb8c9cc134621ff5ace8c98d',
    '590baa25bb1cc16c31fd02395edf6835',
    '593cb5020613a4695859130542f7fc94',
    '59f8514f6db132207ba9e5828f73d706',
    '5bac42475431a87070720e94b27cfd99',
    '5bb3c2b1094912a6df7e862bb2981481',
    '5bbe1c6185296d179b95810e48ee3834',
    '5c29f9e575b94c61db8ed52bdfa53843',
    '5c59566e9132c060423cad5b2d1bac1e',
    '5c7ea2b51202d80ee37eba8a182afad3',
    '5cd7d603e1cf8d2c134d039dc90112f0',
    '5d7b429073c60d53acba21bb6e7e6caa',
    '5dfd5bfee062cd5896b619a2b1309766',
    '5e3fbf49f8301654bb4954c0f1e386a9',
    '5fa0f2a7f323a781640b126978ca8a42',
    '5fa7fbe87758a02a1e4591f88175ccf3',
    '609d5112c0386dc4e5f2e90b93cb7a5f',
    '6154640fdb94510274583591cad7b379',
    '61d2b0dcc730f0b4e92ae0d1929b3caf',
    '635bde2afdaaf20a0bcdc3b5f79578c9',
    '63878a2b6d34b576361d2a2778f321a6',
    '644706e2d97c9a9a1f9874510180f136',
    '648abb9000309b9807cc8b212c11254f',
    '648fc5834f73b4196b4ceb3daad954f9',
    '6521f6bd1eb405232a5e852423722bac',
    '661ece467567ffbb54b551dfc1c2c254',
    '66fba4f92d2f9d8c3bee5dfad3af9828',
    '670b5425fcd1700e2c27af5f09244cb1',
    '7677d625b58ce649c8aeda2ff4a56389',
    '96bf72399b104346f3e79022e0c08e5a',
    'a68c8d0ef75bbbd2923bf7aa78b72d3e',
    'a9318b72c7a2ff32d459af958c7defe1',
    'aa003ea934a97bac86cee52b7122f1f8',
    'aa32f4f9534045b9f33a9599d0c1b580',
    'af18d29036ab0a9f8cf2742a5a1b4804',
    'b6b443777e5ca92aa5152f5593960fd9',
    'c06e8bbdf69f73a69cd3d5dbb4d06a21',
    'c1f185252a2837aa464e36f263d1ebe9',
    'c397ecd66789b905c6b1c5ef21af03ec',
    'ca2a6fbf721ca102c149ad6a90d5b00a',
    'cb156ad2a5458fabc9e093b6b5e0f97f',
    'cca700aed62fd497e64e507752409b41',
    'cf88887857b155d8822f82cad3597744',
    'd4698e3ad06f896058ade2e8f3a09577',
    'd5825f99faec1ae48589b98560a98d61',
    'd6bc66d7c8423368aaa8d789b5bdf5db',
    'dd0b65f632f64369c530f9bbb4b024b4',
    'edb392c8323a4f5f27cc0e59df409c68',
#]


#newswire_ids = [
    'AFP_ENG_20100414.0615',
    'AFP_ENG_20100601.0724',
    'APW_ENG_20090611.0697',
    'APW_ENG_20101231.0037',
    'NYT_ENG_20130422.0048',
    'NYT_ENG_20130428.0140',
    'NYT_ENG_20130501.0255',
    'NYT_ENG_20130504.0098',
    'NYT_ENG_20130506.0045',
    'NYT_ENG_20130525.0040',
    'NYT_ENG_20130603.0111',
    'NYT_ENG_20130613.0153',
    'NYT_ENG_20130619.0092',
    'NYT_ENG_20130625.0044',
    'NYT_ENG_20130625.0192',
    'NYT_ENG_20130703.0214',
    'NYT_ENG_20130709.0087',
    'NYT_ENG_20130710.0155',
    'NYT_ENG_20130712.0047',
    'NYT_ENG_20130716.0036',
    'NYT_ENG_20130716.0217',
    'NYT_ENG_20130731.0133',
    'NYT_ENG_20130813.0006',
    'NYT_ENG_20130816.0151',
    'NYT_ENG_20130822.0136',
    'NYT_ENG_20130828.0147',
    'NYT_ENG_20130910.0002',
    'NYT_ENG_20130910.0191',
    'NYT_ENG_20130914.0094',
    'NYT_ENG_20131003.0269',
    'NYT_ENG_20131118.0019',
    'NYT_ENG_20131121.0040',
    'NYT_ENG_20131121.0250',
    'NYT_ENG_20131122.0237',
    'NYT_ENG_20131128.0177',
    'NYT_ENG_20131210.0203',
    'NYT_ENG_20131220.0283',
#]


#df_weird_ids = [
    'ENG_DF_000170_20150322_F00000082_0-3740',
    'ENG_DF_000170_20150327_F0000007J_0-4822',
    'ENG_DF_000183_20150318_F0000009G_13215-19498',
    'ENG_DF_000183_20150318_F0000009G_19500-21349',
    'ENG_DF_000183_20150318_F0000009G_7866-13213',
    'ENG_DF_000183_20150407_F0000009E_0-3528',
    'ENG_DF_000183_20150407_F0000009E_13238-18593',
    'ENG_DF_000183_20150408_F0000009B_0-6145',
    'ENG_DF_000183_20150408_F0000009B_11977-13151',
    'ENG_DF_000183_20150408_F0000009B_6147-11975',
    'ENG_DF_000183_20150408_F0000009C_0-5958',
    'ENG_DF_000183_20150408_F0000009C_14631-19938',
    'ENG_DF_000183_20150408_F0000009C_19940-25210',
    'ENG_DF_000183_20150408_F0000009C_25212-31026',
    'ENG_DF_000183_20150408_F0000009C_31028-35268',
    'ENG_DF_000183_20150408_F0000009C_5960-9869',
    'ENG_DF_000183_20150409_F0000009F_0-7277',
    'ENG_DF_000183_20150410_F0000009H_0-6659',
    'ENG_DF_000183_20150410_F0000009H_6661-11865',
    'ENG_DF_000261_20150319_F00000084_0-4910',
    'ENG_DF_000261_20150321_F00000081_0-5527'
]



# BUG warning: the 4 ids that are commented out were previously wrongly
# concatenated because I forgot the comma between them. Because of how I wrote
# the code, this made them show up in the TRAINING set for all experiments
# until now. I am commenting them out for clarity, to clarify that they are
# really training, not test docs.

# But we could uncomment them and re-run all experiments if you prefer.

test_fnames = [
 # ~33% of the new forums such that the first number doesn't reappear

 'ENG_DF_000170_20150322_F00000082',
 'ENG_DF_000170_20150327_F0000007J',
 'ENG_DF_000261_20150319_F00000084',
 # 'ENG_DF_000261_20150321_F00000081'

 # all non-NYT ones

 # 'AFP_ENG_20100414.0615',
 'AFP_ENG_20100601.0724',
 'APW_ENG_20090611.0697',
 'APW_ENG_20101231.0037',

 # 8 random ones from np.random.RandomState(0).choice(nyt, size=8)

 'NYT_ENG_20130703.0214',
 'NYT_ENG_20130822.0136',
 'NYT_ENG_20130914.0094',
 'NYT_ENG_20130731.0133',
 'NYT_ENG_20130501.0255',
 'NYT_ENG_20131003.0269',
 'NYT_ENG_20131118.0019',
 # 'NYT_ENG_20130716.0217'

 # random 1/3 of old discussion forum data

 # In [50]: np.random.RandomState(1).choice(old_forum, size=64, replace=False)
 # '178e7de35eccad0df800f0c7539cf614',
 '2d8d3572658fdb8754fdc84d2b15f302',
 '4622b60202cf3944119daf2be53aa74f',
 '4a3d067b19686b281e0beb437573a28c',
 '37b56b6dd846ad0dd6e8cd00ba2efaf4',
 '5dfd5bfee062cd5896b619a2b1309766',
 '18a89cdd00dadc593a88c924111575f1',
 '428e1e095b4e6e830b47e72f133faf87',
 '41404718f9c1e94cf58aad1fc90c70a7',
 '40f1f697a457e39c30ad94b7cc712c96',
 '4edd239ce7d1f7274154cd05081f8995',
 '0cfdfe102b7a4cb34e1a181c1d36d23d',
 '3446f8cbcf53eaca5692913ced012b11',
 '61d2b0dcc730f0b4e92ae0d1929b3caf',
 '52a77871923a7f86bb1a52812bc7f2e1',
 '2f5ee4e363c30678dc3b55caf43bc63d',
 '48c498c9762046efbece8d183ed996ca',
 '1b268b27094ba9c5feb11192dad940ab',
 '02905b7ce3a6b8b0961c6c2310392ef9',
 '23987125927d321ec6f0c30c8f453cb3',
 '2701285c791f423cd2f8fd827df9c2c9',
 '3f0e2f2fb9b773bc178522a6535a9651',
 'cb156ad2a5458fabc9e093b6b5e0f97f',
 '5c59566e9132c060423cad5b2d1bac1e',
 '15ba31cca04cc5300361f46319247c40',
 '2a54459212636289034af844f8634e37',
 '08ebdc5f0ec8588af38ab1684318d99c',
 '44b011cd504c9ed71beb851324db886a',
 '96bf72399b104346f3e79022e0c08e5a',
 '2ac3b55a10d5395ded9e8e54c345553b',
 '0ba982819aaf9f5b94a7cebd48ac6018',
 '44a65adb7f74e6c99d05eb2721fd0baf',
 'dd0b65f632f64369c530f9bbb4b024b4',
 '0f565d3822dca80336582ffac4adaf78',
 '36b12cef6f7a805e3e74a4f430129028',
 '3dc7812b2b39ed067cc7c8ab1218e128',
 '4683e6affe801713ed4cc9d596b57fac',
 '39280a4d31d81837e17469e18a854116',
 '2bdb9d86091c6f412ffa767bdc749be9',
 '648abb9000309b9807cc8b212c11254f',
 '17f98f0c6cda0227e732e6761f396d1f',
 '3e9bbf75058a3f16585889bb9c64a903',
 '4eb58398a5c2ef35b16d885c5573b3d4',
 '2ba8bbf004fe30c0a01f6fcd25f01dcc',
 '3e6c7121211de578d7fd831eae801438',
 '1f60eb9697e240af089b134b69c2042d',
 '0eb03fc279066b84ed49d44b2405469a',
 '04134f2be20afbb868d7a8292f49e277',
 '4743a10c1d5f1ad35c31646049acb9db',
 '24d93564f48ae17904aa82f937db8c21',
 '644706e2d97c9a9a1f9874510180f136',
 '66fba4f92d2f9d8c3bee5dfad3af9828',
 '565fa81d640f451b20955887a43b3a23',
 '670b5425fcd1700e2c27af5f09244cb1',
 '52355a4167e6ac3a80d19c94ad6259a7',
 '0f03cc5a508d630c6c8c8c61396e31a9',
 '6521f6bd1eb405232a5e852423722bac',
 '324274e50f2d07757e2d88ff58a0c33b',
 '2aaa319d1e1a0600837d013cb84290ea',
 '44fd27d40ae65547c3b584c2ff360cd7',
 '47c26ba3563092e41c5a42252931baf1',
 '1b0f90c029f75d326ea39c0371901ef4',
 '1473ea2ded50c05b29b4f55f1b83ada3',
 '2251a78817e67a2adaf0722fd05c7ac0'
] 

# N_TRAIN = 173
# N_TEST = 73

def ret_all_ids():
    return all_ids

def ret_test_fnames():
    return test_fnames

def ret_train_fnames():
    train = []
    for x in all_ids:
        if x not in test_fnames:
            train.append(x)
    return train