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

N_TRAIN = 173
N_TEST = 73
