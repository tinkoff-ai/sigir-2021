import os

import path

PROJECT_ROOT = path.Path(os.path.dirname(__file__)).abspath()
DATA_ROOT = PROJECT_ROOT.joinpath("data")

TRAIN_SESSION_SAMPLE = str(DATA_ROOT.joinpath("sample/browsing_train_sample.csv"))
SUBMISSION_TEST_SAMPLE = str(DATA_ROOT.joinpath("sample/rec_submission_sample.json"))

TRAIN_SESSION_FILE = str(DATA_ROOT.joinpath("train/browsing_train.csv"))
SKU_CONTENT_FILE = str(DATA_ROOT.joinpath("train/sku_to_content.csv"))
TEST_FILE_1 = str(DATA_ROOT.joinpath("rec_test_phase_1.json"))
TEST_FILE_2 = str(DATA_ROOT.joinpath("rec_test_phase_2.json"))


# submission

most_popular_sku = [
    "093f87a220846c0c454de3ccb752bfe4ba00f94dfad925f0f1c5b56ca6033891",
    "1789f1cf10bb9203718ad8ccf139fbf041ee9fbe97e710eed0282f52b8c38ac3",
    "308d4a723466c68a87b151c8dd85533882595da0d5ffb26fcdd16535ff97dbee",
    "1f9f0d4e51cefeff2de650cadf1f7b15a6081576b6405d7c73ec7235e7a21b2a",
    "99c131c0d5565a6f2c6b04187150f1c949ec73cdf99d3b26cbcf275fb598f6ae",
    "62b0b638712cd4acef5f9810684cd9aefbae2503bc9a0379bd334a0fbac5837c",
    "0221eb41620ed9eaa6ab33092f6e4e365648924d07edd6957b15b7d6a384a069",
    "312882833ec43db8aacc896311529f81bf962227029a2366bc5179fa57a8c8fb",
    "12fcdeb510fa500c3ba80cbfe595c56602a966ebb59772740df8970aa8f6230f",
    "55a3a145aa1c7541736794ece457cb6c52ddc71857fa78c3f841d48485481ff6",
    "602134898d9ef7c38879a6edc117f46aa0174cd4d4e94233fdceb398485c7eb2",
    "f48105e564f1077da3a33e178e850453652eded75cd402013c3da84835f0f611",
    "710dc33ed61ae82db5ac89e11dfae101f168a35bc9f963aa887553c79e74f8b6",
    "692c940abb3d2a99db0c8d75b787080741d670abbf4e732375a3cdb4bb7a4ca2",
    "2ab1ba02f7ca5ec008fd4286405f1f1ef7ae86f144010c8d353e1d2dfbab5431",
    "8a6da04ddea4b8967b1d16faaf7eb2a416fefa7e3d7a25da519a1f08b4a8152b",
    "0801bda8c47e45e91ea33414f16311cb775583a93fda17bd142b04af70e57257",
    "fcf59ea06e00b39e7474fbcab000e31929476783d4035a7ede8817be0a63b3d4",
    "52a27241e700bc9f5e3806a7a3632b693d4a17164ded1998fd93de6b4b37c5b3",
    "06d9ffde0f43aa4f5f1b17cfb6acfac59f46b25ef018e4efdcd859ae71775f04",
]

# Secrets

BUCKET_NAME = ""
EMAIL = ""
PARTICIPANT_ID = ""
AWS_ACCESS_KEY = ""
AWS_SECRET_KEY = ""
