activeLearning_config = {
    "kernel": "exponential",
    "gamma": 10,
    "sigmaN": 1,
    "verbose": False,
    "norm": 1,
    "numKernelCores": 1,
}

AMT_config = {
    "Title":"Adjust the photo retouching parameters",
    "Description":"Please adjust the photo retouching parameters for the best results.",
    "Keywords":"image, tag, picture, tagging, photo",
    "Reward":"0.1",
    "MaxAssignments":7,
    "LifetimeInSeconds":3600*12,
    "AssignmentDurationInSeconds":300,
    "AutoApprovalDelayInSeconds":3600*24,
    "sandbox":False}

AMTAPI_config = {
    "aws_access_key_id": "abc",
    "aws_secret_access_key": "abc",
}

fileServer_config = {
    "sshIP": '000.000.000.0',
    "sshPort": 0000,
    "sshUsername": "abc",
    "sshDirectory": 'abc',
    "httpURL": "https://www.abc",
}
